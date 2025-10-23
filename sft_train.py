 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: MIT

import os
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import time
import json
import dataclasses

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from dataset.sft_mimicgen_dataset import MimicGenDataset

# Global constants
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    """Configuration class for OpenVLA fine-tuning parameters."""
    # fmt: off
    # Model Configuration
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Dataset Configuration
    data_root_dir: str = "robotic_data/mimicgen"                    # Path to dataset directory      
    dataset_name: str = "mimicgen"                                  # Name of fine-tuning dataset
    data_list_names: str = "coffee_d0.json"                        # Data list file names
    image_shape: int = 224                                          # Input image size
    demo_number: int = 300                                          # Number of demonstrations
    
    # Directory Configuration
    run_root_dir: Path = Path("runs")                               # Directory for logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights

    # Training Hyperparameters
    batch_size: int = 16                                            # Training batch size
    max_epoch: int = 3                                              # Maximum training epochs
    save_steps: int = 5000                                          # Checkpoint saving interval
    learning_rate: float = 5e-4                                     # Learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Enable image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size
    save_latest_checkpoint_only: bool = False                       # Save only latest checkpoint

    # LoRA Configuration
    use_lora: bool = True                                           # Enable LoRA fine-tuning
    lora_rank: int = 32                                             # LoRA rank
    lora_dropout: float = 0.0                                       # LoRA dropout rate
    use_quantization: bool = False                                  # Enable 4-bit quantization

    # Logging Configuration
    wandb_project: str = "xwk_openvla"                              # W&B project name
    wandb_entity: str = "libero_spatial"                            # W&B entity name
    wandb_name: str = "pytorch_libero"                              # W&B run name
    # fmt: on

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary with proper Path handling."""
        config_dict = asdict(self)
        # Convert Path objects to strings
        config_dict["run_root_dir"] = str(config_dict["run_root_dir"])
        config_dict["adapter_tmp_dir"] = str(config_dict["adapter_tmp_dir"])
        return config_dict


class MetricsTracker:
    """Helper class to track and compute training metrics."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.recent_losses = deque(maxlen=window_size)
        self.recent_action_accuracies = deque(maxlen=window_size)
        self.recent_l1_losses = deque(maxlen=window_size)
    
    def update(self, loss: float, action_accuracy: float, l1_loss: float) -> None:
        """Update metrics with new values."""
        self.recent_losses.append(loss)
        self.recent_action_accuracies.append(action_accuracy)
        self.recent_l1_losses.append(l1_loss)
    
    def get_smoothed_metrics(self) -> Dict[str, float]:
        """Get smoothed metrics over the recent window."""
        return {
            "train_loss": sum(self.recent_losses) / len(self.recent_losses) if self.recent_losses else 0.0,
            "action_accuracy": sum(self.recent_action_accuracies) / len(self.recent_action_accuracies) if self.recent_action_accuracies else 0.0,
            "l1_loss": sum(self.recent_l1_losses) / len(self.recent_l1_losses) if self.recent_l1_losses else 0.0,
        }


def setup_model_and_processor(cfg: FinetuneConfig) -> Tuple[Any, Any]:
    """Setup and configure the VLA model and processor."""
    # Quantization configuration
    quantization_config = None
    if cfg.use_quantization:
        if not cfg.use_lora:
            raise ValueError("Quantized training only supported for LoRA fine-tuning!")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16, 
            bnb_4bit_quant_type="nf4"
        )
    
    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load processor and model
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    
    return vla, processor


def setup_lora_and_ddp(vla: Any, cfg: FinetuneConfig, device_id: int) -> Any:
    """Setup LoRA configuration and DDP wrapper."""
    # Device placement and quantization preparation
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # LoRA configuration
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap in DDP for multi-GPU training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    
    return vla


def compute_action_metrics(
    output: CausalLMOutputWithPast, 
    batch: Dict[str, torch.Tensor], 
    action_tokenizer: ActionTokenizer,
    vla: Any,
    device_id: int
) -> Tuple[float, float]:
    """Compute action accuracy and L1 loss metrics."""
    # Extract action predictions
    num_patches = vla.module.vision_backbone.featurizer.patch_embed.num_patches
    action_logits = output.logits[:, num_patches:-1]
    action_preds = action_logits.argmax(dim=2)
    action_gt = batch["labels"][:, 1:].to(action_preds.device)
    mask = action_gt > action_tokenizer.action_token_begin_idx

    # Compute accuracy
    correct_preds = (action_preds == action_gt) & mask
    action_accuracy = (correct_preds.sum().float() / mask.sum().float()).item() if mask.sum() > 0 else 0.0
    
    # Compute L1 loss on continuous actions
    if mask.sum() > 0:
        try:
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy()),
                dtype=torch.float32
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()),
                dtype=torch.float32
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt).item()
        except Exception as e:
            print(f"Warning: Failed to compute L1 loss: {e}")
            action_l1_loss = 0.0
    else:
        action_l1_loss = 0.0
    
    return action_accuracy, action_l1_loss


def save_checkpoint(
    cfg: FinetuneConfig,
    vla: Any,
    processor: Any,
    run_dir: Path,
    adapter_dir: Path,
    gradient_step_idx: int,
    distributed_state: PartialState
) -> None:
    """Save model checkpoint."""
    if distributed_state.is_main_process:
        print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
        
        # Determine save directory
        save_dir = adapter_dir if cfg.use_lora else run_dir
        
        # Save processor and model weights
        processor.save_pretrained(run_dir)
        processor.save_pretrained(save_dir)
        vla.module.save_pretrained(save_dir)
    
    # Synchronize across processes
    dist.barrier()


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """Main fine-tuning function for OpenVLA model."""
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    
    # Validate GPU availability and setup distributed context
    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning requires at least one GPU!")
    
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    
    # Configure experiment ID and directories
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
        f"+number-{cfg.demo_number}"
    )
    
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.image_aug:
        exp_id += "--image_aug"
    
    exp_id += f"--{cfg.wandb_name}"
    
    # Setup directories
    run_dir = cfg.run_root_dir / exp_id / "model_dir"
    adapter_dir = cfg.run_root_dir / exp_id / "adapter_dir"
    
    # Initialize logging
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft_{cfg.wandb_name}")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        
        # Save configuration
        with open(adapter_dir / "setting.json", "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
    
    # Setup model and processor
    vla, processor = setup_model_and_processor(cfg)
    vla = setup_lora_and_ddp(vla, cfg, device_id)
    
    # Setup optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    
    # Setup action tokenizer and dataset
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    # Ensure data statistics directory exists
    data_statistics_dir = os.path.join(cfg.data_root_dir, str(cfg.demo_number))
    os.makedirs(data_statistics_dir, exist_ok=True)
    
    # Create dataset
    prompt_builder = PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder
    dataset = MimicGenDataset(
        action_tokenizer=action_tokenizer,
        tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=prompt_builder,
        data_root_dir=cfg.data_root_dir,
        data_name=cfg.dataset_name,
        data_list_names=cfg.data_list_names,
        image_shape=cfg.image_shape,
        demo_number=cfg.demo_number
    )
    
    # Save dataset statistics
    if distributed_state.is_main_process:
        dataset.save_data_statistics(adapter_dir)
    
    # Setup data loader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, 
        processor.tokenizer.pad_token_id, 
        padding_side="right"
    )
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(window_size=cfg.grad_accumulation_steps)
    
    # Training loop
    batch_idx = 0
    total_steps = cfg.max_epoch * len(dataloader)
    
    print(f"Starting training for {cfg.max_epoch} epochs, {total_steps} total steps")
    
    with tqdm.tqdm(total=total_steps, desc="Training Progress") as progress_bar:
        for epoch in range(cfg.max_epoch):
            vla.train()
            optimizer.zero_grad()
            
            for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.max_epoch}", leave=False):
                # Forward pass
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output: CausalLMOutputWithPast = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                        labels=batch["labels"],
                    )
                    loss = output.loss
                
                # Normalize loss for gradient accumulation
                normalized_loss = loss / cfg.grad_accumulation_steps
                normalized_loss.backward()
                
                # Compute metrics
                action_accuracy, action_l1_loss = compute_action_metrics(
                    output, batch, action_tokenizer, vla, device_id
                )
                
                # Update metrics tracker
                metrics_tracker.update(loss.item(), action_accuracy, action_l1_loss)
                progress_bar.update(1)
                
                # Compute gradient step index
                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
                
                # Log metrics periodically
                if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                    smoothed_metrics = metrics_tracker.get_smoothed_metrics()
                    wandb.log(smoothed_metrics, step=gradient_step_idx)
                    print(f"Step {gradient_step_idx}: {smoothed_metrics}")
                
                # Optimizer step
                batch_idx += 1
                if batch_idx % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Save checkpoint
                if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                    save_checkpoint(cfg, vla, processor, run_dir, adapter_dir, gradient_step_idx, distributed_state)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    finetune()