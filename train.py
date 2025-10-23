 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: MIT

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate import PartialState
import hydra
from omegaconf import OmegaConf, DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import os
from typing import Optional, Set
import json
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.distributed as dist
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from tokenizer.action_tokenizer import ActionTokenizer
from tokenizer.prompt import PurePromptBuilder
from dataset.collator import PaddedCollatorForActionPrediction
from dataset.robotic_dataloader import NewRoboticDataLoader
from dataset.eval_mimicgen import EvalMimicgen
from torch.utils.data import DistributedSampler
from dataset.sampler import BalancedInteractionDistributedSampler
import dataset
import trainer
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import wandb
import tqdm
import numpy as np
from dataset.balance_hapo_dataset import BalanceHAPODataset
from trainer.ddp_robotic_trainer import HAPOTrainer
import time
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
def main(config: DictConfig):

    """Main entry point for training. Validates config, creates/initializes model(s), and starts training."""
    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    
    set_seed(config.seed)

    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    


    vla_path = config.model.name_or_path
    base_path = config.model.base_path
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    train_dataset = eval(config.dataset._target_)(
        action_tokenizer,
        processor.tokenizer,
        processor.image_processor.apply_transform,
        PurePromptBuilder,
        use_precomputed_statistics = True,
        statistic_path = vla_path,
        **config.dataset.kwargs,
    )

    if distributed_state.is_main_process:
        
        base_dir = os.path.join(config.local_run_dir, config.exp_name,str(config.lr), DATE_TIME)
        os.makedirs(base_dir, exist_ok=True)
        print("Making experiment directory", config.local_run_dir)
        
        adapter_save_dir = os.path.join(base_dir,"adapter_dir")
        model_save_dir = os.path.join(base_dir, "model_dir")
        
        os.makedirs(adapter_save_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)
        
        config_path = os.path.join(base_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config, f)

        print('=' * 80)
        print(f'Writing to {base_dir}')
        print('=' * 80)
        
        if config.wandb.enabled:
            os.environ['WANDB_CACHE_DIR'] = config.cache_dir
            wandb.init(
                entity=config.wandb.entity,
                project=config.wandb.project,
                config=OmegaConf.to_container(config),
                dir=config.cache_dir,
                name=config.exp_name,
            )
        train_dataset.save_data_statistics(adapter_save_dir)
        
    dist.barrier()
    
    eval_dataset = eval(config.eval_dataset._target_)(
        action_tokenizer,
        processor.tokenizer,
        processor.image_processor.apply_transform,
        PurePromptBuilder,
        use_precomputed_statistics = True,
        statistic_path = vla_path,
        **config.eval_dataset.kwargs
    )
    
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    sampler = BalancedInteractionDistributedSampler(
            train_dataset,
            correct_ratio = config.correct_ratio,
            interaction_ratio = config.interaction_ratio, 
            batch_size= config.batch_size,
            num_replicas=dist.get_world_size(),  
            rank=dist.get_rank(),  
            shuffle=True)
    
    
    test_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=dist.get_world_size(),  
            rank=dist.get_rank(),  
            shuffle=True)

    train_dataloader = NewRoboticDataLoader(
                 train_dataset,
                 collator,
                 batch_size = config.batch_size,
                 max_length = 512,
                 max_prompt_length = 128,
                 n_epochs = config.n_epochs,
                 seed = 0,
                 num_workers=4,
                 sampler = sampler
    )

    eval_dataloader = NewRoboticDataLoader(
                 eval_dataset,
                 collator,
                 batch_size = config.eval_batch_size,
                 max_length = 512,
                 max_prompt_length = 128,
                 n_epochs = 1,
                 seed = 0,
                 num_workers=4,
                 sampler = test_sampler
    )
    
    
    print(f'Loading train model ')

    base_model = AutoModelForVision2Seq.from_pretrained(
        base_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


    model = PeftModel.from_pretrained(base_model, vla_path)
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    model.print_trainable_parameters()
        
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    

    reference_base_model = AutoModelForVision2Seq.from_pretrained(
        base_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    reference_model = PeftModel.from_pretrained(reference_base_model, vla_path)

    reference_model = reference_model.to(device_id)

    
    # Loading optimizer, scheduler
    print("Creating optimizer and scheduler")
    optimizer = torch.optim.Adam(trainable_params, lr=config.lr)
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_steps)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=train_dataloader.num_training_steps - config.warmup_steps, eta_min=0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[config.warmup_steps])  
    
    num_skip_batches = 0
    
    trainer = eval(config.trainer._target_)(
        config = config,
        action_tokenizer = action_tokenizer,
        train_iterator = train_dataloader, 
        eval_iterator = eval_dataloader, 
        optimizer = optimizer,
        scheduler = scheduler,
        policy = model, 
        device_id = device_id,
        reference_model = reference_model,
        num_skip_batches=0,
    )

    eval_number = 0
    reference_model.eval()
    total_steps = train_dataloader.num_training_steps
    with tqdm.tqdm(total=total_steps) as progress_bar:
        trainer.policy.train()
        for step, batch in enumerate(train_dataloader):
            mode = 'train'

            batch = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            total_loss, metrics = trainer.get_batch_metrics(batch)

            norm_loss = total_loss / config.grad_accumulation_steps

            norm_loss.backward()

            
            gradient_step_idx = step // config.grad_accumulation_steps

            if gradient_step_idx % 10 == 0 and distributed_state.is_main_process:
                print("metric is:", metrics)
                save_json_dir = os.path.join(base_dir, "metric_json")
                os.makedirs(save_json_dir, exist_ok=True)
                save_file_path = os.path.join(save_json_dir, f"metric_step_{gradient_step_idx}.json")
                for key in metrics.keys():
                    metrics[key] = metrics[key].item()
                with open(save_file_path, "w") as f:
                    json.dump(metrics, f)
                if config.wandb.enabled:
                    wandb.log(
                    metrics,
                    step = gradient_step_idx
                    )
                    
            progress_bar.update(1) 

            if gradient_step_idx % config.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


            if gradient_step_idx % config.save_steps == 0 and gradient_step_idx > 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                    
                    processor.save_pretrained(adapter_save_dir)
                    model.module.save_pretrained(adapter_save_dir)
                    print("finish")
                    
                dist.barrier()
                
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {step}")
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        base_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    print("merging model")
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_save_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    merged_vla.save_pretrained(model_save_dir)
                    
                dist.barrier()


    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    

@hydra.main(version_base=None, config_path="config", config_name="balance_config")
def hydra_main(config: DictConfig):
    main(config)
    
if __name__ == "__main__":
    hydra_main()