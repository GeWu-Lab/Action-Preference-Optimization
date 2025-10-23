 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: MIT

import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
import gc
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer
from accelerate import Accelerator
from collections import defaultdict
import os
import time
import json
from tqdm import tqdm
import numpy as np
import random
from typing import Optional, Dict, List, Union, Tuple
import wandb
from utils import (
    formatted_dict,
    pad_to_length,
    masked_mean,
    masked_var,
    entropy_from_logits,
    delete_dicts,
    rowwise_product,
    get_base_model_state_dict_from_peft
)

class DDPRoboticBasicTrainer(object):
    def __init__(self, 
                 config: DictConfig, 
                 train_iterator, 
                 action_tokenizer,
                 eval_iterator, 
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 policy: nn.Module, 
                 device_id,
                 reference_model: Optional[nn.Module] = None,
                 num_skip_batches=0,
                 **kwargs):
        self.action_tokenizer = action_tokenizer
        self.seed = config.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.device_id = device_id
        
        
        self.config = config
        self.run_dir = config.local_run_dir

        self.example_counter = 0
        self.batch_counter = 0

        self.policy = policy
        self.policy_dtype = getattr(torch, config.model.policy_dtype)

        self.reference_model = reference_model
        self.train_iterator = train_iterator
        self.eval_iterator = eval_iterator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_skip_batches = num_skip_batches # when loading from checkpoint

        self.kl_max = config.loss.kl_max
        self.kl_min = config.loss.kl_min
        self.token_distance = config.loss.token_distance

    def get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor):
        """Compute the token-level log probabilities of the given labels under the given logits."""
        # ignoring vocab size, batch size x length should be equal
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        origin_logits = logits.float()
        masked_per_token_logits = origin_logits * loss_mask.unsqueeze(-1)


        distribution_logps = logits.float().log_softmax(-1)
        
        # import pdb;pdb.set_trace()
        if self.token_distance == 1:
            per_token_logps = torch.gather(distribution_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)    
        else:
            soft_labels = labels.unsqueeze(2).repeat(1,1,self.token_distance)
            add_tensor = torch.arange(-(self.token_distance//2), self.token_distance//2+1).to(labels.device)
            soft_labels = soft_labels + add_tensor
            soft_labels = soft_labels.clamp(self.action_tokenizer.action_token_begin_idx, self.action_tokenizer.tokenizer.vocab_size)
            multi_token_logps = torch.gather(distribution_logps, dim=2, index=soft_labels)
            per_token_logps = multi_token_logps.mean(dim=2)
        
        # print("the per_token_logps is:",per_token_logps)
        # print("the loss_mask is:", loss_mask)
        # soft the action prob
        
        return masked_per_token_logits , per_token_logps * loss_mask


    def loss(self,
             batch: Dict,
             policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            batch: batch of data, mapping keys to Tensors
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the losses, one for each example (sif chosen_only or rejected_only, only n/2 losses).
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively, for reporting.
            Note that rejected responses do not factor into the loss, only the reward calculation.
        """
        raise NotImplementedError

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str=None) -> Tuple[torch.FloatTensor, Dict]:
        """Compute the loss and other metrics for the given batch of inputs.
        
        Arg:
            batch: dictionary of inputs for the batch (what is required will vary depending on the trainer)
            mode: one of 'train', 'eval', 'sample'

        Returns:
            A tuple of a scalar loss and a dict of metrics.
        """
        raise NotImplementedError


    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = {}, final_save=True):
        """Save tokenizer, policy model, optimizer, scheduler state to disk."""
        if output_dir is None:
            output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')

        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            metrics['counter'] = self.example_counter
            json.dump(metrics, f)
    
        print(f"Saving state...")

        # torch.save(optimizer_state, os.path.join(output_dir, "optimizer.pt"))
        # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        
        print(f"Saving model...")

    def forward(self, model, batch):
        raise NotImplementedError


from transformers.modeling_outputs import CausalLMOutputWithPast



class HAPOTrainer(DDPRoboticBasicTrainer):

    def loss(self,
        batch: Dict,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
        *args,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the Kahneman-Tversky loss for a batch of policy and reference model log probabilities.

        If generation y ~ p_desirable, we have the 'desirable' loss:
            L(x, y) := 1 - sigmoid(beta * ([log p_policy(y|x) - log p_reference(y|x)] - KL(p_policy || p_reference)))
        If generation y ~ p_undesirable, we have the 'undesirable' loss:
            L(x, y) := 1 - sigmoid(beta * (KL(p_policy || p_reference) - [log p_policy(y|x) - log p_reference(y|x)]))

        The desirable losses are weighed by config.loss.desirable_weight.
        The undesirable losses are weighed by config.loss.undesirable_weight.
        This should be used to address imbalances in the ratio of desirable:undesirable examples respectively.

        The KL term is estimated by matching x with unrelated outputs y', then calculating the average log ratio
        log p_policy(y'|x) - log p_reference(y'|x). Doing so avoids the requirement that there be equal numbers of 
        desirable and undesirable examples in the microbatch.
        """
        
        # import pdb;pdb.set_trace()

        # print("the policy kl logps is:", policy_KL_logps[0], "the reference kl logps is:", reference_KL_logps[0])
        KL_rewards = policy_KL_logps[:,:-1].sum(-1) - reference_KL_logps[:,:-1].sum(-1)
        # take mean of the KL estimates across all devices in this step
        KL = KL_rewards.detach().mean().clamp(min=self.kl_min, max=self.kl_max)

        if policy_chosen_logps.shape[0] != 0:
            chosen_rewards = (policy_chosen_logps.sum(-1) - reference_chosen_logps.sum(-1))
            chosen_losses = self.config.loss.desirable_weight * (1 - F.sigmoid(self.config.loss.beta * (chosen_rewards - KL)))
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(self.policy_dtype).to(self.device_id)
            chosen_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.device_id)
        
        if policy_rejected_logps.shape[0] != 0:
            rejected_rewards = (policy_rejected_logps[:,:-1].sum(-1) - reference_rejected_logps[:,:-1].sum(-1))
            rejected_losses = self.config.loss.undesirable_weight * (1 - F.sigmoid(self.config.loss.beta * (KL - rejected_rewards)))
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(self.policy_dtype).to(self.device_id)
            rejected_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.device_id)

        losses = torch.cat((chosen_losses, rejected_losses), 0)

        return losses, chosen_rewards.detach(), rejected_rewards.detach(), KL.detach()

    def forward(self, model, batch):
        
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"].to(torch.bfloat16),
                labels=batch["labels"],
            )
            
            
            output_logits = output.logits.to(self.policy_dtype)
            labels = batch["labels"]
            mismatch_labels = batch['mismatch_label']
            label_len = labels.shape[-1]
            output_logits = output_logits[:,-label_len:]
            
            all_logits, all_logps = self.get_batch_logps(output_logits, labels)
            mismatch_label_logits, mismatch_label_logps = self.get_batch_logps(output_logits, mismatch_labels)
            

    
        
        is_correct = batch["is_correct"]

        assert all_logps.shape[0] == len(is_correct)
        true_idx = [idx for idx in range(len(is_correct)) if is_correct[idx]]
        false_idx = [idx for idx in range(len(is_correct)) if not is_correct[idx]]
        
        chosen_logps = all_logps[true_idx, ...]

        rejected_logps = all_logps[false_idx, ...]
        
        # shape: B,Token,vocal_size
        chosen_logits = all_logits[true_idx, ...]

        # get l1 loss
        action_logits = output.logits[:, self.policy.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
        action_preds = action_logits.argmax(dim=2)
        true_action_preds = action_preds[true_idx, ...]
        false_action_preds = action_preds[false_idx, ...]
        action_gt = batch["labels"][:, 1:].to(action_preds.device)
        true_action_gt = action_gt[true_idx, ...]
        false_action_gt = action_gt[false_idx, ...]
        
        
        true_mask = true_action_gt > self.action_tokenizer.action_token_begin_idx
        false_mask = false_action_gt > self.action_tokenizer.action_token_begin_idx
        
        true_continuous_actions_pred = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(true_action_preds[true_mask].cpu().numpy())
        )
        true_continuous_actions_gt = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(true_action_gt[true_mask].cpu().numpy())
        )
        
        false_continuous_actions_pred = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(false_action_preds[false_mask].cpu().numpy())
        )
        false_continuous_actions_gt = torch.tensor(
            self.action_tokenizer.decode_token_ids_to_actions(false_action_gt[false_mask].cpu().numpy())
        )

        
        # action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
        true_action_l1_loss = torch.nn.functional.l1_loss(true_continuous_actions_pred, true_continuous_actions_gt, reduction='none')
        false_action_l1_loss = torch.nn.functional.l1_loss(false_continuous_actions_pred, false_continuous_actions_gt, reduction='none')
        true_action_l1_loss = true_action_l1_loss.view(-1, self.config.action_space).mean(dim=-1)
        false_action_l1_loss = false_action_l1_loss.view(-1, self.config.action_space).mean(dim=-1)
        
        
        return chosen_logits, chosen_logps, rejected_logps, mismatch_label_logps, true_action_l1_loss.detach(), false_action_l1_loss.detach()

    def get_batch_metrics(self, batch, mode= "train"):

        metrics = {}
        policy_chosen_logits, policy_chosen_logps, policy_rejected_logps, policy_KL_logps, policy_action_l1_loss,false_policy_action_l1_loss = self.forward(self.policy, batch)
        with torch.no_grad():
            _, reference_chosen_logps, reference_rejected_logps, reference_KL_logps, reference_action_l1_loss, false_reference_action_l1_loss = self.forward(self.reference_model, batch)
        losses, chosen_rewards, rejected_rewards, KL = self.loss(
            batch,
            policy_chosen_logps,
            policy_rejected_logps,
            policy_KL_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_KL_logps,
        )
        combined_rewards = torch.cat((chosen_rewards.detach(), rejected_rewards.detach()), 0)
        combined_statuses = torch.Tensor([1] * len(chosen_rewards) + [0] * len(rejected_rewards)).to(self.device_id)
        
        all_rewards = combined_rewards
        all_statuses = combined_statuses     
        all_KL = KL   
        chosen_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 1 ]
        rejected_rewards_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 0 ]
        
        if mode == "train":

            true_beta = 1
            false_beta = 1
            gamma = 1
            true_error = torch.abs(policy_action_l1_loss)
            batch_level_weight = (true_error + 1e-4) / (true_error.sum() + 1e-4)
            true_weight = (1 - torch.exp(-true_beta * batch_level_weight)) ** gamma

            false_error = torch.abs(false_policy_action_l1_loss)
            batch_level_weight = (false_error + 1e-4) / (false_error.sum() + 1e-4)
            false_weight = (torch.exp(-false_beta * batch_level_weight)) ** gamma


            total_weight = torch.cat((true_weight, false_weight), 0).to(self.device_id)
            total_loss = (losses * total_weight).sum()

        else:
            total_loss = losses.mean()
        
        if mode == "train":
            metrics[f'rewards_{mode}/chosen'] = all_rewards[chosen_rewards_idx].mean()
            metrics[f'rewards_{mode}/rejected'] = all_rewards[rejected_rewards_idx].mean()
            metrics[f'rewards_{mode}/margins'] = torch.Tensor([(all_rewards[chosen_rewards_idx].mean().nan_to_num(0) - all_rewards[rejected_rewards_idx].mean().nan_to_num(0)).item()])
            metrics[f'rewards_{mode}/KL_estimate'] = all_KL
            metrics[f'policy_action_l1_loss/{mode}'] = policy_action_l1_loss.mean().detach().mean()
            metrics[f'reference_action_l1_loss/{mode}'] = reference_action_l1_loss.mean().detach().mean()
            metrics[f'l1_difference/{mode}'] = (reference_action_l1_loss - policy_action_l1_loss).mean().detach().mean()
            metrics[f'loss/{mode}'] = losses.mean().detach().mean()
            metrics[f'weight_loss/{mode}'] = (losses * total_weight).mean().detach().mean()
        else:
            metrics[f'policy_action_l1_loss/{mode}'] = policy_action_l1_loss.mean().detach().mean()
            metrics[f'reference_action_l1_loss/{mode}'] = reference_action_l1_loss.mean().detach().mean()
            metrics[f'l1_difference/{mode}'] = (reference_action_l1_loss - policy_action_l1_loss).mean().detach().mean()
            metrics[f'loss/{mode}'] = losses.mean().detach().mean()

        del policy_chosen_logps, policy_rejected_logps, policy_KL_logps, reference_chosen_logps, reference_rejected_logps, reference_KL_logps
        del combined_rewards, combined_statuses, all_rewards, all_statuses, chosen_rewards_idx, rejected_rewards_idx, all_KL


        return total_loss, metrics

