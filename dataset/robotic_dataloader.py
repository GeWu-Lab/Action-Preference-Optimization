 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: MIT
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from typing import Dict, List, Optional, Tuple
import torch
from typing import Callable, Dict, Sequence, Tuple
import random
from torch.utils.data import DataLoader


class RoboticDataLoader:
    def __init__(self,
                 dataset,
                 collate_fn,
                 batch_size,
                 max_length: int = 512,
                 max_prompt_length: int = 128,
                 max_prompt_count: int = None,
                 n_epochs: Optional[int] = None,
                 n_examples: Optional[int] = None,
                 seed: int = 0,
                 **kwargs
                 ):
        self.dataset = dataset
        torch.manual_seed(seed)
        self.seed = seed
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_prompt_count = max_prompt_count
        self.kwargs = kwargs
        assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
        self.n_epochs = n_epochs
        self.epoch_idx = 0
        self.n_examples = n_examples
        self.collate_fn = collate_fn
        
        self.num_training_steps = self.get_num_training_steps()
    
    def collate(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict:
        return self.collate_fn(instances)
        
        
    def get_num_training_steps(self):
        return len(self.dataset) // self.batch_size * self.n_epochs
    
    def __iter__(self):
        data_idx_list = list(range(len(self.dataset)))
        epoch_idx = 0
        example_idx = 0
        done = False
        
        while True:
            if done: break
            random.Random(self.seed + epoch_idx).shuffle(data_idx_list)
            batch = []
            example_queue = []
            for data_idx in data_idx_list:
                example = self.dataset[data_idx]
                example_queue.append(example)
                if len(example_queue) == self.batch_size:
                    batch = self.collate(example_queue)
                    example_queue = []
                    yield batch
                    example_idx += self.batch_size
                    if self.n_examples is not None and example_idx >= self.n_examples:
                        done = True
                        break

            # if batch != []:
            #     yield self.collate(batch) # flush
            #     batch = []
            
            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break
        
        
        
        
class NewRoboticDataLoader:
    def __init__(self,
                 dataset,
                 collate_fn,
                 batch_size,
                 max_length: int = 512,
                 max_prompt_length: int = 128,
                 max_prompt_count: int = None,
                 n_epochs: Optional[int] = None,
                 n_examples: Optional[int] = None,
                 seed: int = 0,
                 num_workers: int = 0,
                 sampler = None,
                 **kwargs):
        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_prompt_count = max_prompt_count
        self.n_epochs = n_epochs
        self.n_examples = n_examples
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.sampler = sampler

        self.wrapped_dataset = dataset
        self.data_loader = DataLoader(
            self.wrapped_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Shuffle at DataLoader level
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            sampler = self.sampler,
            drop_last=True
        )
        self.num_training_steps = self.get_num_training_steps()
        
    def get_num_training_steps(self):
        return len(self.data_loader) * self.n_epochs
    
    def worker_init_fn(self, worker_id):
        # Ensure each worker has a unique random seed
        random.seed(self.seed + worker_id)
    
    
    def __iter__(self):
        for epoch_idx in range(self.n_epochs or 1):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch_idx)
                
            for batch in self.data_loader:
                yield batch

