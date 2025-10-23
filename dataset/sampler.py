 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: MIT
import math
import torch
from torch.utils.data import Sampler,DistributedSampler
import numpy as np

class BalancedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, correct_ratio, batch_size, num_replicas=None, rank=None, shuffle=True, seed=0):

        self.dataset = dataset
        
        
        self.correct_data_idx_list, self.incorrect_data_idx_list = dataset.get_balance_list()
        self.batch_size = batch_size
        
        self.correct_ratio = correct_ratio
        self.incorrect_ratio = 1.0 - correct_ratio

        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank or torch.distributed.get_rank()
        self.shuffle = shuffle
        self.seed = seed

        self.correct_size = len(self.correct_data_idx_list)
        self.incorrect_size = len(self.incorrect_data_idx_list)

        print("the incorrect_size is:", self.incorrect_size)
        print("the correct size is:", self.correct_size)


        self.total_size = self.correct_size + self.incorrect_size
        self.num_samples = math.ceil(self.total_size / self.num_replicas)


        self.correct_sample_count = round(self.num_samples * self.correct_ratio)
        self.incorrect_sample_count = self.num_samples - self.correct_sample_count


        self.correct_select_num = int(self.correct_ratio * self.batch_size)
        self.incorrect_select_num = int((1 - self.correct_ratio) * self.batch_size)
        self.epoch = 0

        print("num_replicas is", num_replicas)

        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        
    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)


        correct_indices_list = np.array(self.correct_data_idx_list)
        incorrect_indices_list = np.array(self.incorrect_data_idx_list)


        if self.shuffle:
            correct_indices = torch.randperm(self.correct_size, generator=g).tolist()
            incorrect_indices = torch.randperm(self.incorrect_size, generator=g).tolist()
            
            correct_indices_list = correct_indices_list[correct_indices]
            incorrect_indices_list = incorrect_indices_list[incorrect_indices]



        use_correct_indices_list = list(correct_indices_list[self.rank:self.correct_size:self.num_replicas])
        use_incorrect_indices_list = list(incorrect_indices_list[self.rank:self.incorrect_size:self.num_replicas])
        

        temp_incorrect_indices_list = use_incorrect_indices_list.copy()
        
        
        batch_indices = []
        while len(use_correct_indices_list) != 0:
            # print("the correct_select_num is:", self.correct_select_num, "the incorrect_select_num is:", self.incorrect_select_num)
            batch_correct = use_correct_indices_list[:self.correct_select_num]
            use_correct_indices_list = use_correct_indices_list[self.correct_select_num:]
            
            batch_incorrect = temp_incorrect_indices_list[:self.incorrect_select_num]
            temp_incorrect_indices_list = temp_incorrect_indices_list[self.incorrect_select_num:]
            
            if len(temp_incorrect_indices_list) == 0:
                temp_incorrect_indices_list = use_incorrect_indices_list.copy()

            batch = batch_correct + batch_incorrect
            if len(batch) < self.batch_size:
                continue
            batch_indices.extend(batch)


        return iter(batch_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class BalancedInteractionDistributedSampler(DistributedSampler):
    def __init__(self, dataset, correct_ratio, interaction_ratio, batch_size, num_replicas=None, rank=None, shuffle=True, seed=0):

        self.dataset = dataset
        
        
        self.correct_data_idx_list, self.interaction_data_idx_list ,self.incorrect_data_idx_list = dataset.get_balance_list()
        self.batch_size = batch_size
        
        self.correct_ratio = correct_ratio
        self.interaction_ratio = interaction_ratio
        self.incorrect_ratio = 1.0 - correct_ratio - interaction_ratio

        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank or torch.distributed.get_rank()
        self.shuffle = shuffle
        self.seed = seed

        self.correct_size = len(self.correct_data_idx_list)
        self.incorrect_size = len(self.incorrect_data_idx_list)
        self.interaction_size = len(self.interaction_data_idx_list)
    

        self.total_size = self.correct_size + self.incorrect_size + self.interaction_size
        # self.num_samples = math.ceil(self.total_size / self.num_replicas)

        self.num_samples = math.ceil(self.interaction_size / self.interaction_ratio / self.num_replicas)

        self.correct_select_num = round(self.correct_ratio * self.batch_size)
        self.interaction_select_num = round(self.interaction_ratio * self.batch_size)
        self.incorrect_select_num = round(self.incorrect_ratio * self.batch_size)
        self.epoch = 0

        print("the need correct num is:", self.correct_select_num)
        print("the need interaction_select_num num is:", self.interaction_select_num)
        print("the need incorrect number is:", self.incorrect_select_num)
        
        print("the correct size is:", self.correct_size)
        print("the incorrect size is:", self.incorrect_size)
        print("the interaction_size is:", self.interaction_size)
        print("num_replicas is", num_replicas)

        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        
    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)


        correct_indices_list = np.array(self.correct_data_idx_list)
        incorrect_indices_list = np.array(self.incorrect_data_idx_list)
        interaction_indices_list =  np.array(self.interaction_data_idx_list)


        if self.shuffle:
            correct_indices = torch.randperm(self.correct_size, generator=g).tolist()
            incorrect_indices = torch.randperm(self.incorrect_size, generator=g).tolist()
            interaction_indices = torch.randperm(self.interaction_size, generator=g).tolist()

            correct_indices_list = correct_indices_list[correct_indices]
            incorrect_indices_list = incorrect_indices_list[incorrect_indices]
            interaction_indices_list = interaction_indices_list[interaction_indices]



        use_correct_indices_list = list(correct_indices_list[self.rank:self.correct_size:self.num_replicas])
        use_incorrect_indices_list = list(incorrect_indices_list[self.rank:self.incorrect_size:self.num_replicas])
        use_interaction_indices_list = list(interaction_indices_list[self.rank: self.interaction_size:self.num_replicas])

        # Temporary lists for cycling through correct and incorrect indices
        temp_correct_indices_list = use_correct_indices_list.copy()
        temp_incorrect_indices_list = use_incorrect_indices_list.copy()

        # List to store the final batch indices for this epoch for this replica
        batch_indices = []

        # Check if interaction list is empty from the start for this replica
        if not use_interaction_indices_list:
             pass # If no interaction data, this replica yields nothing.


        # Iterate as long as there are interaction samples left for this replica
        while len(use_interaction_indices_list) > 0:
            # Determine how many interaction samples to take in this step
            actual_interaction_num = min(self.interaction_select_num, len(use_interaction_indices_list))
            if actual_interaction_num <= 0: # Safety break if interaction_select_num is 0 or negative
                break

            batch_interaction = use_interaction_indices_list[:actual_interaction_num]
            # Consume the interaction samples from the main list for this replica
            use_interaction_indices_list = use_interaction_indices_list[actual_interaction_num:]


            # --- Sample Correct Data ---
            batch_correct = []
            needed_correct = self.correct_select_num
            # Only sample if needed_correct > 0 and there are correct indices available for this replica
            if needed_correct > 0 and use_correct_indices_list:
                collected_correct = 0
                # Keep collecting until we have enough correct samples for this batch step
                while collected_correct < needed_correct:
                    # If the temporary list is exhausted, replenish it from the replica's full list
                    if not temp_correct_indices_list:
                        temp_correct_indices_list = use_correct_indices_list.copy()


                    # Determine how many samples to take in this sub-step
                    take_now = min(needed_correct - collected_correct, len(temp_correct_indices_list))
                    batch_correct.extend(temp_correct_indices_list[:take_now])
                    # Consume samples from the temporary list
                    temp_correct_indices_list = temp_correct_indices_list[take_now:]
                    collected_correct += take_now


            # --- Sample Incorrect Data ---
            batch_incorrect = []
            needed_incorrect = self.incorrect_select_num
            # Only sample if needed_incorrect > 0 and there are incorrect indices available for this replica
            if needed_incorrect > 0 and use_incorrect_indices_list:
                collected_incorrect = 0
                # Keep collecting until we have enough incorrect samples for this batch step
                while collected_incorrect < needed_incorrect:
                     # If the temporary list is exhausted, replenish it
                    if not temp_incorrect_indices_list:
                        temp_incorrect_indices_list = use_incorrect_indices_list.copy()
                        # Optional: Reshuffle here
                        # if self.shuffle:
                        #    np.random.shuffle(temp_incorrect_indices_list)

                    # Determine how many samples to take in this sub-step
                    take_now = min(needed_incorrect - collected_incorrect, len(temp_incorrect_indices_list))
                    batch_incorrect.extend(temp_incorrect_indices_list[:take_now])
                     # Consume samples from the temporary list
                    temp_incorrect_indices_list = temp_incorrect_indices_list[take_now:]
                    collected_incorrect += take_now


            # Combine the batch parts
            # The order might matter depending on how the dataset's __getitem__ handles mixed indices
            batch = batch_correct + batch_incorrect + batch_interaction

            # Ensure the generated batch isn't empty before extending
            if batch:
                 batch_indices.extend(batch)
            # Removed the check: if len(batch) < self.batch_size: continue


        # Return iterator over the collected indices for this replica
        return iter(batch_indices)

    def __len__(self):
        # Standard DistributedSampler length: samples per replica
        # This tells DataLoader roughly how many samples to expect per epoch per replica.
        # The actual number yielded by __iter__ is determined by the interaction data size for this replica.
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch