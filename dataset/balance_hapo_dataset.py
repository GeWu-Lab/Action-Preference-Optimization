 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: MIT
from torch.utils.data import Dataset, IterableDataset,DataLoader
import pickle
from pathlib import Path
import numpy as np
import time
import json
import h5py
import os
import imageio
import torchvision.transforms as transforms
import torch
from torchvision import transforms
from PIL import Image
IGNORE_INDEX = -100

from tokenizer.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from typing import Any, Dict, Tuple, Type
from tokenizer.prompt import PromptBuilder

from transformers import PreTrainedTokenizerBase
from typing import Any, Dict, Tuple, Type

class BalanceHAPODataset(Dataset):
    def __init__(self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform,
        prompt_builder_fn: Type[PromptBuilder],
        data_root_dir: str,
        interaction_root_dir: str,
        wrong_root_dir= None,
        data_name: str = "coffee_d0",
        data_list_names: str = "task_name.json",
        interaction_list_names: str = "interaction_data_name.json",
        wrong_list_names: str = "interaction_data_list.json",
        image_shape: int = 256,
        demo_number: int = 50,
        predict_stop_token: bool = True,
        is_validation: bool = False,
        use_precomputed_statistics: bool = True,
        statistic_path = None,
        previous_K=5
        ):
        super().__init__()
        self.TASK_MAPPING = {
            "coffee_d0": ["Coffee_D0", "make coffee"],
            "coffee_d1": ["Coffee_D1", "make coffee"],
            "stack_d0": ["Stack_D0", "stack the red block on top of the green block"],
            "stack_d1": ["Stack_D1", "stack the red block on top of the green block"],
            "stack_three_d0": ["StackThree_D0", "stack the blocks in the order of blue, red, and green from top to bottom"],
            "stack_three_d1": ["StackThree_D1", "stack the blocks in the order of blue, red, and green from top to bottom"],
            "threading_d0": ["Threading_D0", "insert the needle into the needle hole"],
            "square_d0": ["Square_D0", "slide the square block onto the wooden stick"],
            "square_d1": ["Square_D1", "slide the square block onto the wooden stick"],
            "square_d2": ["Square_D2", "slide the square block onto the wooden stick"],
            "three_piece_assembly_d0":["ThreePieceAssembly_D0","stack the three pieces"],
            "three_piece_assembly_d1":["ThreePieceAssembly_D1","stack the three pieces"],
        }
        
        self.predict_stop_token = predict_stop_token
        self.image_shape = image_shape
        self.data_root_dir = data_root_dir
        self.is_validation = is_validation
        data_list = json.load(open(os.path.join(data_root_dir,data_list_names)))
        interaction_data_list = json.load(open(os.path.join(interaction_root_dir,interaction_list_names)))
        self.demo_number = demo_number
        
        if wrong_root_dir is not None:
            wrong_data_list = json.load(open(os.path.join(wrong_root_dir,wrong_list_names)))
        
        
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.statistic_path = statistic_path
        self.data_name = data_name
        
        self.augment_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 0.9), ratio=(1.0, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=1),
            transforms.RandomApply([transforms.ColorJitter(contrast=(0.8, 1.2))], p=1),
            transforms.RandomApply([transforms.ColorJitter(saturation=(0.8, 1.2))], p=1),
            transforms.RandomApply([transforms.ColorJitter(hue=0.05)], p=1),
        ])

        data_statistic = {
                "action" : [],
                "num_trajectories": 0,
                "num_transitions": 0
        }
        
        data_corresponding = []
        total_idx = 0
        st_time = time.time()
        
        self.task_name_h5py_mapping = {
            
        }
        self.interaction_name_mapping = {
            
        }
        self.correct_data_idx_list = []
        self.incorrect_data_idx_list = []
        self.interaction_data_idx_list = []
        self.total_idxs = 0
        
        for robotic_data_name in data_list:
            annotation_path = os.path.join(data_root_dir , robotic_data_name)
            task_name = robotic_data_name.split(".")[0]
            # print("annotation_path is:", annotation_path)
            robotic_data = h5py.File(annotation_path, "r")
            self.task_name_h5py_mapping[task_name] = robotic_data
            task_language_prompt = self.TASK_MAPPING[task_name][1]
            print("the task_language_prompt is:", task_language_prompt)
            print("the demo_number is:", self.demo_number)
            for demo_number_id, demo in enumerate(robotic_data.keys()):
                timestep_len = len(robotic_data[demo].keys()) - 1
                for timestep in range(timestep_len):
                    action = robotic_data[demo][f'timestep_{timestep}']['action'][:]
                    if max(action) > 1 or min(action) < -1:
                        continue
                    if sum(abs(action[:-1])) < 5e-2:
                        continue
                    data_statistic['action'].append(action)
                    data_corresponding.append({
                    "action": action,
                    "timestep": timestep,
                    "language": task_language_prompt,
                    "task_name": task_name,
                    "demo_id": demo,
                    "is_expert": True
                    })
                    self.correct_data_idx_list.append(self.total_idxs)
                    self.total_idxs += 1
                    
                data_statistic['num_transitions'] += timestep_len
                data_statistic['num_trajectories'] += 1
                
                if demo_number_id >= self.demo_number:
                    break
            
        for interaction_data_name in interaction_data_list:
            interaction_data_path = os.path.join(interaction_root_dir, interaction_data_name)
            interaction_data = h5py.File(interaction_data_path, "r")['data']
            interaction_data_name = interaction_data_name.split(".")[0]
            task_language_prompt = interaction_data['task_description'][()].decode('utf-8')
            # task_language_prompt = task_language_prompt.replace("_"," ")
            self.interaction_name_mapping[interaction_data_name] = interaction_data
            # for demo_number_id, demo in enumerate(interaction_data.keys()):
                # skip the task language
            timestep_len = len(interaction_data.keys()) - 1
            is_first_human = True
            for timestep in range(timestep_len):
                action = interaction_data[f'timestep_{timestep}']['action'][:]
                if max(action) > 1 or min(action) < -1:
                    continue
                if sum(abs(action[:-1])) < 5e-2:
                    continue
                is_human = interaction_data[f'timestep_{timestep}']['is_human'][()]
                if is_first_human and is_human == 1:
                    is_first_human = False
                    # change the previous data to false
                    for k in range(1,previous_K + 1):
                        # print(data_corresponding[-k]['is_human'])
                        data_corresponding[-k]['is_human'] = 0
                        data_idx = self.correct_data_idx_list.pop()
                        self.incorrect_data_idx_list.append(data_idx)
                if is_human == 2:
                    is_first_human = True

                if is_human == 2:
                    self.correct_data_idx_list.append(self.total_idxs)
                elif is_human == 1:
                    self.interaction_data_idx_list.append(self.total_idxs)
                
                if is_human == 0:
                    continue
                
                data_statistic['action'].append(action)
                data_corresponding.append({
                "action": action,
                "timestep": timestep,
                "language": task_language_prompt,
                "task_name": interaction_data_name,
                "demo_id": -1,
                "is_expert": False,
                "is_human": is_human
                })
                self.total_idxs += 1


        
        self.data_corresponding = data_corresponding
                        
        print("the length is:", len(data_corresponding))
        print("the data analysis time is:", time.time() - st_time)
        actions = [d['action'] for d in data_corresponding]
        
        action_mean = np.mean(actions, axis=0)
        action_std = np.std(actions, axis=0)
        action_min = np.min(actions, axis=0)
        action_max = np.max(actions, axis=0)
        q01 = np.quantile(actions, 0.01, axis=0)
        q99 = np.quantile(actions, 0.99, axis=0)
        

        if use_precomputed_statistics:
            # use the train dataset statistics
            data_statistics_path = os.path.join(self.statistic_path ,"statistic.json")
            record_data = json.load(open(data_statistics_path))
            self.data_statistics = {
                self.data_name:{key: np.array(record_data[self.data_name]['action'][key]) for key in record_data[self.data_name]['action'].keys()}
            }
            print("load finish, the data statistics is:", self.data_statistics)
        else:
            self.data_statistics = {
                self.data_name:{
                "q01": q01,
                "q99": q99,
                "mean": action_mean,
                "std": action_std,
                "min": action_min,
                "max": action_max,
                'mask': np.array([
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False
                    ])
            }}
            # self.save_data_statistics()
    
    def save_data_statistics(self, data_path):
        statistic_path = os.path.join(data_path, "statistic.json")
        save_data_statistics = {
            self.data_name:{
                "action":{
                key: self.data_statistics[self.data_name][key].tolist() for key in self.data_statistics[self.data_name].keys()
            }}
        }
        
        with open(statistic_path ,"w") as f:
            json.dump(save_data_statistics,f)
            
        
        
    def get_balance_list(self):
        return self.correct_data_idx_list, self.interaction_data_idx_list ,self.incorrect_data_idx_list
    
    def __len__(self):
        return len(self.data_corresponding)
    
    def __getitem__(self, index):
        data = self.data_corresponding[index]
        action = data['action']
        timestep = data['timestep']
        demo_id = data['demo_id']
        language_promot = data['language'].replace("_"," ")
        data_statistics = self.data_statistics[self.data_name]
        # print("data_statistics is:", data_statistics)
        norm_action = np.clip(2 * (action - data_statistics['q01']) / (data_statistics['q99'] - data_statistics['q01'] + 1e-8) - 1, -1, 1)
        is_correct = True
        if data['is_expert']:
            obs_img = self.task_name_h5py_mapping[data['task_name']][demo_id][f'timestep_{timestep}']['third_obs'][:]
            is_correct = True
        else:
            obs_img = self.interaction_name_mapping[data['task_name']][f'timestep_{timestep}']['third_obs'][:]
            if data['is_human'] == 0:
                is_correct = False
            else:
                is_correct = True
            

        # if self.is_validation:
        resized_img = Image.fromarray(obs_img)
        # resized_img = self.image_resize_transform(resized_img)
        if not self.is_validation:
            resized_img = self.augment_transforms(resized_img)

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {language_promot}?"},
            {"from": "gpt", "value": self.action_tokenizer(norm_action)},
        ]
        
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(resized_img)
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        
        return_data = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": self.data_name,
            "is_correct": is_correct
        }        

        return return_data

from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

