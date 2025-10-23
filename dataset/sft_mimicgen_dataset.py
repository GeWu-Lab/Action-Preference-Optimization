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
# import cv2
import torch
from torchvision import transforms
from PIL import Image
IGNORE_INDEX = -100


from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform
from typing import Any, Dict, Tuple, Type
from prismatic.models.backbones.llm.prompting import PromptBuilder

class MimicGenDataset(Dataset):
    def __init__(self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        data_root_dir: str,
        data_name: str = "libero_spatial",
        data_list_names: str = "task_name.json",
        image_shape: int = 256,
        demo_number: int = 50,
        predict_stop_token: bool = True,
        is_validation: bool = False,
        use_precomputed_statistics = False,
        statistic_path = None
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
        self.demo_number = demo_number
        
        self.statistic_path = statistic_path
        self.use_precomputed_statistics = use_precomputed_statistics
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        
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
        # self.image_resize_transform = transforms.Resize(
        #     (self.image_shape,self.image_shape)
        # )
        for robotic_data_name in data_list:
            annotation_path = os.path.join(data_root_dir , robotic_data_name)
            task_name = robotic_data_name.split(".")[0]
            # print("annotation_path is:", annotation_path)
            robotic_data = h5py.File(annotation_path, "r")
            self.task_name_h5py_mapping[task_name] = robotic_data
            # task_language_prompt = " ".join(task_name.split("_")[:-1])
            task_language_prompt = self.TASK_MAPPING[task_name][1]
            print("the task_language_prompt is:", task_language_prompt)
            print("the demo_number is:", self.demo_number)
            for demo_number_id, demo in enumerate(robotic_data.keys()):
                timestep_len = len(robotic_data[demo])
                interaction_data = robotic_data[demo]
                for timestep in range(timestep_len):
                    action = interaction_data[f'timestep_{timestep}']['action'][:]
                    if max(action) > 1 or min(action) < -1:
                        continue
                    data_statistic['action'].append(action)
                    data_corresponding.append({
                        "action": action,
                        "timestep": timestep,
                        "language": task_language_prompt,
                        "task_name": task_name,
                        'demo_id': demo,
                    })

                data_statistic['num_transitions'] += timestep_len
                data_statistic['num_trajectories'] += 1
                
                if demo_number_id >= self.demo_number:
                    break
                

        
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
        
        print("the q01 is:", q01)
        print("the q99 is:", q99)
        print("the action min is:", action_min)
        print("the action max is:", action_max)
        print("the action mean is:", action_mean)
        print("the action std is:", action_std)


        if self.use_precomputed_statistics:
            # use the train dataset statistics
            data_statistics_path = os.path.join(self.statistic_path, "statistic.json")
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
    
    def save_data_statistics(self, save_path):
        statistic_path = os.path.join(save_path, "statistic.json")
        save_data_statistics = {
            self.data_name:{
                "action":{
                key: self.data_statistics[self.data_name][key].tolist() for key in self.data_statistics[self.data_name].keys()
            }}
        }
        
        with open(statistic_path ,"w") as f:
            json.dump(save_data_statistics,f)
            
        
        
    
    def __len__(self):
        return len(self.data_corresponding)
    
    def __getitem__(self, index):
        data = self.data_corresponding[index]
        action = data['action']
        timestep = data['timestep']
        demo_id = data['demo_id']
        task_name = data['task_name']
        language_promot = data['language']
        data_statistics = self.data_statistics[self.data_name]
        # print("data_statistics is:", data_statistics)
        norm_action = np.clip(2 * (action - data_statistics['q01']) / (data_statistics['q99'] - data_statistics['q01'] + 1e-8) - 1, -1, 1)
        
        obs_img = self.task_name_h5py_mapping[task_name][demo_id][f'timestep_{timestep}']['third_obs'][:]



        resized_img = Image.fromarray(obs_img)

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
            "dataset_name": self.data_name
        }        

        return return_data

from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

