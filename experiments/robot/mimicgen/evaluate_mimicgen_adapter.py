 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: MIT
"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import json
import draccus
import numpy as np
import tqdm

import wandb
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")

from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from robosuite.controllers import load_controller_config
import robosuite as suite
import mimicgen
import torch
import h5py
import imageio
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from PIL import Image
from robosuite.wrappers import VisualizationWrapper

def save_rollout_video(rollout_dir, rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    # os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    adapter_path: Optional[str] = None  
    
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "coffee_d0"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    dataset_dir: str = "robotic_data/"
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    demo_num: int = 50
    # fmt: on
    task_name: str = "coffee"
    rollout_dir: str = "mimicgen_rollout"

import torchvision.transforms as transforms

image_resize_transform = transforms.Resize(
            (224,224)
)

def get_pytorch_libero_image(obs, resize_size):
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img

def make_env(env_name):
    options = {}
    options["env_name"] = env_name
    options["robots"] = "Panda"
    options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")

    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        control_freq=20,
        camera_heights=256,
        camera_widths=256,
        reward_shaping= False,
        camera_names= ["agentview","robot0_eye_in_hand"]
    )
    return env

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    TASK_MAPPING = {
    "coffee_d0": ["Coffee_D0", "make coffee"],
    "coffee_d1": ["Coffee_D1", "make coffee"],
    "stack_d0": ["Stack_D0", "stack the red block on top of the green block"],
    "stack_d1": ["Stack_D1", "stack the red block on top of the green block"],
    "stack_three_d0": ["StackThree_D0", "stack the blocks in the order of blue, red, and green from top to bottom"],
    "stack_three_d1": ["StackThree_D1", "stack the blocks in the order of blue, red, and green from top to bottom"],
    "threading_d0": ["Threading_D0", "insert the needle into the needle hole"],
    "three_piece_assembly_d0":["ThreePieceAssembly_D0","stack the three pieces"],
    "three_piece_assembly_d1":["ThreePieceAssembly_D1","stack the three pieces"],
    "square_d0": ["Square_D0","insert the square into the wooden stick"]
    }    
    
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # add model statistic data
    norm_stats = json.load(open(os.path.join(cfg.adapter_path ,"statistic.json")))
    model.norm_stats = norm_stats
    print("the model norm_stats is:", model.norm_stats)

    model = PeftModel.from_pretrained(model, cfg.adapter_path)
    model = model.to(dtype = torch.bfloat16)
    # import pdb;pdb.set_trace()
    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        # processor = get_processor(cfg)
        processor = AutoProcessor.from_pretrained(cfg.adapter_path, trust_remote_code=True)

    # Initialize local logging
    rollout_dir = f"./{cfg.rollout_dir}/{cfg.adapter_path}/{DATE}/{cfg.task_suite_name}/{DATE_TIME}"

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(rollout_dir, exist_ok=True)
    local_log_filepath = os.path.join(rollout_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    print("the processor is:", processor)
    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    # resize_size = get_image_resize_size(cfg)
    resize_size = 256

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_name in tqdm.tqdm(TASK_MAPPING.keys()):
        # Get task
        if cfg.task_name not in task_name: 
            continue
        print("the cfg.task_name is:", cfg.task_name)
        # Get default LIBERO initial states


        # Initialize LIBERO environment and task description
        env = make_env(TASK_MAPPING[task_name][0])
        env = VisualizationWrapper(env)
        # Start episodes
        task_description = TASK_MAPPING[task_name][1]
        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            is_succ = False
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            obs = env.reset()


            # Setup
            t = 0
            replay_images = []
            max_steps = 550  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:

                print("the timestep is:", t)

                img = get_pytorch_libero_image(obs, resize_size)

                replay_images.append(img)

                # Prepare observations dict
                # Note: OpenVLA does not take proprio state as input
                observation = {
                    "full_image": img
                }
 
                action = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                )

                # print("the action is:", action)
                # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                action = normalize_gripper_action(action, binarize=True)


                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                if reward > 0.9:
                    print("the reward is:", reward)
                    task_successes += 1
                    total_successes += 1
                    is_succ = True
                    break
                t += 1

                # except Exception as e:
                #     print(f"Caught exception: {e}")
                #     log_file.write(f"Caught exception: {e}\n")
                #     break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                rollout_dir,replay_images, total_episodes, success=is_succ, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {is_succ}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {is_succ}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
