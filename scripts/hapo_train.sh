#!/bin/bash
task_name=$1
echo ${task_name}
torchrun --standalone --nnodes 1 --nproc-per-node 4 --master_port=12363 train.py \
 batch_size=8 eval_batch_size=8 trainer=hapo_trainer\
  dataset=balance_hapo_dataset eval_dataset=balance_hapo_eval_dataset\
  dataset.kwargs.interaction_root_dir="robotic_data/mimicgen_interaction_data/mimicgen_${task_name}_interaction"\
  dataset.kwargs.data_list_names=${task_name}.json\
  dataset.kwargs.demo_number=50 \
  exp_name=hapo_train_${task_name} dataset.kwargs.previous_K=10 loss.kl_max=5 loss.kl_min=-5\
  loss.token_distance=1 lr=5e-5 correct_ratio=0.5 interaction_ratio=0.25 loss.desirable_weight=1\
  n_epochs=50 model.name_or_path="ckpt/coffee_d0_300" \
  eval_dataset.kwargs.interaction_root_dir="robotic_data/mimicgen_interaction_data/mimicgen_${task_name}_interaction"\
  eval_dataset.kwargs.data_list_names=${task_name}.json\  

