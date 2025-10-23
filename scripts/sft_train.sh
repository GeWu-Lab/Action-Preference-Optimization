

task_name=$1
demo_number=$2

torchrun --standalone --nnodes 1 --nproc-per-node 4 sft_train.py \
  --vla_path "~/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0" \
  --data_root_dir "robotic_data/mimicgen" \
  --dataset_name "mimicgen" \
  --run_root_dir "nips_ckpts/base_policy" \
  --data_list_names ${task_name}.json\
  --adapter_tmp_dir "./temp"\
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --save_latest_checkpoint_only True \
  --image_aug True \
  --wandb_project xwk_nips_mimicgen \
  --wandb_entity mimicgen_${task_name}_${demo_number} \
  --wandb_name mimicgen_${task_name}_${demo_number} \
  --save_steps 500 \
  --max_epoch 10 \
  --demo_number ${demo_number}