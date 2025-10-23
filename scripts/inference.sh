

model_path="~/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0"


task_name=$1
adapter_path=$2
python experiments/robot/mimicgen/evaluate_mimicgen_adapter.py \
  --model_family openvla \
  --pretrained_checkpoint ${model_path} \
  --adapter_path ${adapter_path}\
  --task_suite_name mimicgen \
  --center_crop True \
  --task_name ${task_name} \
  --num_trials_per_task 50 \