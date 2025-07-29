#!/bin/bash

# Replace absolute paths with relative paths
MODEL_DIR="../../models"

# Create difficulty-based curriculum data (Easy → Medium → Hard)
# python curriculum_organizer.py --data_dir ../SFT_Data --strategy difficulty --output_dir curriculum_data/difficulty

# Create task-type curriculum data (similar tasks grouped together)
# python curriculum_organizer.py --data_dir ../SFT_Data --strategy task_type --output_dir curriculum_data/task_type

# Create mixed curriculum data (Easy tasks of all types, then Medium, then Hard)
# python curriculum_organizer.py --data_dir ../SFT_Data --strategy mixed --output_dir curriculum_data/mixed

# # Baseline SFT training with low learning rate
python run_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --learning_rate 1e-5 --max_steps 600 --output_dir outputs/sft_baseline_low_lr --experiment_name baseline_low_lr

# # Baseline SFT training with high learning rate
# python run_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --learning_rate 5e-5 --max_steps 600 --output_dir outputs/sft_baseline_high_lr --experiment_name baseline_high_lr

# # Baseline SFT training with medium learning rate
# python run_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --learning_rate 2e-5 --max_steps 600 --output_dir outputs/sft_baseline_medium_lr --experiment_name baseline_medium_lr
python run_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --learning_rate 2e-5 --max_steps 2400 --output_dir outputs/sft_baseline_medium_lr --experiment_name baseline_medium_lr

python run_sft.py --base_model ../pretrain/outputs/qwen_7b_described_filtered_full/checkpoint-500/ --learning_rate 2e-5 --max_steps 600 --output_dir outputs/sft_pretrained_model_medium_lr --experiment_name pretrained_medium_lr


# # Curriculum learning: difficulty-based with low learning rate (Easy→Medium→Hard)
# python train_sft_curriculum.py --base_model Qwen/Qwen2.5-7B-Instruct --curriculum_dir curriculum_data/difficulty --steps_per_phase 200 --output_dir outputs/sft_curriculum_difficulty --experiment_name curriculum_difficulty


# # Curriculum learning: task-type based with low learning rate (similar tasks together)
# python train_sft_curriculum.py --base_model Qwen/Qwen2.5-7B-Instruct --curriculum_dir curriculum_data/task_type --steps_per_phase 150 --output_dir outputs/sft_curriculum_tasktyper --experiment_name curriculum_tasktype


python train_sft_curriculum.py --base_model Qwen/Qwen2.5-7B-Instruct --curriculum_dir curriculum_data/tag_based --steps_per_phase 120 --output_dir outputs/sft_curriculum_tags_balanced --experiment_name curriculum_tags



python run_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --learning_rate 2e-5 --max_steps 600 --output_dir outputs/sft_baseline_lora --experiment_name baseline_lora --use_lora


python run_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --learning_rate 2e-5 --max_steps 400 --output_dir outputs/sft_baseline_medium_lr_400 --experiment_name baseline_medium_lr_400

python run_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --learning_rate 2e-5 --max_steps 1000 --output_dir outputs/sft_baseline_medium_lr_1000 --experiment_name baseline_medium_lr_1000

python run_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --learning_rate 2e-5 --max_steps 1600 --output_dir outputs/sft_baseline_medium_lr_1600 --experiment_name baseline_medium_lr_1600



## Run best sft setting with best pretrained model 
python run_sft.py --base_model ${MODEL_DIR}/pretrain/outputs/qwen_7b_described_filtered_full_800/checkpoint-800/ --learning_rate 2e-5 --max_steps 1000 --output_dir outputs/sft_best_pretrain_medium_lr_1000 --experiment_name sft_best_pretrain_medium_lr_1000




python run_sft.py --base_model Qwen/Qwen2.5-1.5B-Instruct --learning_rate 2e-5 --max_steps 1000 --output_dir outputs/1.5/sft_baseline_medium_lr_1000 --experiment_name 1.5_baseline_medium_lr_1000
python run_sft.py --base_model Qwen/Qwen2.5-3B-Instruct --learning_rate 2e-5 --max_steps 1000 --output_dir outputs/3/sft_baseline_medium_lr_1000 --experiment_name 3_baseline_medium_lr_1000
python run_sft.py --base_model Qwen/Qwen2.5-14B-Instruct --learning_rate 2e-5 --max_steps 1000 --output_dir outputs/14/sft_baseline_medium_lr_1000 --experiment_name 14_baseline_medium_lr_1000




accelerate launch --num_processes=4 run_sft.py --base_model Qwen/Qwen2.5-14B-Instruct --learning_rate 5e-5 --max_steps 1000 --output_dir outputs/14/sft_baseline_medium_lr_1000 --experiment_name baseline_14b_high_lr



accelerate launch --num_processes=5 --main_process_port=29502 run_sft.py --base_model Qwen/Qwen2.5-32B-Instruct --learning_rate 5e-5 --max_steps 1000 --output_dir outputs/32/sft_baseline_medium_lr_1000 --experiment_name baseline_32b_high_lr --save_every_n_steps 50

accelerate launch --num_processes=4 run_sft.py --base_model Qwen/Qwen2.5-14B-Instruct --learning_rate 5e-5 --max_steps 100 --output_dir outputs/14/sft_baseline_medium_lr_100 --experiment_name baseline_14b_high_lr_100


# Reasonalbe runs for 14 and 32B 
accelerate launch --num_processes=4 run_sft.py --base_model Qwen/Qwen2.5-14B-Instruct --learning_rate 8e-6 --max_steps 1000 --output_dir outputs/14/sft_baseline_correct_lr_1000 --experiment_name sft_baseline_correct_lr_1000 --save_every_n_steps 50

accelerate launch --num_processes=4 --main_process_port=29502 run_sft.py --base_model Qwen/Qwen2.5-14B-Instruct --learning_rate 2e-5  --max_steps 1000 --output_dir outputs/14/sft_baseline_same_lr_as_best_model_1000 --experiment_name sft_baseline_correct_lr_1000 --save_every_n_steps 50


accelerate launch --num_processes=4 --main_process_port=29503 run_sft.py --base_model Qwen/Qwen2.5-14B-Instruct --learning_rate 2e-5  --max_steps 1000 --output_dir outputs/14/sft_baseline_same_lr_as_best_model_1000_no_test_cases --experiment_name sft_baseline_correct_lr_1000_no_test_case --save_every_n_steps 50 --use_no_test_cases







accelerate launch --num_processes=5 --main_process_port=29502 run_sft.py --base_model Qwen/Qwen2.5-32B-Instruct --learning_rate 5e-5 --max_steps 1000 --output_dir outputs/32/sft_baseline_medium_lr_1000 --experiment_name baseline_32b_high_lr --save_every_n_steps 50





# Lower learning rate full 
# Lower learning rate just test case #1
# Lower learning rate just test case #2



accelerate launch --num_processes=4 --main_process_port=29501 run_sft.py --base_model Qwen/Qwen2.5-14B-Instruct --learning_rate 5e-6 --max_steps 1000 --output_dir outputs/14/full_data_5e6lr --experiment_name full_data_5e6lr --save_every_n_steps 50

accelerate launch --num_processes=4 --main_process_port=29502 run_sft.py --base_model Qwen/Qwen2.5-14B-Instruct --learning_rate 5e-6  --max_steps 1000 --output_dir outputs/14/no_testcase_5e6lr --experiment_name no_testcase_5e6lr --save_every_n_steps 50 --use_no_test_cases

accelerate launch --num_processes=4 --main_process_port=29503 run_sft.py --base_model Qwen/Qwen2.5-14B-Instruct --learning_rate 8e-6  --max_steps 1000 --output_dir outputs/14/no_testcase_8e6lr --experiment_name no_testcase_8e6lr --save_every_n_steps 50 --use_no_test_cases



 accelerate launch --num_processes=4 --main_process_port=29502 run_sft.py --base_model Qwen/Qwen2.5-14B-Instruct --learning_rate 2e-6  --max_steps 1000 --output_dir outputs/14/no_testcase_2e6lr --experiment_name no_testcase_2e6lr --save_every_n_steps 50 --use_no_test_cases




accelerate launch --num_processes=4 --main_process_port=29503 run_sft.py --base_model Qwen/Qwen2.5-14B-Instruct --learning_rate 2e-6  --max_steps 1000 --output_dir outputs/14/full_data_2e6lr --experiment_name full_data_2e6lr --save_every_n_steps 50

## 32 

 accelerate launch --num_processes=4 --main_process_port=29505 run_sft.py --base_model Qwen/Qwen2.5-32B-Instruct --learning_rate 2e-6  --max_steps 1000 --output_dir outputs/14/no_testcase_2e6lr --experiment_name 32b_no_testcase_2e6lr --save_every_n_steps 50 --use_no_test_cases


accelerate launch --num_processes=4 --main_process_port=29501 run_sft.py --base_model Qwen/Qwen2.5-32B-Instruct --learning_rate 4e-6 --max_steps 1000 --output_dir outputs/32/full_data_5e6lr --experiment_name 32b_full_data_5e6lr --save_every_n_steps 50



accelerate launch --num_processes=4 --main_process_port=29501 run_sft.py --base_model ${MODEL_DIR}/pretrain/outputs/qwen_32b_described_filtered_full_800/checkpoint-160/consolidated_model/ --learning_rate 4e-6 --max_steps 1000 --output_dir outputs/32/full_data_5e6lr --experiment_name 32b_full_data_5e6lr_from_pretrain --save_every_n_steps 50



#### FINAL EXPS 
# 1.5b
python run_sft.py --base_model ${MODEL_DIR}/pretrain/final_outputs/1.5b/checkpoint-800 --learning_rate 2e-5 --max_steps 1000 --output_dir final_outputs/1.5b/

#3b
python run_sft.py --base_model ${MODEL_DIR}/pretrain/final_outputs/3b/checkpoint-800 --learning_rate 2e-5 --max_steps 1000 --output_dir final_outputs/3b/


#7b
python run_sft.py --base_model ${MODEL_DIR}/pretrain/final_outputs/7b/checkpoint-200 --learning_rate 2e-5 --max_steps 1000 --output_dir final_outputs/7b/



accelerate launch --num_processes=4 --main_process_port=29501 run_sft.py --base_model ${MODEL_DIR}/pretrain/final_outputs/14b/checkpoint-50/consolidated/ --learning_rate 4e-6 --max_steps 1000 --output_dir final_outputs/14b/ --experiment_name q-sft --save_every_n_steps 50


accelerate launch --num_processes=4 --main_process_port=29503 run_sft.py --base_model ${MODEL_DIR}/pretrain/final_outputs/32b/checkpoint-50/consolidated --learning_rate 4e-6 --max_steps 1000 --output_dir final_outputs/32b/ --experiment_name q-sft --save_every_n_steps 50










#