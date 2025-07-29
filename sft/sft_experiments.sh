#!/bin/bash

# Baseline SFT training with low learning rate (1e-5)
python run_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --learning_rate 1e-5 --max_steps 300 --output_dir outputs/sft_baseline_low_lr --experiment_name baseline_low_lr

# Baseline SFT training with high learning rate (5e-5)  
python run_sft.py --base_model Qwen/Qwen2.5-7B-Instruct --learning_rate 5e-5 --max_steps 300 --output_dir outputs/sft_baseline_high_lr --experiment_name baseline_high_lr

# Create difficulty-based curriculum data
python curriculum_organizer.py --data_dir ../SFT_Data --strategy difficulty --output_dir curriculum_data/difficulty

# Curriculum learning: difficulty-based with low learning rate
python train_sft_curriculum.py --base_model Qwen/Qwen2.5-7B-Instruct --curriculum_dir curriculum_data/difficulty --learning_rate 1e-5 --steps_per_phase 100 --output_dir outputs/sft_curriculum_difficulty_low_lr --experiment_name curriculum_difficulty_low_lr

# Curriculum learning: difficulty-based with high learning rate
python train_sft_curriculum.py --base_model Qwen/Qwen2.5-7B-Instruct --curriculum_dir curriculum_data/difficulty --learning_rate 5e-5 --steps_per_phase 100 --output_dir outputs/sft_curriculum_difficulty_high_lr --experiment_name curriculum_difficulty_high_lr

# Create task-type curriculum data
python curriculum_organizer.py --data_dir ../SFT_Data --strategy task_type --output_dir curriculum_data/task_type

# Curriculum learning: task-type based with low learning rate
python train_sft_curriculum.py --base_model Qwen/Qwen2.5-7B-Instruct --curriculum_dir curriculum_data/task_type --learning_rate 1e-5 --steps_per_phase 100 --output_dir outputs/sft_curriculum_tasktype_low_lr --experiment_name curriculum_tasktype_low_lr

# Curriculum learning: task-type based with high learning rate
python train_sft_curriculum.py --base_model Qwen/Qwen2.5-7B-Instruct --curriculum_dir curriculum_data/task_type --learning_rate 5e-5 --steps_per_phase 100 --output_dir outputs/sft_curriculum_tasktype_high_lr --experiment_name curriculum_tasktype_high_lr 