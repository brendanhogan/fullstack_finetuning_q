#!/bin/bash

# Full fine-tuning experiment with raw dataset
# python run_pretraining.py --model_name Qwen/Qwen2.5-7B-Instruct --data_type raw --training_method full --max_steps 500 --save_steps 50 --learning_rate 1e-5 --output_dir outputs/qwen_7b_raw_full --wandb
python run_pretraining.py --model_name Qwen/Qwen2.5-7B-Instruct --data_type raw --training_method full --max_steps 3500 --save_steps 350 --learning_rate 1e-5 --output_dir outputs/qwen_7b_raw_full --wandb

# # Full fine-tuning experiment with filtered dataset  
# python run_pretraining.py --model_name Qwen/Qwen2.5-7B-Instruct --data_type filtered --training_method full --max_steps 500 --save_steps 50 --learning_rate 1e-5 --output_dir outputs/qwen_7b_filtered_full --wandb

# # Full fine-tuning experiment with described_filtered dataset
# python run_pretraining.py --model_name Qwen/Qwen2.5-7B-Instruct --data_type described_filtered --training_method full --max_steps 500 --save_steps 50 --learning_rate 1e-5 --output_dir outputs/qwen_7b_described_filtered_full --wandb

# # LoRA experiment with raw dataset
# python run_pretraining.py --model_name Qwen/Qwen2.5-7B-Instruct --data_type raw --training_method lora --max_steps 500 --save_steps 50 --learning_rate 5e-5 --lora_rank 16 --lora_alpha 32 --output_dir outputs/qwen_7b_raw_lora --wandb


# Run pretraining for different numbers of steps
python run_pretraining.py --model_name Qwen/Qwen2.5-7B-Instruct --data_type described_filtered --training_method full --max_steps 250 --save_steps 25 --learning_rate 1e-5 --output_dir outputs/qwen_7b_described_filtered_full_250 --wandb

python run_pretraining.py --model_name Qwen/Qwen2.5-7B-Instruct --data_type described_filtered --training_method full --max_steps 800 --save_steps 80 --learning_rate 1e-5 --output_dir outputs/qwen_7b_described_filtered_full_800 --wandb

python run_pretraining.py --model_name Qwen/Qwen2.5-7B-Instruct --data_type described_filtered --training_method full --max_steps 1600 --save_steps 160 --learning_rate 1e-5 --output_dir outputs/qwen_7b_described_filtered_full_1600 --wandb

python run_pretraining.py --model_name Qwen/Qwen2.5-7B-Instruct --data_type described_filtered --training_method full --max_steps 2400 --save_steps 240 --learning_rate 1e-5 --output_dir outputs/qwen_7b_described_filtered_full_2400 --wandb



### For bigger model signle rung 
python run_pretraining.py --model_name Qwen/Qwen2.5-1.5B-Instruct --data_type described_filtered --training_method full --max_steps 800 --save_steps 80 --learning_rate 1e-5 --output_dir outputs/qwen_1.5b_described_filtered_full_800 --wandb
python run_pretraining.py --model_name Qwen/Qwen2.5-3B-Instruct --data_type described_filtered --training_method full --max_steps 800 --save_steps 80 --learning_rate 1e-5 --output_dir outputs/qwen_3b_described_filtered_full_800 --wandb
python run_pretraining.py --model_name Qwen/Qwen2.5-14B-Instruct --data_type described_filtered --training_method full --max_steps 800 --save_steps 80 --learning_rate 1e-5 --output_dir outputs/qwen_14b_described_filtered_full_800 --wandb





accelerate launch --num_processes=6 train_pretrain.py \
  --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
  --train_file described_filtered_train.jsonl \
  --eval_file described_filtered_test.jsonl \
  --max_steps 800 \
  --save_steps 80 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --output_dir outputs/qwen_14b_described_filtered_full_800 \
  --report_to wandb \
  --wandb_project q-pretraining


accelerate launch --num_processes=6 --main_process_port=29501 train_pretrain.py \
  --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
  --train_file described_filtered_train.jsonl \
  --eval_file described_filtered_test.jsonl \
  --max_steps 800 \
  --save_steps 80 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --output_dir outputs/qwen_32b_described_filtered_full_800 \
  --report_to wandb \
  --wandb_project q-pretraining


accelerate launch --num_processes=6 --main_process_port=29502 train_pretrain.py \
  --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
  --train_file described_filtered_train.jsonl \
  --eval_file described_filtered_test.jsonl \
  --max_steps 800 \
  --save_steps 80 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --output_dir outputs/qwen_32b_described_filtered_full_800 \
  --report_to wandb \
  --wandb_project q-pretraining





##### FINAL EXPS 
python run_pretraining.py --model_name Qwen/Qwen2.5-1.5B-Instruct --data_type licensed --training_method full --max_steps 800 --save_steps 100 --learning_rate 1e-5 --output_dir final_outputs/1.5b --wandb

python run_pretraining.py --model_name Qwen/Qwen2.5-3B-Instruct --data_type licensed --training_method full --max_steps 800 --save_steps 100 --learning_rate 1e-5 --output_dir final_outputs/3b --wandb

python run_pretraining.py --model_name Qwen/Qwen2.5-7B-Instruct --data_type licensed --training_method full --max_steps 800 --save_steps 100 --learning_rate 1e-5 --output_dir final_outputs/7b --wandb

accelerate launch --num_processes=4 train_pretrain.py \
  --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
  --train_file kdb_license_processed/train_license_kdbsite.jsonl \
  --eval_file kdb_license_processed/eval_license_kdbsite.jsonl \
  --max_steps 800 \
  --save_steps 80 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --output_dir final_outputs/14b \
  --report_to wandb \
  --wandb_project q-pretraining

accelerate launch --num_processes=4 --main_process_port=29502 train_pretrain.py \
  --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
  --train_file kdb_license_processed/train_license_kdbsite.jsonl \
  --eval_file kdb_license_processed/eval_license_kdbsite.jsonl \
  --max_steps 800 \
  --save_steps 80 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --output_dir final_outputs/32b \
  --report_to wandb \
  --wandb_project q-pretraining




#