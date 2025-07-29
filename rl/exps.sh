#!/bin/bash

# Replace absolute paths with relative paths
MODEL_DIR="../../models"

# Simple eval
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/zero3.yaml simple_q_training.py \
    --model ${MODEL_DIR}/sft_training/outputs/14/best_14_model_100ckpt_2e6l/checkpoint-100/consol \
    --use_vllm \
    --output_dir q_language_simple_eval_2e6 \
    --learning_rate 2e-6 \
    --simple

# Start vLLM server
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model ${MODEL_DIR}/sft_training/outputs/14/best_14_model_100ckpt_2e6l/checkpoint-100/consol



##### DEFINITELY WORKED ^ 

# Dimensions - reasoning vs non reasoning, temperature low, med, high, binary vs test 

# may want binary vs test vs binary + test 


# Need to redo above once with better SFT model 



#### DIMENSION 1 Reasoning vs Non Reasoning 
# Non reasoning 
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/ --use_vllm --output_dir final_outputs/14b/ablations/reasonining/non_reasoning/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_wandb --wandb_name non_reasoning

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/

# Reasoning 
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/ --use_vllm --output_dir final_outputs/14b/ablations/reasonining/reasoning/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_reasoning_format --use_wandb --wandb_name reasoning

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/



### DIMENSION 2 BINARY VS TEST CASE REWARD 

# Just test case reward 
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/ --use_vllm --output_dir final_outputs/14b/ablations/rewards/only_test_case_reward/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_reasoning_format --use_wandb --wandb_name only_test_case_reward --perfect_reward_weight 0.0

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/

# Just perfect
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/ --use_vllm --output_dir final_outputs/14b/ablations/rewards/only_perfect_reward/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_reasoning_format --use_wandb --wandb_name only_test_case_reward --base_reward_weight 0.0

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/



### DIMENSION 2 TEMP Low, Med, High 

# Low
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/ --use_vllm --output_dir final_outputs/14b/ablations/temp/low/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_wandb --wandb_name low_temp --generation_temp .8

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/


# high  
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/ --use_vllm --output_dir final_outputs/14b/ablations/temp/high/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_wandb --wandb_name low_temp --generation_temp 1.2

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/


# very Low
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/ --use_vllm --output_dir final_outputs/14b/ablations/temp/v_low/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_wandb --wandb_name low_temp --generation_temp .6

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/


# .7
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/ --use_vllm --output_dir final_outputs/14b/ablations/temp/7/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_wandb --wandb_name 7_temp --generation_temp .7

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/





###### FULL TRAININGS 

## 1.5 B  115
# Non Reasaoning 
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/1.5b/checkpoint-400/ --use_vllm --output_dir final_outputs/1.5b/non_reasoning/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_wandb --wandb_name 1.5b_non_reason --generation_temp .8

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/1.5b/checkpoint-400/


## 3B  115
# Non Reasaoning 
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/3b/checkpoint-400/ --use_vllm --output_dir final_outputs/3b/non_reasoning/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_wandb --wandb_name 3b_non_reason --generation_temp .8

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/3b/checkpoint-400/


## 7B  115
# Non Reasaoning 
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/7b/checkpoint-400/ --use_vllm --output_dir final_outputs/7b/non_reasoning/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_wandb --wandb_name 7b_non_reason --generation_temp .8

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/7b/checkpoint-400/

# 14b reasoning
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/ --use_vllm --output_dir final_outputs/14b/reaoning/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_wandb --wandb_name 14b_reasoning --generation_temp .8 --use_reasoning_format 

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/14b/checkpoint-100/consolidated/



# 32b reasoning
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/32b/checkpoint-100/consolidated/ --use_vllm --output_dir final_outputs/32b/reaoning/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_wandb --wandb_name 32b_reasoning --generation_temp .8 --use_reasoning_format 

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/32b/checkpoint-100/consolidated/ --max-model-len 8192


# 32b no reasoning
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num-processes 7 --config-file config/zero3.yaml simple_q_training.py --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/32b/checkpoint-100/consolidated/ --use_vllm --output_dir final_outputs/32b/no_reaoning/ --learning_rate 2e-6 --simple_eval_problems 15 --repeat_eval_problems 4 --eval_steps 25 --use_wandb --wandb_name 32b_no_reasoning --generation_temp .8

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /mnt/home/brendan/FINALQ/full_stack_q_training/sft_training/final_outputs/32b/checkpoint-100/consolidated/ --max-model-len 8192



#
#



