# Pretraining Pipeline

This component handles the pretraining of language models on Q programming language code, supporting multiple training methods and configurations.

## Overview

The pretraining pipeline offers flexible training approaches:

1. **Training Methods**
   - Full fine-tuning
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized LoRA)

2. **Data Types**
   - Raw code
   - Filtered examples
   - Described/documented code

3. **Model Support**
   - Qwen models (recommended)
   - Other HuggingFace models

## Directory Structure

```
pretrain/
├── run_pretraining.py    # Main training script
├── train_pretrain.py     # Core training logic
├── pretrain_exps.sh      # Example experiment scripts
└── output/
    └── [model_name]/     # Training outputs
        ├── checkpoints/  # Model checkpoints
        ├── logs/        # Training logs
        └── results/     # Evaluation results
```

## Usage

1. **Basic Training**:
   ```bash
   python run_pretraining.py \
       --model_name Qwen/Qwen2.5-7B-Instruct \
       --data_type filtered \
       --training_method lora \
       --max_steps 500
   ```

2. **Training Methods**:
   ```bash
   # LoRA training
   python run_pretraining.py --training_method lora
   
   # QLoRA training
   python run_pretraining.py --training_method qlora
   
   # Full fine-tuning
   python run_pretraining.py --training_method full
   ```

3. **Data Types**:
   ```bash
   # Raw dataset
   python run_pretraining.py --data_type raw
   
   # Filtered dataset
   python run_pretraining.py --data_type filtered
   
   # Described/documented dataset
   python run_pretraining.py --data_type described_filtered
   ```

## Configuration

Key configuration options in `config.yaml`:

```yaml
model:
  base_model: "Qwen/Qwen2.5-7B-Instruct"
  max_seq_length: 2048
  learning_rate: 2.0e-5
  batch_size: 1
  gradient_accumulation_steps: 8
  warmup_steps: 100
  max_steps: 1000
  save_steps: 50
  
  # LoRA settings
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.1
```

## Training Parameters

Important training parameters:

1. **Model Selection**
   - `--model_name`: HuggingFace model ID
   - `--max_seq_length`: Maximum sequence length

2. **Training Settings**
   - `--learning_rate`: Learning rate
   - `--batch_size`: Batch size per GPU
   - `--gradient_accumulation_steps`: Steps for gradient accumulation
   - `--max_steps`: Total training steps
   - `--save_steps`: Checkpoint frequency

3. **LoRA Parameters**
   - `--lora_rank`: LoRA rank
   - `--lora_alpha`: LoRA alpha
   - `--lora_dropout`: LoRA dropout rate

## Experiment Tracking

The pipeline integrates with Weights & Biases:

```bash
python run_pretraining.py \
    --wandb \
    --wandb_project q-pretraining \
    --experiment_name my_experiment
```

## Output Structure

Training produces:

```
output/[model_name]/
├── checkpoints/
│   ├── checkpoint-100/   # Checkpoint at step 100
│   ├── checkpoint-200/   # Checkpoint at step 200
│   └── ...
├── logs/
│   ├── training_log.txt  # Training progress
│   └── eval_results.json # Evaluation metrics
└── config.yaml          # Training configuration
```

## Memory Optimization

Tips for managing memory usage:

1. **Gradient Checkpointing**
   - Enabled by default
   - Trades computation for memory

2. **Batch Size Tuning**
   - Adjust `batch_size`
   - Use `gradient_accumulation_steps`

3. **LoRA vs Full Fine-tuning**
   - LoRA for memory efficiency
   - QLoRA for extreme efficiency

## Troubleshooting

Common issues and solutions:

1. **Out of Memory**
   - Reduce batch size
   - Increase gradient accumulation
   - Switch to LoRA/QLoRA
   - Enable gradient checkpointing

2. **Training Instability**
   - Adjust learning rate
   - Modify warmup steps
   - Check gradient clipping

3. **Slow Training**
   - Optimize sequence length
   - Use appropriate batch size
   - Check GPU utilization

## Contributing

When adding features:
1. Update training scripts
2. Add new configuration options
3. Document parameters
4. Include example usage 