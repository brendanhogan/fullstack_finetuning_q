# Supervised Fine-tuning (SFT)

This component handles supervised fine-tuning of language models on high-quality Q programming examples.

## Overview

The SFT pipeline fine-tunes pretrained models using:
- High-quality filtered examples
- Instruction-following format
- Multiple training approaches

## Directory Structure

```
sft/
├── run_sft.py           # Main training script
├── train_sft.py         # Core training logic
├── run_sft_exps.sh      # Example experiments
└── output/
    └── [model_name]/    # Training outputs
        ├── checkpoints/ # Model checkpoints
        ├── logs/       # Training logs
        └── results/    # Evaluation results
```

## Usage

1. **Basic Training**:
   ```bash
   python run_sft.py \
       --base_model /path/to/pretrained \
       --learning_rate 2e-5 \
       --max_steps 1000 \
       --output_dir outputs/sft_run
   ```

2. **Training Methods**:
   ```bash
   # LoRA training
   python run_sft.py --use_lora

   # QLoRA training
   python run_sft.py --use_qlora

   # Full fine-tuning
   python run_sft.py  # No adaptation flags
   ```

## Configuration

Key configuration options in `config.yaml`:

```yaml
model:
  # Training parameters
  learning_rate: 2.0e-5
  batch_size: 1
  gradient_accumulation_steps: 8
  warmup_steps: 100
  max_steps: 1000
  save_steps: 50
  
  # LoRA parameters (if using)
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.1
```

## Training Data

The SFT process uses:
1. Filtered, high-quality examples
2. Instruction-following format
3. Test case validation
4. Code quality metrics

## Training Parameters

Important parameters:

1. **Model Settings**
   - `--base_model`: Pretrained model path
   - `--max_seq_length`: Sequence length
   - `--bf16`: Mixed precision training

2. **Training Settings**
   - `--learning_rate`: Learning rate
   - `--batch_size`: Batch size per GPU
   - `--gradient_accumulation_steps`: Gradient accumulation
   - `--max_steps`: Total steps
   - `--save_steps`: Checkpoint frequency

3. **Adaptation Settings**
   - `--use_lora`: Enable LoRA
   - `--use_qlora`: Enable QLoRA
   - `--lora_rank`: LoRA rank
   - `--lora_alpha`: LoRA alpha
   - `--lora_dropout`: LoRA dropout

## Experiment Tracking

Weights & Biases integration:

```bash
python run_sft.py \
    --base_model /path/to/model \
    --wandb \
    --wandb_project q-sft \
    --experiment_name my_experiment
```

## Output Structure

Training produces:

```
output/[model_name]/
├── checkpoints/
│   ├── checkpoint-100/  # Checkpoint at step 100
│   ├── checkpoint-200/  # Checkpoint at step 200
│   └── ...
├── logs/
│   ├── training_log.txt # Training progress
│   └── eval_results.json # Evaluation metrics
└── config.yaml         # Training configuration
```

## Memory Optimization

Tips for managing memory:

1. **Gradient Checkpointing**
   - Enabled by default
   - Trades computation for memory

2. **Batch Size Tuning**
   - Adjust `batch_size`
   - Use `gradient_accumulation_steps`

3. **Mixed Precision**
   - Use `--bf16` for mixed precision
   - Reduces memory usage

4. **Model Parallelism**
   - Use `accelerate` for multi-GPU
   - Shard model across devices

## Troubleshooting

Common issues and solutions:

1. **Out of Memory**
   - Reduce batch size
   - Increase gradient accumulation
   - Enable mixed precision
   - Use LoRA/QLoRA

2. **Training Instability**
   - Adjust learning rate
   - Modify warmup steps
   - Check gradient clipping
   - Monitor loss curves

3. **Slow Training**
   - Check GPU utilization
   - Optimize batch size
   - Use appropriate precision

## Contributing

When adding features:
1. Update training scripts
2. Add configuration options
3. Document parameters
4. Include example usage 