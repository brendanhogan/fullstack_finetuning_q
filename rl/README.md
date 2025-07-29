# Reinforcement Learning (RL)

This component implements reinforcement learning for Q language model training, using reward modeling to optimize code generation.

## Overview

The RL pipeline uses:
- Direct Preference Optimization (DPO)
- Test case success as rewards
- Code quality metrics
- Efficient training methods

## Directory Structure

```
rl/
├── run_rl.py           # Main training script
├── train_rl.py         # Core RL implementation
├── reward_model.py     # Reward modeling
├── q_trl_training.py   # TRL integration
└── output/
    └── [model_name]/   # Training outputs
        ├── checkpoints/ # Model checkpoints
        ├── logs/       # Training logs
        └── results/    # Evaluation results
```

## Usage

1. **Basic Training**:
   ```bash
   python run_rl.py \
       --base_model /path/to/sft_model \
       --output_dir outputs/rl_run \
       --learning_rate 2e-6
   ```

2. **Training Methods**:
   ```bash
   # With reward modeling
   python run_rl.py --use_reward_model

   # Direct optimization
   python run_rl.py --use_dpo

   # Test case rewards only
   python run_rl.py --test_case_rewards_only
   ```

## Configuration

Key configuration options in `config.yaml`:

```yaml
model:
  # Training parameters
  learning_rate: 2.0e-6
  batch_size: 1
  gradient_accumulation_steps: 8
  max_steps: 1000
  save_steps: 50
  
  # RL parameters
  reward_scale: 1.0
  kl_coef: 0.1
  entropy_coef: 0.01
```

## Reward Components

The RL process uses multiple reward signals:

1. **Test Case Success**
   - Binary pass/fail signals
   - Partial credit for subtasks
   - Time efficiency bonuses

2. **Code Quality**
   - Style conformance
   - Documentation quality
   - Complexity metrics

3. **Preference Learning**
   - Human preferences
   - Expert demonstrations
   - Quality rankings

## Training Parameters

Important parameters:

1. **Model Settings**
   - `--base_model`: SFT model path
   - `--max_seq_length`: Sequence length
   - `--bf16`: Mixed precision training

2. **RL Settings**
   - `--learning_rate`: Learning rate
   - `--reward_scale`: Reward scaling
   - `--kl_coef`: KL penalty coefficient
   - `--entropy_coef`: Entropy bonus

3. **Training Control**
   - `--max_steps`: Total steps
   - `--save_steps`: Checkpoint frequency
   - `--eval_steps`: Evaluation interval

## Experiment Tracking

Weights & Biases integration:

```bash
python run_rl.py \
    --base_model /path/to/model \
    --wandb \
    --wandb_project q-rl \
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
│   ├── rewards.json    # Reward statistics
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

## Reward Model Training

To train a custom reward model:

```bash
python train_reward_model.py \
    --data_path /path/to/preferences \
    --output_dir reward_model \
    --learning_rate 1e-5
```

## Troubleshooting

Common issues and solutions:

1. **Training Instability**
   - Adjust reward scale
   - Modify KL coefficient
   - Check gradient clipping
   - Monitor value estimates

2. **Reward Sparsity**
   - Use intermediate rewards
   - Add auxiliary objectives
   - Implement reward shaping

3. **Performance Issues**
   - Optimize batch size
   - Check reward computation
   - Monitor GPU utilization

## Contributing

When adding features:
1. Update training scripts
2. Add reward components
3. Document parameters
4. Include example usage 