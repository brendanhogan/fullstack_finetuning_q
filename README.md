# Q Language Model Training Pipeline

A comprehensive pipeline for training large language models on the Q programming language, from dataset creation to evaluation.

## Overview

This project provides a complete pipeline for training language models to understand and generate Q programming language code. It includes:

1. **Dataset Building**: Converting and validating Python solutions to Q
2. **Pretraining**: Efficient pretraining using various methods (LoRA, QLoRA)
3. **Fine-tuning**: Supervised and reinforcement learning approaches
4. **Evaluation**: Robust two-phase evaluation system

## Project Structure

```
.
├── build_dataset/      # Dataset creation and processing
├── pretrain/          # Pretraining scripts and configs
├── sft/               # Supervised fine-tuning
├── rl/                # Reinforcement learning
├── eval/              # Model evaluation
├── config.py          # Configuration management
├── config.yaml        # User configuration
└── SETUP.md          # Detailed setup instructions
```

## Quick Start

1. **Setup**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Configure
   cp config.example.yaml config.yaml
   # Edit config.yaml as needed
   ```

2. **Build Dataset**:
   ```bash
   cd build_dataset
   python process_dataset.py
   python convert_to_q.py
   ```

3. **Train**:
   ```bash
   # Pretrain (optional)
   cd pretrain
   python run_pretraining.py
   
   # Fine-tune
   cd ../sft
   python run_sft.py
   ```

4. **Evaluate**:
   ```bash
   cd eval
   python run_full_evaluation.py
   ```

See [SETUP.md](SETUP.md) for detailed instructions.

## Components

### Dataset Building

The dataset pipeline:
1. Processes LeetCode problems
2. Converts Python solutions to Q using LLMs
3. Validates and filters examples
4. Creates training/validation splits

### Pretraining

Supports multiple training approaches:
- Full fine-tuning
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)

### Fine-tuning

Two-stage fine-tuning process:
1. Supervised Fine-tuning (SFT)
2. Reinforcement Learning (RL)

### Evaluation

Robust evaluation system:
- Two-phase: generation and testing
- Multiple metrics (pass@k, test case coverage)
- Task-specific performance analysis

## Requirements

- Python 3.8+
- Q interpreter
- CUDA-capable GPU (recommended)
- Access to LLM APIs (optional, for dataset building)

## Configuration

See [config.example.yaml](config.example.yaml) for available settings.

Key configuration areas:
- Dataset paths and processing
- Model selection and parameters
- Training hyperparameters
- Evaluation settings

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{q-language-model-training,
  author = {Your Name},
  title = {Q Language Model Training Pipeline},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/username/repo}
}
``` 