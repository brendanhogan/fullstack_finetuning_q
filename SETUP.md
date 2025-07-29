# Setup Guide

This guide explains how to set up and configure the Q language model training pipeline.

## Prerequisites

1. Python 3.8+ with pip
2. Q interpreter installed and accessible
3. CUDA-capable GPU (recommended for training)
4. Access to required APIs (if using external models for dataset building)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy and configure settings:
   ```bash
   cp config.example.yaml config.yaml
   ```
   Edit `config.yaml` to match your environment.

## Configuration

### Environment Variables

The following environment variables can be set either in your shell or in a `.env` file:

```bash
# Q interpreter path (if not in system PATH)
Q_INTERPRETER_PATH=/path/to/q/interpreter

# API keys for external services (if using)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
XAI_API_KEY=your_xai_key
GEMINI_API_KEY=your_gemini_key
```

### Configuration File

The `config.yaml` file controls various aspects of the training pipeline:

1. **Dataset Configuration**
   - `initial_dataset_dir`: Directory for raw dataset
   - `validated_dataset_dir`: Directory for validated examples
   - `final_dataset_dir`: Directory for final processed dataset
   - `q_interpreter_path`: Path to Q interpreter

2. **Model Configuration**
   - `base_model`: HuggingFace model ID or local path
   - Training parameters (batch size, learning rate, etc.)
   - LoRA parameters for efficient fine-tuning

3. **Output Directories**
   - `output_dir`: Directory for model checkpoints and outputs
   - `log_dir`: Directory for training logs

## Pipeline Components

The codebase is organized into several components:

1. **Dataset Building** (`build_dataset/`)
   - Processes LeetCode problems
   - Converts Python solutions to Q
   - Validates and filters examples

2. **Pretraining** (`pretrain/`)
   - Pretrains language models on Q code
   - Supports multiple training methods (LoRA, QLoRA, full)

3. **Supervised Fine-tuning** (`sft/`)
   - Fine-tunes models on specific tasks
   - Uses high-quality filtered examples

4. **Reinforcement Learning** (`rl/`)
   - Further improves model performance
   - Uses reward modeling for optimization

5. **Evaluation** (`eval/`)
   - Two-phase evaluation system
   - Comprehensive metrics collection

## Usage

Each component has its own README with specific instructions. The general workflow is:

1. Build and process the dataset:
   ```bash
   cd build_dataset
   python process_dataset.py
   python convert_to_q.py
   ```

2. Run pretraining (if needed):
   ```bash
   cd pretrain
   python run_pretraining.py
   ```

3. Run supervised fine-tuning:
   ```bash
   cd sft
   python run_sft.py
   ```

4. Run RL training (optional):
   ```bash
   cd rl
   python run_rl.py
   ```

5. Evaluate the model:
   ```bash
   cd eval
   python run_full_evaluation.py
   ```

## Troubleshooting

1. **Q Interpreter Not Found**
   - Ensure Q is installed and in PATH
   - Set Q_INTERPRETER_PATH in .env or config.yaml

2. **GPU Memory Issues**
   - Adjust batch_size and gradient_accumulation_steps
   - Use LoRA/QLoRA instead of full fine-tuning

3. **API Rate Limits**
   - Adjust processing delays in dataset building
   - Use local models where possible

## Contributing

Please see CONTRIBUTING.md for guidelines on contributing to this project. 