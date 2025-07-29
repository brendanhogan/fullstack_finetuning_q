# Dataset Building Pipeline

This component handles the creation and processing of the Q programming language dataset, converting Python solutions from LeetCode into Q code.

## Overview

The dataset building pipeline consists of several stages:

1. **Initial Collection** (`process_dataset.py`)
   - Downloads LeetCode problems and solutions
   - Organizes into structured format
   - Creates initial validation dataset

2. **Python to Q Conversion** (`convert_to_q.py`)
   - Uses LLMs to convert Python solutions to Q
   - Supports multiple LLM providers:
     - OpenAI (GPT-4, GPT-3.5)
     - Anthropic (Claude)
     - Local models
   - Includes validation and error checking

3. **Final Verification** (`final_q_verify.py`)
   - Validates Q code solutions
   - Runs test cases
   - Filters invalid or incorrect solutions

## Directory Structure

```
build_dataset/
├── process_dataset.py     # Initial dataset processing
├── convert_to_q.py        # Python to Q conversion
├── final_q_verify.py      # Solution verification
└── output/
    ├── initial_dataset/   # Raw processed data
    ├── validated_dataset/ # Verified solutions
    └── final_dataset/     # Final cleaned dataset
```

## Usage

1. **Process Initial Dataset**:
   ```bash
   python process_dataset.py
   ```
   This creates the initial dataset structure from LeetCode problems.

2. **Convert to Q**:
   ```bash
   python convert_to_q.py --llm_to_use gpt4
   ```
   Options for `--llm_to_use`:
   - `gpt4`: OpenAI GPT-4
   - `gpt3.5`: OpenAI GPT-3.5
   - `claude`: Anthropic Claude
   - `local`: Local model (requires setup)

3. **Verify Solutions**:
   ```bash
   python final_q_verify.py
   ```

## Configuration

The component uses the root `config.yaml` for settings. Key configurations:

```yaml
dataset:
  initial_dataset_dir: "initial_dataset"
  validated_dataset_dir: "validated_dataset"
  final_dataset_dir: "final_dataset"
  q_interpreter_path: "q"  # Path to Q interpreter
```

## Output Format

Each problem in the final dataset contains:

```
problem_id_task_id/
├── entry.json           # Problem metadata
├── sol.py              # Original Python solution
├── sol.q               # Converted Q solution
├── problem_description.txt  # Problem description
└── test_cases.txt      # Test cases
```

## Filtering Criteria

Solutions are filtered based on:
1. Syntax validity
2. Test case success
3. Code quality metrics
4. Execution time limits

## Error Handling

The pipeline includes robust error handling:
- LLM API failures
- Q interpreter errors
- Invalid syntax detection
- Timeout management

## Contributing

When adding new features:
1. Update the appropriate processing script
2. Add any new dependencies to requirements.txt
3. Update this README with new options/features
4. Add tests for new functionality

## Troubleshooting

Common issues and solutions:

1. **LLM API Errors**
   - Check API key configuration
   - Verify rate limits
   - Ensure proper network connectivity

2. **Q Interpreter Issues**
   - Verify Q installation
   - Check Q_INTERPRETER_PATH setting
   - Ensure proper permissions

3. **Memory Issues**
   - Adjust batch processing size
   - Use incremental processing
   - Clean up temporary files 