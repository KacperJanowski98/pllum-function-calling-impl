# Function Calling Dataset Analysis and PLLuM Fine-tuning

This project provides tools for analyzing the [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset, a collection of 60,000 function calling examples created with the APIGen pipeline. It includes functionality to translate queries to Polish for fine-tuning the PLLuM model, and implements fine-tuning of the PLLuM 8B model for function calling tasks using QLoRA and the Unsloth framework.

## Overview

The dataset contains 60,000 data points collected by APIGen, an automated data generation pipeline designed to produce verifiable high-quality datasets for function-calling applications. Each data point is verified through three hierarchical stages: format checking, actual function executions, and semantic verification, ensuring reliability and correctness.

According to human evaluation of 600 sampled data points, the dataset has a correctness rate above 95% (with the remaining 5% having minor issues like inaccurate arguments).

## Project Structure

```
.
├── .env                  # Environment variables (add your HF token here)
├── .gitignore            # Git ignore file
├── data/                 # Directory for generated datasets
│   └── .gitkeep          # Placeholder to ensure directory is tracked
├── docs/                 # Documentation directory
│   └── instruction.md    # Project instructions
├── models/               # Directory for storing fine-tuned models
│   └── .gitkeep          # Placeholder to ensure directory is tracked
├── pyproject.toml        # Project dependencies and metadata
├── README.md             # Project documentation
├── src/                  # Source code
│   ├── __init__.py       # Makes src a package
│   ├── auth.py           # Hugging Face authentication utilities
│   ├── dataset.py        # Dataset loading and processing utilities
│   ├── fine_tuning.py    # Utilities for fine-tuning the PLLuM model
│   ├── translator.py     # Utilities for translating text
│   └── translation_dataset.py  # Create translated datasets
└── notebooks/            # Jupyter notebooks
    ├── create_translated_dataset.ipynb  # Create a dataset with Polish translations
    ├── dataset_exploration.ipynb        # Notebook for exploring the dataset
    ├── fine_tuning.ipynb                # Notebook for fine-tuning PLLuM model
    ├── test_model.ipynb                 # Test notebook for the fine-tuned model
    └── translation_test.ipynb           # Test notebook for translation
```

## Setup

### Prerequisites

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) for dependency management
- A Hugging Face account with access to the dataset
- NVIDIA GPU (RTX 4060 or better) with CUDA support

### Installation

1. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
uv pip install -e .
```

2. Create a `.env` file with your Hugging Face token:
```bash
cp .env.example .env
# Edit .env to add your Hugging Face token
```

## Usage

### Loading the Dataset

```python
from src.dataset import load_function_calling_dataset

# Load the dataset
dataset = load_function_calling_dataset()

# Access a sample
sample = dataset['train'][0]
print(sample)
```

### Translating Queries to Polish

```python
from src.translator import translate_text, translate_query_in_sample

# Translate a single text
english_text = "Find the nearest restaurant to my location"
polish_text = translate_text(english_text, src='en', dest='pl')
print(polish_text)

# Translate a query in a dataset sample
from src.dataset import load_function_calling_dataset, parse_json_entry

dataset = load_function_calling_dataset()
sample = dataset['train'][0]
translated_sample = translate_query_in_sample(sample, src='en', dest='pl')
parsed = parse_json_entry(translated_sample)
print(f"Translated query: {parsed['query']}")
```

### Creating a Translated Dataset

```python
from src.translation_dataset import create_translated_dataset

# Create a dataset with 40% queries translated to Polish
output_path = "data/translated_dataset.json"
translated_dataset_path = create_translated_dataset(
    output_path=output_path,
    sample_size=1000,  # Optional: limit dataset size
    translation_percentage=0.4,
    random_seed=42
)
print(f"Dataset created at: {translated_dataset_path}")
```

### Fine-tuning PLLuM for Function Calling

The project includes functionality to fine-tune the [CYFRAGOVPL/Llama-PLLuM-8B-instruct](https://huggingface.co/CYFRAGOVPL/Llama-PLLuM-8B-instruct) model for function calling:

```python
from src.fine_tuning import (
    PLLuMFineTuningConfig,
    setup_model_and_tokenizer,
    prepare_dataset,
    train_model
)

# Configure fine-tuning parameters
config = PLLuMFineTuningConfig(
    model_name_or_path="CYFRAGOVPL/Llama-PLLuM-8B-instruct",
    output_dir="models/pllum-function-calling",
    dataset_path="data/translated_dataset.json",
    # QLoRA settings for efficient training on consumer GPUs
    use_4bit=True,
    lora_r=16,
    lora_alpha=32
)

# Load model and tokenizer with QLoRA and Unsloth optimizations
model, tokenizer = setup_model_and_tokenizer(config)

# Prepare dataset for training
train_dataset = prepare_dataset(
    dataset_path=config.dataset_path,
    tokenizer=tokenizer,
    max_length=config.max_seq_length
)

# Train the model
trained_model = train_model(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    config=config
)
```

### Using the Fine-tuned Model

```python
from src.fine_tuning import (
    load_fine_tuned_model,
    generate_function_call
)

# Load a fine-tuned model
model, tokenizer = load_fine_tuned_model("models/pllum-function-calling")

# Define tools
weather_tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "location": {
                "type": "string",
                "description": "The city and state or country",
                "required": True
            },
            "unit": {
                "type": "string",
                "description": "Unit of temperature: 'celsius' or 'fahrenheit'",
                "required": False
            }
        }
    }
]

# Generate function call for a query
query = "Jaka jest pogoda w Warszawie?"  # Polish query: "What's the weather in Warsaw?"
function_call = generate_function_call(
    model=model,
    tokenizer=tokenizer,
    query=query,
    tools=weather_tools,
    temperature=0.1
)

print(function_call)
```

### Exploring with Notebooks

Start Jupyter to explore the notebooks:

```bash
jupyter notebook
```

Available notebooks:
- `notebooks/dataset_exploration.ipynb` - Examples of working with the dataset
- `notebooks/translation_test.ipynb` - Testing the translation functionality
- `notebooks/create_translated_dataset.ipynb` - Creating a dataset with Polish translations
- `notebooks/fine_tuning.ipynb` - Fine-tuning PLLuM 8B with QLoRA and Unsloth
- `notebooks/test_model.ipynb` - Testing the fine-tuned model for function calling

## Fine-tuning Features

The project includes functionality to fine-tune the PLLuM 8B model for function calling:

- `src/fine_tuning.py` - Utilities for fine-tuning the PLLuM model with QLoRA and Unsloth
- Optimized for consumer GPUs (tested on NVIDIA RTX 4060)
- Using 4-bit quantization for memory efficiency
- Support for both English and Polish queries
- Customizable hyperparameters via the `PLLuMFineTuningConfig` class
- Generation utilities for using the fine-tuned model

## Translation Features

The project includes functionality to translate dataset queries from English to Polish:

- `src/translator.py` - Utilities for translating text using the Googletrans library
- `src/translation_dataset.py` - Functions for creating datasets with translated queries
- Ability to create a dataset with a specified percentage (default 40%) of queries translated to Polish
- Preservation of original dataset structure, with only the query field translated

## Data Format

Each entry in the dataset follows this JSON format:

- `query` (string): The query or problem statement
- `tools` (array): Available tools to solve the query
  - Each tool has `name`, `description`, and `parameters`
- `answers` (array): Corresponding answers showing which tools were used with what arguments

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset
- [APIGen pipeline](https://apigen-pipeline.github.io/) for dataset generation
- [CYFRAGOVPL/Llama-PLLuM-8B-instruct](https://huggingface.co/CYFRAGOVPL/Llama-PLLuM-8B-instruct) model
- [Unsloth](https://github.com/unslothai/unsloth) framework for optimized LLM fine-tuning
