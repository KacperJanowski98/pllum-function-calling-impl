# Function Calling Dataset Analysis

This project provides tools for analyzing the [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset, a collection of 60,000 function calling examples created with the APIGen pipeline. It also includes functionality to translate queries to Polish for fine-tuning the PLLUM model.

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
│   └── project_instruction.md  # Project instructions
├── pyproject.toml        # Project dependencies and metadata
├── README.md             # Project documentation
├── src/                  # Source code
│   ├── __init__.py       # Makes src a package
│   ├── auth.py           # Hugging Face authentication utilities
│   ├── dataset.py        # Dataset loading and processing utilities
│   ├── translator.py     # Utilities for translating text
│   └── translation_dataset.py  # Create translated datasets
└── notebooks/            # Jupyter notebooks
    ├── dataset_exploration.ipynb     # Notebook for exploring the dataset
    ├── translation_test.ipynb        # Test notebook for translation
    └── create_translated_dataset.ipynb  # Notebook for creating translated dataset
```

## Setup

### Prerequisites

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) for dependency management
- A Hugging Face account with access to the dataset

### Installation

1. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
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

### Exploring with Notebooks

Start Jupyter to explore the notebooks:

```bash
jupyter notebook
```

Available notebooks:
- `notebooks/dataset_exploration.ipynb` - Examples of working with the dataset
- `notebooks/translation_test.ipynb` - Testing the translation functionality
- `notebooks/create_translated_dataset.ipynb` - Creating a dataset with Polish translations

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
