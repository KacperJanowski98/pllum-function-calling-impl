# Function Calling Dataset Analysis

This project provides tools for analyzing the [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset, a collection of 60,000 function calling examples created with the APIGen pipeline.

## Overview

The dataset contains 60,000 data points collected by APIGen, an automated data generation pipeline designed to produce verifiable high-quality datasets for function-calling applications. Each data point is verified through three hierarchical stages: format checking, actual function executions, and semantic verification, ensuring reliability and correctness.

According to human evaluation of 600 sampled data points, the dataset has a correctness rate above 95% (with the remaining 5% having minor issues like inaccurate arguments).

## Project Structure

```
.
├── .env                  # Environment variables (add your HF token here)
├── pyproject.toml        # Project dependencies and metadata
├── README.md             # Project documentation
├── src/                  # Source code
│   ├── __init__.py       # Makes src a package
│   ├── auth.py           # Hugging Face authentication utilities
│   └── dataset.py        # Dataset loading and processing utilities
└── notebooks/            # Jupyter notebooks
    └── dataset_exploration.ipynb  # Example notebook for exploring the dataset
```

## Setup

### Prerequisites

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) for dependency management
- A Hugging Face account with access to the dataset

### Installation

1. Clone this repository:
```bash
git clone https://github.com/KacperJanowski98/pllum-function-calling-impl.git
cd pllum-function-calling-impl
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

3. Create a `.env` file with your Hugging Face token:
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

### Exploring with Notebooks

Start Jupyter to explore the notebooks:

```bash
jupyter notebook
```

Then navigate to the `notebooks/dataset_exploration.ipynb` notebook to see examples of working with the dataset.

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
