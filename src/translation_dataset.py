"""Utilities for creating translated dataset."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from datasets import Dataset, load_dataset

from src.auth import login_to_huggingface
from src.dataset import parse_json_entry
from src.translator import batch_translate_queries

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_data_directory():
    """
    Create the data directory if it doesn't exist.
    """
    os.makedirs('data', exist_ok=True)
    logger.info("Ensured data directory exists")

def load_and_sample_dataset(
    dataset_name: str = "Salesforce/xlam-function-calling-60k",
    split: str = "train",
    sample_size: Optional[int] = None,
    random_seed: int = 42,
    cache_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load the dataset and sample a specified number of entries.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        split: Dataset split to use
        sample_size: Number of samples to take (None for all)
        random_seed: Random seed for reproducibility
        cache_dir: Directory to cache the dataset
        
    Returns:
        List of sampled dataset entries
    """
    # Ensure we're logged in to Hugging Face
    login_to_huggingface()
    
    logger.info(f"Loading dataset {dataset_name}")
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    
    # Convert to list for easier handling
    samples = list(dataset)
    total_samples = len(samples)
    logger.info(f"Dataset loaded with {total_samples} samples")
    
    # Sample if needed
    if sample_size is not None and sample_size < total_samples:
        random.seed(random_seed)
        samples = random.sample(samples, sample_size)
        logger.info(f"Sampled {sample_size} entries from the dataset")
        
    return samples

def create_translated_dataset(
    output_path: Union[str, Path],
    sample_size: Optional[int] = None,
    translation_percentage: float = 0.4,
    dataset_name: str = "Salesforce/xlam-function-calling-60k",
    split: str = "train",
    source_lang: str = "en",
    target_lang: str = "pl",
    random_seed: int = 42,
    cache_dir: Optional[str] = None
) -> str:
    """
    Create a dataset with translated queries and save it to a file.
    
    Args:
        output_path: Path to save the translated dataset
        sample_size: Number of samples to use (None for all)
        translation_percentage: Percentage of samples to translate
        dataset_name: Name of the dataset on Hugging Face
        split: Dataset split to use
        source_lang: Source language code
        target_lang: Target language code
        random_seed: Random seed for reproducibility
        cache_dir: Directory to cache the dataset
        
    Returns:
        Path to the saved dataset file
    """
    # Make sure output directory exists
    ensure_data_directory()
    
    # Convert output_path to Path object
    output_path = Path(output_path)
    
    # Load and sample the dataset
    samples = load_and_sample_dataset(
        dataset_name=dataset_name,
        split=split,
        sample_size=sample_size,
        random_seed=random_seed,
        cache_dir=cache_dir
    )
    
    # Translate a percentage of queries
    logger.info(f"Starting translation of {translation_percentage*100:.1f}% of the queries")
    translated_samples = batch_translate_queries(
        samples=samples,
        percentage=translation_percentage,
        src=source_lang,
        dest=target_lang
    )
    
    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the translated dataset
    logger.info(f"Saving translated dataset to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated_samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Successfully saved {len(translated_samples)} samples with {translation_percentage*100:.1f}% translated queries")
    return str(output_path)

def analyze_translated_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Analyze a translated dataset to check language distribution.
    
    Args:
        file_path: Path to the translated dataset file
        
    Returns:
        DataFrame with analysis results
    """
    logger.info(f"Analyzing translated dataset at {file_path}")
    
    # Load the dataset
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Count English vs Polish samples
    # This is a simplistic approach - a more robust approach would use a language detector
    results = []
    
    for i, sample in enumerate(data):
        query = sample.get('query', '')
        
        # Very basic language detection based on Polish characters
        has_polish_chars = any(c in query for c in 'ąćęłńóśźżĄĆĘŁŃÓŚŹŻ')
        
        results.append({
            'index': i,
            'language': 'Polish' if has_polish_chars else 'English',
            'query': query[:100] + '...' if len(query) > 100 else query
        })
    
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary = df['language'].value_counts().reset_index()
    summary.columns = ['Language', 'Count']
    summary['Percentage'] = 100 * summary['Count'] / len(df)
    
    logger.info(f"Analysis complete - found {len(df)} samples")
    return summary
