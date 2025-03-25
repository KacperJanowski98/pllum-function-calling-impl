"""Utilities for translating dataset content."""

import logging
import time
from typing import Any, Dict, List, Union

from googletrans import Translator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def translate_text(text: str, src: str = 'en', dest: str = 'pl', retries: int = 3, delay: int = 1) -> str:
    """
    Translate text from source language to destination language using googletrans.
    
    Args:
        text: The text to translate
        src: Source language code (default: 'en' for English)
        dest: Destination language code (default: 'pl' for Polish)
        retries: Number of retry attempts if translation fails
        delay: Delay in seconds between retries
        
    Returns:
        Translated text
    
    Raises:
        Exception: If translation fails after all retries
    """
    if not text or not isinstance(text, str):
        return text
    
    translator = Translator()
    
    for attempt in range(retries):
        try:
            result = translator.translate(text, src=src, dest=dest)
            return result.text
        except Exception as e:
            logger.warning(f"Translation attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Translation failed after {retries} attempts: {str(e)}")
                # Return original text if all translation attempts fail
                return text

def translate_query_in_sample(sample: Dict[str, Any], src: str = 'en', dest: str = 'pl') -> Dict[str, Any]:
    """
    Translate only the query field in a dataset sample.
    
    Args:
        sample: Dictionary representing a dataset sample
        src: Source language code
        dest: Destination language code
        
    Returns:
        Sample with translated query
    """
    result = sample.copy()
    
    # Extract query (might be a string or JSON string)
    query = sample.get('query', '')
    
    if query:
        # Translate the query
        translated_query = translate_text(query, src=src, dest=dest)
        result['query'] = translated_query
    
    return result

def batch_translate_queries(samples: List[Dict[str, Any]], 
                          percentage: float = 0.4, 
                          src: str = 'en', 
                          dest: str = 'pl',
                          batch_size: int = 10) -> List[Dict[str, Any]]:
    """
    Translate a percentage of queries in a dataset.
    
    Args:
        samples: List of dataset samples
        percentage: Percentage of samples to translate (0.0 to 1.0)
        src: Source language code
        dest: Destination language code
        batch_size: Number of samples to translate in a batch before logging progress
        
    Returns:
        Updated list of samples with some queries translated
    """
    if not samples:
        return []
    
    # Determine number of samples to translate
    num_to_translate = max(1, int(len(samples) * percentage))
    num_to_translate = min(num_to_translate, len(samples))
    
    logger.info(f"Translating queries for {num_to_translate} out of {len(samples)} samples ({percentage*100:.1f}%)")
    
    results = []
    translated_count = 0
    
    for i, sample in enumerate(samples):
        if i < num_to_translate:
            # Translate this sample
            translated_sample = translate_query_in_sample(sample, src=src, dest=dest)
            results.append(translated_sample)
            translated_count += 1
            
            # Log progress after each batch
            if translated_count % batch_size == 0 or translated_count == num_to_translate:
                logger.info(f"Translated {translated_count}/{num_to_translate} queries")
                
            # Add a small delay to avoid hitting API limits
            time.sleep(0.1)
        else:
            # Keep the rest unchanged
            results.append(sample)
    
    logger.info(f"Translation completed. Translated {translated_count} queries.")
    return results
