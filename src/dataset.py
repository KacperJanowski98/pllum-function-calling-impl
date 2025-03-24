"""Utilities for loading and processing the function calling dataset."""

import json
from typing import Dict, List, Optional, Union

from datasets import Dataset, load_dataset

from src.auth import login_to_huggingface


def load_function_calling_dataset(cache_dir: Optional[str] = None) -> Dataset:
    """Load the Salesforce/xlam-function-calling-60k dataset.
    
    Args:
        cache_dir: Directory to cache the dataset in. If None, uses the default cache.
        
    Returns:
        The loaded dataset.
    """
    # Ensure we're logged in to Hugging Face
    login_to_huggingface()
    
    # Load the dataset
    return load_dataset("Salesforce/xlam-function-calling-60k", cache_dir=cache_dir)


def parse_json_entry(entry: Dict) -> Dict:
    """Parse JSON strings in the dataset entry into Python objects.
    
    Args:
        entry: A dataset entry with JSON strings.
        
    Returns:
        The entry with JSON strings parsed into Python objects.
    """
    parsed_entry = {}
    
    for key, value in entry.items():
        if isinstance(value, str) and (key in ["query", "tools", "answers"]):
            try:
                parsed_entry[key] = json.loads(value)
            except json.JSONDecodeError:
                parsed_entry[key] = value
        else:
            parsed_entry[key] = value
    
    return parsed_entry


def get_tool_names(dataset: Dataset) -> List[str]:
    """Extract all unique tool names from the dataset.
    
    Args:
        dataset: The function calling dataset.
        
    Returns:
        List of unique tool names.
    """
    tool_names = set()
    
    for entry in dataset["train"]:
        parsed = parse_json_entry(entry)
        tools = parsed.get("tools", [])
        
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict) and "name" in tool:
                    tool_names.add(tool["name"])
    
    return sorted(list(tool_names))


def filter_by_tool(dataset: Dataset, tool_name: str) -> List[Dict]:
    """Filter dataset entries that use a specific tool.
    
    Args:
        dataset: The function calling dataset.
        tool_name: Name of the tool to filter by.
        
    Returns:
        List of entries that use the specified tool.
    """
    filtered_entries = []
    
    for entry in dataset["train"]:
        parsed = parse_json_entry(entry)
        tools = parsed.get("tools", [])
        
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict) and tool.get("name") == tool_name:
                    filtered_entries.append(parsed)
                    break
    
    return filtered_entries
