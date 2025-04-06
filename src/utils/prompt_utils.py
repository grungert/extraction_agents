"""Utilities for working with prompt files."""
import os
from typing import Dict, Optional
from pathlib import Path


def get_prompt_path(file_name: str) -> str:
    """
    Get the absolute path to a prompt file.
    
    Args:
        file_name: Name of the prompt file
        
    Returns:
        Absolute path to the prompt file
    """
    base_dir = Path(__file__).resolve().parent.parent
    return os.path.join(base_dir, "prompts", file_name)


def load_prompt(file_name: str) -> str:
    """
    Load a prompt from a file.
    
    Args:
        file_name: Name of the prompt file
        
    Returns:
        Contents of the prompt file
    """
    path = get_prompt_path(file_name)
    with open(path, 'r') as f:
        return f.read()


def load_template_prompt(file_name: str, **kwargs) -> str:
    """
    Load a prompt template from a file and format it.
    
    Args:
        file_name: Name of the prompt template file
        **kwargs: Keys and values to format the template
        
    Returns:
        Formatted prompt template
    """
    template = load_prompt(file_name)
    return template.format(**kwargs)
