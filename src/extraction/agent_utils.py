"""Utility functions for dynamic agents."""
import logging
from typing import Dict, Type

import tiktoken

from ..utils.display import logger
from src.models import BaseExtraction
from ..config_manager import ConfigurationManager
from ..dynamic_model_factory import create_extraction_models_dict as factory_create_models


# Helper function to get tokenizer
def get_tokenizer(tokenizer_name: str = "cl100k_base"):
    try:
        return tiktoken.get_encoding(tokenizer_name)
    except Exception as e:
        logger.warning(f"Error loading tiktoken encoding '{tokenizer_name}': {e}. Token counting may be inaccurate.")
        return None

# Helper function to count tokens
def count_tokens(encoding, text):
    if encoding:
        return len(encoding.encode(text))
    # Estimate if no encoding (very rough)
    return len(text) // 4

# Helper function to get extraction models from config manager
def create_extraction_models_dict(
    config_manager: ConfigurationManager,
    include_examples: bool = True
) -> Dict[str, Type[BaseExtraction]]:
    """
    Create a dictionary mapping section names to model classes.

    Args:
        config_manager: Configuration manager instance
        include_examples: Whether to include examples in model descriptions

    Returns:
        Dictionary mapping section names to model classes
    """
    # Call the factory function, passing the include_examples flag
    return factory_create_models(config_manager, include_examples=include_examples)
