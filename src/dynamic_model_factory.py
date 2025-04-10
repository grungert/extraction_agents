"""Dynamic model factory for creating Pydantic models from JSON configuration."""
from typing import Dict, List, Type, Optional, Any
from pydantic import Field, create_model

from .models import BaseExtraction
from .config_manager import ConfigurationManager


def create_model_from_config(model_def: Dict[str, Any]) -> Type[BaseExtraction]:
    """
    Dynamically create a Pydantic model from a configuration definition.
    
    Args:
        model_def: Dictionary containing model definition
        
    Returns:
        A dynamically created Pydantic model class
    """
    fields = {}
    
    # Add fields from the configuration definition
    for field_name, field_def in model_def.get("fields", {}).items():
        # Use PascalCase field names directly
        description = field_def.get("description", "")
        examples = field_def.get("examples", [])
        if examples:
            description += f". Possible headers: {', '.join(examples)}"
        
        type_str = field_def.get("type", "string").lower()

        type_map = {
            "string": str,
            "int": int,
            "float": float,
            "bool": bool,
            "date": "date",       # Placeholder, will be replaced below
            "datetime": "datetime" # Placeholder
        }

        py_type = type_map.get(type_str, str)

        # Import datetime types
        from datetime import date as dt_date, datetime as dt_datetime
        if py_type == "date":
            py_type = dt_date
        elif py_type == "datetime":
            py_type = dt_datetime

        fields[field_name] = (Optional[py_type], Field(None, description=description))
    
    # Add ValidationConfidence field that's standard in BaseExtraction
    fields["ValidationConfidence"] = (Optional[float], Field(None, description="Confidence in validation (0.0-1.0)"))
    
    # Create the model class dynamically
    model_name = model_def.get("name", "DynamicModel")
    model_doc = model_def.get("description", "")
    
    # Create the model class with proper inheritance using create_model
    model_class = create_model(
        model_name,
        __base__=BaseExtraction,
        __doc__=model_doc,
        **fields
    )
    
    return model_class


def create_models_from_config(config_manager: ConfigurationManager) -> Dict[str, Type[BaseExtraction]]:
    """
    Create all models defined in the configuration.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Dictionary mapping model names to model classes
    """
    models = {}
    
    for model_def in config_manager.get_extraction_models():
        model_name = model_def.get("name")
        model_class = create_model_from_config(model_def)
        models[model_name] = model_class
    
    return models


def format_examples_from_config(config_manager: ConfigurationManager, model_name: str) -> List[Dict[str, Any]]:
    """
    Format examples from configuration for use with LLM.
    
    Args:
        config_manager: Configuration manager instance
        model_name: Name of the model
        
    Returns:
        List of formatted examples
    """
    examples = []
    
    # Get the raw examples from the configuration
    raw_examples = config_manager.get_model_examples(model_name)
    
    for example in raw_examples:
        # Use PascalCase keys directly from the expected fields
        expected = example.get("expected", {}).copy()
        
        # Add validation confidence if not present
        if "ValidationConfidence" not in expected:
            expected["ValidationConfidence"] = 0.9
        
        examples.append({
            "table": example.get("table", ""),
            "json": expected
        })
    
    return examples


def create_extraction_models_dict(config_manager: ConfigurationManager) -> Dict[str, Type[BaseExtraction]]:
    """
    Create a dictionary mapping section names to model classes.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Dictionary mapping section names to model classes
    """
    # Create all models
    models = create_models_from_config(config_manager)
    
    # Map section names to model classes
    extraction_models = {}
    for model_name, model_class in models.items():
        # Use the model name as the section name
        extraction_models[model_name] = model_class
    
    return extraction_models
