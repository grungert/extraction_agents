"""Dynamic model factory for creating Pydantic models from JSON configuration."""
from typing import Dict, List, Type, Optional, Any
from pydantic import Field, create_model

from .models import BaseExtraction
from .config_manager import ConfigurationManager
# Assuming AppConfig is accessible, adjust import if needed
try:
    from .models import AppConfig
except ImportError:
    AppConfig = None


def create_model_from_config(
    model_def: Dict[str, Any], 
    include_examples: bool = True  # Flag to control example inclusion in description
) -> Type[BaseExtraction]:
    """
    Dynamically create a Pydantic model from a configuration definition.
    
    Args:
        model_def: Dictionary containing model definition
        include_examples: Whether to include examples in field descriptions
        
    Returns:
        A dynamically created Pydantic model class
    """
    fields = {}
    
    # Add fields from the configuration definition
    for field_name, field_def in model_def.get("fields", {}).items():
        # Use PascalCase field names directly
        description = field_def.get("description", "")
        examples = field_def.get("examples", [])
        # Conditionally add examples to the description
        if examples and include_examples:
            description += f". Possible headers values: {', '.join(examples)}"
        
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


def create_models_from_config(
    config_manager: ConfigurationManager, 
    include_examples: bool = True  # Pass flag down
) -> Dict[str, Type[BaseExtraction]]:
    """
    Create all models defined in the configuration.
    
    Args:
        config_manager: Configuration manager instance
        include_examples: Whether to include examples in field descriptions
        
    Returns:
        Dictionary mapping model names to model classes
    """
    models = {}
    
    for model_def in config_manager.get_extraction_models():
        model_name = model_def.get("name")
        # Pass the flag when creating individual models
        model_class = create_model_from_config(model_def, include_examples)
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


def create_extraction_models_dict(
    config_manager: ConfigurationManager, 
    include_examples: bool = True # Pass flag down
) -> Dict[str, Type[BaseExtraction]]:
    """
    Create a dictionary mapping section names to model classes.
    
    Args:
        config_manager: Configuration manager instance
        include_examples: Whether to include examples in field descriptions
        
    Returns:
        Dictionary mapping section names to model classes
    """
    # Create all models, passing the flag
    models = create_models_from_config(config_manager, include_examples)
    
    # Map section names to model classes (using model name as section name)
    extraction_models = {name: model for name, model in models.items()}
    
    return extraction_models
