"""Dynamic model factory for creating Pydantic models from JSON configuration."""
import json
from typing import Dict, List, Type, Optional, Any, get_type_hints
from pydantic import BaseModel, Field, create_model

from .models import BaseExtraction


def create_model_from_json(model_def: Dict) -> Type[BaseExtraction]:
    """
    Dynamically create a Pydantic model from a JSON definition.
    
    Args:
        model_def: Dictionary containing model definition
        
    Returns:
        A dynamically created Pydantic model class
    """
    fields = {}
    
    # Add fields from the JSON definition
    for field_def in model_def["fields"]:
        field_name = field_def["name"]
        field_type = field_def["type"]
        optional = field_def.get("optional", False)
        description = field_def.get("description", "")
        
        # Map JSON types to Python types
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            # Add more type mappings as needed
        }
        
        python_type = type_mapping.get(field_type, str)
        
        # Create the field with proper type annotation
        if optional:
            fields[field_name] = (Optional[python_type], Field(None, description=description))
        else:
            fields[field_name] = (python_type, Field(..., description=description))
    
    # Add ValidationConfidence field that's in BaseExtraction
    fields["ValidationConfidence"] = (Optional[float], Field(None, description="Confidence in validation (0.0-1.0)"))
    
    # Create the model class dynamically
    model_name = model_def["name"]
    model_doc = model_def.get("description", "")
    
    # Create the model class with proper inheritance using create_model
    model_class = create_model(
        model_name,
        __base__=BaseExtraction,
        __doc__=model_doc,
        **fields
    )
    
    return model_class


def load_models_from_config(config_path: str) -> Dict[str, Type[BaseExtraction]]:
    """
    Load model definitions from a JSON configuration file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dictionary mapping model names to model classes
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    models = {}
    
    # Create model classes from definitions
    for model_def in config["models"]:
        model_class = create_model_from_json(model_def)
        models[model_def["name"]] = model_class
    
    # Create the extraction models dictionary
    extraction_models = {}
    for key, model_name in config.get("extraction_models", {}).items():
        if model_name in models:
            extraction_models[key] = models[model_name]
    
    return extraction_models


def validate_model_config(config_path: str) -> List[str]:
    """
    Validate a model configuration file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {str(e)}"]
    except FileNotFoundError:
        return [f"File not found: {config_path}"]
    
    # Check required top-level keys
    if "models" not in config:
        errors.append("Missing required key: 'models'")
    elif not isinstance(config["models"], list):
        errors.append("'models' must be a list")
    
    if "extraction_models" not in config:
        errors.append("Missing required key: 'extraction_models'")
    elif not isinstance(config["extraction_models"], dict):
        errors.append("'extraction_models' must be a dictionary")
    
    # Check model definitions
    if "models" in config and isinstance(config["models"], list):
        model_names = set()
        
        for i, model_def in enumerate(config["models"]):
            # Check required model keys
            if "name" not in model_def:
                errors.append(f"Model at index {i} is missing required key: 'name'")
            elif not isinstance(model_def["name"], str):
                errors.append(f"Model 'name' at index {i} must be a string")
            else:
                if model_def["name"] in model_names:
                    errors.append(f"Duplicate model name: {model_def['name']}")
                model_names.add(model_def["name"])
            
            if "fields" not in model_def:
                errors.append(f"Model at index {i} is missing required key: 'fields'")
            elif not isinstance(model_def["fields"], list):
                errors.append(f"Model 'fields' at index {i} must be a list")
            else:
                # Check field definitions
                for j, field_def in enumerate(model_def["fields"]):
                    if "name" not in field_def:
                        errors.append(f"Field at index {j} in model {model_def.get('name', i)} is missing required key: 'name'")
                    elif not isinstance(field_def["name"], str):
                        errors.append(f"Field 'name' at index {j} in model {model_def.get('name', i)} must be a string")
                    
                    if "type" not in field_def:
                        errors.append(f"Field at index {j} in model {model_def.get('name', i)} is missing required key: 'type'")
                    elif not isinstance(field_def["type"], str):
                        errors.append(f"Field 'type' at index {j} in model {model_def.get('name', i)} must be a string")
                    elif field_def["type"] not in ["string", "integer", "number", "boolean"]:
                        errors.append(f"Field 'type' at index {j} in model {model_def.get('name', i)} must be one of: string, integer, number, boolean")
    
    # Check extraction_models references
    if "extraction_models" in config and isinstance(config["extraction_models"], dict) and "models" in config and isinstance(config["models"], list):
        model_names = {model_def["name"] for model_def in config["models"] if "name" in model_def}
        
        for key, model_name in config["extraction_models"].items():
            if not isinstance(model_name, str):
                errors.append(f"Extraction model value for key '{key}' must be a string")
            elif model_name not in model_names:
                errors.append(f"Extraction model '{model_name}' referenced by key '{key}' is not defined in 'models'")
    
    return errors
