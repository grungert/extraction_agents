#!/usr/bin/env python3
"""Test script for dynamic model loading."""
import json
import os
from pprint import pprint

from src.model_factory import load_models_from_config, validate_model_config


def main():
    """Test dynamic model loading."""
    # Path to the model configuration file
    config_path = "config/models.json"
    
    # Validate the configuration
    print("Validating model configuration...")
    errors = validate_model_config(config_path)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    print("Configuration is valid.")
    
    # Load models from configuration
    print("\nLoading models from configuration...")
    models = load_models_from_config(config_path)
    
    # Print loaded models
    print(f"\nLoaded {len(models)} models:")
    for key, model_class in models.items():
        print(f"  - {key}: {model_class.__name__}")
        
        # Print model fields
        print("    Fields:")
        for field_name, field in model_class.__fields__.items():
            if field_name != "__root__":
                field_type = field.annotation.__name__ if hasattr(field.annotation, "__name__") else str(field.annotation)
                print(f"      - {field_name}: {field_type}")
                if field.description:
                    print(f"        Description: {field.description}")
    
    # Create an instance of each model
    print("\nCreating model instances:")
    for key, model_class in models.items():
        instance = model_class()
        print(f"  - {key}: {instance}")
        
        # Convert to dict
        instance_dict = instance.model_dump()
        print(f"    Dict: {instance_dict}")
        
        # Convert to JSON
        instance_json = instance.model_dump_json()
        print(f"    JSON: {instance_json}")
        
        print()


if __name__ == "__main__":
    main()
