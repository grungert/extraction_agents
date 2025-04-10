"""Configuration manager for the dynamically configurable extraction system."""
import os
import json
from typing import Dict, List, Any, Optional

class ConfigurationManager:
    """Manager for loading and validating configuration from JSON."""
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from a JSON file.
        
        Returns:
            Dictionary containing configuration
        
        Raises:
            FileNotFoundError: If the configuration file is not found
            json.JSONDecodeError: If the configuration file is not valid JSON
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {str(e)}")
    
    def _validate_config(self) -> None:
        """
        Validate the configuration.
        
        Raises:
            ValueError: If the configuration is invalid
        """
        errors = []
        
        # Check required sections
        if "extraction_models" not in self.config:
            errors.append("Missing required section: extraction_models")
        elif not isinstance(self.config["extraction_models"], list):
            errors.append("extraction_models must be a list")
        
        # Check header detection section
        if "header_detection" not in self.config:
            errors.append("Missing required section: header_detection")
        elif not isinstance(self.config["header_detection"], dict):
            errors.append("header_detection must be a dictionary")
        
        # Check validation section
        if "validation" not in self.config:
            errors.append("Missing required section: validation")
        elif not isinstance(self.config["validation"], dict):
            errors.append("validation must be a dictionary")
        
        # Check extraction models
        if "extraction_models" in self.config and isinstance(self.config["extraction_models"], list):
            model_names = set()
            
            for i, model in enumerate(self.config["extraction_models"]):
                # Check required model keys
                if "name" not in model:
                    errors.append(f"Model at index {i} is missing required key: name")
                elif not isinstance(model["name"], str):
                    errors.append(f"Model name at index {i} must be a string")
                else:
                    if model["name"] in model_names:
                        errors.append(f"Duplicate model name: {model['name']}")
                    model_names.add(model["name"])
                
                # Check fields
                if "fields" not in model:
                    errors.append(f"Model {model.get('name', f'at index {i}')} is missing required key: fields")
                elif not isinstance(model["fields"], dict):
                    errors.append(f"Model fields for {model.get('name', f'at index {i}')} must be a dictionary")
                else:
                    for field_name, field in model["fields"].items():
                        if not isinstance(field, dict):
                            errors.append(f"Field {field_name} in model {model.get('name', f'at index {i}')} must be a dictionary")
                        else:
                            if "description" not in field:
                                errors.append(f"Field {field_name} in model {model.get('name', f'at index {i}')} is missing required key: description")
                
                # Check examples
                if "examples" not in model:
                    errors.append(f"Model {model.get('name', f'at index {i}')} is missing required key: examples")
                elif not isinstance(model["examples"], list):
                    errors.append(f"Model examples for {model.get('name', f'at index {i}')} must be a list")
                else:
                    for j, example in enumerate(model["examples"]):
                        if "table" not in example:
                            errors.append(f"Example at index {j} in model {model.get('name', f'at index {i}')} is missing required key: table")
                        if "expected" not in example:
                            errors.append(f"Example at index {j} in model {model.get('name', f'at index {i}')} is missing required key: expected")
        
        if errors:
            raise ValueError("\n".join(errors))
    
    def get_header_detection_config(self) -> Dict[str, Any]:
        """
        Get the header detection configuration.
        
        Returns:
            Dictionary containing header detection configuration
        """
        return self.config.get("header_detection", {}).get("config", {})
    
    def get_header_examples(self) -> List[Dict[str, Any]]:
        """
        Get the header detection examples.
        
        Returns:
            List of header example dictionaries
        """
        return self.config.get("header_detection", {}).get("examples", [])
    
    def get_validation_config(self) -> Dict[str, Any]:
        """
        Get the validation configuration.
        
        Returns:
            Dictionary containing validation configuration
        """
        return self.config.get("validation", {})
        
    
    def get_extraction_models(self) -> List[Dict[str, Any]]:
        """
        Get the extraction model definitions.
        
        Returns:
            List of dictionaries containing model definitions
        """
        return self.config.get("extraction_models", [])
    
    def get_model_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a model definition by name.
        
        Args:
            name: Name of the model
            
        Returns:
            Dictionary containing model definition, or None if not found
        """
        for model in self.get_extraction_models():
            if model.get("name") == name:
                return model
        return None
    
    def get_model_fields(self, name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get the fields for a model.
        
        Args:
            name: Name of the model
            
        Returns:
            Dictionary mapping field names to field definitions
            
        Raises:
            ValueError: If the model is not found
        """
        model = self.get_model_by_name(name)
        if model is None:
            raise ValueError(f"Model not found: {name}")
        return model.get("fields", {})
    
    def get_model_examples(self, name: str) -> List[Dict[str, Any]]:
        """
        Get the examples for a model.
        
        Args:
            name: Name of the model
            
        Returns:
            List of dictionaries containing examples
            
        Raises:
            ValueError: If the model is not found
        """
        model = self.get_model_by_name(name)
        if model is None:
            raise ValueError(f"Model not found: {name}")
        return model.get("examples", [])


def get_configuration_manager(config_path: Optional[str] = None) -> ConfigurationManager:
    """
    Get a configuration manager instance.
    
    Args:
        config_path: Path to the configuration file, or None to use the default path
        
    Returns:
        ConfigurationManager instance
    """
    if config_path is None:
        # Use the default path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "full_config.json")
    
    return ConfigurationManager(config_path)
