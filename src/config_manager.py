"""Configuration manager for the dynamically configurable extraction system."""
import os
import json
from typing import Dict, List, Any, Optional, Type, Union
from pydantic import ValidationError as PydanticValidationError # Import Pydantic's ValidationError

# Import the new Pydantic models for configuration
from .models import (
    ExtractionConfig,
    HeaderDetectionConfig,
    ValidationConfig,
    ExtractionModelDefinition,
    PromptsConfig,
    FieldDefinition
)

# Import custom exceptions
from .exceptions import ConfigurationError, FileProcessingError # Import ConfigurationError

class ConfigurationManager:
    """Manager for loading and validating configuration from JSON using Pydantic models."""

    def __init__(self, config_path: str):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the JSON configuration file
        """
        self.config_path = config_path
        self.config: Optional[ExtractionConfig] = None # Store as Pydantic model
        self._load_and_validate_config()

    def _load_and_validate_config(self) -> None:
        """
        Load and validate the configuration from a JSON file using Pydantic.

        Raises:
            ConfigurationError: If the configuration file is not found, invalid JSON, or fails Pydantic validation
            RuntimeError: For other unexpected errors during loading/validation
        """
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            # Use Pydantic model for validation and structuring
            self.config = ExtractionConfig(**config_data)
        except FileNotFoundError:
            # Raise custom ConfigurationError
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            # Raise custom ConfigurationError
            raise ConfigurationError(f"Invalid JSON in configuration file: {str(e)}")
        except PydanticValidationError as e:
            # Raise custom ConfigurationError with validation details
            raise ConfigurationError(f"Configuration validation failed in {self.config_path}:\n{e}")
        except Exception as e:
            # Catch any other unexpected errors during loading/validation and raise as RuntimeError
            # Keep RuntimeError for truly unexpected issues not related to config format/existence
            raise RuntimeError(f"An unexpected error occurred while loading configuration from {self.config_path}: {e}")


    def get_header_detection_config(self) -> HeaderDetectionConfig:
        """
        Get the header detection configuration.

        Returns:
            HeaderDetectionConfig instance
        """
        if self.config is None:
             raise RuntimeError("Configuration not loaded.")
        return self.config.header_detection

    def get_header_examples(self) -> List[Dict[str, Any]]:
        """
        Get the header detection examples.

        Returns:
            List of header example dictionaries
        """
        if self.config is None:
             raise RuntimeError("Configuration not loaded.")
        return self.config.header_detection.examples

    def get_validation_config(self) -> ValidationConfig:
        """
        Get the validation configuration.

        Returns:
            ValidationConfig instance
        """
        if self.config is None:
             raise RuntimeError("Configuration not loaded.")
        return self.config.validation

    def get_prompts_config(self) -> PromptsConfig:
        """Get the prompts configuration section."""
        if self.config is None:
             raise RuntimeError("Configuration not loaded.")
        return self.config.prompts

    def get_header_detection_prompt_file(self) -> str:
        """Get the filename for the header detection system prompt."""
        if self.config is None:
             raise RuntimeError("Configuration not loaded.")
        return self.config.prompts.header_detection_system_prompt_file

    def get_header_validation_prompt_file(self) -> str:
        """Get the filename for the header validation system prompt."""
        if self.config is None:
             raise RuntimeError("Configuration not loaded.")
        return self.config.prompts.header_validation_system_prompt_file

    def get_deduplication_system_prompt_file(self) -> str:
        """Get the filename for the deduplication system prompt."""
        if self.config is None:
             raise RuntimeError("Configuration not loaded.")
        return self.config.prompts.deduplication_system_prompt_file

    def get_deduplication_instruction_prompt_file(self) -> str:
        """Get the filename for the deduplication instruction prompt."""
        if self.config is None:
             raise RuntimeError("Configuration not loaded.")
        return self.config.prompts.deduplication_instruction_prompt_file

    def get_section_extraction_template_file(self) -> str:
        """Get the filename for the section extraction template."""
        if self.config is None:
             raise RuntimeError("Configuration not loaded.")
        return self.config.prompts.section_extraction_template_file

    def get_section_validation_template_file(self) -> str:
        """Get the filename for the section validation template."""
        if self.config is None:
             raise RuntimeError("Configuration not loaded.")
        return self.config.prompts.section_validation_template_file


    def get_extraction_models(self) -> List[ExtractionModelDefinition]:
        """
        Get the extraction model definitions.

        Returns:
            List of ExtractionModelDefinition instances
        """
        if self.config is None:
             raise RuntimeError("Configuration not loaded.")
        return self.config.extraction_models

    def get_model_by_name(self, name: str) -> Optional[ExtractionModelDefinition]:
        """
        Get a model definition by name.

        Args:
            name: Name of the model

        Returns:
            ExtractionModelDefinition instance, or None if not found
        """
        if self.config is None:
             raise RuntimeError("Configuration not loaded.")
        for model in self.config.extraction_models:
            if model.name == name:
                return model
        return None

    def get_model_fields(self, name: str) -> Dict[str, FieldDefinition]:
        """
        Get the fields for a model.

        Args:
            name: Name of the model

        Returns:
            Dictionary mapping field names to FieldDefinition instances

        Raises:
            ConfigurationError: If the model is not found
        """
        model = self.get_model_by_name(name)
        if model is None:
            # Raise custom ConfigurationError
            raise ConfigurationError(f"Model definition not found in configuration: {name}")
        return model.fields

    def get_model_examples(self, name: str) -> List[Dict[str, Any]]:
        """
        Get the examples for a model.

        Args:
            name: Name of the model

        Returns:
            List of dictionaries containing examples

        Raises:
            ConfigurationError: If the model is not found
        """
        model = self.get_model_by_name(name)
        if model is None:
            # Raise custom ConfigurationError
            raise ConfigurationError(f"Model definition not found in configuration: {name}")
        return model.examples


def get_configuration_manager(config_path: str) -> ConfigurationManager:
    """
    Get a configuration manager instance.

    Args:
        config_path: Path to the configuration file (required)

    Returns:
        ConfigurationManager instance

    Raises:
        ConfigurationError: If the configuration file is not found or invalid
        RuntimeError: For other unexpected errors during loading
    """
    # Check if path exists before creating manager
    if not os.path.exists(config_path):
         # Raise custom ConfigurationError
         raise ConfigurationError(f"Configuration file specified does not exist: {config_path}")

    # ConfigurationManager now handles loading and Pydantic validation internally
    return ConfigurationManager(config_path)
