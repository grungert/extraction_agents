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
from .exceptions import ConfigurationError, FileProcessingError, PipelineException # Import enhanced exceptions

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
            ConfigurationError: If the configuration file is not found, invalid JSON, or fails Pydantic validation.
        """
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            # Use Pydantic model for validation and structuring
            self.config = ExtractionConfig(**config_data)
        except FileNotFoundError:
            raise ConfigurationError(
                message=f"Configuration file not found",
                error_code="CONFIG_FILE_NOT_FOUND",
                context={"path": self.config_path}
            )
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                message=f"Invalid JSON in configuration file: {e}",
                error_code="CONFIG_INVALID_JSON",
                context={"path": self.config_path, "error_details": str(e)}
            )
        except PydanticValidationError as e:
            raise ConfigurationError(
                message=f"Configuration validation failed",
                error_code="CONFIG_VALIDATION_FAILED",
                context={"path": self.config_path, "validation_errors": str(e)}
            )
        except Exception as e:
            # Catch any other unexpected errors during loading/validation
            raise ConfigurationError(
                message=f"An unexpected error occurred while loading configuration: {e}",
                error_code="CONFIG_UNEXPECTED_LOAD_ERROR",
                context={"path": self.config_path, "exception_type": e.__class__.__name__}
            )

    def _ensure_config_loaded(self):
        """Raise an error if the configuration hasn't been loaded."""
        if self.config is None:
             # This should ideally not happen if __init__ succeeded, but good practice
             raise ConfigurationError(
                 message="Configuration accessed before successful loading.",
                 error_code="CONFIG_NOT_LOADED",
                 severity=PipelineException.CRITICAL # This indicates a programming error
             )

    def get_header_detection_config(self) -> HeaderDetectionConfig:
        """
        Get the header detection configuration.

        Returns:
            HeaderDetectionConfig instance
        """
        self._ensure_config_loaded()
        return self.config.header_detection

    def get_header_examples(self) -> List[Dict[str, Any]]:
        """
        Get the header detection examples.

        Returns:
            List of header example dictionaries
        """
        self._ensure_config_loaded()
        # Ensure examples exist, return empty list if not (or raise if required)
        return getattr(self.config.header_detection, 'examples', [])


    def get_validation_config(self) -> ValidationConfig:
        """
        Get the validation configuration.

        Returns:
            ValidationConfig instance
        """
        self._ensure_config_loaded()
        return self.config.validation

    def get_prompts_config(self) -> PromptsConfig:
        """Get the prompts configuration section."""
        self._ensure_config_loaded()
        return self.config.prompts

    def get_header_detection_prompt_file(self) -> str:
        """Get the filename for the header detection system prompt."""
        self._ensure_config_loaded()
        return self.config.prompts.header_detection_system_prompt_file

    def get_header_validation_prompt_file(self) -> str:
        """Get the filename for the header validation system prompt."""
        self._ensure_config_loaded()
        return self.config.prompts.header_validation_system_prompt_file

    def get_deduplication_system_prompt_file(self) -> str:
        """Get the filename for the deduplication system prompt."""
        self._ensure_config_loaded()
        return self.config.prompts.deduplication_system_prompt_file

    def get_deduplication_instruction_prompt_file(self) -> str:
        """Get the filename for the deduplication instruction prompt."""
        self._ensure_config_loaded()
        return self.config.prompts.deduplication_instruction_prompt_file

    def get_section_extraction_template_file(self) -> str:
        """Get the filename for the section extraction template."""
        self._ensure_config_loaded()
        return self.config.prompts.section_extraction_template_file

    def get_section_validation_template_file(self) -> str:
        """Get the filename for the section validation template."""
        self._ensure_config_loaded()
        return self.config.prompts.section_validation_template_file


    def get_extraction_models(self) -> List[ExtractionModelDefinition]:
        """
        Get the extraction model definitions.

        Returns:
            List of ExtractionModelDefinition instances
        """
        self._ensure_config_loaded()
        return self.config.extraction_models

    def get_model_by_name(self, name: str) -> Optional[ExtractionModelDefinition]:
        """
        Get a model definition by name.

        Args:
            name: Name of the model

        Returns:
            ExtractionModelDefinition instance, or None if not found
        """
        self._ensure_config_loaded()
        for model in self.config.extraction_models:
            if model.name == name:
                return model
        # Return None if not found, let caller handle it or raise specific error
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
            raise ConfigurationError(
                message=f"Model definition not found in configuration",
                error_code="CONFIG_MODEL_NOT_FOUND",
                context={"model_name": name, "config_path": self.config_path}
            )
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
            raise ConfigurationError(
                message=f"Model definition not found in configuration",
                error_code="CONFIG_MODEL_NOT_FOUND",
                 context={"model_name": name, "config_path": self.config_path}
            )
        # Ensure examples exist, return empty list if not
        return getattr(model, 'examples', [])


def get_configuration_manager(config_path: str) -> ConfigurationManager:
    """
    Get a configuration manager instance.

    Args:
        config_path: Path to the configuration file (required)

    Returns:
        ConfigurationManager instance

    Raises:
        ConfigurationError: If the configuration file is not found or invalid.
    """
    # Check if path exists before creating manager
    if not os.path.exists(config_path):
         raise ConfigurationError(
             message=f"Configuration file specified does not exist",
             error_code="CONFIG_FILE_NOT_FOUND",
             context={"path": config_path}
         )

    # ConfigurationManager constructor now handles loading, validation, and specific errors
    try:
        return ConfigurationManager(config_path)
    except ConfigurationError:
        # Re-raise ConfigurationError directly
        raise
    except Exception as e:
        # Wrap unexpected errors during instantiation
        raise ConfigurationError(
            message=f"Unexpected error creating ConfigurationManager: {e}",
            error_code="CONFIG_MANAGER_INIT_FAILED",
            context={"path": config_path, "exception_type": e.__class__.__name__}
        )
