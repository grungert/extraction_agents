"""Dynamic model factory for creating Pydantic models from JSON configuration."""
from typing import Dict, List, Type, Optional, Any
from pydantic import Field, create_model
import logging # Import logging

from .models import BaseExtraction, ExtractionModelDefinition, FieldDefinition # Import Pydantic models
from .config_manager import ConfigurationManager
# Import PipelineException for wrapping errors
from .exceptions import PipelineException
# Assuming AppConfig is accessible, adjust import if needed
try:
    from .models import AppConfig
except ImportError:
    AppConfig = None # Keep this fallback for now

# Get the logger instance (assuming it's configured elsewhere, e.g., in display.py)
# If not, basicConfig might be needed here, but prefer centralized config.
logger = logging.getLogger(__name__) # Use module-level logger

def create_model_from_config(
    model_def: ExtractionModelDefinition, # Type hint with Pydantic model
    include_examples: bool = True  # Flag to control example inclusion in description
) -> Type[BaseExtraction]:
    """
    Dynamically create a Pydantic model from a configuration definition.

    Args:
        model_def: ExtractionModelDefinition instance containing model definition
        include_examples: Whether to include examples in field descriptions

    Returns:
        A dynamically created Pydantic model class

    Raises:
        PipelineException: If model creation fails.
    """
    try:
        fields = {}
        # Iterate over model_def.fields which is a Dict[str, FieldDefinition]
        for field_name, field_def in model_def.fields.items():
            # Use dot notation to access attributes of FieldDefinition
            description = field_def.description or "" # Use .attribute_name
            examples = field_def.examples or [] # Use .attribute_name
            # Conditionally add examples to the description
            if examples and include_examples:
                # Ensure examples are strings for joining
                examples_str_list = [str(ex) for ex in examples]
                description += f". Possible headers values: {', '.join(examples_str_list)}"

            type_str = field_def.type.lower() # Use .attribute_name

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
        model_name = model_def.name # Use dot notation
        model_doc = model_def.description or "" # Use dot notation

        # Create the model class with proper inheritance using create_model
        model_class = create_model(
            model_name,
            __base__=BaseExtraction,
            __doc__=model_doc,
            **fields
        )
        return model_class
    except Exception as e:
        # Enhanced Exception for model creation failure
        logger.error(f"Error creating model {model_def.name}: {e}", exc_info=True) # Log error
        raise PipelineException(
            message=f"Failed to dynamically create Pydantic model '{model_def.name}': {e}",
            error_code="MODEL_CREATION_FAILED",
            context={
                "model_name": model_def.name,
                "exception_type": e.__class__.__name__
            }
        )


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
    # Iterate over the list of ExtractionModelDefinition objects
    for model_def in config_manager.get_extraction_models():
        # model_def is already an ExtractionModelDefinition Pydantic object
        model_name = model_def.name # Use dot notation
        # Pass the flag when creating individual models
        # create_model_from_config now raises PipelineException on failure
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

    Raises:
        PipelineException: If formatting fails.
    """
    try:
        examples = []
        # Get the raw examples from the configuration
        # get_model_examples now raises ConfigurationError if model not found
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
    except Exception as e:
        # Enhanced Exception for example formatting failure
        logger.error(f"Error formatting examples for model {model_name}: {e}", exc_info=True) # Log error
        raise PipelineException(
            message=f"Failed to format examples for model '{model_name}': {e}",
            error_code="EXAMPLE_FORMATTING_FAILED",
            context={
                "model_name": model_name,
                "exception_type": e.__class__.__name__
            }
        )


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
    # create_models_from_config now propagates exceptions from create_model_from_config
    models = create_models_from_config(config_manager, include_examples)

    # Map section names to model classes (using model name as section name)
    extraction_models = {name: model for name, model in models.items()}

    return extraction_models
