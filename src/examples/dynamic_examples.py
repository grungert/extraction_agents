"""Dynamic example management for LLM extraction agents."""
import json
from typing import Dict, List, Any, Optional, Type, Union

from ..config_manager import ConfigurationManager, convert_pascal_to_snake
from ..models import BaseExtraction


def get_extraction_examples_from_config(
    config_manager: ConfigurationManager, model_name: str
) -> List[Dict[str, Any]]:
    """
    Get examples for a specific model from configuration.
    
    Args:
        config_manager: Configuration manager instance
        model_name: Name of the model
        
    Returns:
        List of examples for the specified model
    """
    examples = []
    
    # Get the raw examples from the configuration
    raw_examples = config_manager.get_model_examples(model_name)
    
    for example in raw_examples:
        # Use PascalCase keys directly from expected fields
        expected = example.get("expected", {}).copy()
        
        # Add validation confidence if not present
        if "ValidationConfidence" not in expected:
            expected["ValidationConfidence"] = 0.9
        
        examples.append({
            "table": example.get("table", ""),
            "json": expected
        })
    
    return examples


def create_validation_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a validation example from an extraction example.
    
    Args:
        example: The extraction example
        
    Returns:
        A validation example with modified data
    """
    # Copy the example
    validation_example = example.copy()
    
    # Create a modified version of the JSON data
    modified_json = example["json"].copy()
    
    # Modify some fields to demonstrate validation
    for field, value in modified_json.items():
        if field == "ValidationConfidence":
            continue
            
        # For string fields, modify them slightly
        if isinstance(value, str) and value:
            # Uppercase the value to demonstrate case correction
            modified_json[field] = value.upper()
        # For non-null fields that aren't strings, set to None to demonstrate adding missing values
        elif value is not None and not isinstance(value, (int, float, bool)):
            modified_json[field] = None
    
    # Create the validation input
    validation_example["modified_json"] = modified_json
    
    # Create the expected validation output
    validation_example["validation_output"] = {
        "ValidatedData": example["json"],
        "ValidationConfidence": example["json"].get("ValidationConfidence", 0.9),
        "CorrectionsMade": ["Corrected data based on table context"]
    }
    
    return validation_example


def format_messages(examples: List[Dict[str, Any]], system_message: Dict[str, str], 
                   message_type: str = "extraction") -> List[Dict[str, str]]:
    """
    Format examples for LLM agents.
    
    Args:
        examples: List of examples
        system_message: System message for the LLM
        message_type: Type of formatting to apply ("extraction" or "validation")
        
    Returns:
        List of formatted messages for the LLM
    """
    formatted_examples = []
    
    for idx, example in enumerate(examples, start=1):
        if message_type == "extraction":
            # Simple extraction format
            input_content = f"""
# Example {idx}
# Input Table 
{example["table"]}"""
            output_content = json.dumps(example["json"])
            
        elif message_type == "validation":
            # Create a validation example from the extraction example
            validation_example = create_validation_example(example)
            
            input_content = f"""
# Example {idx}
# Original Table
{example["table"]}

# Extracted Data
```json
{json.dumps(validation_example["modified_json"], indent=2)}
```"""
            output_content = json.dumps(validation_example["validation_output"])
            
        else:
            raise ValueError(f"Unknown message type: {message_type}")

        formatted_examples.extend([
            {"role": "user", "content": input_content},
            {"role": "assistant", "content": output_content}
        ])
    
    return [system_message] + formatted_examples


def format_extraction_messages(
    config_manager: ConfigurationManager, 
    model_name: str, 
    system_message: Dict[str, str]
) -> List[Dict[str, str]]:
    """
    Format examples for extraction agent.
    
    Args:
        config_manager: Configuration manager instance
        model_name: Name of the model
        system_message: System message for the LLM
        
    Returns:
        List of formatted messages for the LLM
    """
    examples = get_extraction_examples_from_config(config_manager, model_name)
    return format_messages(examples, system_message, "extraction")


def format_validation_messages(
    config_manager: ConfigurationManager,
    system_message: Dict[str, str]
) -> List[Dict[str, str]]:
    """
    Format examples for validation agent.
    
    Args:
        config_manager: Configuration manager instance
        system_message: System message for the LLM
        
    Returns:
        List of formatted messages for the LLM
    """
    # Get examples for all models
    all_examples = []
    for model_def in config_manager.get_extraction_models():
        model_name = model_def.get("name")
        examples = get_extraction_examples_from_config(config_manager, model_name)
        all_examples.extend(examples)
    
    return format_messages(all_examples, system_message, "validation")
