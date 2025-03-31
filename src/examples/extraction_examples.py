from typing import List, Dict, Any, Type, Union
import json
from ..models import BaseExtraction, EXTRACTION_MODELS

def get_denomination_examples() -> List[Dict[str, Any]]:
    """
    Return examples for DenominationModel extraction.
    
    Returns:
        List of dicts with keys: 'table' and 'json'
    """
    raw_examples = [
        # Example 1 - Table with clear denomination fields
        ("""| Fund Name | ISIN Code | Currency | NAV |
| --- | --- | --- | --- |
| UFF AVENIR SECURITE | FR0007436969 | EUR | 75.42 |
| UFF AVENIR DIVERSIFIE | FR0007437124 | EUR | 90.64 |""",
        {
            "vehicule_name": "UFF",
            "compartment_name": None,
            "instrument_name": "Fund Name",
            "share_type": None,
            "validation_confidence": 0.92
        })
    ]
    
    return [{"table": table, "json": metadata} for table, metadata in raw_examples]

def get_identifier_examples() -> List[Dict[str, Any]]:
    """
    Return examples for IdentifierModel extraction.
    
    Returns:
        List of dicts with keys: 'table' and 'json'
    """
    raw_examples = [
        # Example 1
        ("""| CODE ISIN | FCP | DATE | COUPON | CREDIT D'IMPÔT | VL APRES DETACH. |
| --- | --- | --- | --- | --- | --- |
| FR0007436969 | UFF AVENIR SECURITE | 1989-12-27 | 5.526277 | 0.030490 | 75.418053 |""",
        {
            "code": "CODE ISIN",
            "code_type": "Isin",
            "currency": "EUR",
            "cic_code": None,
            "validation_confidence": 0.95
        })
    ]
    
    return [{"table": table, "json": metadata} for table, metadata in raw_examples]

def get_model_examples(model_class: Type[BaseExtraction]) -> List[Dict[str, Any]]:
    """
    Get examples for a specific model class.
    
    Args:
        model_class: The model class to get examples for
        
    Returns:
        List of examples for the specified model
    """
    model_name = model_class.__name__
    
    # Return examples based on model name
    if model_name == "DenominationModel":
        return get_denomination_examples()
    elif model_name == "IdentifierModel":
        return get_identifier_examples()
    
    # If no examples are defined for this model, return an empty list
    return []

def get_extraction_examples(model_name_or_class: Union[str, Type[BaseExtraction]]) -> List[Dict[str, Any]]:
    """
    Get examples for a specific model.
    
    Args:
        model_name_or_class: Name of the model or model class to get examples for
        
    Returns:
        List of examples for the specified model
    """
    # Handle both string names and actual model classes
    if isinstance(model_name_or_class, str):
        # Look up the model class by name
        if model_name_or_class == "DenominationModel":
            return get_denomination_examples()
        elif model_name_or_class == "Identifier":
            return get_identifier_examples()
        elif model_name_or_class in EXTRACTION_MODELS:
            model_class = EXTRACTION_MODELS[model_name_or_class]
            return get_model_examples(model_class)
        else:
            return []
    else:
        # It's already a model class
        return get_model_examples(model_name_or_class)

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
        if field == "validation_confidence":
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
        "validated_data": example["json"],
        "validation_confidence": example["json"].get("validation_confidence", 0.9),
        "corrections_made": ["Corrected data based on table context"]
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

# Keep these functions for backward compatibility
def format_extraction_messages(examples: List[Dict[str, Any]], system_message: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Format examples for extraction agent (wrapper for format_messages).
    """
    return format_messages(examples, system_message, "extraction")

def format_validation_messages(examples: List[Dict[str, Any]], system_message: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Format examples for validation agent (wrapper for format_messages).
    """
    return format_messages(examples, system_message, "validation")
