"""Utilities for working with prompt files."""
import os
from typing import Dict, Optional, List, Any
from pathlib import Path

# Assuming AppConfig and ConfigurationManager are accessible
# Adjust imports based on your project structure if needed
try:
    from ..models import AppConfig
    from ..config_manager import ConfigurationManager, get_configuration_manager
except ImportError:
    # Handle cases where this might be run standalone or structure differs
    AppConfig = None
    ConfigurationManager = None
    get_configuration_manager = None

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

def load_prompt(file_name: str) -> str:
    """
    Load a raw prompt from a file.

    Args:
        file_name: Name of the prompt file in the prompts directory

    Returns:
        Contents of the prompt file
    """
    try:
        with open(PROMPT_DIR / file_name, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file '{file_name}' not found in {PROMPT_DIR}")
        return ""


def load_template_prompt(
    file_name: str,
    config_manager: Optional[ConfigurationManager] = None,
    app_config: Optional[AppConfig] = None,
    **kwargs
) -> str:
    """
    Load a prompt template from a file and format it using standard Python formatting.
    Conditionally includes field examples based on AppConfig.

    Args:
        file_name: Name of the prompt template file
        config_manager: Configuration manager instance (required if examples are needed)
        app_config: Application configuration instance (required if examples are needed)
        **kwargs: Additional keys and values to format the template (e.g., section_name)

    Returns:
        Formatted prompt template string
    """
    template_string = load_prompt(file_name)
    if not template_string:
        return "" # Return empty if template loading failed

    format_args = kwargs.copy()
    examples_text = "" # Default to empty string

    # Check if we should include examples
    include_examples_flag = False
    if app_config and hasattr(app_config, 'include_header_examples_in_prompt'):
        include_examples_flag = app_config.include_header_examples_in_prompt

    # Build the examples text string if needed
    if include_examples_flag and config_manager and 'section_name' in kwargs:
        section_name = kwargs['section_name']
        model_def = config_manager.get_model_by_name(section_name)

        if model_def and 'fields' in model_def:
            field_lines = []
            has_any_examples = False
            for field_name, field_def in model_def['fields'].items():
                desc = field_def.get('description', '')
                field_examples = field_def.get('examples', [])
                # Use Python f-string for formatting each line
                line = f"- **`{field_name}`**: {desc}"
                if field_examples:
                    has_any_examples = True
                    # Join examples with ', '
                    examples_str = ', '.join(field_examples)
                    line += f" - _e.g._: {examples_str}"
                field_lines.append(line)

            # Only add the section if the flag is true AND there were examples
            if has_any_examples:
                 # Determine the header based on the template file name
                 if "extraction" in file_name:
                     examples_header = "6. Example headers to look for:"
                 elif "validation" in file_name:
                     examples_header = "7. Expected fields and example headers:"
                 else:
                     examples_header = "Field Examples:" # Default header
                 # Construct the final text block with header and formatted lines
                 examples_text = f"\n{examples_header}\n" + "\n".join(field_lines) + "\n" # Add newlines for spacing

    # Add the generated examples text (or empty string) to the format arguments
    format_args['field_examples_section'] = examples_text

    # Ensure section_name is present if needed by the template placeholders
    if 'section_name' in kwargs and 'section_name' not in format_args:
         format_args['section_name'] = kwargs['section_name']

    try:
        # Use standard Python string formatting
        return template_string.format(**format_args)
    except KeyError as e:
        print(f"Error formatting template '{file_name}': Missing key {e}")
        print(f"Available keys: {list(format_args.keys())}")
        return template_string # Return unformatted template on error
    except Exception as e:
        print(f"Error formatting template '{file_name}': {e}")
        return template_string # Return unformatted template on error
