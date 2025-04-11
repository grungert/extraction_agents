"""Utilities for working with prompt files."""
import os
from typing import Dict, Optional, List, Any
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

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

# Setup Jinja2 environment
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(PROMPT_DIR), trim_blocks=True, lstrip_blocks=True)


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
    Load a prompt template from a file and render it using Jinja2.
    Conditionally includes field examples based on AppConfig.
    
    Args:
        file_name: Name of the prompt template file
        config_manager: Configuration manager instance (required if examples are needed)
        app_config: Application configuration instance (required if examples are needed)
        **kwargs: Additional keys and values to render the template (e.g., section_name)
        
    Returns:
        Rendered prompt template string
    """
    try:
        template = jinja_env.get_template(file_name)
    except Exception as e:
        print(f"Error loading template '{file_name}': {e}")
        return ""

    render_context = kwargs.copy()
    
    # Check if we need to include examples
    include_examples = False
    if app_config and hasattr(app_config, 'include_header_examples_in_prompt'):
        include_examples = app_config.include_header_examples_in_prompt
        
    render_context['include_examples'] = include_examples

    # If examples are included and config manager is provided, fetch field info
    if include_examples and config_manager and 'section_name' in kwargs:
        section_name = kwargs['section_name']
        # Use the correct method name: get_model_by_name
        model_def = config_manager.get_model_by_name(section_name) 
        
        if model_def and 'fields' in model_def:
            fields_info = []
            for field_name, field_def in model_def['fields'].items():
                fields_info.append({
                    'name': field_name,
                    'description': field_def.get('description', ''),
                    'examples': field_def.get('examples', [])
                })
            render_context['fields'] = fields_info
        else:
             render_context['fields'] = [] # Ensure fields is always present
    elif 'section_name' in kwargs:
         render_context['fields'] = [] # Ensure fields is always present if section_name exists

    try:
        return template.render(render_context)
    except Exception as e:
        print(f"Error rendering template '{file_name}' with context {render_context}: {e}")
        return ""
