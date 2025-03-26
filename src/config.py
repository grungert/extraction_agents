"""Configuration handling for Excel Header Mapper."""
from .models import AppConfig, ModelConfig

def get_app_config(**kwargs):
    """
    Create application configuration with overrides from keyword arguments.
    
    Args:
        **kwargs: Configuration overrides
        
    Returns:
        AppConfig: Application configuration with overrides applied
    """
    # Start with default config
    config = AppConfig()
    
    # Update model configuration if provided
    if 'model_name' in kwargs and kwargs['model_name']:
        config.model.model_name = kwargs['model_name']
    if 'base_url' in kwargs and kwargs['base_url']:
        config.model.base_url = kwargs['base_url']
    if 'api_key' in kwargs and kwargs['api_key']:
        config.model.api_key = kwargs['api_key']
    if 'temperature' in kwargs and kwargs['temperature'] is not None:
        config.model.temperature = kwargs['temperature']
    if 'max_retries' in kwargs and kwargs['max_retries'] is not None:
        config.model.max_retries = kwargs['max_retries']
    
    # Update app configuration if provided
    if 'input_dir' in kwargs and kwargs['input_dir']:
        config.input_dir = kwargs['input_dir']
    if 'output_dir' in kwargs and kwargs['output_dir']:
        config.output_dir = kwargs['output_dir']
    if 'start_row' in kwargs and kwargs['start_row'] is not None:
        config.start_row = kwargs['start_row']
    if 'end_row' in kwargs and kwargs['end_row'] is not None:
        config.end_row = kwargs['end_row']
    if 'all_sheets' in kwargs and kwargs['all_sheets'] is not None:
        config.all_sheets = kwargs['all_sheets']
    
    return config
