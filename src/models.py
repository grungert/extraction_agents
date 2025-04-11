"""Data models for Excel Header Mapper."""
import os
import json
from typing import Dict, List, TypedDict, Optional, Any, Type
from pydantic import BaseModel, Field, create_model

# Configuration models
class ModelConfig(BaseModel):
    """Configuration for the LLM model."""
    model_name: str = "llama-3.2-3b-instruct"
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "null"
    temperature: float = 0.3
    max_retries: int = 2

class AppConfig(BaseModel):
    """Application configuration."""
    input_dir: str = "input"
    output_dir: str = "json_outputs"
    start_row: int = 0
    end_row: int = 15
    all_sheets: bool = False
    model: ModelConfig = ModelConfig()
    config_path: str = "config/full_config.json"
    include_header_examples_in_prompt: bool = True

# Extraction models - base class
class BaseExtraction(BaseModel):
    """Base model that all extraction models should inherit from."""
    ValidationConfidence: Optional[float] = Field(None, description="Confidence in validation (0.0-1.0)")

# Validation model base class
class ValidationResult(BaseModel):
    """Base model for validation results."""
    ValidationConfidence: float = Field(0.0, description="Confidence in validation (0.0-1.0)")
    CorrectionsMade: List[str] = Field([], description="List of corrections made during validation")

# Context section
class ContextModel(BaseExtraction):
    """Model for extracting context information."""
    FileName: Optional[str] = Field(None, description="File name of the Excel document")
    HeaderStartLine: Optional[int] = Field(None, description="Line where headers start (1-based)")
    HeaderEndLine: Optional[int] = Field(None, description="Line where headers end (1-based)")
    ContentStartLine: Optional[int] = Field(None, description="Line where content starts (1-based)")
    FileType: Optional[str] = Field(None, description="File type (xlsx, csv)")
    # header_detection_confidence field removed as requested

# Container for LLM extraction results
class Data(BaseModel):
    """Container for extracted data."""
    data_extracted: List[Any]  # Changed from IdentifierModel to Any for flexibility

class Example(TypedDict):
    """
    A representation of an example consisting of text input and expected tool calls.
    For extraction, the tool calls are represented as instances of pydantic model.
    """
    input: str  # This is the example text
    tool_calls: List[Any]  # Instances of pydantic model that should be extracted

# Get dynamic models from config
try:
    from .config_manager import get_configuration_manager
    from .dynamic_model_factory import create_extraction_models_dict
    
    # Get configuration manager
    config_manager = get_configuration_manager()
    
    # Load dynamic models, passing the flag from AppConfig
    # Create a default AppConfig instance to access the flag if not otherwise available
    app_config_instance = AppConfig()
    EXTRACTION_MODELS = create_extraction_models_dict(
        config_manager, 
        include_examples=app_config_instance.include_header_examples_in_prompt
    )
except Exception as e:
    # Fall back to empty dictionary if loading fails
    print(f"Error loading dynamic models: {str(e)}")
    EXTRACTION_MODELS = {}
