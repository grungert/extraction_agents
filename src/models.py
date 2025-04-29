"""Data models for Excel Header Mapper."""
import os
import json
from typing import Dict, List, TypedDict, Optional, Any, Type, Union
from pydantic import BaseModel, Field, create_model, validator, ValidationError

# Configuration models

class LLMConfig(BaseModel):
    """Base Configuration for an LLM model."""
    model_name: str = "llama-3.2-3b-instruct"
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "null"
    temperature: float = 0.2
    max_retries: int = 2
    # Add context window size if available from the model provider or config
    context_window: Optional[int] = Field(None, description="Maximum number of tokens the model can handle")


class ModelConfig(LLMConfig):
    """Configuration for the main LLM model."""
    # Specific settings for the main model if needed, otherwise inherits LLMConfig
    pass

class ClassificationModelConfig(LLMConfig):
    """Configuration specific to the classification LLM."""
    model_name: str = "claude2-alpaca-7b" # Example classification model
    temperature: float = 0.3
    max_tokens: int = 4000 # This might be redundant if context_window is used
    context_window_percentage: float = 0.45 # Percentage of context_window to use for prompt


class AppConfig(BaseModel):
    """Application configuration."""
    input_dir: str = "input"
    output_dir: str = "json_outputs"
    start_row: int = 0
    end_row: int = 15
    all_sheets: bool = False
    model: ModelConfig = Field(default_factory=ModelConfig) # Main model config
    classification_model: ClassificationModelConfig = Field(default_factory=ClassificationModelConfig) # Classification model config
    include_header_examples_in_prompt: bool = True
    enable_deduplication_agent: bool = True
    classification_prompt: str = "classifier_agent_v2_2.md"
    classification_validation_prompt: str = "classifier_validation_agent_v1.md"
    classification_labels: List[str] = Field(default_factory=lambda: [
        "Mutual Funds", "ETF", "Stocks", "Bonds", "Real State",
        "Crypto Asset", "Commodities", "Private Equities", "Index",
        "Currencies Pairs", "None of those"
    ])
    # Add other application-level settings here

    @classmethod
    def load_from_file(cls, file_path: str) -> "AppConfig":
        """Loads application configuration from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Application configuration file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in application configuration file: {str(e)}")
        except ValidationError as e:
            raise ValueError(f"Application configuration validation failed in {file_path}:\n{e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading application configuration from {file_path}: {e}")


# --- Dynamic Extraction Configuration Models ---

class FieldDefinition(BaseModel):
    """Defines a field to be extracted."""
    description: str = Field(..., description="Description of the data to extract for this field.")
    type: str = Field("string", description="Expected data type (e.g., string, number, boolean, array).")
    examples: Optional[List[Any]] = Field(None, description="Examples of expected values for this field.")
    # Add other field-specific settings if needed (e.g., format, constraints)

class ExtractionModelDefinition(BaseModel):
    """Defines a Pydantic model to be created dynamically for extraction."""
    name: str = Field(..., description="Name of the extraction model (used as section name).")
    description: Optional[str] = Field(None, description="Description of the data section this model represents.")
    fields: Dict[str, FieldDefinition] = Field(..., description="Definitions of the fields within this model.")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Examples for few-shot learning for this model.")

class HeaderDetectionConfig(BaseModel):
    """Configuration for the header detection agent."""
    config: Dict[str, Any] = Field(default_factory=dict, description="Specific configuration settings for the header detection agent.")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Examples for few-shot learning for header detection.")
    header_detection_system_prompt_file: str = Field('header_detection_system_v1.md', description="Filename for the header detection system prompt.")


class ValidationConfig(BaseModel):
    """Configuration for validation agents."""
    confidence_threshold: float = Field(0.7, description="Confidence threshold for validation.")
    header_validation_prompt_file: str = Field('header_validation_system_v1.md', description="Filename for the header validation system prompt.")
    section_validation_template_file: str = Field('section_validation_system_template.md', description="Filename for the section validation template.")
    # Add other validation-specific settings if needed

class PromptsConfig(BaseModel):
    """Configuration for prompt file names."""
    header_detection_system_prompt_file: str = Field('header_detection_system_v1.md', description="Filename for the header detection system prompt.")
    header_validation_system_prompt_file: str = Field('header_validation_system_v1.md', description="Filename for the header validation system prompt.")
    deduplication_system_prompt_file: str = Field('deduplication_system.md', description="Filename for the deduplication system prompt.")
    deduplication_instruction_prompt_file: str = Field('deduplication_agent.md', description="Filename for the deduplication instruction prompt.")
    section_extraction_template_file: str = Field('section_extraction_system_template.md', description="Filename for the section extraction template.")
    section_validation_template_file: str = Field('section_validation_system_template.md', description="Filename for the section validation template.")

class ExtractionConfig(BaseModel):
    """Overall dynamic extraction configuration for a specific document type."""
    header_detection: HeaderDetectionConfig = Field(..., description="Configuration for header detection.")
    validation: ValidationConfig = Field(..., description="Configuration for validation agents.")
    extraction_models: List[ExtractionModelDefinition] = Field(..., description="Definitions of the extraction models.")
    prompts: PromptsConfig = Field(default_factory=PromptsConfig, description="Configuration for prompt file names.")
    # Add other top-level configuration sections if needed


# --- Output Models ---

# Base for extractions
class BaseExtraction(BaseModel):
    """Base model that all extraction models should inherit from."""
    ValidationConfidence: Optional[float] = Field(None, description="Confidence score (0.0-1.0)")

# Classification Output
class ClassificationOutput(BaseModel):
    """Output from the classification agent."""
    predicted_class: str = Field(..., description="The class predicted by the LLM.")
    confidence: str = Field(..., description="The confidence description (e.g., High, Medium, Low).")

# Classification Validation Output
class ClassificationValidationOutput(BaseModel):
    """Output from the classification validation agent."""
    predicted_class: str = Field(..., description="The validated or corrected class.")
    confidence: str = Field(..., description="The confidence description after validation.")
    validation_reason: Optional[str] = Field(None, description="Reason if validation changed the class or confidence.")

# General Validation Result (for section validation)
class ValidationResult(BaseModel):
    """Base model for section validation results."""
    ValidationConfidence: float = Field(0.0, description="Confidence in validation (0.0-1.0)")
    CorrectionsMade: List[str] = Field([], description="List of corrections made during validation")

# Context section (inherits BaseExtraction for ValidationConfidence)
class ContextModel(BaseExtraction):
    """Model for extracting context information."""
    FileName: Optional[str] = Field(None, description="File name of the Excel document")
    HeaderStartLine: Optional[int] = Field(None, description="Line where headers start (1-based)")
    HeaderEndLine: Optional[int] = Field(None, description="Line where headers end (1-based)")
    ContentStartLine: Optional[int] = Field(None, description="Line where content starts (1-based)")
    FileType: Optional[str] = Field(None, description="File type (xlsx, csv)")

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

# NOTE: EXTRACTION_MODELS are now loaded dynamically within the
# DynamicAgentPipelineCoordinator based on classification results,
# so the module-level loading has been removed.
