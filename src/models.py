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
    input_dir: str = "testing"
    output_dir: str = "json_outputs"
    start_row: int = 0
    end_row: int = 15
    all_sheets: bool = False
    model: ModelConfig = ModelConfig()
    config_path: str = "config/full_config.json"

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

# Identifier section - primary model for LLM extraction
class IdentifierModel(BaseExtraction):
    """Model for extracting identifier information."""
    code: Optional[str] = Field(None, description="Header for ISIN Code. Examples: CODE ISIN, Code ISIN, ISIN, Account")
    code_type: Optional[str] = Field(None, description="Type of code, typically 'Isin'")
    currency: Optional[str] = Field(None, description="Header for currency. Examples: Devise, Currency")
    cic_code: Optional[str] = Field(None, description="Header for CIC Code if present")

# Denomination section
class DenominationModel(BaseExtraction):
    """Model for extracting denomination information."""
    vehicule_name: Optional[str] = Field(None, description="Header for vehicle name. Examples: Umbrella name")
    compartment_name: Optional[str] = Field(None, description="Header for compartment name")
    instrument_name: Optional[str] = Field(None, description="Header for instrument name. Examples: FCP")
    share_type: Optional[str] = Field(None, description="Header for share type")

# Valorisation section
class ValorisationModel(BaseExtraction):
    """Model for extracting valorisation information."""
    nav: Optional[str] = Field(None, description="Header for NAV. Examples: Valeur Liquidative, VL, NAV/share, NAV per share")
    nav_date: Optional[str] = Field(None, description="Header for NAV date. Example: Date de publication, Valuation Date")

# MarketCap section
class MarketCapModel(BaseExtraction):
    """Model for extracting market cap information."""
    reference_date: Optional[str] = Field(None, description="Header for reference date")
    compartment_currency: Optional[str] = Field(None, description="Header for compartment currency") 
    compartment_asset_value: Optional[str] = Field(None, description="Header for asset value. Example: Actif Net")
    share_asset_value: Optional[str] = Field(None, description="Header for share asset value")
    number_of_shares: Optional[str] = Field(None, description="Header for number of shares. Example: Nombre de Parts")

# CorporateAction section
class CorporateActionModel(BaseExtraction):
    """Model for extracting corporate action information."""
    currency: Optional[str] = Field(None, description="Header for currency")
    type: Optional[str] = Field(None, description="Header for type. Example: Coupon")
    value: Optional[str] = Field(None, description="Header for value. Example: COUPON")
    execution_date: Optional[str] = Field(None, description="Header for execution date. Example: DATE")
    payment_date: Optional[str] = Field(None, description="Header for payment date")
    record_date: Optional[str] = Field(None, description="Header for record date")
    distribution_rate: Optional[str] = Field(None, description="Header for distribution rate")

# Characteristics section
class CharacteristicsModel(BaseExtraction):
    """Model for extracting characteristics information."""
    strategy: Optional[str] = Field(None, description="Header for strategy")
    asset_manager: Optional[str] = Field(None, description="Header for asset manager")
    portfolio_manager: Optional[str] = Field(None, description="Header for portfolio manager")
    hosting_country: Optional[str] = Field(None, description="Header for hosting country")
    legal_status: Optional[str] = Field(None, description="Header for legal status")
    under_legal_status: Optional[str] = Field(None, description="Header for under legal status")
    inception_date: Optional[str] = Field(None, description="Header for inception date")
    distribution_policy: Optional[str] = Field(None, description="Header for distribution policy")
    payment_frequency: Optional[str] = Field(None, description="Header for payment frequency")

# Define which models to extract
EXTRACTION_MODELS = {
    "DenominationModel": DenominationModel,
    "Identifier": IdentifierModel,
    "Valorisation": ValorisationModel,
    "MarketCap": MarketCapModel,
    "CorporateAction": CorporateActionModel,
    "Characteristics": CharacteristicsModel
}

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
    
    # Load dynamic models
    EXTRACTION_MODELS = create_extraction_models_dict(config_manager)
except Exception as e:
    # Fall back to empty dictionary if loading fails
    print(f"Error loading dynamic models: {str(e)}")
    EXTRACTION_MODELS = {}
