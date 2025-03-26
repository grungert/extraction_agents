"""Data models for Excel Header Mapper."""
from typing import Dict, List, TypedDict, Optional, Any
from pydantic import BaseModel, Field

# Configuration models
class ModelConfig(BaseModel):
    """Configuration for the LLM model."""
    model_name: str = "gemma-3-12b-it"
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

# Extraction models
class BaseExtraction(BaseModel):
    """Base model that all extraction models should inherit from."""
    pass

# Context section
class ContextModel(BaseExtraction):
    """Model for extracting context information."""
    FileName: Optional[str] = Field(None, description="File name of the Excel document")
    HeaderStartLine: Optional[int] = Field(None, description="Line where headers start (1-based)")
    HeaderEndLine: Optional[int] = Field(None, description="Line where headers end (1-based)")
    ContentStartLine: Optional[int] = Field(None, description="Line where content starts (1-based)")
    FileType: Optional[str] = Field(None, description="File type (xlsx, csv)")

# Identifier section
class IdentifierModel(BaseExtraction):
    """Model for extracting identifier information."""
    Code: Optional[str] = Field(None, description="Header for ISIN Code. Examples: CODE ISIN, Code ISIN, ISIN")
    CodeType: Optional[str] = Field(None, description="Type of code, typically 'Isin'")
    Currency: Optional[str] = Field(None, description="Header for currency. Examples: Devise, Currency")
    CIC_Code: Optional[str] = Field(None, description="Header for CIC Code if present")

# Denomination section
class DenominationModel(BaseExtraction):
    """Model for extracting denomination information."""
    VehiculeName: Optional[str] = Field(None, description="Header for vehicle name")
    CompartmentName: Optional[str] = Field(None, description="Header for compartment name")
    InstrumentName: Optional[str] = Field(None, description="Header for instrument name. Examples: FCP")
    ShareType: Optional[str] = Field(None, description="Header for share type")

# Valorisation section
class ValorisationModel(BaseExtraction):
    """Model for extracting valorisation information."""
    Nav: Optional[str] = Field(None, description="Header for NAV. Examples: Valeur Liquidative, VL")
    NavDate: Optional[str] = Field(None, description="Header for NAV date. Example: Date de publication")

# MarketCap section
class MarketCapModel(BaseExtraction):
    """Model for extracting market cap information."""
    ReferenceDate: Optional[str] = Field(None, description="Header for reference date")
    CompartmentCurrency: Optional[str] = Field(None, description="Header for compartment currency") 
    CompartmentAssetValue: Optional[str] = Field(None, description="Header for asset value. Example: Actif Net")
    ShareAssetValue: Optional[str] = Field(None, description="Header for share asset value")
    Number_of_Shares: Optional[str] = Field(None, description="Header for number of shares. Example: Nombre de Parts")

# CorporateAction section
class CorporateActionModel(BaseExtraction):
    """Model for extracting corporate action information."""
    Currency: Optional[str] = Field(None, description="Header for currency")
    Type: Optional[str] = Field(None, description="Header for type. Example: Coupon")
    Value: Optional[str] = Field(None, description="Header for value. Example: COUPON")
    ExecutionDate: Optional[str] = Field(None, description="Header for execution date. Example: DATE")
    PaymentDate: Optional[str] = Field(None, description="Header for payment date")
    RecordDate: Optional[str] = Field(None, description="Header for record date")
    DistributionRate: Optional[str] = Field(None, description="Header for distribution rate")

# Characteristics section
class CharacteristicsModel(BaseExtraction):
    """Model for extracting characteristics information."""
    Strategy: Optional[str] = Field(None, description="Header for strategy")
    AssetManager: Optional[str] = Field(None, description="Header for asset manager")
    PortfolioManager: Optional[str] = Field(None, description="Header for portfolio manager")
    HostingCountry: Optional[str] = Field(None, description="Header for hosting country")
    LegalStatus: Optional[str] = Field(None, description="Header for legal status")
    UnderLegalStatus: Optional[str] = Field(None, description="Header for under legal status")
    InceptionDate: Optional[str] = Field(None, description="Header for inception date")
    DistributionPolicy: Optional[str] = Field(None, description="Header for distribution policy")
    PaymentFrequency: Optional[str] = Field(None, description="Header for payment frequency")

# Define which models to extract
EXTRACTION_MODELS = {
    "Context": ContextModel,
    "Identifier": IdentifierModel, 
}

# LLM extraction models
class DataToExtract(BaseModel):
    """Model for extracted data."""
    code: Optional[str] = Field(..., description="ISIN Code from header. Example of values: SEDOL, ISIN code, FMSedols, BloombergTicker, Instrument Identifier/ISIN code, CODE ISIN, ISIN, AFCCodes, ISIN Code, CODE_WPK, Isin Code, WKN, USD.ISIN, VN, Bundesamt, ISIN CODE, Code ISIN, CodIsin, ISESedols, Sedol, EuroClearCedel, Column2, CODE_ISIN")
    code_type: Optional[str] = Field(..., description="Isin from header")
    currency: Optional[str] = Field(..., description="Share currency from header")
    cic_code: Optional[str] = Field(..., description="CIC Code from header")

class Data(BaseModel):
    """Container for extracted data."""
    dataExtracted: List[DataToExtract]

class Example(TypedDict):
    """
    A representation of an example consisting of text input and expected tool calls.
    For extraction, the tool calls are represented as instances of pydantic model.
    """
    input: str  # This is the example text
    tool_calls: List[Any]  # Instances of pydantic model that should be extracted
