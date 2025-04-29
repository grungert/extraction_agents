"""Custom exception types for the Excel Extraction Pipeline."""

class PipelineException(Exception):
    """Base exception for pipeline-related errors."""
    pass

class ConfigurationError(PipelineException):
    """Exception raised for errors in configuration loading or validation."""
    pass

class FileProcessingError(PipelineException):
    """Exception raised for errors during file reading or processing."""
    pass

class LLMInteractionError(PipelineException):
    """Exception raised for errors during interaction with the LLM."""
    pass

class ExtractionError(PipelineException):
    """Exception raised for errors during the data extraction process."""
    pass

class ValidationError(PipelineException):
    """Exception raised for errors during data validation."""
    pass

class DeduplicationError(PipelineException):
    """Exception raised for errors during the deduplication process."""
    pass

# Add other specific exception types as needed
