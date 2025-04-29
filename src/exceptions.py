"""Custom exception types for the Excel Extraction Pipeline."""

class PipelineException(Exception):
    """Base exception for pipeline-related errors.

    Attributes:
        message (str): Human-readable error description
        error_code (str): Unique code identifying the error type
        severity (str): Error severity level (INFO, WARNING, ERROR, CRITICAL)
        context (dict): Additional context information about the error
    """

    # Define severity levels as class constants
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def __init__(self, message, error_code=None, severity=ERROR, context=None):
        """Initialize a new PipelineException.

        Args:
            message (str): Human-readable error description
            error_code (str, optional): Unique code identifying the error type.
                Defaults to a generic code based on the exception class.
            severity (str, optional): Error severity level. Defaults to ERROR.
            context (dict, optional): Additional context information. Defaults to empty dict.
        """
        super().__init__(message)
        self.message = message  # Store message explicitly for serialization
        # Generate a default error code based on the class name if none provided
        self.error_code = error_code or f"PIPELINE_{self.__class__.__name__.upper()}"
        self.severity = severity
        self.context = context or {}

    def to_dict(self):
        """Convert exception to dictionary for API responses.

        Returns:
            dict: Structured representation of the exception
        """
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "severity": self.severity,
                "context": self.context
            }
        }

    def __str__(self):
        """Create string representation of the exception.

        Returns:
            str: Formatted error message
        """
        base_msg = f"[{self.error_code}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base_msg} (Context: {context_str})"
        return base_msg


class ConfigurationError(PipelineException):
    """Exception raised for errors in configuration loading or validation."""

    def __init__(self, message, error_code=None, severity=PipelineException.ERROR, context=None):
        # Provide a more specific default error code
        error_code = error_code or "CONFIG_ERROR"
        super().__init__(message, error_code, severity, context)


class FileProcessingError(PipelineException):
    """Exception raised for errors during file reading or processing."""

    def __init__(self, message, error_code=None, severity=PipelineException.ERROR, context=None):
        # Provide a more specific default error code
        error_code = error_code or "FILE_PROCESSING_ERROR"
        super().__init__(message, error_code, severity, context)


class LLMInteractionError(PipelineException):
    """Exception raised for errors during interaction with the LLM."""

    def __init__(self, message, error_code=None, severity=PipelineException.ERROR, context=None):
        # Provide a more specific default error code
        error_code = error_code or "LLM_INTERACTION_ERROR"
        super().__init__(message, error_code, severity, context)


class ExtractionError(PipelineException):
    """Exception raised for errors during the data extraction process."""

    def __init__(self, message, error_code=None, severity=PipelineException.ERROR, context=None):
        # Provide a more specific default error code
        error_code = error_code or "EXTRACTION_ERROR"
        super().__init__(message, error_code, severity, context)


class ValidationError(PipelineException):
    """Exception raised for errors during data validation."""

    def __init__(self, message, error_code=None, severity=PipelineException.ERROR, context=None):
        # Provide a more specific default error code
        error_code = error_code or "VALIDATION_ERROR"
        super().__init__(message, error_code, severity, context)


class DeduplicationError(PipelineException):
    """Exception raised for errors during the deduplication process."""

    def __init__(self, message, error_code=None, severity=PipelineException.ERROR, context=None):
        # Provide a more specific default error code
        error_code = error_code or "DEDUPLICATION_ERROR"
        super().__init__(message, error_code, severity, context)

# Add other specific exception types as needed
# Example:
# class ClassificationError(PipelineException):
#     """Exception raised for errors during document classification."""
#     def __init__(self, message, error_code=None, severity=PipelineException.ERROR, context=None):
#         error_code = error_code or "CLASSIFICATION_ERROR"
#         super().__init__(message, error_code, severity, context)

# class HeaderDetectionError(PipelineException):
#     """Exception raised for errors during header detection."""
#     def __init__(self, message, error_code=None, severity=PipelineException.ERROR, context=None):
#         error_code = error_code or "HEADER_DETECTION_ERROR"
#         super().__init__(message, error_code, severity, context)

# class InvalidInputError(PipelineException):
#     """Exception raised for malformed or invalid input data."""
#     def __init__(self, message, error_code=None, severity=PipelineException.WARNING, context=None):
#         error_code = error_code or "INVALID_INPUT_ERROR"
#         super().__init__(message, error_code, severity, context)

# class ThirdPartyServiceError(PipelineException):
#     """Exception raised for errors interacting with external services."""
#     def __init__(self, message, error_code=None, severity=PipelineException.CRITICAL, context=None):
#         error_code = error_code or "THIRD_PARTY_SERVICE_ERROR"
#         super().__init__(message, error_code, severity, context)
