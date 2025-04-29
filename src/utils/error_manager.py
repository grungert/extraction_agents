"""Centralized error handling and logging utilities."""
import logging
import traceback
import sys
from typing import Dict, Any, Optional, Type, Union

# Use relative import for exceptions
from ..exceptions import PipelineException
# Use relative import for display logger
from .display import logger

class ErrorManager:
    """Centralized error handling and reporting."""

    @staticmethod
    def handle_exception(exc: Exception, log_traceback: bool = True) -> Dict[str, Any]:
        """Handle exceptions with standardized logging and formatting.

        Args:
            exc: The exception to handle
            log_traceback: Whether to include the traceback in the log

        Returns:
            Dict containing a formatted error response
        """
        original_exc = exc # Keep track of the original exception

        # Convert any non-PipelineException to a PipelineException
        if not isinstance(exc, PipelineException):
            exc = PipelineException(
                message=str(exc),
                error_code="UNHANDLED_ERROR",
                severity=PipelineException.ERROR,
                context={"exception_type": original_exc.__class__.__name__}
            )

        # Log the exception with appropriate level
        log_method = {
            PipelineException.INFO: logger.info,
            PipelineException.WARNING: logger.warning,
            PipelineException.ERROR: logger.error,
            PipelineException.CRITICAL: logger.critical
        }.get(exc.severity, logger.error) # Default to error level

        # Use exc_info=True to automatically include traceback for ERROR and CRITICAL
        # Only log traceback explicitly if requested and severity is lower
        should_log_traceback = log_traceback and exc.severity in [PipelineException.INFO, PipelineException.WARNING]
        if exc.severity in [PipelineException.ERROR, PipelineException.CRITICAL]:
             log_method(f"{exc}", exc_info=True) # Always log traceback for ERROR/CRITICAL
        elif should_log_traceback:
             log_method(f"{exc}", exc_info=True)
        else:
             log_method(f"{exc}")

        # Convert to API-friendly format using the exception's method
        return exc.to_dict()

    @staticmethod
    def map_exception_to_http_status(exc: Exception) -> int:
        """Map exception types to appropriate HTTP status codes.

        Args:
            exc: The exception to map

        Returns:
            int: HTTP status code appropriate for this exception
        """
        # Import specific exceptions locally to avoid potential circular imports at module level
        from ..exceptions import (
            ConfigurationError, FileProcessingError, LLMInteractionError,
            ExtractionError, ValidationError, DeduplicationError
            # Add any other specific exceptions here if needed
        )

        # Map specific exceptions to HTTP status codes
        # More specific exceptions should come first
        status_map = {
            ValidationError: 400,         # Bad Request (client-side validation issue)
            ConfigurationError: 400,      # Bad Request (client might provide bad config?) or 500 if server-side
            FileProcessingError: 400,     # Bad Request (issue with uploaded file)
            # InvalidInputError: 400,     # Example if added
            LLMInteractionError: 502,     # Bad Gateway (external service failure)
            # ThirdPartyServiceError: 502, # Example if added
            DeduplicationError: 500,      # Internal Server Error
            ExtractionError: 500,         # Internal Server Error
            PipelineException: 500,       # Default Internal Server Error for pipeline issues
        }

        # Find the most specific matching exception type in the map
        for exc_type, status_code in status_map.items():
            if isinstance(exc, exc_type):
                return status_code

        # Default status code for unhandled exceptions (should be caught by the base PipelineException mapping)
        return 500
