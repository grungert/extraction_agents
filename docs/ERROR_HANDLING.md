# Error Handling Guide

This document explains the error handling strategy used in the Excel Extraction Pipeline.

## 1. Overview

The pipeline utilizes a custom exception hierarchy based on the `PipelineException` class defined in `src/exceptions.py`. This provides a standardized way to handle and report errors throughout the application.

## 2. Exception Hierarchy

-   **`PipelineException` (Base Class):**
    -   All custom exceptions inherit from this base class.
    -   Attributes:
        -   `message` (str): Human-readable error description.
        -   `error_code` (str): A unique code identifying the error type (e.g., `CONFIG_ERROR`, `FILE_NOT_FOUND`). Defaults to a generic code based on the exception class name if not provided.
        -   `severity` (str): Indicates the severity (`INFO`, `WARNING`, `ERROR`, `CRITICAL`). Defaults to `ERROR`.
        -   `context` (dict): A dictionary for additional relevant information (e.g., file path, specific parameters).
    -   Methods:
        -   `to_dict()`: Returns a structured dictionary representation suitable for API responses.
        -   `__str__()`: Provides a formatted string representation including the error code and context.

-   **Specific Exception Subclasses:**
    -   `ConfigurationError`: For errors related to loading or validating configuration files. (Default code: `CONFIG_ERROR`)
    -   `FileProcessingError`: For errors during file reading, parsing, or conversion (e.g., unsupported format, read errors). (Default code: `FILE_PROCESSING_ERROR`)
    -   `LLMInteractionError`: For errors communicating with the Large Language Model (e.g., API timeouts, connection issues, invalid responses). (Default code: `LLM_INTERACTION_ERROR`)
    -   `ExtractionError`: For errors specifically during the data extraction phase by an agent. (Default code: `EXTRACTION_ERROR`)
    -   `ValidationError`: For errors during the data validation phase (e.g., validation rules failed, low confidence). (Default code: `VALIDATION_ERROR`)
    -   `DeduplicationError`: For errors during the conflict resolution/deduplication phase. (Default code: `DEDUPLICATION_ERROR`)
    -   *(Additional specific exceptions can be added as needed)*

## 3. Centralized Error Handling (`ErrorManager`)

The `src/utils/error_manager.py` module provides the `ErrorManager` class for consistent error processing:

-   **`ErrorManager.handle_exception(exc, log_traceback=True)`:**
    -   Takes any exception (`exc`) as input.
    -   If the exception is not already a `PipelineException`, it wraps it in a generic `PipelineException` with `error_code="UNHANDLED_ERROR"`.
    -   Logs the exception using the configured logger (`src.utils.display.logger`) with the appropriate severity level. Tracebacks are automatically included for `ERROR` and `CRITICAL` levels, or if `log_traceback=True` for lower severities.
    -   Returns a standardized dictionary using the exception's `to_dict()` method.

-   **`ErrorManager.map_exception_to_http_status(exc)`:**
    -   Takes an exception (`exc`) as input.
    -   Maps specific `PipelineException` subclasses (and the base class) to appropriate HTTP status codes (e.g., 400 for client errors like `ValidationError`, 500 for server errors like `ExtractionError`, 502 for `LLMInteractionError`).
    -   Returns the corresponding integer status code.

## 4. Usage in API (`api_main.py`)

The main API endpoint (`/extract`) uses a single `try...except Exception as e:` block. Inside the `except` block:
1.  `ErrorManager.handle_exception(e)` is called to log the error and get the formatted dictionary.
2.  `ErrorManager.map_exception_to_http_status(e)` is called to get the correct HTTP status code.
3.  A `FastAPI.HTTPException` is raised with the determined `status_code` and the error dictionary as the `detail`.

This ensures all errors result in consistent logging and structured JSON responses for API clients.

## 5. Best Practices

-   **Raise Specific Exceptions:** When possible, raise the most specific subclass of `PipelineException` that matches the error condition.
-   **Provide Context:** Use the `error_code`, `severity`, and `context` parameters when raising exceptions to provide meaningful details for logging and debugging.
    ```python
    raise FileProcessingError(
        f"File not found at path: {file_path}",
        error_code="FILE_NOT_FOUND",
        context={"path": file_path}
    )
    ```
-   **Handle Exceptions Appropriately:** Catch specific exceptions where recovery or alternative logic is possible. Let unhandled exceptions bubble up to the main API handler.
-   **Use `logger.exception` in `except` blocks:** When catching and logging unexpected errors manually (outside the `ErrorManager`), use `logger.exception()` to include traceback information automatically.
