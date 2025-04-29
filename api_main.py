import os
import sys
import json
import logging # Import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import ValidationError

# Add the parent directory to the sys.path to allow importing src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import AppConfig # Import AppConfig
from src.extraction.dynamic_agents import DynamicAgentPipelineCoordinator
from src.utils.display import console, logger # Import logger
from src.extraction.excel import excel_to_markdown
from src.utils.error_manager import ErrorManager # Import the new ErrorManager

# Import custom exceptions
from src.exceptions import (
    PipelineException,
    ConfigurationError,
    FileProcessingError,
    LLMInteractionError,
    ExtractionError,
    ValidationError as CustomValidationError, # Alias custom ValidationError
    DeduplicationError
)


app = FastAPI()

# Load application configuration at startup
APP_CONFIG_PATH = os.environ.get("APP_CONFIG_PATH", "config/app_config.json") # Allow override via env var
try:
    app_config = AppConfig.load_from_file(APP_CONFIG_PATH)
    logger.info(f"Successfully loaded application configuration from {APP_CONFIG_PATH}") # Use logger.info
except FileNotFoundError as e:
    logger.error(f"Error loading application configuration: {e}") # Use logger.error
    logger.error("Please ensure app_config.json exists or APP_CONFIG_PATH environment variable is set correctly.") # Use logger.error
    sys.exit(1) # Exit if config cannot be loaded
except ConfigurationError as e: # Catch custom ConfigurationError
    logger.error(f"Configuration Error loading application configuration: {e}") # Use logger.error
    sys.exit(1)
except Exception as e: # Catch any other unexpected errors
    logger.exception(f"An unexpected error occurred during application configuration loading: {e}")
    sys.exit(1)


# Initialize the pipeline coordinator with the loaded app config
pipeline = DynamicAgentPipelineCoordinator(app_config)
logger.info("Dynamic agent pipeline initialized successfully.") # Use logger.info


@app.get("/")
async def read_root():
    return {"message": "Excel Extraction Pipeline API"}

@app.post("/extract")
async def extract_excel(
    file: UploadFile = File(...),
    # config_json: Optional[str] = Form(None) # Config is now dynamic based on classification
):
    """
    Extracts structured data from an uploaded Excel or CSV file.

    The extraction configuration is determined dynamically based on document classification.
    """
    logger.info(f"Received file for extraction: {file.filename}") # Use logger.info

    # Save the uploaded file temporarily
    temp_file_path = f"/tmp/{file.filename}"
    try:
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        logger.info(f"Saved temporary file: {temp_file_path}") # Use logger.info

        # Determine file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.xlsx', '.xls', '.csv']:
             logger.error(f"Unsupported file type: {file_ext}. Supported types are .xlsx, .xls, .csv") # Use logger.error
             # Raise HTTPException for client error
             raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}. Supported types are .xlsx, .xls, .csv")

        # Convert Excel file to markdown
        # Pass app_config to excel_to_markdown
        # excel_to_markdown now raises FileProcessingError
        markdown_content = excel_to_markdown(temp_file_path, app_config)
        logger.info("Converted file to markdown.") # Use logger.info

        # Process the markdown content through the dynamic pipeline
        # The pipeline coordinator handles dynamic config loading based on classification
        # process_markdown now raises custom PipelineExceptions
        logger.info("Processing markdown content through pipeline...") # Use logger.info
        results = pipeline.process_markdown(markdown_content, temp_file_path)
        logger.info("Pipeline processing complete.") # Use logger.info

        # Check for errors in results (should be handled by exceptions now, but keep for safety)
        if "error" in results:
             logger.error(f"Pipeline returned an error in result: {results['error']}") # Use logger.error
             raise HTTPException(status_code=500, detail=results["error"])


        return JSONResponse(content=results)

    # --- Updated Exception Handling using ErrorManager ---
    except Exception as e:
        # Use ErrorManager to handle logging and formatting
        error_response = ErrorManager.handle_exception(e, log_traceback=True)
        # Use ErrorManager to map exception to HTTP status code
        status_code = ErrorManager.map_exception_to_http_status(e)
        # Raise HTTPException with the standardized response and status code
        raise HTTPException(status_code=status_code, detail=error_response)
    # --- End Updated Exception Handling ---

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Cleaned up temporary file: {temp_file_path}") # Use logger.info

# Add other API endpoints as needed
