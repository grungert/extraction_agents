#!/usr/bin/env python3
# /// script
# dependencies = [
#   "openai>=1.63.0",
#   "rich>=13.7.0",
#   "pandas>=2.0.0",
#   "typer>=0.4.0",
#   "langchain>=0.1.0",
#   "langchain-core>=0.1.0",
#   "langchain-openai>=0.1.0",
#   "xlrd >= 2.0.1",
#   "markitdown[all] >= 0.1.0",
#   "langfuse>=0.1.0",
#   "jinja2>=3.0.0",
# ]
# ///
"""LLM-based extraction pipeline for parsing Excel files."""
import json
import os
import sys
from pydantic import BaseModel
import argparse

from src.models import AppConfig # Import AppConfig
from src.config_manager import get_configuration_manager
from src.extraction.pipeline import DynamicAgentPipelineCoordinator
from src.utils.display import console, logger # Import logger
from src.extraction.excel import excel_to_markdown

# Import custom exceptions
from src.exceptions import ( # Import custom exceptions
    PipelineException,
    ConfigurationError,
    FileProcessingError,
    LLMInteractionError,
    ExtractionError,
    ValidationError,
    DeduplicationError
)


def main():
    """Run the dynamic agent pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM-based extraction pipeline for Excel files")
    # Config argument is no longer needed as it's determined dynamically
    # parser.add_argument("--config", type=str, default=None,
    #                     help="Path to base configuration file (optional, specific config loaded based on classification)")
    parser.add_argument("--app-config", type=str, default="config/app_config.json",
                        help="Path to the application configuration file (default: config/app_config.json)")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to specific Excel file to process (if not provided, processes all Excel files in input_dir)")
    parser.add_argument("--sheet", type=str, default=None,
                        help="Sheet name to process (if not provided, uses the first sheet)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output JSON file (if not provided, prints to console)")

    args = parser.parse_args()

    # Load application configuration from file
    logger.info(f"Loading application configuration from {args.app_config}...") # Use logger.info
    try:
        app_config = AppConfig.load_from_file(args.app_config)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}") # Use logger.error
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}") # Use logger.error
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime Error loading config: {e}") # Use logger.error
        sys.exit(1)


    # Initialize the agent pipeline
    logger.info("Initializing dynamic agent pipeline...") # Use logger.info
    pipeline = DynamicAgentPipelineCoordinator(app_config)

    # Determine files to process
    if args.file:
        # Process a specific Excel file
        logger.info(f"Processing Excel file: {args.file}") # Use logger.info
        # Check if the file exists
        if not os.path.exists(args.file):
            logger.error(f"Error: File not found: {args.file}") # Use logger.error
            return

        # Process single file
        files_to_process = [args.file]
    else:
        # Process all Excel files in the input directory
        input_dir = app_config.input_dir
        logger.info(f"Processing all Excel files in directory: {input_dir}") # Use logger.info

        # Check if directory exists
        if not os.path.exists(input_dir):
            logger.error(f"Error: Input directory not found: {input_dir}") # Use logger.error
            return

        # Get all Excel files in the directory
        files_to_process = []
        for filename in os.listdir(input_dir):
            if filename.endswith(('.xlsx', '.xls', '.csv', '.XLS')):
                files_to_process.append(os.path.join(input_dir, filename))

        if not files_to_process:
            logger.warning(f"No Excel files found in {input_dir}") # Use logger.warning
            return

    # Process each file
    for file_path in files_to_process:
        source_file = file_path
        logger.info(f"Processing file: {source_file}") # Use logger.info

        try: # Add try block for pipeline execution
            # Convert Excel file to markdown
            # excel_to_markdown now raises FileProcessingError
            markdown_content = excel_to_markdown(source_file, app_config, args.sheet)

            # Process the markdown content
            # process_markdown now raises custom PipelineExceptions
            logger.info("Processing markdown content...") # Use logger.info
            results = pipeline.process_markdown(markdown_content, source_file)

            # Output the results
            if args.output and len(files_to_process) == 1:
                # Use the specified output path (only for single file mode)
                output_path = args.output
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                logger.info(f"Writing results to specified location: {output_path}") # Use logger.info
            else:
                # Create output directory if it doesn't exist
                os.makedirs(app_config.output_dir, exist_ok=True)

                # Generate a default filename based on the source file
                output_filename = f"{os.path.basename(source_file)}.json"

                # Combine with the output directory
                output_path = os.path.join(app_config.output_dir, output_filename)
                logger.info(f"Writing results to default location: {output_path}") # Use logger.info

            # Write the results to file
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results written to {output_path}") # Use logger.info

        except (FileProcessingError, ConfigurationError, LLMInteractionError, ExtractionError, ValidationError, DeduplicationError, PipelineException) as e: # Catch custom exceptions
            logger.error(f"Error processing file {source_file}: {e}") # Log the custom exception
            # Continue to the next file
            continue
        except Exception as e: # Catch any other unexpected errors
            logger.exception(f"An unexpected error occurred while processing file {source_file}: {e}") # Log with traceback
            # Continue to the next file
            continue


    logger.info("Processing completed successfully!") # Use logger.info


if __name__ == "__main__":
    main()
