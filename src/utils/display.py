"""Display and logging utilities for Excel Header Mapper."""
import logging
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler # Import RichHandler for better console logging

# Initialize rich console for general output
console = Console()

# Configure logging
# Create a logger
logger = logging.getLogger("excel_extraction_pipeline")
logger.setLevel(logging.INFO) # Set minimum logging level

# Create handlers
# Use RichHandler for console output
console_handler = RichHandler(
    console=console, # Use the existing rich console
    show_time=True,
    show_path=False, # Hide path for cleaner output
    keywords=["Error", "Warning", "Configuration Error", "Runtime Error"] # Highlight key terms
)
console_handler.setLevel(logging.INFO) # Set minimum level for console output

# Create a formatter (optional, RichHandler has a default)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)

# Add handlers to the logger
# Prevent duplicate handlers if the module is reloaded
if not logger.handlers:
    logger.addHandler(console_handler)

# You can add other handlers here, e.g., FileHandler for logging to a file
# file_handler = logging.FileHandler("pipeline.log")
# file_handler.setLevel(logging.ERROR) # Log only errors and above to file
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# Example usage:
# logger.info("This is an informational message.")
# logger.warning("This is a warning message.")
# logger.error("This is an error message.")
# logger.exception("This is an error with traceback.") # Use inside an except block

# Keep the rich console for non-logging display like tables and panels
# console.print("This is a general message using console.print")

# Export the logger for use in other modules
__all__ = ["console", "logger", "create_summary_table", "create_progress_panel"]
