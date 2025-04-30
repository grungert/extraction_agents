"""Display and logging utilities for Excel Header Mapper."""
import logging
import sys
import os  # Import os for path manipulation
from logging.handlers import TimedRotatingFileHandler  # Import TimedRotatingFileHandler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler  # Import RichHandler for better console logging

# Initialize rich console for general output
console = Console()

# Configure logging
# Create a logger
logger = logging.getLogger("excel_extraction_pipeline")
logger.setLevel(logging.INFO)  # Set minimum logging level

class CustomFormatter(logging.Formatter):
    def formatException(self, record):
        s = super().formatException(record)
        return repr(s)

    def format(self, record):
        record.message = record.getMessage()
        if self.usesTime() :
            record.asctime = self.formatTime(record, self.datefmt)
        s = f"{record.asctime} - {record.message}"
        if record.exc_info:
            s = s + ' | Exception: ' + self.formatException(record)
        return s

# Create handlers
# Use RichHandler for console output
console_handler = RichHandler(
    console=console,  # Use the existing rich console
    show_time=True,
    show_path=False,  # Hide path for cleaner output
    keywords=["Error", "Warning", "Configuration Error", "Runtime Error"]  # Highlight key terms
)
console_handler.setLevel(logging.INFO)  # Set minimum level for console output

# Create a rotating file handler
log_file_path = "pipeline.log"  # Default log file path
# Rotate logs daily
file_handler = TimedRotatingFileHandler(log_file_path, when="D", interval=1, backupCount=7)
file_handler.setLevel(logging.ERROR)  # Log only errors and above to file

# Apply the custom formatter
formatter = CustomFormatter('%(asctime)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
# Prevent duplicate handlers if the module is reloaded
try:
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)  # Add file handler
except Exception as e:
    console.print(f"Error adding handlers to logger: {e}")

# Example usage:
# logger.info("This is an informational message.")
# logger.warning("This is a warning message.")
# logger.error("This is an error message.")
# logger.exception("This is an error with traceback.") # Use inside an except block

# Keep the rich console for non-logging display like tables and panels
# console.print("This is a general message using console.print")

# Export the logger for use in other modules
__all__ = ["console", "logger", "create_summary_table", "create_progress_panel"]
