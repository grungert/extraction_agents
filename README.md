# Excel Header Mapper

A dynamic LLM-based system for extracting structured data from Excel files.

## Features

- Automatic header detection
- Configurable field extraction
- Validation with confidence scoring
- Local LLM integration
- Dynamic model configuration via JSON

## Getting Started

### Prerequisites

- Python 3.10+
- Local LLM (e.g., Gemma 3)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Process all Excel files in input/ directory
python run_pipeline.py

# Process a specific Excel file
python run_pipeline.py --file path/to/your/file.xlsx

# Specify a particular sheet
python run_pipeline.py --file path/to/your/file.xlsx --sheet "Sheet2"

# Specify a different config file
python run_pipeline.py --config config/alternative_config.json

# Specify a custom output location
python run_pipeline.py --file path/to/your/file.xlsx --output custom_output.json
```

## Directory Structure

- `config/` - Configuration files
- `input/` - Place Excel files here for processing
- `json_outputs/` - Extracted JSON results
- `src/` - Source code
  - `extraction/` - Core extraction logic
  - `examples/` - Example data for LLM prompting
  - `utils/` - Utility functions

## Configuration

Update the `config/full_config.json` file to add or modify extraction models and their fields.
