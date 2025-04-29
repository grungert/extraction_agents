# Configuration Guide

This document explains how to configure the Excel Extraction Pipeline.

## 1. Overview

The pipeline uses a dynamic configuration system managed by `src/config_manager.py`. The core idea is:

1.  A main application configuration (`config/app_config.json`) defines global settings like LLM parameters, logging levels, and paths to specific prompt files.
2.  The document classification agent determines the type of the input document (e.g., "Mutual Funds").
3.  Based on the classified type, a specific detailed configuration file (e.g., `config/full_config_Mutual_Funds.json`) is loaded.
4.  This detailed configuration defines the specific extraction models, fields, examples, and prompt variations needed for that document type.

## 2. Main Application Configuration (`config/app_config.json`)

This file defines global settings using the `AppConfig` model (`src/models.py`). Key sections include:

-   `model`: Parameters for the main LLM used for extraction, validation, etc. (model name, API key, temperature).
-   `classification_model`: Parameters for the LLM used specifically for classification.
-   `classification_labels`: List of possible document types the classifier can predict.
-   `classification_prompt`: Filename of the main classification prompt template (in `src/prompts/`).
-   `classification_validation_prompt`: Filename of the classification validation prompt template.
-   `enable_deduplication_agent`: Boolean flag to enable/disable the deduplication step.
-   `include_header_examples_in_prompt`: Boolean flag to control whether header examples are included in prompts.
-   `excel_processing`: Settings for how Excel files are converted to Markdown (e.g., `max_rows`).

## 3. Document-Specific Configuration (`config/full_config_*.json`)

These files define the extraction details for a specific document type identified by the classifier. The filename should match the classification label (e.g., `full_config_Mutual_Funds.json` for the "Mutual Funds" label).

The structure follows the `ExtractionConfig` model (`src/models.py`) and contains:

-   **`description`**: A brief description of this configuration.
-   **`prompts`**: Specifies the filenames (within `src/prompts/`) for various system prompts used by the agents for this document type:
    -   `header_detection_system_prompt_file`
    -   `header_validation_system_prompt_file`
    -   `section_extraction_template_file`
    -   `section_validation_template_file`
    -   `deduplication_instruction_prompt_file`
    -   `deduplication_system_prompt_file`
-   **`header_detection`**: Configuration for the header detection agent (currently empty but can be extended).
-   **`validation`**: Configuration for validation agents:
    -   `confidence_threshold`: Minimum confidence score required for validation steps to pass.
-   **`header_examples`**: A list of few-shot examples used for header detection and validation. Each example includes:
    -   `table`: A string representation of an example input table (Markdown format).
    -   `json`: The expected `ContextModel` JSON output for that table.
-   **`extraction_models`**: A list defining the sections to be extracted. Each item represents a section and follows the `ExtractionModelConfig` structure:
    -   `name`: The name of the section (e.g., "Identifier", "Denomination"). This corresponds to a key in the final JSON output.
    -   `description`: A description of what this section represents.
    -   `fields`: A dictionary where keys are the field names (as they will appear in the output JSON) and values are `FieldDefinition` objects defining:
        -   `description`: Description of the field.
        -   `type`: Expected data type (e.g., "string", "number", "date").
        -   `examples`: Optional list of example values for this field.
    -   `examples`: A list of few-shot examples specific to this section's extraction and validation. Each example includes:
        -   `table`: Example input table snippet (Markdown).
        -   `extracted_json`: The expected JSON output from the extraction agent for this snippet.
        -   `validated_json`: The expected JSON output from the validation agent for this snippet (including `ValidationConfidence` and potentially `ValidatedData`).

## 4. Adding New Document Types

To add support for a new document type:

1.  **Add Label:** Add the new document type label (e.g., "Invoice", "Balance Sheet") to the `classification_labels` list in `config/app_config.json`.
2.  **Train/Update Classifier (Optional but Recommended):** If necessary, fine-tune or update the classification model/prompts (`classification_prompt`, `classification_validation_prompt`) to accurately recognize the new document type.
3.  **Create Configuration File:** Create a new `config/full_config_YourNewLabel.json` file.
4.  **Define Prompts:** Create specific prompt template files in `src/prompts/` for header detection, extraction, validation, etc., tailored to the new document type. Update the `prompts` section in the new config file to point to these templates.
5.  **Define Header Examples:** Add relevant `header_examples` to the new config file.
6.  **Define Extraction Models:** Define the `extraction_models` list in the new config file, specifying the sections, fields, and examples relevant to the new document type. Use the `ExtractionModelConfig` and `FieldDefinition` structures.
7.  **Test:** Thoroughly test the pipeline with examples of the new document type.

## 5. Dynamic Model Factory (`src/dynamic_model_factory.py`)

This module reads the `extraction_models` configuration for a given document type and dynamically creates Pydantic models based on the defined fields. This allows the extraction and validation agents to work with strongly-typed models specific to the document being processed without needing to hardcode them.
