# Progress

This document summarizes the current status of the project, what is working, what is left to build, and known issues.

## Current Status
- The core pipeline structure is in place, supporting initial document classification and validation, markdown conversion, header processing, extraction, and validation via configurable agents.
- CLI and API interfaces are implemented.
- Basic file type support (.xlsx, .xls, .csv) and formatting features (currency, trimming) are functional.
- Dynamic configuration loading based on classification is implemented.
- Configuration management has been refactored to use Pydantic models and separate application config.
- Basic logging is integrated.
- An initial test suite structure was in place but has been removed to focus on core refactoring.
- **Custom exception types have been defined and partially implemented.**
- **LLM token counting has been added to agent logs.**

## What Works
- Markdown conversion of Excel/CSV files.
- Document classification using `DynamicClassificationAgent`.
- Classification validation using `DynamicClassificationValidationAgent`.
- Dynamic loading of extraction configuration based on classification results.
- Configurable agent-based processing flow.
- CLI and API entry points.
- Extraction of currency symbols and formats.
- Trimming of empty columns.
- Pydantic-based configuration loading and validation.
- Basic logging to console.
- **Custom exceptions are defined and used in key areas.**
- **LLM input and output token counts are logged for each agent.**

## What's Left to Build
- Addressing duplicate header data in outputs.
- Accessing the "Hill git".
- Establishing ground truth status tracking.
- Integrating Ollama.
- Creating an agent for CSV to markdown table conversion if needed.
- Adding information about different extraction types (column, row, fixed, deductive).
- Improving dynamic example creation.
- **Completing the implementation of custom exception types throughout the codebase.**
- **Further performance optimization.**
- **Completing code quality improvements (type hints, docstrings, refactoring).**
- **Writing detailed documentation.**
- **Implementing security measures.**
- **Defining deployment procedures.**
- **Conducting a full dynamism review.**
- Revisit and improve the test suite later.

## Known Issues
- None currently preventing core functionality, but further testing is needed to uncover potential issues.
