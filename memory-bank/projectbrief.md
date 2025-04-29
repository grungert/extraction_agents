# Project Brief

This is the main document outlining the core requirements and goals of the project.

## Project Goals
- **Classify document type as the first step in the pipeline.**
- **Validate the document classification.**
- Extract structured data from Excel/CSV files using LLM agents.
- Support header detection, extraction, validation.
- Provide a fully configurable pipeline via JSON.
- Offer both CLI and HTTP API interfaces.
- Address duplicate header data in extraction outputs.
- Integrate Ollama.
- Improve dynamic example creation.

## Key Requirements
- Accurate data extraction from various Excel/CSV formats.
- **Accurate and reliable document classification.**
- Flexible configuration through JSON files, dynamically loaded based on classification.
- Robust header detection and validation.
- Efficient handling of large files.
- Clear and usable CLI and API interfaces.

## Scope
- **Included:** LLM-based data extraction pipeline, **document classification and validation**, header processing, validation, JSON configuration, CLI and API interfaces, handling of `.xlsx`, `.xls`, `.csv` files, currency/format extraction, column trimming.
- **Excluded:** Access to the "Hill git" (currently a separate task), Ground truth status tracking (currently a separate task).
