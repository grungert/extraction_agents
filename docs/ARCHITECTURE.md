# Excel Extraction Pipeline Architecture

This document provides an overview of the system architecture for the Excel Extraction Pipeline.

## 1. High-Level Overview

*(A diagram and brief description of the main components and their interactions will be added here.)*

The pipeline is designed to dynamically process Excel or CSV files, classify the document type, and extract structured data based on configurable models. It leverages Large Language Models (LLMs) for classification, header detection, data extraction, validation, and deduplication tasks.

## 2. Data Flow

*(A diagram and description of the data flow from file upload to final JSON output will be added here.)*

1.  **File Upload:** An Excel/CSV file is uploaded via the API.
2.  **Preprocessing:** The file is converted into Markdown format.
3.  **Classification:** The Markdown content is classified to determine the document type (e.g., "Mutual Funds").
4.  **Dynamic Configuration Loading:** Based on the classification, the specific configuration (models, prompts, examples) for that document type is loaded.
5.  **Header Detection & Validation:** The header rows and content start are identified and validated.
6.  **Section Extraction & Validation:** Data is extracted for each defined section using the corresponding Pydantic model and validated against the original content.
7.  **Deduplication:** Conflicts where multiple fields might map to the same source data are resolved.
8.  **Output Generation:** The final structured data is returned as a JSON response.

## 3. Agent-Based Architecture

The core logic is implemented using a series of specialized agents, each responsible for a specific task in the pipeline:

-   **`DynamicClassificationAgent` & `DynamicClassificationValidationAgent`:** Classify the document type and validate the classification.
-   **`DynamicHeaderDetectionAgent` & `DynamicHeaderValidationAgent`:** Detect and validate header rows.
-   **`DynamicExtractionAgent`:** Extracts data for specific sections based on loaded models.
-   **`DynamicValidationAgent`:** Validates and potentially corrects extracted data against the source.
-   **`DynamicDeduplicationAgent`:** Resolves conflicts in extracted data.
-   **`DynamicAgentPipelineCoordinator`:** Orchestrates the flow between agents and manages dynamic configuration loading.

This modular design allows for flexibility and extensibility.

## 4. Key Components

-   **`api_main.py`:** FastAPI application providing the main API endpoint (`/extract`).
-   **`src/extraction/pipeline.py`:** Contains the `DynamicAgentPipelineCoordinator` which manages the overall process.
-   **`src/extraction/agents/`:** Directory containing the individual agent classes.
-   **`src/extraction/llm.py`:** Utilities for configuring and interacting with the LLM.
-   **`src/models.py`:** Defines core Pydantic models (AppConfig, BaseExtraction, etc.).
-   **`src/dynamic_model_factory.py`:** Creates Pydantic models dynamically based on configuration.
-   **`src/config_manager.py`:** Handles loading and accessing configuration files.
-   **`src/exceptions.py`:** Defines custom exception types.
-   **`src/utils/`:** Contains utility modules (display, error handling, formatting, prompts).
-   **`config/`:** Directory storing configuration files (e.g., `app_config.json`, `full_config_*.json`).
-   **`src/prompts/`:** Directory storing prompt templates used by the agents.

## 5. Design Patterns

*(Details on specific design patterns like Strategy (for agents), Factory (for models), Configuration Management, etc., will be added here.)*
