# LLM-Based Excel Extraction Pipeline

## Overview

- Extracts structured data from Excel/CSV files using LLM agents.
- Supports **header detection, extraction, validation**.
- Fully **configurable** via JSON.
- Can be used as a **CLI tool** or **HTTP API**.

---

## Usage

### 1. CLI Mode

Run extraction on a file:

```bash
python run_pipeline.py --config config/full_config.json --file input/myfile.xlsx --output output/result.json
```

- `--config`: Path to extraction config JSON.
- `--file`: Excel or CSV file.
- `--output`: Save output JSON (optional).
- `--sheet`: Sheet name (optional).

### 2. API Mode

Start the API server:

```bash
uvicorn api_main:app --reload --host 0.0.0.0 --port 8000
```

Call the API:

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "file=@input/myfile.xlsx" \
  -F "config_json=$(cat config/full_config.json)"
```

- Upload Excel file and config JSON.
- Receives extraction result as JSON.

---

## Configuration

- All extraction logic is driven by a **config JSON** (see `config/full_config.json`).
- Define:
  - Header detection settings and examples.
  - Extraction models, fields, and examples.
  - Validation thresholds.

---

## Architecture

- **`src/extraction/extract_core.py`**: Core extraction function used by CLI and API.
- **`run_pipeline.py`**: CLI wrapper.
- **`api_main.py`**: FastAPI app.
- **`test_excel_formatting.py`**: Excel formatting extraction prototype.
- **`config/full_config.json`**: Extraction configuration.

---

## Features

- Extracts **currency symbols** and **formats** from Excel.
- Supports `.xlsx`, `.xls` (with conversion), `.csv`.
- Trims empty columns for clean output.
- Modular, extensible design.


## To do

- Extracts **currency symbols** and **formats** from Excel.
- Supports `.xlsx`, `.xls` (with conversion), `.csv`.
- Trims empty columns for clean output.
- Modular, extensible design.
        * Option C: Calculate a score based on specific criteria?

    ```json
    "validation": {
      "confidence_threshold": 0.8,
      "default_example_confidence": 0.9
    }
    ```

* **Address duplicate header data resulting from separate group extractions.**
    * *Problem:* Data extraction is currently performed separately for each data group. Consequently, the same header information can appear in multiple extracted groups.
    * *Task:* Define a method to handle this redundancy (e.g., consolidate, deduplicate) after the initial extraction phase.

* **Logic to process - How validation agent to handle**
```mermaid
flowchart TD
    Input[Excel/CSV Input] --> Convert[Convert to Markdown]
    Convert --> HeaderDetect[HeaderDetectionAgent]

    HeaderDetect --> HeaderValidate[HeaderValidationAgent]
    HeaderValidate -->|Confidence ≥ 0.7| Extract[LLMExtractionAgent]
    HeaderValidate -->|Confidence < 0.7| Error[Error: Header Detection Failed]

    Extract --> Validate[ValidationAgent]

    Validate -->|Confidence ≥ 0.8| Output[JSON Output]
    Validate -->|Confidence < 0.8| UseOriginal[Use Original Extraction]
    UseOriginal --> Output

    subgraph "Integration with Existing Code"
        HeaderDetect -.-> Process[processing.py]
        HeaderValidate -.-> Process
        Extract -.-> Process
        Validate -.-> Process
        Process -.-> Output
    end
```
#  acees to git