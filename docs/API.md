# Excel Extraction Pipeline API Documentation

This document describes the API endpoints provided by the Excel Extraction Pipeline.

## Base URL

*(The base URL will depend on the deployment environment. Example: `http://localhost:8000`)*

## Authentication

*(Currently, no authentication is implemented. This section should be updated if authentication is added.)*

## Endpoints

### POST /extract

Extracts structured data from an uploaded Excel or CSV file.

-   **Request:**
    -   Method: `POST`
    -   Content-Type: `multipart/form-data`
    -   Body Parameters:
        -   `file`: The Excel (`.xlsx`, `.xls`) or CSV (`.csv`) file to process. (Required)

-   **Successful Response (200 OK):**
    -   Content-Type: `application/json`
    -   Body: A JSON object containing the extracted data, structured according to the dynamically determined configuration for the classified document type. The structure includes sections defined in the configuration (e.g., "Identifier", "Denomination") along with a "Context" section.

    **Example Success Response Body:**
    ```json
    {
      "Classification": {
        "predicted_class": "Mutual Funds",
        "confidence": "High",
        "validation_reason": "The text contains data points like \"FCP\"..."
      },
      "Context": {
        "ValidationConfidence": 0.95,
        "FileName": "8206743_Diffusion VL externe.2025.01.27",
        "HeaderStartLine": 0,
        "HeaderEndLine": 0,
        "ContentStartLine": 1,
        "FileType": "xlsx",
        "ProcessingTimeSeconds": 235.913
      },
      "Identifier": {
        "ISIN Code": "FR0000979678",
        "ValidationConfidence": 0.99
      },
      "Denomination": {
        "Fund Name": "UFF ACTIONS EUROPE EVOLUTIF",
        "Currency": "EUR",
        "ValidationConfidence": 1.00
      },
      "Valorisation": {
        "Net Asset Value": 100.5,
        "Publication Date": "2025-01-27",
        "ValidationConfidence": 0.98
      },
      // ... other extracted sections ...
    }
    ```

-   **Error Responses:**
    -   Uses standard HTTP status codes (4xx for client errors, 5xx for server errors).
    -   Content-Type: `application/json`
    -   Body: A JSON object following the structure defined by `PipelineException.to_dict()`:
        ```json
        {
          "detail": { // Note: FastAPI wraps the detail in a "detail" key
            "error": {
              "code": "ERROR_CODE", // e.g., "FILE_PROCESSING_ERROR", "LLM_INTERACTION_ERROR"
              "message": "Human-readable error description.",
              "severity": "ERROR", // Or WARNING, CRITICAL
              "context": {
                // Optional additional context key-value pairs
                "file_type": ".docx",
                "original_exception": "UnsupportedMediaType"
              }
            }
          }
        }
        ```
    -   **Common Status Codes:**
        -   `400 Bad Request`: Invalid input file type, validation errors, configuration issues.
        -   `500 Internal Server Error`: General pipeline processing errors, extraction failures, deduplication issues.
        -   `502 Bad Gateway`: Errors communicating with the underlying LLM service.

## Example Request (curl)

```bash
curl -X POST "http://localhost:8000/extract" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/spreadsheet.xlsx"
```

*(Replace `http://localhost:8000` with the actual API base URL and `/path/to/your/spreadsheet.xlsx` with the correct file path.)*
