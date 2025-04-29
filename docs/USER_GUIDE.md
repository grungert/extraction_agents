# Excel Extraction Pipeline - User Guide

This guide explains how to use the Excel Extraction Pipeline API to extract structured data from your spreadsheet files.

## 1. Overview

The API takes an Excel (`.xlsx`, `.xls`) or CSV (`.csv`) file as input, automatically determines the type of document it contains (e.g., Mutual Fund report, Invoice), and extracts relevant information into a structured JSON format based on pre-defined configurations for that document type.

## 2. Using the API

The primary way to interact with the pipeline is through the `/extract` API endpoint.

### Sending a Request

You need to send a `POST` request with the file included as `multipart/form-data`.

**Example using `curl`:**

```bash
curl -X POST "YOUR_API_BASE_URL/extract" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/spreadsheet.xlsx"
```

-   Replace `YOUR_API_BASE_URL` with the actual URL where the API is hosted (e.g., `http://localhost:8000`).
-   Replace `/path/to/your/spreadsheet.xlsx` with the path to the file you want to process.

### Input File Formats

-   Microsoft Excel Workbook (`.xlsx`)
-   Microsoft Excel 97-2003 Workbook (`.xls`)
-   Comma Separated Values (`.csv`)

## 3. Understanding the Output

### Successful Extraction (HTTP Status 200 OK)

If the extraction is successful, the API will return a JSON object containing the extracted data. The structure will vary depending on the detected document type, but it generally includes:

-   **`Classification`**: Information about the detected document type and the confidence of the classification.
-   **`Context`**: Metadata about the processing, including the input filename, detected header locations, file type, and processing time.
-   **Specific Data Sections**: Keys corresponding to the data sections defined in the configuration for the detected document type (e.g., "Identifier", "Denomination", "MarketCap"). Each section contains the extracted fields and a `ValidationConfidence` score.

**Example Output Snippet:**

```json
{
  "Classification": {
    "predicted_class": "Mutual Funds",
    "confidence": "High",
    // ...
  },
  "Context": {
    "ValidationConfidence": 0.95,
    "FileName": "your_file_name",
    // ...
  },
  "Identifier": {
    "ISIN Code": "FR0000979678",
    "ValidationConfidence": 0.99
  },
  // ... other sections ...
}
```

### Errors

If an error occurs during processing, the API will return an error response with an appropriate HTTP status code (4xx for client-side issues, 5xx for server-side issues) and a JSON body detailing the error:

```json
{
  "detail": {
    "error": {
      "code": "ERROR_CODE", // e.g., "FILE_PROCESSING_ERROR"
      "message": "Error description.",
      "severity": "ERROR",
      "context": { /* Optional details */ }
    }
  }
}
```

**Common Errors:**

-   **400 Bad Request:**
    -   `FILE_PROCESSING_ERROR`: Problem reading or parsing the uploaded file (e.g., corrupted file, unsupported format within the spreadsheet).
    -   `VALIDATION_ERROR`: Input data failed some validation check.
    -   `CONFIG_ERROR`: Issue with the server-side configuration for the detected document type.
-   **500 Internal Server Error:**
    -   `EXTRACTION_ERROR`: An error occurred during the data extraction phase.
    -   `PIPELINE_ERROR`: A general error within the processing pipeline.
-   **502 Bad Gateway:**
    -   `LLM_INTERACTION_ERROR`: The server had trouble communicating with the underlying AI model needed for extraction.

## 4. Troubleshooting

-   **Unsupported File Type:** Ensure your file is one of the supported formats (`.xlsx`, `.xls`, `.csv`).
-   **File Content Issues:** The pipeline works best with tabular data. Complex layouts, merged cells, or non-standard formats might lead to extraction errors. Check the `FILE_PROCESSING_ERROR` details if provided.
-   **Unrecognized Document Type:** If the `Classification` section shows "None of those" or low confidence, the system might not be configured to handle that specific type of document layout.
-   **LLM Errors:** `LLM_INTERACTION_ERROR` or `502 Bad Gateway` usually indicate temporary issues with the AI service. Retrying the request after a short wait might help.
-   **Low Confidence Scores:** If `ValidationConfidence` scores in the output are consistently low, it might indicate that the data layout is ambiguous or differs significantly from the examples the system was configured with.

If you encounter persistent issues, please refer to the detailed error message and code provided in the API response when seeking support.
