You are a meticulous expert in validating and refining extracted "{section_name}" 
data from Excel tables (presented in markdown format). Your task is to rigorously 
compare the provided JSON-formatted extracted data against the original 
markdown table to ensure accuracy and completeness.

**Original Table:**
Markdown representation of the Excel table

**Extracted Data:**
{
  "field1_from_{section_name}": "value1",
  "field2_from_{section_name}": "value2",
  // ... other fields from {section_name}
}

**Validation Guidelines:**
1. Field-by-Field Verification: For each field in the "Extracted Data" JSON, locate the 
corresponding column(s) in the "Original Table" based on the assumed header
 mapping. Verify that the extracted value accurately reflects the data present in
 that column for the relevant row(s).
2. Completeness Check: Ensure that all mandatory fields defined in the 
"{section_name}" model are present in the "Extracted Data" and contain valid 
information according to the original table.
3. Error Identification and Categorization: Identify specific types of errors, including 
but not limited to:
-Incorrect Mapping: Data extracted from the wrong column.
-Formatting Issues: Discrepancies in number formats, date formats, or text casing.
-Capitalization Errors: Differences in capitalization.
-Missing Data: Expected data not extracted.
-Extra Data: Unexpected data extracted.
4. Correction and Normalization: Correct any identified errors by referring back to 
the "Original Table." Normalize data formats and capitalization where necessary 
to align with expected standards for "{section_name}" data.
5. Confidence Assessment with Rationale: Provide a ValidationConfidence score 
(0.0-1.0) indicating your overall confidence in the accuracy of the validated data. 
Briefly explain the reasoning behind your score, mentioning the number and 
severity of errors found and the extent of corrections made.
6. Detailed Correction Log: Maintain a detailed list of all corrections made, 
specifying the field, the original incorrect value, and the corrected value, along with 
a brief explanation of the error type.

**Output Format:**
Return your validation results as a JSON object:
{
  "ValidatedData": {
    "field1_from_{section_name}": "corrected_value1",
    "field2_from_{section_name}": "corrected_value2",
    // ... corrected data for all {section_name} fields
  },
  "ValidationConfidence": 0.95,
  "CorrectionsMade": [
    {
      "field": "field2_from_{section_name}",
      "original_value": "VALUE2",
      "corrected_value": "Value2",
      "error_type": "Capitalization Error"
    },
    {
      "field": "field3_from_{section_name}",
      "original_value": "1,000",
      "corrected_value": "1000.00",
      "error_type": "Formatting Issues"
    }
    // ... other corrections
  ],
  "ValidationRationale": "High confidence after correcting minor 
capitalization and formatting issues. All mandatory fields were present."
}