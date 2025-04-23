As an expert in analyzing the structure of Excel tables (presented in markdown) for 
information extraction, your task is to identify the column headers that best correspond 
to the fields within the "{section_name}" data structure.

Your process should involve a detailed examination of the table's header row(s) to
 determine the most likely mapping to the fields in "{section_name}".

**Guidelines:**
1. **Structural Analysis:** Analyze the first few rows of the table (especially the header
 rows and any potential sub-header rows) to understand the organization and naming 
conventions used for columns.

2. **Best-Match Identification:** For each field in the "{section_name}" model, identify the 
column header(s) that most closely align in meaning and context. Consider potential 
abbreviations, variations in capitalization, and the overall theme of the column.

3. **Header-Centric Output:** Your output should only contain the identified column headers. 
Do not include any data rows from the table.

4. **Null for Missing Fields:** If a semantically relevant column header cannot be identified for 
a specific field in the "{section_name}" model after analyzing the table structure, return null
for that field.

5. **Confidence Scoring with Rationale:** Assign a `ValidationConfidence` score (0.0-1.0) based 
on the clarity and strength of the matches found. Provide a brief `ValidationRationale`
 explaining the factors that influenced your confidence score, such as the directness of the 
matches or any ambiguities encountered.

**Output Format:**
Return your extraction as a JSON object where the keys are the field names from 
the "{section_name}" model, and the values are the corresponding identified column 
headers (or null). Include a final field named `ValidationConfidence` and a `ValidationRationale`.
{
  "field_a_from_{section_name}": "Column A",
  "field_b_from_{section_name}": "Label B",
  "field_c_from_{section_name}": null,
  "ValidationConfidence": 0.92,
  "ValidationRationale": "Strong matches found for 'field_a' and 'field_b'. 'field_c' was absent
 from the table headers."
}