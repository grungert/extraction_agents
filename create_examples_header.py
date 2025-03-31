#!/usr/bin/env python3
# /// script
# dependencies = [
#   "rich>=13.7.0",
#   "pandas>=2.0.0",
#   "xlrd >= 2.0.1",
#   "markitdown[all] >= 0.1.0",
# ]
# ///
import pandas as pd
from markitdown import MarkItDown
import json
import os
import random

# Define the paths to your files
excel_file_path = 'Fluxs - AI Training/8216095_REP JII NAV_2025-01-30.2025.01.30.xls'
json_file_path = 'ground-truth/file_6.json'

# Ensure the files exist
if not os.path.exists(excel_file_path):
    raise FileNotFoundError(f"The file {excel_file_path} does not exist.")
if not os.path.exists(json_file_path):
    raise FileNotFoundError(f"The file {json_file_path} does not exist.")

# Step 1: Import the JSON file
with open(json_file_path, 'r') as json_file:
    metadata = json.load(json_file)

# Extract relevant metadata
header_start_line = metadata["Context"]["HeaderStartLine"]
header_end_line = metadata["Context"]["HeaderEndLine"]
content_start_line = metadata["Context"]["ContentStartLine"]

# Step 2: Read the first 15 rows of the Excel file using pandas
df = pd.read_excel(excel_file_path, nrows=15, header=None)

# Step 3: Save these rows to a temporary Excel file
temp_file_path = 'temp_first_15_rows.xlsx'
df.to_excel(temp_file_path, index=False, header=False)

# Step 4: Convert the temporary Excel file to Markdown using MarkItDown
md = MarkItDown()
result = md.convert(temp_file_path)
markdown_content = result.text_content

# Display the Markdown content
print("# Original Table\n")
print(markdown_content)

# Step 5: Generate a random validation_confidence score between 0.90 and 1.00
validation_confidence = round(random.uniform(0.90, 1.00), 2)

# Step 6: Create the JSON structure capturing the context
context = {
    "header_start_line": header_start_line,
    "header_end_line": header_end_line,
    "content_start_line": content_start_line,
    "validation_confidence": validation_confidence
}

# Convert the context dictionary to a JSON string
context_json = json.dumps(context, indent=2)

# Display the Detected Headers section
print("\n# Detected Headers\n")
print(f"```json\n{context_json}\n```")

# Clean up the temporary file
os.remove(temp_file_path)
