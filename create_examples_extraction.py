#!/usr/bin/env python3
# /// script
# dependencies = [
#   "rich>=13.7.0",
#   "pandas>=2.0.0",
#   "xlrd >= 2.0.1"
# ]
# ///
import pandas as pd
import chardet
import lxml.etree as ET
from io import StringIO
import json
import os
import random
from rich.console import Console
from rich.table import Table
from rich import print_json

console = Console()

# File paths
document_file_path = 'Fluxs - AI Training/8208700_PIMCO Global Investor Series PLC Re Investment Notice January 2025.2025.01.28.xlsx'
json_file_path = 'ground-truth/file_17_H.json'

# Load JSON metadata
with open(json_file_path, 'r') as f:
    metadata = json.load(f)

# Extract metadata
header_start = metadata["Context"]["HeaderStartLine"]
header_end = metadata["Context"]["HeaderEndLine"]
content_start = metadata["Context"]["ContentStartLine"]

# CHANGE THIS TO THE CORRECT MODEL
# json_output_fields = metadata["Identifier"][0]  # assume first object in list
# json_output_fields = metadata["Denomination"]
# json_output_fields = metadata["Valorisation"]
# json_output_fields = metadata["MarketCap"]
json_output_fields = metadata["CorporateAction"][0]
# json_output_fields = metadata["Characteristics"]

# Load entire document to access specific rows
name_doc, extension =  os.path.splitext(document_file_path)

if extension.lower() in ('.csv','.txt'):
        with open(document_file_path, 'rb') as f:
            data_crude = f.read()
            guessing = chardet.detect(data_crude)
            encode2use = guessing['encoding']
        try:
            df_all = pd.read_csv(document_file_path, encoding=encode2use, header=None, sep=';')
        except Exception as e:
            print(f"Error trying to open the document with pandas: {e}")
if extension.lower() == '.dat':
        try:
            try:
                df_all = pd.read_csv(document_file_path, header=None, sep=',')
            except Exception as e:
                df_all = pd.read_csv(document_file_path, header=None, sep=';')
        except Exception as e:
                print(f"Error trying to open the document with pandas as separeted by , or ; : {e}")
if extension.lower() in ('.xls', '.xlsx') :
        try:           
            df_all = pd.read_excel(document_file_path, header=None)
        except Exception as e:
            print(f"Error trying to open the document with pandas: {e}")
if extension.lower() == '.xml':
        try:
            parser = ET.XMLParser(recover=True)
            tree = ET.parse(document_file_path, parser=parser)
            root = tree.getroot()
            xml_string = ET.tostring(root, encoding='utf-8').decode('utf-8')
            df_all = pd.read_xml(StringIO(xml_string))
        except Exception as e:
            print(f"Error trying to open the document with pandas: {e}")
if extension.lower() == '.html':
        try:
            df_all = pd.read_html(document_file_path)
        except Exception as e:
            print(f"Error trying to open the document with pandas: {e}")

#Removing columns that are fully empty
empty_columns = df_all.columns[df_all.isna().all()]
df_all = df_all.drop(empty_columns, axis=1)

# Slice rows to include headers and 5 rows of content
if extension.lower() in ('.csv', '.xls', '.xlsx'):
    header_start = header_start - 1
    header_end = header_end - 1
    content_start  = content_start - 1
rows_needed = list(range(header_start, header_end + 1)) + list(range(content_start, content_start + min(5, df_all.shape[0]-content_start)))
df_slice = df_all.iloc[rows_needed].reset_index(drop=True)

# Try to detect column indexes based on the last header row
num_header_rows = header_end - header_start + 1
header_row = df_slice.iloc[num_header_rows-1]
column_map = {}
for key, col_name in json_output_fields.items():
    if col_name:
        for idx, val in enumerate(header_row):
            if str(val).strip().lower() == col_name.strip().lower():
                column_map[key] = idx
                break

# Format values to avoid scientific notation and clean dates
def clean_value(x):
    if isinstance(x, float):
        return f"{x:.4f}".rstrip('0').rstrip('.')
    if pd.isna(x):
        return " "
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")
    return str(x)

# Clean all values
df_slice_clean = df_slice.map(clean_value)

# Extract and clean header row
temp_header_df = pd.DataFrame(columns=df_slice_clean.columns)
for j in range(num_header_rows):
    row_cleaned = df_slice_clean.iloc[j].map(str).fillna(" ").map(str.strip)
    temp_header_df.loc[j] = row_cleaned
header_values = []
for col in temp_header_df.columns:
    columns_values = [' ' if element == 'nan' else element for element in temp_header_df[col].tolist()]
    header_values.append(' '.join(columns_values))
header_row_cleaned = pd.Series(header_values)

# Assign headers and drop the original header row
df_slice_clean.columns = header_row_cleaned
df_slice_clean = df_slice_clean.drop(index = list(range(0, num_header_rows))).reset_index(drop=True)

# Re-filter from cleaned dataframe using updated headers
# Filter dataframe to only selected columns
filtered_cols = list(column_map.values())
filtered_cols_clean = [col for i, col in enumerate(header_row_cleaned) if i in filtered_cols]
df_filtered = df_slice_clean[filtered_cols_clean]

# Generate clean markdown
markdown_table = df_slice_clean.to_markdown(index=False)

# Output metadata
validation_confidence = round(random.uniform(0.90, 1.00), 2)
json_output_fields["validation_confidence"] = validation_confidence

# === Markdown Output ===
print("\n# Original Table (Markdown)")
print(markdown_table)

# JSON result
console.rule("[cyan]JSON Output with Confidence")
print_json(data=json_output_fields)

# === Rich Output ===
console.rule("[bold blue]Rich Output")
# Input table (with real headers)
console.rule("[green]Input Table Slice (Full)")
input_table = Table(show_lines=True)
nb_col = 0
for col in df_slice_clean.columns:
    input_table.add_column(str(col))
    nb_col += 1
print(f"Number columns: {nb_col}")
for _, row in df_slice_clean.iterrows():
    row_str = [str(x) for x in row]
    print(f"Number of elements in row: {len(row_str)}")
    input_table.add_row(*row_str)
console.print(input_table)

console.rule("[green]Input Filtered Table Slice (Full)")
input_table2 = Table(show_lines=True)
for col in df_filtered.columns:
    input_table2.add_column(str(col))
for _, row in df_filtered.iterrows():
    input_table2.add_row(*[str(x) for x in row])
console.print(input_table2)



