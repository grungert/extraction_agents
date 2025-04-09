# test_excel_formatting.py
#!/usr/bin/env python3
# /// script
# dependencies = [
#   "openai>=1.63.0",
#   "rich>=13.7.0",
#   "pandas>=2.0.0",
#   "typer>=0.4.0",
#   "langchain>=0.1.0",
#   "langchain-core>=0.1.0",
#   "langchain-openai>=0.1.0",
#   "xlrd >= 2.0.1",
#   "markitdown[all] >= 0.1.0",
#   "langfuse>=0.1.0",
#   "tabulate>=0.9.0",
#   "openpyxl>=3.0.10",
#   "pyexcel>=0.6.7",
#   "pyexcel-xls>=0.6.7"
# ]
# ///
import pandas as pd
import openpyxl
from tabulate import tabulate
import argparse
import os
import re
from decimal import Decimal, ROUND_HALF_UP

# Regex to find currency symbols/codes in Excel format strings
# Looks for patterns like [$€-1], [$$], [$USD-409], etc.
CURRENCY_REGEX = re.compile(r'\[\$([^-\]]+)(?:-[0-9A-F]+)?\]')

def format_value_based_on_excel_format(value, num_format):
    """
    Attempts to format a numeric value based on its Excel number format string,
    focusing on currency and percentages.
    """
    if value is None:
        return ""
        
    is_numeric = isinstance(value, (int, float, Decimal))
    
    if not is_numeric or not num_format or num_format == 'General':
        return str(value) # Return as string if not numeric or no specific format

    # --- Currency Handling ---
    currency_match = CURRENCY_REGEX.search(num_format)
    if currency_match:
        currency_symbol = currency_match.group(1)
        # Use the extracted symbol or ISO code directly (no manual mapping)
        
        # Determine decimal places (simple check for '.00')
        decimal_places = 2 if '.00' in num_format else 0
        
        # Format the number
        try:
            # Use Decimal for accurate rounding
            num_decimal = Decimal(str(value))
            # Format with appropriate decimal places
            formatted_num = "{:,.{dp}f}".format(num_decimal, dp=decimal_places)
            
            # Determine symbol position (simple check for symbol at start/end)
            if num_format.startswith('['): # Symbol likely at start e.g., [$€-1] #,##0.00
                return f"{currency_symbol}{formatted_num}"
            else: # Symbol likely at end e.g., #,##0.00 [$€-1]
                return f"{formatted_num} {currency_symbol}"
        except Exception:
             return str(value) # Fallback if formatting fails

    # --- Percentage Handling ---
    if '%' in num_format:
        try:
            # Use Decimal for accurate rounding
            num_decimal = Decimal(str(value)) * 100 # Multiply by 100 for percentage
            # Determine decimal places (simple check)
            decimal_places = 0
            if '.0%' in num_format:
                decimal_places = 1
            elif '.00%' in num_format:
                decimal_places = 2
                
            formatted_num = "{:,.{dp}f}".format(num_decimal, dp=decimal_places)
            return f"{formatted_num}%"
        except Exception:
             return str(value) # Fallback

    # --- Default Numeric Formatting (if not currency/percent) ---
    # Basic formatting if specific rules weren't matched
    try:
        num_decimal = Decimal(str(value))
        decimal_places = 2 if '.' in str(value) else 0 # Guess decimals
        if '.00' in num_format: decimal_places = 2
        if '.000' in num_format: decimal_places = 3
        # Add more specific decimal place detection if needed based on format codes
        
        return "{:,.{dp}f}".format(num_decimal, dp=decimal_places)
    except Exception:
        return str(value) # Final fallback

def get_formatted_value_and_format(cell):
    """
    Attempts to get the cell value and its number format string.
    Returns the raw value if formatting info isn't available or applicable.
    """
    value = cell.value
    num_format = cell.number_format
    
    # Basic check if formatting might be relevant (is a number)
    is_numeric = isinstance(value, (int, float))
    
    # Return both value and format if relevant, otherwise just the value
    if is_numeric and num_format and num_format != 'General':
         # For simplicity in this test, just return both. 
         # Actual formatting reconstruction can be complex.
        return value, num_format
    else:
        # Handle non-numeric types or general format
        return value, None # Return None for format if not applicable

def process_excel(file_path, num_rows=15):
    """Reads an Excel file and extracts values and formats."""
    try:
        # Use openpyxl to access formatting
        workbook = openpyxl.load_workbook(filename=file_path, data_only=False) # data_only=False to get formulas if needed, but primarily for format
        sheet = workbook.active
        
        data = []
        headers = []
        
        # Read headers and data row by row
        for r_idx, row in enumerate(sheet.iter_rows(max_row=num_rows + 1)): # Read one extra for header potential
             if r_idx == 0: # Assume first row is header
                 headers = [cell.value if cell.value is not None else "" for cell in row]
                 continue # Skip header row for data processing
                 
             if r_idx > num_rows: # Stop after processing num_rows of data
                 break

             row_data = []
             for cell in row:
                 val = cell.value
                 fmt = cell.number_format
                 # Get the reconstructed display value
                 display_val = format_value_based_on_excel_format(val, fmt)
                 row_data.append(display_val)
             data.append(row_data)
             
        if not headers and data: # Fallback if no header row detected in first row
             headers = [f"Col_{i+1}" for i in range(len(data[0]))]

        return headers, data

    except Exception as e:
        print(f"Error processing Excel with openpyxl: {e}")
        # Fallback to pandas if openpyxl fails (won't get formatting)
        try:
            print("Falling back to pandas (formatting info will be lost)...")
            df = pd.read_excel(file_path, nrows=num_rows, engine='openpyxl') # Try pandas with openpyxl engine
            headers = df.columns.tolist()
            data = df.values.tolist()
            return headers, data
        except Exception as pd_e:
             print(f"Error processing Excel with pandas: {pd_e}")
             return [], []


def process_csv(file_path, num_rows=15):
    """Reads a CSV file."""
    try:
        df = pd.read_csv(file_path, nrows=num_rows)
        headers = df.columns.tolist()
        data = df.values.tolist()
        # Add placeholder for format column to match excel output structure
        data_with_fmt = [[f"{val}" for val in row] for row in data]
        return headers, data_with_fmt
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return [], []

def main():
    parser = argparse.ArgumentParser(description="Test reading formatted Excel/CSV data.")
    parser.add_argument("--file", required=True, help="Path to the input file (xlsx, xls, csv)")
    args = parser.parse_args()

    file_path = args.file
    num_rows = 15

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    headers = []
    data = []

    print(f"Processing file: {file_path}")

    if file_extension == '.xlsx':
        print("Processing as XLSX (attempting to read formats)...")
        headers, data = process_excel(file_path, num_rows)
    elif file_extension == '.xls':
        print("Detected .xls file. Attempting to convert to temporary .xlsx for processing...")
        temp_xlsx_path = file_path + ".converted.xlsx"
        converted = False
        try:
            import pyexcel
            try:
                pyexcel.save_book_as(file_name=file_path, dest_file_name=temp_xlsx_path)
                print(f"Converted using pyexcel: {temp_xlsx_path}")
                converted = True
            except Exception as e:
                print(f"pyexcel conversion failed: {e}")
        except ImportError:
            print("pyexcel not installed, skipping pyexcel conversion attempt.")

        if not converted:
            try:
                # Fallback to pandas conversion
                df = pd.read_excel(file_path, engine='xlrd')
                df.to_excel(temp_xlsx_path, index=False, engine='openpyxl')
                print(f"Converted using pandas: {temp_xlsx_path}")
                converted = True
            except Exception as e:
                print(f"pandas conversion failed: {e}")
                print("Unable to convert .xls file. Aborting.")
                return

        try:
            # Process the converted file
            headers, data = process_excel(temp_xlsx_path, num_rows)
        finally:
            # Delete the temporary file
            try:
                #os.remove(temp_xlsx_path)
                print(f"Deleted temporary file: {temp_xlsx_path}")
            except Exception:
                pass
    elif file_extension == '.csv':
        print("Processing as CSV...")
        headers, data = process_csv(file_path, num_rows)
    else:
        print(f"Error: Unsupported file extension '{file_extension}'. Please use .xlsx, .xls, or .csv.")
        return

    if data:
        print("\n--- Extracted Data (Formatted for Markdown) ---")
        # Ensure all rows have the same number of columns as headers for tabulate
        num_headers = len(headers)
        uniform_data = []
        for row in data:
            row_len = len(row)
            if row_len == num_headers:
                uniform_data.append(row)
            elif row_len < num_headers:
                 uniform_data.append(row + [""] * (num_headers - row_len)) # Pad shorter rows
            else:
                 uniform_data.append(row[:num_headers]) # Truncate longer rows

        # Trim trailing empty columns
        # Transpose to columns
        columns = list(zip(*uniform_data))
        trimmed_columns = []
        trimmed_headers = []
        for idx, col in enumerate(columns):
            # Check if all values in the column are empty or None
            if all((str(val).strip() == "" or val is None) for val in col):
                continue  # Skip this column
            trimmed_columns.append(col)
            trimmed_headers.append(headers[idx])

        # Transpose back to rows
        trimmed_data = list(zip(*trimmed_columns))

        try:
             markdown_table = tabulate(trimmed_data, headers=trimmed_headers, tablefmt="github")
             print(markdown_table)
        except Exception as e:
             print(f"\nError generating markdown table with tabulate: {e}")
             print("Raw data:")
             print("Headers:", trimmed_headers)
             print("Data:", trimmed_data)

    else:
        print("\nNo data extracted.")

if __name__ == "__main__":
    main()
