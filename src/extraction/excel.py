"""Excel file extraction utilities for Excel Header Mapper."""
import os
import tempfile
import pandas as pd
from decimal import Decimal
from markitdown import MarkItDown
from ..utils.display import console
from ..models import AppConfig
from rich.panel import Panel
import re
import openpyxl
import xlrd

CURRENCY_REGEX = re.compile(r'\[\$([^\-\]]+)(?:\-[0-9A-F]+)?\]')

def format_value_based_on_excel_format(value, num_format):
    if value is None:
        return ""

    is_numeric = isinstance(value, (int, float, Decimal))

    if not is_numeric or not num_format or num_format == 'General':
        return str(value)

    currency_match = CURRENCY_REGEX.search(num_format)
    if currency_match:
        currency_symbol = currency_match.group(1)
        decimal_places = 2 if '.00' in num_format else 0
        try:
            num_decimal = Decimal(str(value))
            formatted_num = "{:,.{dp}f}".format(num_decimal, dp=decimal_places)
            if num_format.startswith('['):
                return f"{currency_symbol}{formatted_num}"
            else:
                return f"{formatted_num} {currency_symbol}"
        except Exception:
            return str(value)

    if '%' in num_format:
        try:
            num_decimal = Decimal(str(value)) * 100
            decimal_places = 0
            if '.0%' in num_format:
                decimal_places = 1
            elif '.00%' in num_format:
                decimal_places = 2
            formatted_num = "{:,.{dp}f}".format(num_decimal, dp=decimal_places)
            return f"{formatted_num}%"
        except Exception:
            return str(value)

    try:
        num_decimal = Decimal(str(value))
        decimal_places = 2 if '.' in str(value) else 0
        if '.00' in num_format: decimal_places = 2
        if '.000' in num_format: decimal_places = 3
        return "{:,.{dp}f}".format(num_decimal, dp=decimal_places)
    except Exception:
        return str(value)

def get_formatted_dataframe(file_path, sheet_name, start_row, end_row):
    ext = os.path.splitext(file_path)[-1].lower()
    nrows = end_row - start_row

    if ext == ".xlsx":
        wb = openpyxl.load_workbook(file_path, data_only=True)
        ws = wb[sheet_name] if sheet_name else wb.active

        rows = list(ws.iter_rows(min_row=start_row + 1, max_row=end_row, values_only=False))
        data = []
        for row in rows:
            formatted_row = []
            for cell in row:
                value = cell.value
                num_format = cell.number_format
                formatted_value = format_value_based_on_excel_format(value, num_format)
                formatted_row.append(formatted_value)
            data.append(formatted_row)

        return pd.DataFrame(data)

    elif ext == ".xls":
        wb = xlrd.open_workbook(file_path, formatting_info=True)
        sheet = wb.sheet_by_name(sheet_name) if sheet_name else wb.sheet_by_index(0)
        xf_list = wb.xf_list
        fmt_map = wb.format_map
        data = []

        for r_idx in range(start_row, min(end_row, sheet.nrows)):
            row_data = []
            for c_idx in range(sheet.ncols):
                cell = sheet.cell(r_idx, c_idx)
                val = cell.value
                fmt = None
                if cell.ctype in (xlrd.XL_CELL_NUMBER, xlrd.XL_CELL_DATE):
                    xf_idx = cell.xf_index
                    if xf_idx < len(xf_list):
                        fmt_key = xf_list[xf_idx].format_key
                        fmt_obj = fmt_map.get(fmt_key)
                        if fmt_obj:
                            fmt = fmt_obj.format_str
                formatted = format_value_based_on_excel_format(val, fmt)
                row_data.append(formatted)
            data.append(row_data)

        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported Excel format")

def excel_to_markdown(file_path, config: AppConfig, sheet_name=None):
    try:
        nrows = config.end_row - config.start_row

        if file_path.endswith('.csv'):
            console.print(f"[dim]Processing CSV file: {file_path}[/dim]")
            df = pd.read_csv(file_path, skiprows=config.start_row, nrows=nrows,
                             dtype=str, header=None, keep_default_na=False)
            temp_suffix = '.csv'

            with tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=False) as temp_file:
                temp_path = temp_file.name
            df.to_csv(temp_path, index=False, header=False)

        elif file_path.endswith(('.xls', '.xlsx')):
            sheet_param = sheet_name if sheet_name else (0 if not config.all_sheets else None)
            df = get_formatted_dataframe(file_path, sheet_param, config.start_row, config.end_row)
            temp_suffix = '.xlsx'

            with tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=False) as temp_file:
                temp_path = temp_file.name
            df.to_excel(temp_path, index=False, header=False, engine='openpyxl')

        else:
            raise ValueError("Unsupported file format. Please provide a .csv, .xls, or .xlsx file.")

        md = MarkItDown(enable_plugins=False, nrows=nrows)
        result = md.convert(temp_path)
        os.unlink(temp_path)

        console.print(f"[green]Converted to markdown (rows {config.start_row}-{config.end_row-1})[/green]")
        return result.text_content

    except Exception as e:
        console.print(f"[red]Error converting file to markdown: {str(e)}[/red]")
        raise

def prepare_excel_sheets_markdown(file_path, config: AppConfig):
    sheet_markdowns = {}

    if file_path.endswith('.csv'):
        markdown = excel_to_markdown(file_path, config)
        if markdown:
            console.print(Panel(
                f"[green]Processed CSV file to markdown (rows {config.start_row} to {config.end_row-1})",
                expand=False
            ))
            sheet_markdowns["csv_data"] = markdown
        else:
            console.print("[red]No data found in CSV file[/red]")

    elif file_path.endswith(('.xls', '.xlsx')):
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        console.print(f"[bold]Converting {len(sheet_names)} sheets to markdown...[/bold]")

        for idx, sheet in enumerate(sheet_names):
            console.print(f"[dim]Sheet {idx + 1}/{len(sheet_names)}: [cyan]{sheet}[/cyan][/dim]")
            df = get_formatted_dataframe(file_path, sheet, config.start_row, config.end_row)

            if df.empty:
                console.print(f"  [yellow]⚠ Sheet '{sheet}' is empty - skipping[/yellow]")
                continue

            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
                temp_path = temp_file.name
            df.to_excel(temp_path, index=False, header=False, engine='openpyxl')

            md = MarkItDown(enable_plugins=False, nrows=config.end_row - config.start_row)
            result = md.convert(temp_path)
            sheet_markdowns[sheet] = result.text_content
            os.unlink(temp_path)

            console.print(f"  [green]✓ Converted sheet '{sheet}' to markdown[/green]")

    else:
        raise ValueError("Unsupported file format. Please provide a .csv, .xls, or .xlsx file.")

    return sheet_markdowns
