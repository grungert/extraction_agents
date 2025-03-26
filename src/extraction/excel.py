"""Excel file extraction utilities for Excel Header Mapper."""
import os
import tempfile
import pandas as pd
from markitdown import MarkItDown
from ..utils.display import console
from ..models import AppConfig
from rich.panel import Panel

def excel_to_markdown(file_path, config: AppConfig, sheet_name=None):
    """
    Read specified rows from a file and convert them to markdown.
    
    Args:
        file_path (str): Path to the file (.csv, .xls, or .xlsx)
        config (AppConfig): Application configuration
        sheet_name (str, optional): Specific sheet name to process
        
    Returns:
        str: Markdown representation of the data
    """
    try:
        # Calculate number of rows to read
        nrows = config.end_row - config.start_row
        
        if file_path.endswith('.csv'):
            console.print(f"[dim]Processing CSV file: {file_path}[/dim]")
            # For CSV files, there's only one sheet
            df = pd.read_csv(file_path, skiprows=config.start_row, nrows=nrows, 
                            dtype=str, header=None, keep_default_na=False)
            temp_suffix = '.csv'
            
            # Create a temporary file with the same format
            with tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save the dataframe to the temporary file
            df.to_csv(temp_path, index=False, header=False)
            
            # Convert to markdown
            md = MarkItDown(enable_plugins=False, nrows=nrows)
            result = md.convert(temp_path)
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            console.print(f"[green]Converted CSV to markdown (rows {config.start_row}-{config.end_row-1})[/green]")
            return result.text_content
            
        elif file_path.endswith(('.xls', '.xlsx')):
            temp_suffix = '.xlsx'
            markdown_results = []
            
            if config.all_sheets:
                # Get all sheet names
                xls = pd.ExcelFile(file_path)
                sheet_names = xls.sheet_names
                
                console.print(f"[dim]Processing {len(sheet_names)} sheets in Excel file[/dim]")
                
                # Process each sheet
                for sheet_name in sheet_names:
                    console.print(f"[dim]Processing sheet: {sheet_name}[/dim]")
                    df = pd.read_excel(file_path, sheet_name=sheet_name, 
                                      skiprows=config.start_row, nrows=nrows,
                                      dtype=str, header=None, keep_default_na=False)
                    
                    # Skip empty sheets
                    if df.empty:
                        console.print(f"[yellow]Sheet '{sheet_name}' is empty - skipping[/yellow]")
                        continue
                    
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Save the dataframe to the temporary file
                    df.to_excel(temp_path, index=False, header=False, engine='openpyxl')
                    
                    # Convert to markdown
                    md = MarkItDown(enable_plugins=False, nrows=nrows)
                    result = md.convert(temp_path)
                    
                    # Add sheet name as header and append to results
                    sheet_markdown = f"## Sheet: {sheet_name}\n\n{result.text_content}\n\n"
                    markdown_results.append(sheet_markdown)
                    
                    # Clean up the temporary file
                    os.unlink(temp_path)
                    console.print(f"[green]Converted sheet '{sheet_name}' to markdown[/green]")
                
                # Combine all results
                return "".join(markdown_results)
            else:
                console.print(f"[dim]Processing single sheet in Excel file[/dim]")
                # Process specified sheet or default sheet
                sheet_param = sheet_name if sheet_name else 0
                df = pd.read_excel(file_path, sheet_name=sheet_param,
                                  skiprows=config.start_row, nrows=nrows,
                                  dtype=str, header=None, keep_default_na=False)
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Save the dataframe to the temporary file
                df.to_excel(temp_path, index=False, header=False, engine='openpyxl')
                
                # Convert to markdown
                md = MarkItDown(enable_plugins=False, nrows=nrows)
                result = md.convert(temp_path)
                
                # Clean up the temporary file
                os.unlink(temp_path)
                
                sheet_name_str = sheet_name if sheet_name else "default sheet"
                console.print(f"[green]Converted Excel sheet '{sheet_name_str}' to markdown (rows {config.start_row}-{config.end_row-1})[/green]")
                return result.text_content
        else:
            raise ValueError("Unsupported file format. Please provide a .csv, .xls, or .xlsx file.")
    except Exception as e:
        console.print(f"[red]Error converting file to markdown: {str(e)}[/red]")
        raise

def prepare_excel_sheets_markdown(file_path, config: AppConfig):
    """
    Process each sheet in an Excel file individually and convert to markdown.
    Does NOT run LLM - just prepares the data for later processing.
    
    Args:
        file_path (str): Path to the Excel file
        config (AppConfig): Application configuration
        
    Returns:
        dict: Dictionary with sheet names as keys and markdown content as values
    """
    sheet_markdowns = {}
    
    if file_path.endswith('.csv'):
        # For CSV files, process as a single sheet
        markdown = excel_to_markdown(file_path, config)
        
        # Check if there's data to process
        if markdown:
            console.print(Panel(
                f"[green]Processed CSV file to markdown (rows {config.start_row} to {config.end_row-1})",
                expand=False
            ))
            sheet_markdowns["csv_data"] = markdown
        else:
            console.print("[red]No data found in CSV file[/red]")
        
    elif file_path.endswith(('.xls', '.xlsx')):
        # Get all sheet names
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        
        # Process each sheet individually without using track
        console.print(f"[bold]Converting {len(sheet_names)} sheets to markdown...[/bold]")
        
        for sheet_idx, sheet_name in enumerate(sheet_names):
            console.print(f"[dim]Sheet {sheet_idx+1}/{len(sheet_names)}: [cyan]{sheet_name}[/cyan] (rows {config.start_row} to {config.end_row-1})[/dim]")
            
            # Read data from this sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name,
                              skiprows=config.start_row, nrows=(config.end_row - config.start_row),
                              dtype=str, header=None, keep_default_na=False)
            
            # Skip empty sheets
            if df.empty:
                console.print(f"  [yellow]⚠[/yellow] Sheet '{sheet_name}' is empty - skipping")
                continue
            
            # Create a temporary file
            temp_suffix = '.xlsx'
            with tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save the dataframe to the temporary file
            df.to_excel(temp_path, index=False, header=False, engine='openpyxl')
            
            # Convert to markdown
            md = MarkItDown(enable_plugins=False, nrows=(config.end_row - config.start_row))
            result = md.convert(temp_path)
            sheet_markdown = result.text_content
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            # Store just the markdown content
            sheet_markdowns[sheet_name] = sheet_markdown
            console.print(f"  [green]✓[/green] Converted sheet '{sheet_name}' to markdown")
    else:
        raise ValueError("Unsupported file format. Please provide a .csv, .xls, or .xlsx file.")
    
    return sheet_markdowns
