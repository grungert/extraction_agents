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
#   "markitdown[all] >= 0.1.0"
# ]
# ///
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track, Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.markdown import Markdown
from rich.live import Live
from rich.status import Status
from rich import print as rprint
import json
import time
from datetime import datetime, timedelta
from functools import wraps
import statistics
from contextlib import contextmanager

# Timing and progress tracking utilities
class ProcessingStats:
    """Track processing statistics for files and sheets"""
    def __init__(self):
        self.start_time = time.time()
        self.file_times = {}
        self.sheet_times = {}
        self.field_counts = {}
        self.success_count = 0
        self.error_count = 0
        self.total_fields_extracted = 0
        
    def record_file_start(self, file_path):
        """Record the start time for a file"""
        self.file_times[file_path] = {"start": time.time(), "end": None, "duration": None}
        
    def record_file_end(self, file_path, success=True, fields_extracted=0):
        """Record the end time and status for a file"""
        if file_path in self.file_times:
            end_time = time.time()
            self.file_times[file_path]["end"] = end_time
            self.file_times[file_path]["duration"] = end_time - self.file_times[file_path]["start"]
            self.file_times[file_path]["success"] = success
            self.file_times[file_path]["fields_extracted"] = fields_extracted
            
            if success:
                self.success_count += 1
                self.total_fields_extracted += fields_extracted
            else:
                self.error_count += 1
    
    def record_sheet_time(self, file_path, sheet_name, duration, fields_extracted=0):
        """Record processing time for a sheet"""
        if file_path not in self.sheet_times:
            self.sheet_times[file_path] = {}
        
        self.sheet_times[file_path][sheet_name] = {
            "duration": duration,
            "fields_extracted": fields_extracted
        }
        
    def get_average_file_time(self):
        """Get the average processing time per file"""
        durations = [data["duration"] for data in self.file_times.values() 
                    if data["duration"] is not None]
        return statistics.mean(durations) if durations else 0
    
    def get_estimated_time_remaining(self, files_remaining):
        """Estimate remaining time based on average processing time"""
        avg_time = self.get_average_file_time()
        return avg_time * files_remaining
    
    def get_total_duration(self):
        """Get total processing duration so far"""
        return time.time() - self.start_time
    
    def get_summary(self):
        """Get a summary of processing statistics"""
        total_files = self.success_count + self.error_count
        success_rate = (self.success_count / total_files * 100) if total_files > 0 else 0
        
        return {
            "total_duration": self.get_total_duration(),
            "total_files": total_files,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "total_fields_extracted": self.total_fields_extracted,
            "avg_fields_per_file": (self.total_fields_extracted / self.success_count) 
                                   if self.success_count > 0 else 0
        }

def time_function(func):
    """Decorator to measure and print execution time of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        
        # Format duration nicely
        if duration < 1:
            formatted_time = f"{duration*1000:.2f} ms"
        elif duration < 60:
            formatted_time = f"{duration:.2f} seconds"
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            formatted_time = f"{minutes} min {seconds:.2f} sec"
            
        console.print(f"[dim]Function {func.__name__} completed in {formatted_time}[/dim]")
        return result
    return wrapper

@contextmanager
def timed_section(description):
    """Context manager to time a section of code without using Status (to avoid display conflicts)"""
    console.print(f"[bold blue]{description}...[/bold blue]")
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        
        # Format duration nicely
        if duration < 1:
            formatted_time = f"{duration*1000:.2f} ms"
        elif duration < 60:
            formatted_time = f"{duration:.2f} seconds"
        else:
            minutes = int(duration // 60)
            seconds = duration % 60
            formatted_time = f"{minutes} min {seconds:.2f} sec"
        
        console.print(f"[green]✓[/green] {description} completed in {formatted_time}")

def format_time_delta(seconds):
    """Format seconds into a human-readable time string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

# Initialize rich console
console = Console()
import uuid
from typing import Dict, List, TypedDict, Optional, Callable, Any
import tempfile
import os
from markitdown import MarkItDown
import pandas as pd
from openai import OpenAI
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import (
    HumanMessage,
)
from pydantic import BaseModel, Field

# Define a custom prompt to provide instructions and any additional context.
examples = []
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
"""You are an expert data extraction algorithm. Your task is to map the column headers from an Excel table to the corresponding fields in a structured JSON output.  

### Instructions:  
- Identify the column headers in the given input table.  
- Match each header to the corresponding attribute in the JSON output.  
- Extract only relevant data under each header and assign it to the correct JSON field.  
- If a required value is missing or unavailable, return `null` for that field.  
- Ensure accuracy by strictly following the column-to-attribute mapping.  


Your task is to process the input data and return structured JSON output that follows this mapping.  
"""
        ),
        MessagesPlaceholder("examples"),
        ("human", "{text}"),
    ]
)

class DataToExtract(BaseModel):

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    code: Optional[str] = Field(..., description="ISIN Code from header. Example of values: SEDOL, ISIN code, FMSedols, BloombergTicker, Instrument Identifier/ISIN code, CODE ISIN, ISIN, AFCCodes, ISIN Code, CODE_WPK, Isin Code, WKN, USD.ISIN, VN, Bundesamt, ISIN CODE, Code ISIN, CodIsin, ISESedols, Sedol, EuroClearCedel, Column2, CODE_ISIN")
    code_type: Optional[str] = Field(..., description="Isin from header")
    currency: Optional[str] = Field(..., description="Share currency from header")
    cic_code: Optional[str] = Field(..., description="CIC Code from header")

class Data(BaseModel):

    # Creates a model so that we can extract multiple entities.
    dataExtracted: List[DataToExtract]

class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted

def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    try:
        console.print("[dim]Converting example to messages...[/dim]")
        messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
        
        tool_calls = []
        for tool_call in example["tool_calls"]:
            tool_calls.append(
                {
                    "id": str(uuid.uuid4()),
                    "args": tool_call.model_dump(),
                    "name": tool_call.__class__.__name__,
                },
            )
        
        messages.append(AIMessage(content="", tool_calls=tool_calls))
        
        tool_outputs = example.get("tool_outputs") or [
            "You have correctly called this tool."
        ] * len(tool_calls)
        
        for output, tool_call in zip(tool_outputs, tool_calls):
            messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
            
        console.print(f"[green]Converted example to {len(messages)} messages[/green]")
        return messages
        
    except Exception as e:
        console.print(f"[red]Error converting example to messages: {str(e)}[/red]")
        raise

# Define example data for the tool to learn from
example_data = """CODE ISIN FCP	DATE	COUPON	CREDIT D'IMPÔT	VL APRES DETACH.	Coefficient	Ouvrant à PL	NET du PL
FR0007436969	UFF AVENIR SECURITE	12/27/89	5,53	0,03	75,42	1,0732752522	NR	5,52627687485613 €
FR0007437124	UFF AVENIR DIVERSIFIE	12/27/89	1,30	0,65	90,64	1,0143469851	NR	1,30039011703511 €
FR0010180786	UFF AVENIR FRANCE	12/27/89	1,27	0,41	106,70	1,0118588370	NR	1,26532684307051 €
FR0007437124	UFF AVENIR DIVERSIFIE	12/26/90	0,04	0,02	74,65	1,0005310024	NR	0,0396367444817267 €
FR0007436969	UFF AVENIR SECURITE	12/28/90	7,65	0,04	77,38	1,0988623769	NR	7,65043 €
FR0010180786	UFF AVENIR FRANCE	12/15/91	1,13	0,56	90,24	1,0125183721	NR	1,12964721772921 €
FR0007436969	UFF AVENIR SECURITE	12/23/91	4,73	0,01	78,86	1,0600243577	NR	4,73354198522159 €"""

# Create a proper example with the correct DataToExtract model
example_tool_call = DataToExtract(
    code="CODE ISIN",
    code_type="Isin",
    currency="EUR",
    cic_code=None
)

# Create the example in the format expected by tool_example_to_messages
examples = [
    {
        "input": example_data,
        "tool_calls": [Data(dataExtracted=[example_tool_call])]
    }
]

# Convert examples to messages format
messages = []
for example in examples:
    messages.extend(tool_example_to_messages(example))

# Configure to use local LLM server
llm = ChatOpenAI(
    model_name="gemma-3-12b-it",
    base_url="http://localhost:1234/v1",
    api_key="null",
    temperature=0.3,
    max_retries=2
)

runnable = prompt | llm.with_structured_output(
    schema=Data,
    method="function_calling",
    include_raw=False,
)

def excel_to_markdown(file_path, start_row=0, end_row=15, all_sheets=False):
    """
    Reads a specified range of rows from a file and converts them to markdown.
    
    Args:
        file_path (str): Path to the file (.csv, .xls, or .xlsx)
        start_row (int): First row to read (0-based, default: 0)
        end_row (int): Last row to read (exclusive, default: 15)
        all_sheets (bool): Whether to process all sheets in Excel files (default: False)
        
    Returns:
        str: Markdown representation of the data
    """
    try:
        # Calculate number of rows to read
        nrows = end_row - start_row
        
        if file_path.endswith('.csv'):
            console.print(f"[dim]Processing CSV file: {file_path}[/dim]")
            # For CSV files, there's only one sheet
            df = pd.read_csv(file_path, skiprows=start_row, nrows=nrows, 
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
            
            console.print(f"[green]Converted CSV to markdown (rows {start_row}-{end_row-1})[/green]")
            return result.text_content
            
        elif file_path.endswith(('.xls', '.xlsx')):
            temp_suffix = '.xlsx'
            markdown_results = []
            
            if all_sheets:
                # Get all sheet names
                xls = pd.ExcelFile(file_path)
                sheet_names = xls.sheet_names
                
                console.print(f"[dim]Processing {len(sheet_names)} sheets in Excel file[/dim]")
                
                # Process each sheet
                for sheet_name in sheet_names:
                    console.print(f"[dim]Processing sheet: {sheet_name}[/dim]")
                    df = pd.read_excel(file_path, sheet_name=sheet_name, 
                                      skiprows=start_row, nrows=nrows,
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
                # Process only the default sheet (original behavior)
                df = pd.read_excel(file_path, skiprows=start_row, nrows=nrows,
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
                
                console.print(f"[green]Converted Excel to markdown (rows {start_row}-{end_row-1})[/green]")
                return result.text_content
        else:
            raise ValueError("Unsupported file format. Please provide a .csv, .xls, or .xlsx file.")
    except Exception as e:
        console.print(f"[red]Error converting file to markdown: {str(e)}[/red]")
        raise

def process_excel_sheets_individually(file_path, start_row=0, end_row=15):
    """
    Process each sheet in an Excel file individually and run extraction on each.
    
    Args:
        file_path (str): Path to the Excel file
        start_row (int): First row to read (0-based, default: 0)
        end_row (int): Last row to read (exclusive, default: 15)
        
    Returns:
        dict: Dictionary of extraction results with sheet names as keys
    """
    results = {}
    
    if file_path.endswith('.csv'):
        # For CSV files, process as a single sheet
        markdown = excel_to_markdown(file_path, start_row=start_row, end_row=end_row)
        
        # Check if there's data to process
        if markdown:
            console.print(Panel("[bold green]Processing CSV file...", expand=False))
            sheet_result = runnable.invoke(
                {
                    "text": markdown,
                    "examples": [],
                }
            )
            results["csv_data"] = sheet_result
        else:
            print("No data found in CSV file")
        
    elif file_path.endswith(('.xls', '.xlsx')):
        # Get all sheet names
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        
        # Process each sheet individually without using track
        console.print(f"[bold]Processing {len(sheet_names)} sheets individually...[/bold]")
        
        for sheet_idx, sheet_name in enumerate(sheet_names):
            console.print(f"[dim]Sheet {sheet_idx+1}/{len(sheet_names)}: [cyan]{sheet_name}[/cyan][/dim]")
            
            # Read data from this sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name,
                              skiprows=start_row, nrows=(end_row - start_row),
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
            md = MarkItDown(enable_plugins=False, nrows=(end_row - start_row))
            result = md.convert(temp_path)
            sheet_markdown = result.text_content
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            # Process this individual sheet with the LLM
            console.print(f"  [blue]Running extraction on sheet: {sheet_name}...[/blue]")
            sheet_result = runnable.invoke(
                {
                    "text": sheet_markdown,
                    "examples": messages,
                }
            )
            console.print(f"  [green]✓[/green] Extraction completed for sheet: {sheet_name}")
            
            # Store the result for this sheet
            results[sheet_name] = sheet_result
            console.print(f"  [green]✓[/green] Processed sheet '{sheet_name}'")
    else:
        raise ValueError("Unsupported file format. Please provide a .csv, .xls, or .xlsx file.")
    
    return results

def prepare_excel_sheets_markdown(file_path, start_row=0, end_row=15):
    """
    Processes each sheet in an Excel file individually and converts to markdown.
    Does NOT run LLM - just prepares the data for later processing.
    
    Args:
        file_path (str): Path to the Excel file
        start_row (int): First row to read (0-based, default: 0)
        end_row (int): Last row to read (exclusive, default: 15)
        
    Returns:
        dict: Dictionary with sheet names as keys and markdown content as values
    """
    sheet_markdowns = {}
    
    if file_path.endswith('.csv'):
        # For CSV files, process as a single sheet
        markdown = excel_to_markdown(file_path, start_row=start_row, end_row=end_row)
        
        # Check if there's data to process
        if markdown:
            console.print(Panel(
                f"[green]Processed CSV file to markdown (rows {start_row} to {end_row-1})",
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
            console.print(f"[dim]Sheet {sheet_idx+1}/{len(sheet_names)}: [cyan]{sheet_name}[/cyan] (rows {start_row} to {end_row-1})[/dim]")
            
            # Read data from this sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name,
                              skiprows=start_row, nrows=(end_row - start_row),
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
            md = MarkItDown(enable_plugins=False, nrows=(end_row - start_row))
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

from typing import Dict, List, TypedDict, Optional
from pydantic import BaseModel, Field

# Base model for all extraction sections
class BaseExtraction(BaseModel):
    """Base model that all extraction models should inherit from"""
    pass

# Context section
class ContextModel(BaseExtraction):
    FileName: Optional[str] = Field(None, description="File name of the Excel document")
    HeaderStartLine: Optional[int] = Field(None, description="Line where headers start (1-based)")
    HeaderEndLine: Optional[int] = Field(None, description="Line where headers end (1-based)")
    ContentStartLine: Optional[int] = Field(None, description="Line where content starts (1-based)")
    FileType: Optional[str] = Field(None, description="File type (xlsx, csv)")

# Identifier section
class IdentifierModel(BaseExtraction):
    Code: Optional[str] = Field(None, description="Header for ISIN Code. Examples: CODE ISIN, Code ISIN, ISIN")
    CodeType: Optional[str] = Field(None, description="Type of code, typically 'Isin'")
    Currency: Optional[str] = Field(None, description="Header for currency. Examples: Devise, Currency")
    CIC_Code: Optional[str] = Field(None, description="Header for CIC Code if present")

# Denomination section
class DenominationModel(BaseExtraction):
    VehiculeName: Optional[str] = Field(None, description="Header for vehicle name")
    CompartmentName: Optional[str] = Field(None, description="Header for compartment name")
    InstrumentName: Optional[str] = Field(None, description="Header for instrument name. Examples: FCP")
    ShareType: Optional[str] = Field(None, description="Header for share type")

# Valorisation section
class ValorisationModel(BaseExtraction):
    Nav: Optional[str] = Field(None, description="Header for NAV. Examples: Valeur Liquidative, VL")
    NavDate: Optional[str] = Field(None, description="Header for NAV date. Example: Date de publication")

# MarketCap section
class MarketCapModel(BaseExtraction):
    ReferenceDate: Optional[str] = Field(None, description="Header for reference date")
    CompartmentCurrency: Optional[str] = Field(None, description="Header for compartment currency") 
    CompartmentAssetValue: Optional[str] = Field(None, description="Header for asset value. Example: Actif Net")
    ShareAssetValue: Optional[str] = Field(None, description="Header for share asset value")
    Number_of_Shares: Optional[str] = Field(None, description="Header for number of shares. Example: Nombre de Parts")

# CorporateAction section
class CorporateActionModel(BaseExtraction):
    Currency: Optional[str] = Field(None, description="Header for currency")
    Type: Optional[str] = Field(None, description="Header for type. Example: Coupon")
    Value: Optional[str] = Field(None, description="Header for value. Example: COUPON")
    ExecutionDate: Optional[str] = Field(None, description="Header for execution date. Example: DATE")
    PaymentDate: Optional[str] = Field(None, description="Header for payment date")
    RecordDate: Optional[str] = Field(None, description="Header for record date")
    DistributionRate: Optional[str] = Field(None, description="Header for distribution rate")

# Characteristics section
class CharacteristicsModel(BaseExtraction):
    Strategy: Optional[str] = Field(None, description="Header for strategy")
    AssetManager: Optional[str] = Field(None, description="Header for asset manager")
    PortfolioManager: Optional[str] = Field(None, description="Header for portfolio manager")
    HostingCountry: Optional[str] = Field(None, description="Header for hosting country")
    LegalStatus: Optional[str] = Field(None, description="Header for legal status")
    UnderLegalStatus: Optional[str] = Field(None, description="Header for under legal status")
    InceptionDate: Optional[str] = Field(None, description="Header for inception date")
    DistributionPolicy: Optional[str] = Field(None, description="Header for distribution policy")
    PaymentFrequency: Optional[str] = Field(None, description="Header for payment frequency")

# Define which models to extract
EXTRACTION_MODELS = {
    "Context": ContextModel,
    "Identifier": IdentifierModel, 
}

def extract_section(markdown_content, section_name, model_class):
    """Extract a specific section using a specific model"""
    try:
        # Create a dynamic prompt for this specific section
        section_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are an expert data extraction algorithm. Your task is to map the column headers from an Excel table to the corresponding fields in the {section_name} section.  

### Instructions:  
- Identify the column headers in the given input table that correspond to the {section_name} section.
- Match each header to the corresponding attribute in the JSON output.
- Extract only relevant data under each header and assign it to the correct JSON field.
- If a required value is missing or unavailable, return `null` for that field.
- Ensure accuracy by strictly following the column-to-attribute mapping.
                    """
                ),
                MessagesPlaceholder("examples"),
                ("human", "{text}"),
            ]
        )
        
        # Create a runnable specifically for this model
        section_runnable = section_prompt | llm.with_structured_output(
            schema=model_class,
            method="function_calling",
            include_raw=False,
        )
        
        # Run the extraction
        result = section_runnable.invoke(
            {
                "text": markdown_content,
                "examples": messages,
            }
        )
        
        # Validate the result structure
        if not hasattr(result, 'model_dump'):
            return None
            
        return result
        
    except Exception as e:
        return None

def extract_all_sections(markdown_content, source_file=None):
    """Extract all sections and combine them into a complete result"""
    results = {}
    section_stats = {}

    # Initialize all sections with complete field structure
    for section_name, model_class in EXTRACTION_MODELS.items():
        # Create empty instance to get all fields
        empty_instance = model_class()
        results[section_name] = {k: None for k in empty_instance.model_fields.keys()}
        
        # Initialize section stats
        section_stats[section_name] = {
            "status": "pending",
            "fields": 0,
            "start_time": None,
            "duration": None
        }

    # Process each section without using Live display
    console.print("[bold]Extracting data sections...[/bold]")
    
    for section_name, model_class in EXTRACTION_MODELS.items():
        # Show current section being processed
        console.print(f"[dim]Processing section: [cyan]{section_name}[/cyan][/dim]")
        
        # Record start time
        section_stats[section_name]["start_time"] = time.time()
        
        try:
            # Run extraction without Status indicator
            console.print(f"  [blue]Extracting {section_name} data...[/blue]")
            section_result = extract_section(markdown_content, section_name, model_class)
            
            if section_result:
                result_data = section_result.model_dump()
                # Update only fields that have values
                for field, value in result_data.items():
                    if value is not None:
                        results[section_name][field] = value
                
                # Count fields with values
                fields_with_values = len([v for v in result_data.values() if v is not None])
                
                # Update section stats
                section_stats[section_name].update({
                    "status": "success",
                    "fields": fields_with_values,
                    "duration": time.time() - section_stats[section_name]["start_time"]
                })
                
                # Show success indicator
                console.print(f"  [green]✓[/green] Section [cyan]{section_name}[/cyan]: {fields_with_values} fields extracted")
            else:
                # Update section stats for no data
                section_stats[section_name].update({
                    "status": "no_data",
                    "fields": 0,
                    "duration": time.time() - section_stats[section_name]["start_time"]
                })
                console.print(f"  [yellow]⚠[/yellow] Section [cyan]{section_name}[/cyan]: No data extracted")
            
        except Exception as e:
            # Update section stats for error
            section_stats[section_name].update({
                "status": "error",
                "fields": 0,
                "duration": time.time() - section_stats[section_name]["start_time"],
                "error": str(e)
            })
            console.print(f"  [red]✗[/red] Section [cyan]{section_name}[/cyan]: Error - {type(e).__name__}: {str(e)}")

    # Create section summary table
    section_table = Table(
        title="[bold]Section Extraction Summary[/bold]",
        show_header=True,
        header_style="bold magenta"
    )
    section_table.add_column("Section", style="cyan")
    section_table.add_column("Status", width=12)
    section_table.add_column("Fields", justify="right")
    section_table.add_column("Time", justify="right")
    
    total_fields = 0
    for section_name, stats in section_stats.items():
        status_str = {
            "success": "[green]Success[/green]",
            "no_data": "[yellow]No Data[/yellow]",
            "error": "[red]Error[/red]",
            "pending": "[dim]Pending[/dim]"
        }.get(stats["status"], "[dim]Unknown[/dim]")
        
        section_table.add_row(
            section_name,
            status_str,
            str(stats["fields"]),
            format_time_delta(stats["duration"]) if stats["duration"] else "-"
        )
        
        total_fields += stats["fields"]
    
    console.print(section_table)
    
    # Save results to JSON file if source file is provided
    if source_file and total_fields > 0:
        try:
            # Create json_outputs directory if it doesn't exist
            os.makedirs("json_outputs", exist_ok=True)
            
            # Generate output filename
            base_name = os.path.basename(source_file)
            json_filename = f"json_outputs/{base_name}.json"
            
            # Save results to JSON
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Saved results to {json_filename}[/green]")
                
        except Exception as e:
            console.print(f"[red]Error saving JSON output: {str(e)}[/red]")
    
    return results

@time_function
def process_file(file_path):
    """Process a single file and return results"""
    try:
        file_name = os.path.basename(file_path)
        sheet_stats = {}
        
        # Log start of sheet preparation without using timed_section (to avoid nesting)
        console.print(f"[bold blue]Preparing sheets for {file_name}...[/bold blue]")
        prep_start_time = time.time()
        all_sheet_markdowns = prepare_excel_sheets_markdown(file_path)
        prep_duration = time.time() - prep_start_time
        console.print(f"[green]✓[/green] Preparing sheets for {file_name} completed in {format_time_delta(prep_duration)}")
        
        if not all_sheet_markdowns:
            console.print(Panel(
                "[red]No sheets found to process[/red]",
                title="Error",
                border_style="red"
            ))
            return None, 0, 0

        extraction_results = {}
        sheets_processed = 0
        fields_extracted = 0
        
        # Process each sheet without using a nested Live display
        console.print(f"[bold]Processing {len(all_sheet_markdowns)} sheets for {file_name}...[/bold]")
        
        for sheet_idx, (sheet_name, markdown_content) in enumerate(all_sheet_markdowns.items()):
            # Show current sheet being processed
            console.print(f"[dim]Sheet {sheet_idx+1}/{len(all_sheet_markdowns)}: [cyan]{sheet_name}[/cyan][/dim]")
            
            # Record start time for this sheet
            sheet_start_time = time.time()
            
            try:
                # Process sheet without status indicator
                console.print(f"  [blue]Extracting data from {sheet_name}...[/blue]")
                sheet_results = extract_all_sections(markdown_content, file_path)
                
                if sheet_results:
                    extraction_results[sheet_name] = sheet_results
                    sheets_processed += 1
                    
                    # Count extracted fields
                    sheet_fields = sum(
                        len([v for v in section.values() if v is not None])
                        for section in sheet_results.values()
                    )
                    fields_extracted += sheet_fields
                    
                    # Record sheet statistics
                    sheet_duration = time.time() - sheet_start_time
                    sheet_stats[sheet_name] = {
                        "duration": sheet_duration,
                        "fields": sheet_fields,
                        "status": "success"
                    }
                    
                    # Show sheet success indicator
                    console.print(f"  [green]✓[/green] Sheet [cyan]{sheet_name}[/cyan]: {sheet_fields} fields extracted in {format_time_delta(sheet_duration)}")
                else:
                    # Record sheet failure
                    sheet_stats[sheet_name] = {
                        "duration": time.time() - sheet_start_time,
                        "fields": 0,
                        "status": "no_data"
                    }
                    console.print(f"  [yellow]⚠[/yellow] Sheet [cyan]{sheet_name}[/cyan]: No data extracted")
            
            except Exception as e:
                # Record sheet error
                sheet_stats[sheet_name] = {
                    "duration": time.time() - sheet_start_time,
                    "fields": 0,
                    "status": "error",
                    "error": str(e)
                }
                console.print(f"  [red]✗[/red] Sheet [cyan]{sheet_name}[/cyan]: Error - {type(e).__name__}: {str(e)}")
        
        # Create sheet summary table
        sheet_table = Table(
            title=f"[bold]Sheet Processing Summary for {file_name}[/bold]",
            show_header=True,
            header_style="bold magenta"
        )
        sheet_table.add_column("Sheet", style="cyan")
        sheet_table.add_column("Status", width=12)
        sheet_table.add_column("Fields", justify="right")
        sheet_table.add_column("Time", justify="right")
        
        for sheet_name, stats in sheet_stats.items():
            status_str = {
                "success": "[green]Success[/green]",
                "no_data": "[yellow]No Data[/yellow]",
                "error": "[red]Error[/red]"
            }.get(stats["status"], "[dim]Unknown[/dim]")
            
            sheet_table.add_row(
                sheet_name,
                status_str,
                str(stats["fields"]),
                format_time_delta(stats["duration"])
            )
        
        console.print(sheet_table)
        
        return extraction_results, sheets_processed, fields_extracted
        
    except Exception as e:
        console.print(Panel(
            f"[red]Error processing {os.path.basename(file_path)}[/red]\n"
            f"[yellow]{type(e).__name__}: {str(e)}[/yellow]",
            border_style="red"
        ))
        return None, 0, 0

@time_function
def process_directory(directory_path):
    """Process all Excel/CSV files in a directory sequentially"""
    # Create json_outputs directory if it doesn't exist
    os.makedirs("json_outputs", exist_ok=True)
    
    # Initialize processing statistics
    stats = ProcessingStats()
    
    # Find all Excel/CSV files
    all_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.xlsx', '.xls', '.csv')):
                all_files.append(os.path.join(root, file))

    if not all_files:
        console.print(Panel(
            "[yellow]No Excel/CSV files found in directory[/yellow]",
            border_style="yellow"
        ))
        return

    total_files = len(all_files)
    
    # Print process header
    console.print(f"[bold]Processing {total_files} files...[/bold]")
    
    # Process files without using Progress widget
    for i, file_path in enumerate(all_files):
        file_name = os.path.basename(file_path)
        
        # Show current file being processed
        console.print(f"[cyan]File {i+1}/{total_files}: {file_name}[/cyan]")
        
        # Record start time for this file
        stats.record_file_start(file_path)
        
        try:
            # Process file without using timed_section (to avoid nesting)
            console.print(f"[bold blue]Processing {file_name}...[/bold blue]")
            process_start_time = time.time()
            results, sheets_processed, fields_extracted = process_file(file_path)
            process_duration = time.time() - process_start_time
            console.print(f"[green]✓[/green] Processing {file_name} completed in {format_time_delta(process_duration)}")
            
            if results is not None:
                # Record successful processing
                stats.record_file_end(file_path, success=True, fields_extracted=fields_extracted)
                
                # Calculate time remaining
                files_remaining = total_files - (i + 1)
                time_remaining = stats.get_estimated_time_remaining(files_remaining)
                
                console.print(Panel(
                    f"[green]Successfully processed {file_name}[/green]\n"
                    f"• Sheets: [cyan]{sheets_processed}[/cyan]\n"
                    f"• Fields: [cyan]{fields_extracted}[/cyan]\n"
                    f"• Processing time: [cyan]{format_time_delta(stats.file_times[file_path]['duration'])}[/cyan]",
                    title=f"[bold green]File {i+1}/{total_files}[/bold green]",
                    subtitle=f"[dim]Est. remaining: {format_time_delta(time_remaining)}[/dim]",
                    border_style="green"
                ))
            else:
                # Record failed processing
                stats.record_file_end(file_path, success=False)
                console.print(Panel(
                    f"[yellow]No data extracted from {file_name}[/yellow]",
                    border_style="yellow"
                ))
        except Exception as e:
            # Record error
            stats.record_file_end(file_path, success=False)
            console.print(Panel(
                f"[red]Error processing {file_name}:[/red]\n"
                f"[yellow]{type(e).__name__}: {str(e)}[/yellow]",
                border_style="red"
            ))
        
        # Show progress percentage
        percent_complete = ((i + 1) / total_files) * 100
        console.print(f"[dim]Progress: {percent_complete:.1f}% ({i+1}/{total_files})[/dim]")
        console.print("─" * 80)  # Separator line
    
    # Get summary statistics
    summary = stats.get_summary()
    
    # Show detailed final summary
    console.print(Panel(
        f"[bold]Processing Summary:[/bold]\n"
        f"• Files processed: [cyan]{total_files}[/cyan]\n"
        f"• Successful: [green]{summary['success_count']}[/green]\n"
        f"• Errors: [red]{summary['error_count']}[/red]\n"
        f"• Success rate: [cyan]{summary['success_rate']:.1f}%[/cyan]\n"
        f"• Total fields extracted: [cyan]{summary['total_fields_extracted']}[/cyan]\n"
        f"• Avg. fields per file: [cyan]{summary['avg_fields_per_file']:.1f}[/cyan]\n"
        f"• Total processing time: [cyan]{format_time_delta(summary['total_duration'])}[/cyan]",
        title="[bold]Results[/bold]",
        border_style="blue"
    ))

def main():
    """Main execution function with rich formatting"""
    start_time = time.time()
    
    try:
        # Print fancy header with timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(Panel.fit(
            f"[bold green]Excel Header Mapper[/bold green]\n[dim]Started at: {current_time}[/dim]",
            subtitle="[dim]Process files or directories to extract structured data[/dim]",
            border_style="green",
            padding=(1, 2)
        ))
        
        # Print system info
        console.print(Panel(
            f"[bold]System Configuration:[/bold]\n"
            f"• Model: [cyan]{llm.model_name}[/cyan]\n"
            f"• Temperature: [cyan]{llm.temperature}[/cyan]\n"
            f"• API Base: [cyan]{llm.openai_api_base}[/cyan]",
            title="System Info",
            border_style="blue",
            expand=False
        ))

        # Hardcoded input directory path
        input_dir = "testing"
        
        if os.path.isdir(input_dir):
            # Record start time
            dir_start_time = time.time()
            
            # Process directory
            process_directory(input_dir)
            
            # Calculate total execution time
            total_time = time.time() - dir_start_time
            
            # Print execution time summary
            console.print(Panel(
                f"[bold]Total Execution Time:[/bold] [cyan]{format_time_delta(total_time)}[/cyan]",
                border_style="blue"
            ))
        else:
            console.print(Panel(
                f"[red]Directory not found: {input_dir}[/red]",
                border_style="red"
            ))
        
    except Exception as e:
        # Calculate execution time even if there was an error
        error_time = time.time() - start_time
        
        console.print(Panel(
            f"[red]Error in main execution: {str(e)}[/red]\n"
            f"[yellow]Execution time before error: {format_time_delta(error_time)}[/yellow]",
            title="[bold red]Error[/bold red]",
            border_style="red",
            padding=(1, 2)
        ))
        raise

if __name__ == "__main__":
    main()
