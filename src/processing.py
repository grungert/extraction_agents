"""Main processing logic for Excel Header Mapper."""
import os
import time
import json
from datetime import datetime
from rich.panel import Panel
from rich.table import Table

from .utils.display import console, create_summary_table
from .utils.timing import ProcessingStats, time_function, timed_section
from .utils.formatting import format_time_delta
from .extraction.excel import prepare_excel_sheets_markdown
from .extraction.llm import initialize_llm_pipeline, extract_section, configure_llm
from .models import AppConfig, EXTRACTION_MODELS

def extract_all_sections(markdown_content, source_file, config, llm_pipeline, messages):
    """
    Extract all sections from markdown content using the agent pipeline.
    
    Args:
        markdown_content (str): Markdown content to process
        source_file (str): Source file path (for output)
        config (AppConfig): Application configuration
        llm_pipeline: LLM extraction pipeline (not used with agent pipeline)
        messages (List): Example messages for the LLM (not used with agent pipeline)
        
    Returns:
        dict: Dictionary of extracted sections
    """
    from .extraction.agents import AgentPipelineCoordinator
    
    # Initialize section stats
    section_stats = {}
    for section_name, model_class in EXTRACTION_MODELS.items():
        section_stats[section_name] = {
            "status": "pending",
            "fields": 0,
            "start_time": None,
            "duration": None
        }
    
    # Process each section using the agent pipeline
    console.print("[bold]Extracting data using LLM agent pipeline...[/bold]")
    
    # Record start time for overall processing
    overall_start_time = time.time()
    
    try:
        # Initialize the agent pipeline
        agent_pipeline = AgentPipelineCoordinator(config)
        
        # Process the markdown content
        console.print("[blue]Running agent pipeline...[/blue]")
        results = agent_pipeline.process_markdown(markdown_content, source_file)
        
        # Update section stats based on results
        for section_name, section_data in results.items():
            if section_name in section_stats:
                # Count fields with values
                fields_with_values = len([v for v in section_data.values() if v is not None])
                
                # Update section stats
                section_stats[section_name].update({
                    "status": "success" if fields_with_values > 0 else "no_data",
                    "fields": fields_with_values,
                    "duration": time.time() - overall_start_time  # Approximate duration
                })
        
        console.print(f"[green]✓[/green] Agent pipeline completed in {format_time_delta(time.time() - overall_start_time)}")
        
    except Exception as e:
        console.print(f"[red]Error in agent pipeline: {type(e).__name__}: {str(e)}[/red]")
        # Update all section stats to error
        for section_name in section_stats:
            section_stats[section_name].update({
                "status": "error",
                "fields": 0,
                "duration": time.time() - overall_start_time,
                "error": str(e)
            })
        # Return empty results
        results = {}
        for section_name, model_class in EXTRACTION_MODELS.items():
            empty_instance = model_class()
            results[section_name] = {k: None for k in empty_instance.model_fields.keys()}

    # Create section summary table
    header_columns = [
        ("Section", "cyan"),
        ("Status", None),
        ("Fields", None),
        ("Time", None)
    ]
    
    rows = []
    total_fields = 0
    
    for section_name, stats in section_stats.items():
        status_str = {
            "success": "[green]Success[/green]",
            "no_data": "[yellow]No Data[/yellow]",
            "error": "[red]Error[/red]",
            "pending": "[dim]Pending[/dim]"
        }.get(stats["status"], "[dim]Unknown[/dim]")
        
        rows.append([
            section_name,
            status_str,
            str(stats["fields"]),
            format_time_delta(stats["duration"]) if stats["duration"] else "-"
        ])
        
        total_fields += stats["fields"]
    
    section_table = create_summary_table("Agent Pipeline Extraction Summary", header_columns, rows)
    console.print(section_table)
    
    # Save results to JSON file if source file is provided
    if source_file and total_fields > 0:
        try:
            # Create json_outputs directory if it doesn't exist
            os.makedirs(config.output_dir, exist_ok=True)
            
            # Generate output filename
            base_name = os.path.basename(source_file)
            json_filename = f"{config.output_dir}/{base_name}.json"
            
            # Save results to JSON
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Saved results to {json_filename}[/green]")
                
        except Exception as e:
            console.print(f"[red]Error saving JSON output: {str(e)}[/red]")
    
    return results

@time_function
def process_file(file_path, config, llm_pipeline, messages):
    """
    Process a single file and return results.
    
    Args:
        file_path (str): Path to the file to process
        config (AppConfig): Application configuration
        llm_pipeline: LLM extraction pipeline
        messages (List): Example messages for the LLM
        
    Returns:
        tuple: (extraction_results, sheets_processed, fields_extracted)
    """
    try:
        FileName = os.path.basename(file_path)
        sheet_stats = {}
        
        # Log start of sheet preparation
        console.print(f"[bold blue]Preparing sheets for {FileName}...[/bold blue]")
        prep_start_time = time.time()
        all_sheet_markdowns = prepare_excel_sheets_markdown(file_path, config)
        prep_duration = time.time() - prep_start_time
        console.print(f"[green]✓[/green] Preparing sheets for {FileName} completed in {format_time_delta(prep_duration)}")
        
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
        
        # Process each sheet
        console.print(f"[bold]Processing {len(all_sheet_markdowns)} sheets for {FileName}...[/bold]")
        
        for sheet_idx, (sheet_name, markdown_content) in enumerate(all_sheet_markdowns.items()):
            # Show current sheet being processed
            console.print(f"[dim]Sheet {sheet_idx+1}/{len(all_sheet_markdowns)}: [cyan]{sheet_name}[/cyan][/dim]")
            
            # Record start time for this sheet
            sheet_start_time = time.time()
            
            try:
                # Process sheet
                console.print(f"  [blue]Extracting data from {sheet_name}...[/blue]")
                sheet_results = extract_all_sections(markdown_content, file_path, config, llm_pipeline, messages)
                
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
        header_columns = [
            ("Sheet", "cyan"),
            ("Status", None),
            ("Fields", None),
            ("Time", None)
        ]
        
        rows = []
        for sheet_name, stats in sheet_stats.items():
            status_str = {
                "success": "[green]Success[/green]",
                "no_data": "[yellow]No Data[/yellow]",
                "error": "[red]Error[/red]"
            }.get(stats["status"], "[dim]Unknown[/dim]")
            
            rows.append([
                sheet_name,
                status_str,
                str(stats["fields"]),
                format_time_delta(stats["duration"])
            ])
        
        sheet_table = create_summary_table(
            f"Sheet Processing Summary for {FileName}", 
            header_columns, 
            rows
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
def process_directory(config: AppConfig):
    """
    Process all Excel/CSV files in a directory.
    
    Args:
        config (AppConfig): Application configuration
    """
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize processing statistics
    stats = ProcessingStats()
    
    # Get LLM pipeline
    llm_pipeline, messages = initialize_llm_pipeline(config)
    
    # Find all Excel/CSV files
    all_files = []
    for root, _, files in os.walk(config.input_dir):
        for file in files:
            if file.endswith(('.xlsx', '.xls', '.csv')):
                all_files.append(os.path.join(root, file))

    if not all_files:
        console.print(Panel(
            f"[yellow]No Excel/CSV files found in directory: {config.input_dir}[/yellow]",
            border_style="yellow"
        ))
        return

    total_files = len(all_files)
    
    # Print process header
    console.print(f"[bold]Processing {total_files} files...[/bold]")
    
    # Process files
    for i, file_path in enumerate(all_files):
        FileName = os.path.basename(file_path)
        
        # Show current file being processed
        console.print(f"[cyan]File {i+1}/{total_files}: {FileName}[/cyan]")
        
        # Record start time for this file
        stats.record_file_start(file_path)
        
        try:
            # Process file
            console.print(f"[bold blue]Processing {FileName}...[/bold blue]")
            process_start_time = time.time()
            results, sheets_processed, fields_extracted = process_file(file_path, config, llm_pipeline, messages)
            process_duration = time.time() - process_start_time
            console.print(f"[green]✓[/green] Processing {FileName} completed in {format_time_delta(process_duration)}")
            
            if results is not None:
                # Record successful processing
                stats.record_file_end(file_path, success=True, fields_extracted=fields_extracted)
                
                # Calculate time remaining
                files_remaining = total_files - (i + 1)
                time_remaining = stats.get_estimated_time_remaining(files_remaining)
                
                console.print(Panel(
                    f"[green]Successfully processed {FileName}[/green]\n"
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
                    f"[yellow]No data extracted from {FileName}[/yellow]",
                    border_style="yellow"
                ))
        except Exception as e:
            # Record error
            stats.record_file_end(file_path, success=False)
            console.print(Panel(
                f"[red]Error processing {FileName}:[/red]\n"
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
