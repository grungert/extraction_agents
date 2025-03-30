#!/usr/bin/env python3
"""
Test script for the LLM-based agent pipeline.
This script demonstrates the agent pipeline in action by processing a sample Excel file.
"""
import os
import sys
from rich.panel import Panel

from src.config import get_app_config
from src.extraction.excel import prepare_excel_sheets_markdown
from src.extraction.agents import AgentPipelineCoordinator
from src.utils.display import console
from src.utils.formatting import format_time_delta
import time

def main():
    """
    Main function to test the agent pipeline.
    """
    # Print header
    console.print(Panel.fit(
        "[bold green]Excel Header Mapper - Agent Pipeline Test[/bold green]",
        subtitle="[dim]Testing the LLM-based agent pipeline[/dim]",
        border_style="green",
        padding=(1, 2)
    ))
    
    # Check if a file path was provided
    if len(sys.argv) < 2:
        console.print("[yellow]Please provide a path to an Excel or CSV file.[/yellow]")
        console.print("Usage: python test_agent_pipeline.py <file_path>")
        return
    
    file_path = sys.argv[1]
    
    # Check if the file exists
    if not os.path.exists(file_path):
        console.print(f"[red]File not found: {file_path}[/red]")
        return
    
    # Check if the file is an Excel or CSV file
    if not file_path.endswith(('.xlsx', '.xls', '.csv')):
        console.print(f"[red]Unsupported file format: {file_path}[/red]")
        console.print("Please provide an Excel (.xlsx, .xls) or CSV (.csv) file.")
        return
    
    # Create configuration
    config = get_app_config(
        input_dir=os.path.dirname(file_path),
        output_dir="json_outputs",
        start_row=0,
        end_row=15,
        all_sheets=False,
        model_name="gemma-3-12b-it",
        base_url="http://localhost:1234/v1",
        api_key="null",
        temperature=0.3,
        max_retries=2
    )
    
    # Print configuration
    console.print(Panel(
        f"[bold]Configuration:[/bold]\n"
        f"• File: [cyan]{os.path.basename(file_path)}[/cyan]\n"
        f"• Model: [cyan]{config.model.model_name}[/cyan]\n"
        f"• Temperature: [cyan]{config.model.temperature}[/cyan]\n"
        f"• Row Range: [cyan]{config.start_row}-{config.end_row}[/cyan]",
        title="Test Setup",
        border_style="blue",
        expand=False
    ))
    
    try:
        # Convert Excel/CSV to markdown
        console.print("[bold blue]Converting file to markdown...[/bold blue]")
        start_time = time.time()
        
        # For simplicity, we'll just process the first sheet
        sheet_markdowns = prepare_excel_sheets_markdown(file_path, config)
        
        if not sheet_markdowns:
            console.print("[red]No sheets found in the file.[/red]")
            return
        
        # Get the first sheet
        sheet_name, markdown_content = next(iter(sheet_markdowns.items()))
        
        console.print(f"[green]✓[/green] Converted sheet '{sheet_name}' to markdown in {format_time_delta(time.time() - start_time)}")
        
        # Initialize the agent pipeline
        console.print("[bold blue]Initializing agent pipeline...[/bold blue]")
        agent_pipeline = AgentPipelineCoordinator(config)
        
        # Process the markdown content
        console.print("[bold blue]Processing with agent pipeline...[/bold blue]")
        pipeline_start_time = time.time()
        
        results = agent_pipeline.process_markdown(markdown_content, file_path)
        
        pipeline_duration = time.time() - pipeline_start_time
        console.print(f"[green]✓[/green] Agent pipeline completed in {format_time_delta(pipeline_duration)}")
        
        # Print results
        console.print("[bold]Extraction Results:[/bold]")
        for section_name, section_data in results.items():
            # Count fields with values
            fields_with_values = len([v for v in section_data.values() if v is not None])
            
            if fields_with_values > 0:
                console.print(f"[green]✓[/green] Section [cyan]{section_name}[/cyan]: {fields_with_values} fields extracted")
                
                # Print extracted fields
                for field, value in section_data.items():
                    if value is not None:
                        console.print(f"  • {field}: [cyan]{value}[/cyan]")
            else:
                console.print(f"[yellow]⚠[/yellow] Section [cyan]{section_name}[/cyan]: No data extracted")
        
        # Save results to JSON
        output_dir = "json_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        import json
        json_filename = f"{output_dir}/{os.path.basename(file_path)}_agent_test.json"
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓[/green] Results saved to {json_filename}")
        
        # Print summary
        total_duration = time.time() - start_time
        console.print(Panel(
            f"[bold]Test Summary:[/bold]\n"
            f"• File: [cyan]{os.path.basename(file_path)}[/cyan]\n"
            f"• Sheet: [cyan]{sheet_name}[/cyan]\n"
            f"• Total fields extracted: [cyan]{sum(len([v for v in section.values() if v is not None]) for section in results.values())}[/cyan]\n"
            f"• Total processing time: [cyan]{format_time_delta(total_duration)}[/cyan]",
            title="[bold]Results[/bold]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {type(e).__name__}: {str(e)}[/red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    main()
