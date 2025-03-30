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
#   "langfuse>=0.1.0"
# ]
# ///

import typer
import os
from datetime import datetime
from rich.panel import Panel

from src.config import get_app_config
from src.processing import process_directory
from src.utils.display import console
from src.utils.formatting import format_time_delta

app = typer.Typer()

@app.command()
def main(
    input_dir: str = typer.Option("testing", help="Directory containing Excel/CSV files"),
    output_dir: str = typer.Option("json_outputs", help="Directory for JSON output"),
    start_row: int = typer.Option(0, help="First row to read (0-based)"),
    end_row: int = typer.Option(15, help="Last row to read (exclusive)"),
    all_sheets: bool = typer.Option(False, help="Process all sheets in Excel files"),
    model_name: str = typer.Option("gemma-3-12b-it", help="LLM model name"),
    base_url: str = typer.Option("http://localhost:1234/v1", help="LLM API base URL"),
    api_key: str = typer.Option("null", help="API key for the LLM"),
    temperature: float = typer.Option(0.3, help="Temperature for LLM generation"),
    max_retries: int = typer.Option(2, help="Maximum retries for LLM API calls")
):
    """
    Process Excel/CSV files to extract structured data using LLM.
    """
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
        
        # Create configuration
        config = get_app_config(
            input_dir=input_dir,
            output_dir=output_dir,
            start_row=start_row,
            end_row=end_row,
            all_sheets=all_sheets,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries
        )
        
        # Print system info
        console.print(Panel(
            f"[bold]System Configuration:[/bold]\n"
            f"• Model: [cyan]{config.model.model_name}[/cyan]\n"
            f"• Temperature: [cyan]{config.model.temperature}[/cyan]\n"
            f"• API Base: [cyan]{config.model.base_url}[/cyan]\n"
            f"• Input Directory: [cyan]{config.input_dir}[/cyan]\n"
            f"• Output Directory: [cyan]{config.output_dir}[/cyan]\n"
            f"• Row Range: [cyan]{config.start_row}-{config.end_row}[/cyan]",
            title="System Info",
            border_style="blue",
            expand=False
        ))
        
        if os.path.isdir(config.input_dir):
            # Process directory
            process_directory(config)
            
            # Calculate total execution time
            total_time = time.time() - start_time
            
            # Print execution time summary
            console.print(Panel(
                f"[bold]Total Execution Time:[/bold] [cyan]{format_time_delta(total_time)}[/cyan]",
                border_style="blue"
            ))
        else:
            console.print(Panel(
                f"[red]Directory not found: {config.input_dir}[/red]",
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
    import time
    app()
