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
#   "jinja2>=3.0.0",
# ]
# ///
"""LLM-based extraction pipeline for parsing Excel files."""
import json
import os
import sys
from pydantic import BaseModel
import argparse

from src.models import AppConfig
from src.config_manager import get_configuration_manager
from src.extraction.dynamic_agents import DynamicAgentPipelineCoordinator
from src.utils.display import console
from src.extraction.excel import excel_to_markdown


def main():
    """Run the dynamic agent pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM-based extraction pipeline for Excel files")
    # Config argument is no longer needed as it's determined dynamically
    # parser.add_argument("--config", type=str, default=None, 
    #                     help="Path to base configuration file (optional, specific config loaded based on classification)")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to specific Excel file to process (if not provided, processes all Excel files in input_dir)")
    parser.add_argument("--sheet", type=str, default=None,
                        help="Sheet name to process (if not provided, uses the first sheet)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output JSON file (if not provided, prints to console)")
    
    args = parser.parse_args()
    
    # Load base AppConfig settings from models.py defaults
    console.print("[blue]Loading base application configuration...[/blue]")
    app_config = AppConfig() 
    # Note: Specific extraction config (like full_config_Mutual_Funds.json) 
    # is now loaded inside the pipeline coordinator based on classification.
    
    # Initialize the agent pipeline
    console.print("[blue]Initializing dynamic agent pipeline...[/blue]")
    pipeline = DynamicAgentPipelineCoordinator(app_config)
    
    # Determine files to process
    if args.file:
        # Process a specific Excel file
        console.print(f"[blue]Processing Excel file: {args.file}[/blue]")
        # Check if the file exists
        if not os.path.exists(args.file):
            console.print(f"[red]Error: File not found: {args.file}[/red]")
            return
        
        # Process single file
        files_to_process = [args.file]
    else:
        # Process all Excel files in the input directory
        input_dir = app_config.input_dir
        console.print(f"[blue]Processing all Excel files in directory: {input_dir}[/blue]")
        
        # Check if directory exists
        if not os.path.exists(input_dir):
            console.print(f"[red]Error: Input directory not found: {input_dir}[/red]")
            return
        
        # Get all Excel files in the directory
        files_to_process = []
        for filename in os.listdir(input_dir):
            if filename.endswith(('.xlsx', '.xls', '.csv', '.XLS')):
                files_to_process.append(os.path.join(input_dir, filename))
        
        if not files_to_process:
            console.print(f"[yellow]No Excel files found in {input_dir}[/yellow]")
            return
    
    # Process each file
    for file_path in files_to_process:
        source_file = file_path
        console.print(f"[blue]Processing file: {source_file}[/blue]")
        
        # Convert Excel file to markdown
        markdown_content = excel_to_markdown(source_file, app_config, args.sheet)
        
        # Process the markdown content
        console.print("[blue]Processing markdown content...[/blue]")
        results = pipeline.process_markdown(markdown_content, source_file)
        
        # Output the results
        if args.output and len(files_to_process) == 1:
            # Use the specified output path (only for single file mode)
            output_path = args.output
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            console.print(f"[blue]Writing results to specified location: {output_path}[/blue]")
        else:
            # Create output directory if it doesn't exist
            os.makedirs(app_config.output_dir, exist_ok=True)
            
            # Generate a default filename based on the source file
            output_filename = f"{os.path.basename(source_file)}.json"
            
            # Combine with the output directory
            output_path = os.path.join(app_config.output_dir, output_filename)
            console.print(f"[blue]Writing results to default location: {output_path}[/blue]")
        
        # Write the results to file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Results written to {output_path}[/green]")
    
    console.print("[green]Processing completed successfully![/green]")


if __name__ == "__main__":
    main()
