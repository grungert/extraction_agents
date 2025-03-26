"""Display utilities for Excel Header Mapper."""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Initialize rich console
console = Console()

def create_summary_table(title, header_columns, rows, title_style="bold"):
    """
    Create a summary table with the given data.
    
    Args:
        title (str): The table title
        header_columns (list): List of (name, style) tuples for columns where style can be a color or None
        rows (list): List of row data
        title_style (str): Style for the title
        
    Returns:
        Table: A rich Table object
    """
    table = Table(title=f"[{title_style}]{title}[/{title_style}]", 
                 show_header=True, 
                 header_style="bold magenta")
    
    # Add columns
    for name, style in header_columns:
        # Check if style might be a justification direction
        if style in ["right", "center", "left"]:
            table.add_column(name, justify=style)
        else:
            table.add_column(name, style=style)
    
    # Add rows
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    
    return table

def create_progress_panel(message, title=None, subtitle=None, style="green"):
    """Create a panel with progress information."""
    return Panel(
        message,
        title=title,
        subtitle=subtitle,
        border_style=style
    )
