# Technical Context

This document details the technologies used, development setup, and technical constraints of the project.

## Technologies Used
- **Programming Language:** Python
- **Frameworks/Libraries:**
    - LLM interaction: `openai`, `langchain`, `langchain-core`, `langchain-openai`, `langfuse`
    - CLI: `typer`
    - API: `FastAPI`, `uvicorn`
    - Data Handling: `pandas`, `xlrd`
    - Markdown Conversion: `markitdown`
    - Templating: `jinja2`
    - Other: `rich` (for console output)
- **Configuration:** JSON files
- **Diagramming:** Mermaid

## Development Setup
- Requires Python environment with specified dependencies (listed in `run_pipeline.py` script header and likely `requirements.txt`).
- CLI usage: `python run_pipeline.py ...`
- API usage: `uvicorn api_main:app --reload ...`

## Technical Constraints
- Performance is dependent on LLM response times and token limits.
- Accuracy is dependent on LLM capabilities and the quality of configuration/examples.
- Handling of complex or poorly formatted Excel/CSV files may require specific configuration tuning.

## Dependencies
- Key dependencies are listed in the `run_pipeline.py` script header and `requirements.txt`. These include libraries for LLM interaction, data processing, CLI/API interfaces, and formatting.
