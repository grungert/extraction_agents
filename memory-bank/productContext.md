# Product Context

This document describes the purpose and user experience goals of the project.

## Purpose
- The project exists to automate the extraction of structured data from financial documents provided in Excel or CSV format.
- It solves the problem of manually parsing these documents, which is time-consuming and error-prone.

## How it Works
- The pipeline takes an Excel or CSV file as input.
- The file is converted to a markdown representation.
- LLM agents perform header detection, extraction, and validation based on a configurable JSON file.
- The process can be initiated via a command-line interface or an HTTP API.
- The output is structured JSON data.

## User Experience Goals
- Users should be able to easily configure the extraction process for different document types.
- The tool should provide accurate and reliable data extraction.
- Both technical users (CLI) and developers integrating the functionality (API) should have a straightforward experience.
