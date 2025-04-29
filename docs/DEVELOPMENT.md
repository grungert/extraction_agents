# Development Guide

This guide provides instructions for setting up the development environment, running tests, and contributing to the Excel Extraction Pipeline project.

## 1. Environment Setup

### Prerequisites

-   Python 3.10 or higher
-   `pip` (Python package installer)
-   `git` (Version control system)
-   (Optional) `uv` (Faster virtual environment and package manager)
-   (Optional) Docker and Docker Compose (for containerized development/deployment)

### Setup Steps

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd agent_extraction
    ```

2.  **Create a Virtual Environment:**
    *   Using `venv` (standard):
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```
    *   Using `uv` (recommended):
        ```bash
        uv venv
        source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
        ```

3.  **Install Dependencies:**
    *   Using `pip`:
        ```bash
        pip install -r requirements.txt
        ```
    *   Using `uv`:
        ```bash
        uv pip install -r requirements.txt
        ```

4.  **Set Up Environment Variables:**
    -   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    -   Edit the `.env` file and add your API keys (e.g., `OPENAI_API_KEY`) and configure other settings like Langfuse integration if needed.

5.  **Verify Setup:**
    -   Run the pipeline with an example file:
        ```bash
        python run_pipeline.py --file input/your_test_file.xlsx
        ```
    -   Start the API server:
        ```bash
        uvicorn api_main:app --reload
        ```
        Access the API docs at `http://localhost:8000/docs`.

## 2. Running Tests

*(Test framework setup (e.g., pytest) and instructions for running tests will be added here once implemented.)*

-   **Running all tests:**
    ```bash
    # pytest
    ```
-   **Running specific tests:**
    ```bash
    # pytest path/to/test_file.py::test_function
    ```
-   **Viewing test coverage:**
    ```bash
    # pytest --cov=src
    ```

## 3. Code Style and Linting

*(Details about code style guidelines (e.g., PEP 8, Black, isort) and linting tools (e.g., Flake8, Ruff) will be added here.)*

-   **Formatting:**
    ```bash
    # black .
    # isort .
    ```
-   **Linting:**
    ```bash
    # flake8 src
    # ruff check .
    ```

## 4. Contribution Workflow

1.  Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name` or `bugfix/issue-description`.
2.  Make your changes, following code style guidelines.
3.  Add relevant tests for your changes.
4.  Ensure all tests pass.
5.  Format and lint your code.
6.  Commit your changes with clear commit messages.
7.  Push your branch to the remote repository.
8.  Create a Pull Request (PR) against the main development branch.
9.  Address any feedback or requested changes during the code review process.

## 5. Adding New Document Types

Refer to the `docs/CONFIGURATION.md` guide for detailed steps on adding support for new document types. This typically involves:
-   Updating `config/app_config.json`.
-   Creating a new `config/full_config_*.json` file.
-   Defining new prompt templates in `src/prompts/`.
-   Adding examples to the configuration.
-   Testing thoroughly.

## 6. Key Modules Overview

-   `api_main.py`: FastAPI entry point.
-   `run_pipeline.py`: CLI script for running the pipeline locally.
-   `src/extraction/pipeline.py`: Orchestrates the agent workflow.
-   `src/extraction/agents/`: Contains individual processing agents.
-   `src/config_manager.py`: Loads and manages configurations.
-   `src/dynamic_model_factory.py`: Creates Pydantic models from config.
-   `src/models.py`: Core data structures (Pydantic models).
-   `src/exceptions.py`: Custom exception classes.
-   `src/utils/`: Utility modules.
