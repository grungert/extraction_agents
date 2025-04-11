import os
import json
from src.models import AppConfig
from src.config_manager import ConfigurationManager
from src.extraction.dynamic_agents import DynamicAgentPipelineCoordinator
from src.extraction.excel import excel_to_markdown

def run_extraction(file_path: str, config_dict: dict, sheet_name: str = None) -> dict:
    """
    Run the extraction pipeline on a single file with a given config dict.

    Args:
        file_path (str): Path to the Excel or CSV file.
        config_dict (dict): Extraction configuration (parsed from JSON).
        sheet_name (str, optional): Sheet name to process. Defaults to None.

    Returns:
        dict: Extraction result.
    """
    # Save config dict to a temporary JSON file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp_config_file:
        json.dump(config_dict, tmp_config_file, indent=2)
        tmp_config_path = tmp_config_file.name

    try:
        # Load configuration manager from temp file
        config_manager = ConfigurationManager(tmp_config_path)

        # Create app config
        app_config = AppConfig(config_path=tmp_config_path)

        # Initialize pipeline
        pipeline = DynamicAgentPipelineCoordinator(app_config, config_manager)

        # Convert Excel to markdown
        markdown_content = excel_to_markdown(file_path, app_config, sheet_name)

        # Run extraction
        result = pipeline.process_markdown(markdown_content, file_path)

        return result

    finally:
        # Clean up temp config file
        try:
            os.remove(tmp_config_path)
        except Exception:
            pass
