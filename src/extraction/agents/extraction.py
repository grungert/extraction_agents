"""Dynamic LLM-based extraction agent."""
import json
import logging
from typing import Dict, List, Optional, Type, Any, Union

from ...utils.display import console, logger
from ...models import (
    AppConfig,
    BaseExtraction,
    ContextModel,
    ExtractionConfig,
)
from ...exceptions import (
    ConfigurationError,
    LLMInteractionError,
    ExtractionError,
)
from ..llm import extract_section
from ...config_manager import ConfigurationManager
from ...examples.dynamic_examples import format_extraction_messages

# Import helpers from agent_utils
from ..agent_utils import get_tokenizer, count_tokens


class DynamicExtractionAgent:
    """Agent for extracting structured data from Excel sheets using dynamic configuration."""

    def __init__(self, llm, app_config: Optional[AppConfig] = None):
        """
        Initialize the extraction agent.

        Args:
            llm: LLM instance to use for extraction
            app_config: Application configuration instance
        """
        self.llm = llm
        self.config_manager: Optional[ConfigurationManager] = None
        # Store app_config if provided, otherwise load default AppConfig for standalone use
        self.app_config = app_config or AppConfig()
        self.section_messages = {} # Initialize as empty, loaded at runtime

    def extract_data(self, markdown_content: str, header_info: ContextModel,
                    section_name: str, model_class: Type[BaseExtraction]) -> Optional[BaseExtraction]:
        """
        Extract data for a specific section.

        Args:
            markdown_content: Markdown content to extract from
            header_info: Header detection information
            section_name: Name of the section to extract
            model_class: Model class for this section

        Returns:
            Extracted data or None if extraction failed
        """
        logger.info(f"Extracting {section_name} data...")

        # Ensure config_manager is set
        if self.config_manager is None:
             logger.error(f"Error: Config manager not set for Extraction Agent ({section_name}).")
             raise ConfigurationError(f"Config manager not set for Extraction Agent ({section_name}).")


        try:
            # Focus on relevant rows based on header detection
            focused_content = self._focus_content(markdown_content, header_info)

            # Display the input for the extraction agent
            logger.debug(f"Extraction Agent Input for {section_name}:\n" + focused_content[:500] + "..." if len(focused_content) > 500 else focused_content)

            # Get or create example messages for this section, using the dynamically set config_manager
            if section_name not in self.section_messages or not self.section_messages[section_name]:
                self.section_messages[section_name] = self._create_section_messages(section_name, self.app_config)
                if not self.section_messages[section_name]:
                     logger.error(f"Error: Failed to load messages for {section_name} extraction.")
                     raise ConfigurationError(f"Failed to load messages for {section_name} extraction.")


            # Extract data
            result = extract_section(
                markdown_content=focused_content,
                section_name=section_name,
                model_class=model_class,
                messages=self.section_messages[section_name],
                llm=self.llm
            )

            if result:
                logger.info(f"{section_name} extraction successful")
                return result
            else:
                logger.warning(f"{section_name} extraction returned no data")
                return None

        except LLMInteractionError:
            raise
        except Exception as e:
            logger.exception(f"Error in {section_name} extraction: {str(e)}")
            raise ExtractionError(f"Error in {section_name} extraction: {str(e)}")

    def _focus_content(self, markdown_content: str, header_info: ContextModel) -> str:
        """
        Focus on relevant rows based on header detection.

        Args:
            markdown_content: Full markdown content
            header_info: Header detection information

        Returns:
            Focused markdown content with headers and some content rows
        """
        lines = markdown_content.strip().split('\n')

        # Get header lines
        header_start = header_info.HeaderStartLine or 0
        header_end = header_info.HeaderEndLine or 0
        content_start = header_info.ContentStartLine or (header_end + 1)

        # Get header lines and up to 10 content rows
        content_end = min(content_start + 10, len(lines))
        focused_lines = lines[header_start:content_end]

        return '\n'.join(focused_lines)

    def _create_section_messages(self, section_name: str, app_config: AppConfig) -> List[Dict]:
        """
        Create example messages for a specific section, including conditional examples in the system prompt.

        Args:
            section_name: Name of the section
            app_config: Application configuration instance

        Returns:
            List of message dictionaries for the LLM
        """
        # This method relies on self.config_manager being set
        if not self.config_manager:
             logger.warning(f"Cannot create section messages for {section_name} without config manager.")
             raise ConfigurationError(f"Cannot create section messages for {section_name} without config manager.")


        # Load system message template from prompt file, passing necessary context
        from ...utils.prompt_utils import load_template_prompt
        # Use dot notation for prompt file name
        template_filename = self.config_manager.get_prompts_config().section_extraction_template_file
        system_content = load_template_prompt(
            template_filename,
            section_name=section_name,
            config_manager=self.config_manager,
            app_config=app_config
        )
        system_message = {"role": "system", "content": system_content}

        # Format examples for extraction using the dynamic examples manager
        return format_extraction_messages(self.config_manager, section_name, system_message)
