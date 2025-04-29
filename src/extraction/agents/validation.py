"""Dynamic LLM-based validation agent."""
import json
import logging
from typing import Dict, List, Optional, Type, Any, Union

from ...utils.display import console, logger
from ...models import (
    AppConfig,
    BaseExtraction,
    ContextModel,
    ValidationResult,
)
from ...exceptions import (
    ConfigurationError,
    LLMInteractionError,
    ValidationError,
)
from ..llm import extract_section
from ...config_manager import ConfigurationManager
from ...examples.dynamic_examples import format_validation_messages

# Import helpers from agent_utils
from ..agent_utils import get_tokenizer, count_tokens


class DynamicValidationAgent:
    """Agent for validating and correcting extracted data using dynamic configuration."""

    def __init__(self, llm, app_config: Optional[AppConfig] = None):
        """
        Initialize the validation agent.

        Args:
            llm: LLM instance to use for validation
            app_config: Application configuration instance
        """
        self.llm = llm
        self.config_manager: Optional[ConfigurationManager] = None
        # Store app_config if provided, otherwise load default AppConfig for standalone use
        self.app_config = app_config or AppConfig()
        self.section_messages = {}  # Initialize as empty, loaded at runtime

    def validate(self, extracted_data: BaseExtraction, markdown_content: str, header_info: ContextModel,
                section_name: str, model_class: Type[BaseExtraction]) -> Optional[Any]:
        """
        Validate extracted data for a specific section against the original table.

        Args:
            extracted_data: Data to validate
            markdown_content: Original markdown table content
            header_info: Header detection information
            section_name: Name of the section
            model_class: Model class for this section

        Returns:
            Validated data with confidence score, or None if validation failed
        """
        logger.info(f"Validating {section_name} data...")

        # Ensure config_manager is set
        if self.config_manager is None:
             logger.error(f"Error: Config manager not set for Validation Agent ({section_name}).")
             raise ConfigurationError(f"Config manager not set for Validation Agent ({section_name}).")


        try:
            # Focus on relevant rows based on header detection (same as extraction)
            focused_content = self._focus_content(markdown_content, header_info)

            # Convert extracted data to JSON string
            extracted_json = json.dumps(extracted_data.model_dump())

            # Create a combined input with both the table and extracted data
            # Format the input as a markdown string with the table and extracted data
            combined_input = f"""
# Original Table
{focused_content}

# Extracted Data
```json
{extracted_json}
```
"""

            # Use the combined input directly as markdown
            json_input = combined_input

            # Create validation model class that extends the original
            validation_model = self._create_validation_model(model_class)

            # Get or create example messages for this section, using dynamic config_manager
            if section_name not in self.section_messages or not self.section_messages[section_name]:
                self.section_messages[section_name] = self._create_section_validation_messages(section_name, self.app_config)
                if not self.section_messages[section_name]:
                     logger.error(f"Error: Failed to load messages for {section_name} validation.")
                     raise ConfigurationError(f"Failed to load messages for {section_name} validation.")


            # Validate data using section-specific messages
            result = extract_section(
                markdown_content=json_input,
                section_name=f"{section_name}Validation",
                model_class=validation_model,
                messages=self.section_messages[section_name],
                llm=self.llm
            )

            if result and hasattr(result, 'ValidationConfidence'):
                logger.info(f"Validation successful with confidence {result.ValidationConfidence:.2f})")
                return result
            else:
                logger.warning(f"{section_name} validation returned incomplete data")
                return None

        except LLMInteractionError:
            raise
        except Exception as e:
            logger.exception(f"Error in {section_name} validation: {str(e)}")
            raise ValidationError(f"Error in {section_name} validation: {str(e)}")

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

    def _create_validation_model(self, model_class: Type[BaseExtraction]) -> Type:
        """
        Create a validation model that extends the original model.

        Args:
            model_class: Original model class

        Returns:
            New model class with validation fields
        """
        # This is a simplified implementation
        # In a real implementation, you would dynamically create a new model class
        # that combines the original model with validation fields

        class CombinedValidationModel(ValidationResult):
            ValidatedData: model_class

        return CombinedValidationModel

    def _create_section_validation_messages(self, section_name: str, app_config: AppConfig) -> List[Dict]:
        """
        Create example messages for validation of a specific section, including conditional examples in the system prompt.

        Args:
            section_name: Name of the section
            app_config: Application configuration instance

        Returns:
            List of message dictionaries for the LLM
        """
        # This method relies on self.config_manager being set
        if not self.config_manager:
             logger.warning(f"Cannot create section validation messages for {section_name} without config manager.")
             raise ConfigurationError(f"Cannot create section validation messages for {section_name} without config manager.")


        # Load system message template from prompt file, passing necessary context
        from ...utils.prompt_utils import load_template_prompt
        # Use dot notation for prompt file name
        template_filename = self.config_manager.get_prompts_config().section_validation_template_file
        system_content = load_template_prompt(
            template_filename,
            section_name=section_name,
            config_manager=self.config_manager,
            app_config=app_config
        )
        system_message = {"role": "system", "content": system_content}

        # Format examples for validation using the dynamic examples manager
        # Pass the section_name to get only examples for this specific section
        return format_validation_messages(self.config_manager, system_message, section_name)
