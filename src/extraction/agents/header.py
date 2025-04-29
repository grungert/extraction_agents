"""Dynamic LLM-based header detection and validation agents."""
import json
import re
import logging
from typing import Dict, List, Optional, Type, Any, Union

from ...utils.display import console, logger
from ...models import (
    AppConfig,
    ContextModel,
    ValidationResult,
    HeaderDetectionConfig,
    ValidationConfig,
    PromptsConfig
)
from ...exceptions import (
    PipelineException,
    ConfigurationError,
    FileProcessingError,
    LLMInteractionError,
    ExtractionError,
    ValidationError,
    DeduplicationError
)
from ..llm import extract_section
from ...config_manager import ConfigurationManager
from ...utils.prompt_utils import load_prompt

# Import helpers from agent_utils
from ..agent_utils import get_tokenizer, count_tokens


class DynamicHeaderValidationAgent:
    """Agent for validating header detection results using dynamic configuration."""

    def __init__(self, llm):
        """
        Initialize the header validation agent.

        Args:
            llm: LLM instance to use for validation
        """
        self.llm = llm
        self.config_manager: Optional[ConfigurationManager] = None
        self.messages = []

    def validate(self, header_info: ContextModel, markdown_content: str) -> Optional[ContextModel]:
        # Ensure config_manager is set before creating messages
        if self.config_manager is None:
             # Enhanced Exception
             raise ConfigurationError(
                 message="Configuration manager not set for Header Validation Agent.",
                 error_code="CONFIG_MANAGER_NOT_SET",
                 context={"agent": "DynamicHeaderValidationAgent"}
             )

        # Load messages here, using the dynamically set config_manager
        if not self.messages:
            try:
                self.messages = self._create_validation_messages()
            except ConfigurationError: # Catch specific error from _create_validation_messages
                raise # Re-raise if it's already a ConfigurationError
            except Exception as e:
                # Enhanced Exception for unexpected errors during message creation
                raise ConfigurationError(
                    message=f"Failed to create validation messages for Header Validation: {e}",
                    error_code="HEADER_VALIDATION_MESSAGE_CREATION_FAILED",
                    context={"agent": "DynamicHeaderValidationAgent", "exception_type": e.__class__.__name__}
                )
            if not self.messages:
                 # This case might be redundant if _create_validation_messages raises, but keep for safety
                 raise ConfigurationError(
                     message="Failed to load validation messages for Header Validation (empty result).",
                     error_code="HEADER_VALIDATION_MESSAGES_EMPTY",
                     context={"agent": "DynamicHeaderValidationAgent"}
                 )

        """
        Validate header detection results against the original table.

        Args:
            header_info: Header detection information to validate
            markdown_content: Original markdown table content

        Returns:
            Validated header information with confidence score, or None if validation failed
        """
        logger.info("Validating header detection...")

        try:
            # Convert header info to JSON string
            header_json = json.dumps(header_info.model_dump())

            # Create a combined input with both the table and header info
            # Format the input as a markdown string
            combined_input = f"""
# Original Table
{markdown_content}

# Detected Headers
```json
{header_json}
```
"""
            # Validate header detection
            result = extract_section(
                markdown_content=combined_input,
                section_name="HeaderValidation",
                model_class=ContextModel,
                messages=self.messages,
                llm=self.llm
            )

            if result and hasattr(result, 'ValidationConfidence'):
                logger.info(f"Header validation successful with confidence {result.ValidationConfidence:.2f}")
                return result
            else:
                logger.warning("Header validation returned incomplete data")
                return None

        except LLMInteractionError:
            raise # Re-raise LLMInteractionError
        except Exception as e:
            logger.exception(f"Error during header validation: {str(e)}")
            # Enhanced Exception (wrapping original)
            raise ValidationError(
                message=f"Error during header validation: {e}",
                error_code="HEADER_VALIDATION_RUNTIME_ERROR",
                context={"exception_type": e.__class__.__name__}
            )

    def _create_validation_messages(self) -> List[Dict]:
        """
        Create example messages for header validation.

        Returns:
            List of message dictionaries for the LLM
        """
        # This method now relies on self.config_manager being set before it's called
        if not self.config_manager:
             # Enhanced Exception
             raise ConfigurationError(
                 message="Configuration manager not set when creating header validation messages.",
                 error_code="CONFIG_MANAGER_NOT_SET",
                 context={"method": "_create_validation_messages"}
             )

        # Get header validation configuration using dot notation
        validation_config: ValidationConfig = self.config_manager.get_validation_config();

        # Load system message from prompt file using config manager
        # from ...utils.prompt_utils import load_prompt # Already imported above
        # Use dot notation for prompt file name
        prompt_filename = self.config_manager.get_prompts_config().header_validation_system_prompt_file
        system_message = {
            "role": "system",
            "content": load_prompt(prompt_filename)
        }

        # Load header examples from configuration manager
        header_examples = self.config_manager.get_header_examples();

        # Create examples for the LLM - format them for validation
        examples = []
        for idx, example in enumerate(header_examples[:2], start=1):
            validation_input = f"""
# Example {idx}
# Input Table
{example["table"]}
"""

            examples.extend([
                {"role": "user", "content": validation_input},
                {"role": "assistant", "content": json.dumps(example["json"])}
            ])

        # Combine system message and examples
        return [system_message] + examples


class DynamicHeaderDetectionAgent:
    """Agent for detecting header positions in Excel sheets using dynamic configuration."""

    def __init__(self, llm):
        """
        Initialize the header detection agent.

        Args:
            llm: LLM instance to use for detection
        """
        self.llm = llm
        self.config_manager: Optional[ConfigurationManager] = None
        self.messages = []

    def detect_headers(self, markdown_content: str) -> Optional[ContextModel]:
        # Ensure config_manager is set before creating messages
        if self.config_manager is None:
             # Enhanced Exception
             raise ConfigurationError(
                 message="Configuration manager not set for Header Detection Agent.",
                 error_code="CONFIG_MANAGER_NOT_SET",
                 context={"agent": "DynamicHeaderDetectionAgent"}
             )

        # Load messages here, using the dynamically set config_manager
        # Check if messages need reloading (if config_manager changed or messages are empty)
        if not self.messages:
            try:
                self.messages = self._create_example_messages()
            except ConfigurationError: # Catch specific error from _create_example_messages
                raise # Re-raise if it's already a ConfigurationError
            except Exception as e:
                 # Enhanced Exception for unexpected errors during message creation
                 raise ConfigurationError(
                     message=f"Failed to create example messages for Header Detection: {e}",
                     error_code="HEADER_DETECTION_MESSAGE_CREATION_FAILED",
                     context={"agent": "DynamicHeaderDetectionAgent", "exception_type": e.__class__.__name__}
                 )
            if not self.messages:
                 # This case might be redundant if _create_example_messages raises, but keep for safety
                 raise ConfigurationError(
                     message="Failed to load example messages for Header Detection (empty result).",
                     error_code="HEADER_DETECTION_MESSAGES_EMPTY",
                     context={"agent": "DynamicHeaderDetectionAgent"}
                 )

        """
        Detect header positions in markdown content.

        Args:
            markdown_content: Markdown content to analyze

        Returns:
            ContextModel with header positions, or None if detection failed
        """
        logger.info("Running header detection agent...")

        # Display the input for the header agent
        logger.debug("Header Agent Input:\n" + markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content)

        try:
            # Use a dictionary to capture the raw response including header_detection_confidence
            raw_result = None

            # --- NEW: Token Counting for Input ---
            # Need to format the full prompt including system message and examples to count input tokens accurately
            # Extract the header information
            result = extract_section(
                markdown_content=markdown_content,
                section_name="HeaderDetection",
                model_class=ContextModel,
                messages=self.messages,
                llm=self.llm
            )

            if result and hasattr(result, 'ValidationConfidence'):
                # Store ValidationConfidence as an attribute for internal use
                if hasattr(result, 'ValidationConfidence'):
                    # Store the confidence value as an attribute for internal use
                    result._header_detection_confidence = result.ValidationConfidence
                    logger.info(f"Header detection successful with confidence {result.ValidationConfidence:.2f}")
                else:
                    logger.info("Header detection successful (no confidence score)")

                return result
            else:
                logger.warning("Header detection returned incomplete data")
                return None

        except LLMInteractionError:
            raise # Re-raise LLMInteractionError
        except Exception as e:
            logger.exception(f"Error during header detection: {str(e)}")
            # Enhanced Exception (wrapping original)
            raise ExtractionError(
                message=f"Error during header detection: {e}",
                error_code="HEADER_DETECTION_RUNTIME_ERROR",
                context={"exception_type": e.__class__.__name__}
            )

    def _create_example_messages(self) -> List[Dict]:
        """
        Create example messages for few-shot learning.

        Returns:
            List of message dictionaries for the LLM
        """
        # This method now relies on self.config_manager being set before it's called
        if not self.config_manager:
             # Enhanced Exception
             raise ConfigurationError(
                 message="Configuration manager not set when creating header detection messages.",
                 error_code="CONFIG_MANAGER_NOT_SET",
                 context={"method": "_create_example_messages"}
             )

        # Get header detection configuration using dot notation
        header_config: HeaderDetectionConfig = self.config_manager.get_header_detection_config();

        # Load system message from prompt file using config manager
        # from ...utils.prompt_utils import load_prompt # Already imported above
        # Use dot notation for prompt file name
        prompt_filename = self.config_manager.get_prompts_config().header_detection_system_prompt_file
        system_message = {
            "role": "system",
            "content": load_prompt(prompt_filename)
        }

        # Load header examples from configuration manager
        header_examples = self.config_manager.get_header_examples();

        # Create examples for the LLM - format them for validation
        examples = []
        for idx, example in enumerate(header_examples[:2], start=1):
            validation_input = f"""
# Example {idx}
# Input Table
{example["table"]}
"""

            examples.extend([
                {"role": "user", "content": validation_input},
                {"role": "assistant", "content": json.dumps(example["json"])}
            ])

        # Combine system message and examples
        return [system_message] + examples

    def fallback_header_detection(self, markdown_content: str) -> ContextModel:
        """
        Fallback strategy for header detection when LLM-based detection fails or has low confidence.

        Args:
            markdown_content: Markdown content to analyze

        Returns:
            ContextModel with header positions
        """
        logger.warning("Using fallback header detection strategy...")

        lines = markdown_content.strip().split('\n')

        # Default values
        HeaderStartLine = 0
        HeaderEndLine = 0
        ContentStartLine = 1

        # Look for separator row (contains mainly dashes or pipes)
        for i, line in enumerate(lines):
            if i > 0 and re.match(r'^[\|\-\+\s]+$', line.strip()):
                # Found separator row, headers are above it
                HeaderEndLine = i - 1
                ContentStartLine = i + 1
                break

        # Look for first row with numeric data (likely content)
        for i, line in enumerate(lines):
            if i > HeaderEndLine:
                # Check if line contains numeric data
                if re.search(r'\d+\.\d+|\d+', line):
                    ContentStartLine = i
                    break

        logger.warning(f"Fallback detection: headers at lines {HeaderStartLine}-{HeaderEndLine}, content starts at line {ContentStartLine}")

        # Create ContextModel without header_detection_confidence
        return ContextModel(
            HeaderStartLine=HeaderStartLine,
            HeaderEndLine=HeaderEndLine,
            ContentStartLine=ContentStartLine
        )
