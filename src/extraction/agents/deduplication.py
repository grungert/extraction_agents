"""Dynamic LLM-based deduplication agent."""
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
    DeduplicationError,
)
from ..llm import extract_section
from ...config_manager import ConfigurationManager

# Import helpers from agent_utils
from ..agent_utils import get_tokenizer, count_tokens

# Import necessary components for deduplication agent
from pydantic import create_model, Field


class DynamicDeduplicationAgent:
    """Agent for resolving header conflicts between extraction models."""

    def __init__(self, llm, app_config: Optional[AppConfig] = None):
        """Initialize the deduplication agent."""
        self.llm = llm
        self.config_manager: Optional[ConfigurationManager] = None
        self.app_config = app_config or AppConfig()
        self.messages = [] # Initialize as empty, load at runtime

    def deduplicate(self, extraction_results: Dict, markdown_content: str) -> Dict:
        """
        Resolve conflicts where multiple fields map to the same header cell.

        Args:
            extraction_results: Combined results from all extraction agents
            markdown_content: Original markdown table content

        Returns:
            Deduplicated extraction results
        """
        # Ensure config_manager is set
        if self.config_manager is None:
             logger.error("Error: Config manager not set for Deduplication Agent.")
             raise ConfigurationError("Config manager not set for Deduplication Agent.")


        try:
            # Get the Pydantic model definitions from config (uses self.config_manager)
            pydantic_models = self._get_pydantic_models()
            if not pydantic_models:
                 logger.error("Error: Failed to get Pydantic models for deduplication.")
                 raise ConfigurationError("Failed to get Pydantic models for deduplication.")


            # Convert inputs to formatted strings for the LLM
            extraction_json = json.dumps(extraction_results, indent=2)
            models_json = json.dumps(pydantic_models, indent=2)

            # Load detailed instructions (ROLE, OBJECTIVE, etc.) using config manager
            from ...utils.prompt_utils import load_prompt
            # Use dot notation for prompt file name
            instruction_prompt_filename = self.config_manager.get_prompts_config().deduplication_instruction_prompt_file
            detailed_instructions = load_prompt(instruction_prompt_filename)

            # Prepare JSON string inputs with ```json blocks
            extraction_json_str = f"```json\n{json.dumps(extraction_results, indent=2)}\n```"
            conflicts = self._identify_conflicts(extraction_results)
            conflicts_json_str = f"```json\n{json.dumps(conflicts, indent=2)}\n```"

            # Construct the user message content
            user_message_content = f"{detailed_instructions}\n\n# Original Table\n{markdown_content}\n\n# Extracted Data\n{extraction_json_str}\n\n# Identified Conflicts\n{conflicts_json_str}"

            # Create a detailed schema for the DeduplicatedData property
            # This will include all field definitions from the extraction models
            detailed_schema = self._create_detailed_schema(pydantic_models)

            # Create a validation model class for the output with detailed schema
            DeduplicationResult = create_model(
                "DeduplicationResult",
                DeduplicatedData=(Dict, Field(..., description="Deduplicated data with resolved conflicts", json_schema_extra=detailed_schema)),
                DeduplicationConfidence=(float, Field(..., description="Confidence score for deduplication (0.0-1.0)"))
            )

            # Use the extract_section function for consistency
            from ..llm import extract_section

            # Load messages if not already loaded for this run
            if not self.messages:
                 self.messages = self._create_deduplication_messages()
                 if not self.messages:
                      logger.error("Error: Failed to load messages for Deduplication.")
                      # Assuming prompt_filename is still in scope from load_prompt above
                      raise ConfigurationError(f"Deduplication system prompt file '{instruction_prompt_filename}' not found.")


            # Call extract_section with the structured messages
            # self.messages now contains the concise system prompt loaded just-in-time
            response = extract_section(
                markdown_content=user_message_content, # This is the detailed user message
                section_name="Deduplication",
                model_class=DeduplicationResult,
                messages=self.messages,
                llm=self.llm
            )

            # Return deduplicated results if successful
            if response and hasattr(response, "DeduplicatedData"):
                return response.model_dump()
            else:
                logger.warning("Deduplication failed to return valid data")
                return None

        except LLMInteractionError:
            raise
        except Exception as e:
            logger.exception(f"Error in deduplication: {str(e)}")
            raise DeduplicationError(f"Error in deduplication: {str(e)}")

    def _create_detailed_schema(self, pydantic_models: Dict) -> Dict:
        """
        Create a detailed schema for the DeduplicatedData property.

        Args:
            pydantic_models: Dictionary containing model definitions

        Returns:
            Dictionary containing detailed schema for DeduplicatedData
        """
        # Create a schema that includes all field definitions
        schema = {
            "properties": {},
            "additionalProperties": True,
            "type": "object"
        }

        # Add properties for each model and its fields
        for model_name, model_info in pydantic_models.items():
            model_schema = {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
                "description": model_info.get("description", "")
            }

            # Add properties for each field in the model
            for field_name, field_info in model_info.get("fields", {}).items():
                field_schema = {
                    "anyOf": [
                        {"type": field_info.get("type", "string")},
                        {"type": "null"}
                    ],
                    "default": None,
                    "description": field_info.get("description", "")
                }

                # Add examples if available
                if "examples" in field_info:
                    field_schema["examples"] = field_info["examples"]

                model_schema["properties"][field_name] = field_schema

            schema["properties"][model_name] = model_schema

        return schema

    def _identify_conflicts(self, extraction_results: Dict) -> List[Dict]:
        """
        Identify headers that are mapped to multiple fields across different models.

        Args:
            extraction_results: Combined results from all extraction agents

        Returns:
            List of dictionaries containing header and conflicting fields
        """
        # Create a map of header -> list of model.field references
        header_mappings = {}

        # Scan through all models and their fields
        for model_name, model_data in extraction_results.items():
            if model_name == "Context" or not isinstance(model_data, dict):
                continue

            # Skip ValidationConfidence field
            for field_name, field_value in model_data.items():
                if field_name == "ValidationConfidence" or field_value is None:
                    continue

                # Add this model.field to the list of mappings for this header
                if field_value not in header_mappings:
                    header_mappings[field_value] = []
                header_mappings[field_value].append(f"{model_name}.{field_name}")

        # Extract only the headers with multiple mappings (conflicts)
        conflicts = []
        for header, mappings in header_mappings.items():
            if len(mappings) > 1:
                conflicts.append({
                    "header": header,
                    "conflicts": mappings
                })

        return conflicts

    def _get_pydantic_models(self) -> Dict:
        """
        Extract Pydantic model definitions from the configuration.

        Returns:
            Dictionary containing all model definitions from config/full_config.json
        """
        # This method relies on self.config_manager being set
        if not self.config_manager:
             logger.warning("Cannot get pydantic models without config manager.")
             raise ConfigurationError("Cannot get pydantic models without config manager.")


        # Get the extraction models configuration using dot notation
        extraction_models_config = self.config_manager.get_extraction_models()

        # Extract the relevant information for each model
        models_info = {}
        for model_config in extraction_models_config:
            # Access attributes using dot notation
            model_name = model_config.name
            if model_name:
                models_info[model_name] = {
                    "description": model_config.description or "",
                    "fields": {field_name: field.model_dump() for field_name, field in model_config.fields.items()}, # Convert FieldDefinition to dict
                    # Include examples if they exist
                    "examples": model_config.examples or []
                }

        return models_info

    def _create_deduplication_messages(self) -> List[Dict]:
        """Create example messages for deduplication."""
        # Load the concise system message from the prompt file specified in config
        # Assumes self.config_manager is set before this is called
        from ...utils.prompt_utils import load_prompt
        # Use dot notation for prompt file name
        prompt_filename = self.config_manager.get_prompts_config().deduplication_system_prompt_file
        try:
            system_content = load_prompt(prompt_filename)
        except FileNotFoundError:
             logger.error(f"Error: Deduplication system prompt file '{prompt_filename}' not found.")
             raise ConfigurationError(f"Deduplication system prompt file '{prompt_filename}' not found.")


        system_message = {"role": "system", "content": system_content}

        # For now, we don't include examples as this is a new agent
        # In the future, examples could be added to the config
        # Return only the system message for now
        return [system_message]
