"""Dynamic LLM-based classification agents."""
import json
import re
import logging
from typing import Dict, List, Optional, Type, Any, Union

from ...utils.display import logger
from ...models import (
    AppConfig,
    ClassificationOutput,
    ClassificationValidationOutput,
)
from ...exceptions import (
    ConfigurationError,
    LLMInteractionError,
    ExtractionError,
    ValidationError,
)
from ..agent_utils import get_tokenizer, count_tokens # Corrected import path
from ...utils.prompt_utils import load_prompt


class DynamicClassificationAgent:
    """Agent for classifying the document type."""
    def __init__(self, llm, app_config: AppConfig):
        self.llm = llm
        self.app_config = app_config
        self.labels = app_config.classification_labels
        self.prompt_template = self._load_prompt_template(app_config.classification_prompt)
        self.encoding = get_tokenizer() # Use default tokenizer

    def _load_prompt_template(self, prompt_file_name):
        # from ...utils.prompt_utils import load_prompt # Local import - already imported above
        try:
            # Assuming prompts are in src/prompts/
            prompt_text = load_prompt(prompt_file_name)
            # Use ChatPromptTemplate for potential future structure, though we invoke directly now
            from langchain.prompts import ChatPromptTemplate
            return ChatPromptTemplate.from_template(prompt_text)
        except FileNotFoundError:
            # Enhanced Exception
            raise ConfigurationError(
                message=f"Classification prompt file not found",
                error_code="PROMPT_FILE_NOT_FOUND",
                context={"filename": prompt_file_name}
            )

    def _prepare_input_text(self, markdown_content: str, doc_name: str) -> str:
        """Truncates markdown content based on token limits."""
        if not self.prompt_template:
             return "" # Cannot calculate if prompt failed to load

        # Estimate base prompt tokens (excluding the actual content)
        try:
            base_prompt_messages = self.prompt_template.format_messages(doc_name=doc_name, question="")
            base_prompt_text = "".join([msg.content for msg in base_prompt_messages])
            base_prompt_tokens = count_tokens(self.encoding, base_prompt_text)
        except Exception as e:
             logger.warning(f"Error estimating base prompt tokens: {e}. Using default truncation.")
             base_prompt_tokens = 200 # Default estimate

        # Calculate available tokens for content
        max_llm_tokens = self.app_config.classification_model.max_tokens
        context_percent = self.app_config.classification_model.context_window_percentage
        available_tokens = int(max_llm_tokens * context_percent) - base_prompt_tokens

        if available_tokens <= 0:
            logger.warning(f"Warning: Base prompt tokens ({base_prompt_tokens}) exceed allowed limit ({int(max_llm_tokens * context_percent)}). No content will be used for classification.")
            return ""

        chars_per_token = 4
        max_chars = available_tokens * chars_per_token

        truncated_text = markdown_content[:max_chars]

        # Optional: Verify token count
        # actual_tokens = count_tokens(self.encoding, truncated_text)
        # logger.debug(f"Truncated classification input to ~{actual_tokens} tokens / {len(truncated_text)} chars.[/dim]")

        return truncated_text

    def _parse_response(self, llm_response_text: str) -> Dict:
        """Parses the raw LLM response to extract class and confidence."""
        predicted_class = None
        confidence = "Low" # Default confidence

        # Try regex first (more specific)
        class_match = re.search(r"<sol>(.*?)</sol>", llm_response_text, re.IGNORECASE | re.DOTALL)
        if class_match:
            predicted_class_raw = class_match.group(1).strip()
            # Find the best match from labels
            for label in self.labels:
                if label.lower() == predicted_class_raw.lower():
                    predicted_class = label
                    break
            if not predicted_class: # If raw match not in labels, check if any label is substring
                 for label in self.labels:
                     if label.lower() in predicted_class_raw.lower():
                         predicted_class = label
                         break
            # If still not found, maybe it's "None of those" or similar
            if not predicted_class and "none of those" in predicted_class_raw.lower():
                 predicted_class = "None of those" # Normalize

        # Fallback: Check if any label is directly mentioned if regex fails
        if not predicted_class:
            for label in self.labels:
                # Use word boundaries to avoid partial matches like 'Fund' in 'Refund'
                if re.search(r'\b' + re.escape(label) + r'\b', llm_response_text, re.IGNORECASE):
                    predicted_class = label
                    break

        # Simple confidence extraction (can be enhanced)
        if predicted_class and predicted_class != "None of those":
             # Basic confidence based on presence - could be refined
             confidence = "Medium" # Assume medium if class found

        if predicted_class:
            # Ensure the predicted class is one of the allowed labels
            if predicted_class not in self.labels:
                 logger.warning(f"LLM predicted class '{predicted_class}' not in defined labels. Defaulting to 'None of those'. LLM Response: {llm_response_text}")
                 predicted_class = "None of those"
                 confidence = "Low"
            return {"predicted_class": predicted_class, "confidence": confidence}
        else:
            logger.warning(f"Could not extract valid class from LLM classification response: {llm_response_text}. Defaulting to 'None of those'.")
            return {"predicted_class": "None of those", "confidence": "Low", "validation_reason": "Failed to parse LLM output."}

    def run(self, markdown_content: str, doc_name: str) -> Optional[ClassificationOutput]:
        """Runs the classification agent."""
        logger.info("Running Classification Agent...")
        if not self.prompt_template:
             return None # Cannot run without a prompt

        try:
            prepared_text = self._prepare_input_text(markdown_content, doc_name)

            # Format the prompt using the template
            # Note: Langchain templates expect keyword arguments
            formatted_messages = self.prompt_template.format_messages(question=prepared_text, doc_name=doc_name)
            final_prompt_text = "".join([msg.content for msg in formatted_messages])

            # Call LLM directly (no structured output needed here)
            llm_response = self.llm.invoke(final_prompt_text)

            if hasattr(llm_response, 'content'):
                 llm_response_text = llm_response.content
            elif isinstance(llm_response, str):
                 llm_response_text = llm_response
            else:
                 # Enhanced Exception
                 raise LLMInteractionError(
                     message=f"Unexpected LLM response type during classification",
                     error_code="LLM_UNEXPECTED_RESPONSE_TYPE",
                     context={"doc_name": doc_name, "response_type": type(llm_response).__name__}
                 )

            parsed_data = self._parse_response(llm_response_text)

            if "error" in parsed_data:
                logger.error(f"Error parsing classification response: {parsed_data['error']}")
                # Return a default or raise error? Returning default for now.
                return ClassificationOutput(predicted_class="None of those", confidence="Low")
            else:
                output = ClassificationOutput(**parsed_data)
                logger.info(f"Classification successful: Class='{output.predicted_class}', Confidence='{output.confidence}'")
                return output

        except LLMInteractionError:
            # Re-raise custom exception
            raise
        except Exception as e:
            logger.exception(f"Error during classification for doc '{doc_name}': {e}")
            # Enhanced Exception (wrapping original)
            raise ExtractionError(
                message=f"Error during classification: {e}",
                error_code="CLASSIFICATION_RUNTIME_ERROR",
                context={"doc_name": doc_name, "exception_type": e.__class__.__name__}
            )


class DynamicClassificationValidationAgent:
    """Agent for validating the classification."""
    def __init__(self, llm, app_config: AppConfig):
        self.llm = llm
        self.app_config = app_config
        self.labels = app_config.classification_labels
        self.prompt_template = self._load_prompt_template(app_config.classification_validation_prompt)
        self.encoding = get_tokenizer()

    def _load_prompt_template(self, prompt_file_name):
        # from ...utils.prompt_utils import load_prompt # Local import - already imported above
        try:
            prompt_text = load_prompt(prompt_file_name)
            from langchain.prompts import ChatPromptTemplate
            return ChatPromptTemplate.from_template(prompt_text)
        except FileNotFoundError:
            # Enhanced Exception
            raise ConfigurationError(
                message=f"Classification validation prompt file not found",
                error_code="PROMPT_FILE_NOT_FOUND",
                context={"filename": prompt_file_name}
            )

    def _prepare_input_text(self, markdown_content: str, doc_name: str, previous_class: str) -> str:
        """Truncates markdown content based on token limits for validation prompt."""
        if not self.prompt_template:
            return ""

        try:
            base_prompt_messages = self.prompt_template.format_messages(doc_name=doc_name, previous_class=previous_class, text="")
            base_prompt_text = "".join([msg.content for msg in base_prompt_messages])
            base_prompt_tokens = count_tokens(self.encoding, base_prompt_text)
        except Exception as e:
             logger.warning(f"Error estimating validation base prompt tokens: {e}. Using default truncation.")
             base_prompt_tokens = 250 # Default estimate

        # Calculate available tokens for content
        max_llm_tokens = self.app_config.classification_model.max_tokens
        context_percent = self.app_config.classification_model.context_window_percentage
        available_tokens = int(max_llm_tokens * context_percent) - base_prompt_tokens

        if available_tokens <= 0:
            logger.warning(f"Warning: Validation base prompt tokens ({base_prompt_tokens}) exceed limit. No content used for validation.")
            return ""

        chars_per_token = 4
        max_chars = available_tokens * chars_per_token
        truncated_text = markdown_content[:max_chars]

        # Optional: Verify token count
        # actual_tokens = count_tokens(self.encoding, truncated_text)
        # logger.debug(f"Truncated validation input to ~{actual_tokens} tokens / {len(truncated_text)} chars.[/dim]")

        return truncated_text

    def _parse_response(self, llm_response_text: str) -> Dict:
        """Parses the raw LLM response for validation."""
        predicted_class = None
        confidence = "Low"
        reason = None

        # Extract class
        class_match = re.search(r"<class>(.*?)</class>", llm_response_text, re.IGNORECASE | re.DOTALL)
        if class_match:
            predicted_class_raw = class_match.group(1).strip()
            # Normalize
            for label in self.labels:
                 if label.lower() == predicted_class_raw.lower():
                     predicted_class = label
                     break
            if not predicted_class and "none of those" in predicted_class_raw.lower():
                 predicted_class = "None of those"

        # Fallback class extraction
        if not predicted_class:
             for label in self.labels:
                 if re.search(r'\b' + re.escape(label) + r'\b', llm_response_text, re.IGNORECASE):
                     predicted_class = label
                     break

        # Extract confidence
        confidence_match = re.search(r"<confidence>(.*?)</confidence>", llm_response_text, re.IGNORECASE | re.DOTALL)
        if confidence_match:
            confidence = confidence_match.group(1).strip()
        # else keep default "Low"

        # Extract reason
        reason_match = re.search(r"<reason>(.*?)</reason>", llm_response_text, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()

        if predicted_class:
             if predicted_class not in self.labels:
                 logger.warning(f"Validation LLM predicted class '{predicted_class}' not in labels. Defaulting to 'None of those'. LLM Response: {llm_response_text}")
                 predicted_class = "None of those"
                 confidence = "Low"
             return {"predicted_class": predicted_class, "confidence": confidence, "validation_reason": reason}
        else:
            logger.warning(f"Could not extract valid class from LLM validation response: {llm_response_text}. Defaulting to 'None of those'.")
            return {"predicted_class": "None of those", "confidence": "Low", "validation_reason": "Failed to parse LLM output."}


    def run(self, markdown_content: str, doc_name: str, classification_output: ClassificationOutput) -> Optional[ClassificationValidationOutput]:
        """Runs the classification validation agent."""
        logger.info("Running Classification Validation Agent...")
        if not self.prompt_template:
            return None

        try:
            prepared_text = self._prepare_input_text(markdown_content, doc_name, classification_output.predicted_class)

            formatted_messages = self.prompt_template.format_messages(
                text=prepared_text,
                doc_name=doc_name,
                previous_class=classification_output.predicted_class
            )
            final_prompt_text = "".join([msg.content for msg in formatted_messages])

            llm_response = self.llm.invoke(final_prompt_text)

            if hasattr(llm_response, 'content'):
                 llm_response_text = llm_response.content
            elif isinstance(llm_response, str):
                 llm_response_text = llm_response
            else:
                 # Enhanced Exception
                 raise LLMInteractionError(
                     message=f"Unexpected LLM response type during classification validation",
                     error_code="LLM_UNEXPECTED_RESPONSE_TYPE",
                     context={"doc_name": doc_name, "response_type": type(llm_response).__name__}
                 )

            parsed_data = self._parse_response(llm_response_text)

            if "error" in parsed_data:
                logger.error(f"Error parsing validation response: {parsed_data['error']}")
                # Return default validation output
                return ClassificationValidationOutput(
                    predicted_class=classification_output.predicted_class, # Keep original class
                    confidence="Low",
                    validation_reason="Failed to parse validation response."
                )
            else:
                output = ClassificationValidationOutput(**parsed_data)
                logger.info(f"Validation successful: Class='{output.predicted_class}', Confidence='{output.confidence}'")
                return output

        except LLMInteractionError:
            # Re-raise custom exception
            raise
        except Exception as e:
            logger.exception(f"Error during classification validation for doc '{doc_name}': {e}")
            # Enhanced Exception (wrapping original)
            raise ValidationError(
                message=f"Error during classification validation: {e}",
                error_code="CLASSIFICATION_VALIDATION_RUNTIME_ERROR",
                context={"doc_name": doc_name, "exception_type": e.__class__.__name__}
            )
