"""Dynamic LLM-based agents for Excel Header Mapper."""
import json
import time  # Import the time module
from typing import Dict, List, Optional, Type, Any, Union
import re

from ..utils.display import console
from ..models import (
    AppConfig,
    BaseExtraction,
    ContextModel,
    ValidationResult
)
from .llm import extract_section, configure_llm
from ..config_manager import ConfigurationManager, get_configuration_manager
from ..examples.dynamic_examples import format_extraction_messages, format_validation_messages
# Import necessary components for classification agents
from langchain_core.messages import HumanMessage
import tiktoken 
from ..models import ClassificationOutput, ClassificationValidationOutput, AppConfig # Import new models

# Helper function to get tokenizer (moved here for encapsulation or keep in utils)
def get_tokenizer(tokenizer_name: str = "cl100k_base"):
    try:
        return tiktoken.get_encoding(tokenizer_name)
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Error loading tiktoken encoding '{tokenizer_name}': {e}. Token counting may be inaccurate.")
        return None

# Helper function to count tokens (moved here or keep in utils)
def count_tokens(encoding, text):
    if encoding:
        return len(encoding.encode(text))
    # Estimate if no encoding (very rough)
    return len(text) // 4


class DynamicClassificationAgent:
    """Agent for classifying the document type."""
    def __init__(self, llm, app_config: AppConfig):
        self.llm = llm
        self.app_config = app_config
        self.labels = app_config.classification_labels
        self.prompt_template = self._load_prompt_template(app_config.classification_prompt)
        self.encoding = get_tokenizer() # Use default tokenizer

    def _load_prompt_template(self, prompt_file_name):
        from ..utils.prompt_utils import load_prompt # Local import
        try:
            # Assuming prompts are in src/prompts/
            prompt_text = load_prompt(prompt_file_name)
            # Use ChatPromptTemplate for potential future structure, though we invoke directly now
            from langchain.prompts import ChatPromptTemplate 
            return ChatPromptTemplate.from_template(prompt_text)
        except FileNotFoundError:
            console.print(f"[red]Error: Classification prompt file '{prompt_file_name}' not found.[/red]")
            raise # Re-raise to stop pipeline if prompt is missing

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
             console.print(f"[yellow]⚠[/yellow] Error estimating base prompt tokens: {e}. Using default truncation.")
             base_prompt_tokens = 200 # Default estimate

        # Calculate available tokens for content
        max_llm_tokens = self.app_config.classification_model.max_tokens
        context_percent = self.app_config.classification_model.context_window_percentage
        available_tokens = int(max_llm_tokens * context_percent) - base_prompt_tokens

        if available_tokens <= 0:
            console.print(f"[yellow]⚠[/yellow] Warning: Base prompt tokens ({base_prompt_tokens}) exceed allowed limit ({int(max_llm_tokens * context_percent)}). No content will be used for classification.")
            return ""

        # Estimate characters per token (heuristic)
        chars_per_token = 4 
        max_chars = available_tokens * chars_per_token
        
        truncated_text = markdown_content[:max_chars]
        
        # Optional: Verify token count of truncated text (can be slow)
        # actual_tokens = count_tokens(self.encoding, truncated_text)
        # console.print(f"[dim]Truncated classification input to ~{actual_tokens} tokens / {len(truncated_text)} chars.[/dim]")

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
                 console.print(f"[yellow]⚠[/yellow] LLM predicted class '{predicted_class}' not in defined labels. Defaulting to 'None of those'. LLM Response: {llm_response_text}")
                 predicted_class = "None of those"
                 confidence = "Low"
            return {"predicted_class": predicted_class, "confidence": confidence}
        else:
            console.print(f"[yellow]⚠[/yellow] Could not extract valid class from LLM classification response: {llm_response_text}. Defaulting to 'None of those'.")
            return {"predicted_class": "None of those", "confidence": "Low"} # Default if extraction fails

    def run(self, markdown_content: str, doc_name: str) -> Optional[ClassificationOutput]:
        """Runs the classification agent."""
        console.print("[blue]Running Classification Agent...[/blue]")
        if not self.prompt_template:
             return None # Cannot run without a prompt

        try:
            prepared_text = self._prepare_input_text(markdown_content, doc_name)
            
            # Format the prompt using the template
            # Note: Langchain templates expect keyword arguments
            formatted_messages = self.prompt_template.format_messages(question=prepared_text, doc_name=doc_name)
            # Combine messages into a single string for direct invocation
            final_prompt_text = "".join([msg.content for msg in formatted_messages])

            console.print(f"[dim]Classification Prompt (truncated): {final_prompt_text[:300]}...[/dim]")

            # Call LLM directly (no structured output needed here)
            llm_response = self.llm.invoke(final_prompt_text)
            
            # Ensure response is string content
            if hasattr(llm_response, 'content'):
                 llm_response_text = llm_response.content
            elif isinstance(llm_response, str):
                 llm_response_text = llm_response
            else:
                 raise ValueError(f"Unexpected LLM response type: {type(llm_response)}")

            console.print(f"[dim]Classification LLM Raw Response: {llm_response_text}[/dim]")

            parsed_data = self._parse_response(llm_response_text)

            if "error" in parsed_data:
                console.print(f"[red]Error parsing classification response: {parsed_data['error']}[/red]")
                # Return a default or raise error? Returning default for now.
                return ClassificationOutput(predicted_class="None of those", confidence="Low")
            else:
                output = ClassificationOutput(**parsed_data)
                console.print(f"[green]✓[/green] Classification successful: Class='{output.predicted_class}', Confidence='{output.confidence}'")
                return output

        except Exception as e:
            console.print(f"[red]Error during classification: {e}[/red]")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            return None


class DynamicClassificationValidationAgent:
    """Agent for validating the classification."""
    def __init__(self, llm, app_config: AppConfig):
        self.llm = llm
        self.app_config = app_config
        self.labels = app_config.classification_labels
        self.prompt_template = self._load_prompt_template(app_config.classification_validation_prompt)
        self.encoding = get_tokenizer()

    def _load_prompt_template(self, prompt_file_name):
        from ..utils.prompt_utils import load_prompt # Local import
        try:
            prompt_text = load_prompt(prompt_file_name)
            from langchain.prompts import ChatPromptTemplate
            return ChatPromptTemplate.from_template(prompt_text)
        except FileNotFoundError:
            console.print(f"[red]Error: Classification validation prompt file '{prompt_file_name}' not found.[/red]")
            raise

    def _prepare_input_text(self, markdown_content: str, doc_name: str, previous_class: str) -> str:
        """Truncates markdown content based on token limits for validation prompt."""
        if not self.prompt_template:
            return ""

        try:
            base_prompt_messages = self.prompt_template.format_messages(doc_name=doc_name, previous_class=previous_class, text="")
            base_prompt_text = "".join([msg.content for msg in base_prompt_messages])
            base_prompt_tokens = count_tokens(self.encoding, base_prompt_text)
        except Exception as e:
             console.print(f"[yellow]⚠[/yellow] Error estimating validation base prompt tokens: {e}. Using default truncation.")
             base_prompt_tokens = 250 # Default estimate

        max_llm_tokens = self.app_config.classification_model.max_tokens
        context_percent = self.app_config.classification_model.context_window_percentage
        available_tokens = int(max_llm_tokens * context_percent) - base_prompt_tokens

        if available_tokens <= 0:
            console.print(f"[yellow]⚠[/yellow] Warning: Validation base prompt tokens ({base_prompt_tokens}) exceed limit. No content used for validation.")
            return ""

        chars_per_token = 4
        max_chars = available_tokens * chars_per_token
        truncated_text = markdown_content[:max_chars]
        
        # Optional: Verify token count
        # actual_tokens = count_tokens(self.encoding, truncated_text)
        # console.print(f"[dim]Truncated validation input to ~{actual_tokens} tokens / {len(truncated_text)} chars.[/dim]")

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
                 console.print(f"[yellow]⚠[/yellow] Validation LLM predicted class '{predicted_class}' not in labels. Defaulting to 'None of those'. LLM Response: {llm_response_text}")
                 predicted_class = "None of those"
                 confidence = "Low"
             return {"predicted_class": predicted_class, "confidence": confidence, "validation_reason": reason}
        else:
            console.print(f"[yellow]⚠[/yellow] Could not extract valid class from LLM validation response: {llm_response_text}. Defaulting to 'None of those'.")
            return {"predicted_class": "None of those", "confidence": "Low", "validation_reason": "Failed to parse LLM output."}


    def run(self, markdown_content: str, doc_name: str, classification_output: ClassificationOutput) -> Optional[ClassificationValidationOutput]:
        """Runs the classification validation agent."""
        console.print("[blue]Running Classification Validation Agent...[/blue]")
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

            console.print(f"[dim]Validation Prompt (truncated): {final_prompt_text[:300]}...[/dim]")

            llm_response = self.llm.invoke(final_prompt_text)
            
            if hasattr(llm_response, 'content'):
                 llm_response_text = llm_response.content
            elif isinstance(llm_response, str):
                 llm_response_text = llm_response
            else:
                 raise ValueError(f"Unexpected LLM response type: {type(llm_response)}")

            console.print(f"[dim]Validation LLM Raw Response: {llm_response_text}[/dim]")

            parsed_data = self._parse_response(llm_response_text)

            if "error" in parsed_data:
                console.print(f"[red]Error parsing validation response: {parsed_data['error']}[/red]")
                # Return default validation output
                return ClassificationValidationOutput(
                    predicted_class=classification_output.predicted_class, # Keep original class
                    confidence="Low", 
                    validation_reason="Failed to parse validation response."
                )
            else:
                output = ClassificationValidationOutput(**parsed_data)
                console.print(f"[green]✓[/green] Validation successful: Class='{output.predicted_class}', Confidence='{output.confidence}'")
                return output

        except Exception as e:
            console.print(f"[red]Error during classification validation: {e}[/red]")
            import traceback
            traceback.print_exc()
            return None


class DynamicHeaderValidationAgent:
    """Agent for validating header detection results using dynamic configuration."""
    
    def __init__(self, llm): # Remove config_manager argument
        """
        Initialize the header validation agent.
        
        Args:
            llm: LLM instance to use for validation
        """
        self.llm = llm
        self.config_manager = None # Initialize as None
        self.messages = [] # Initialize as empty list
        
    def validate(self, header_info: ContextModel, markdown_content: str) -> Optional[ContextModel]:
        # Ensure config_manager is set before creating messages
        if self.config_manager is None:
             console.print("[red]Error: Config manager not set for Header Validation Agent.[/red]")
             return None
        # Load messages here, using the dynamically set config_manager
        if not self.messages: # Simple check
            self.messages = self._create_validation_messages()
            if not self.messages: # If loading failed
                 console.print("[red]Error: Failed to load validation messages for Header Validation.[/red]")
                 return None
        """
        Validate header detection results against the original table.
        
        Args:
            header_info: Header detection information to validate
            markdown_content: Original markdown table content
            
        Returns:
            Validated header information with confidence score, or None if validation failed
        """
        console.print("[blue]Validating header detection...[/blue]")
        
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
                console.print(f"[green]✓[/green] Header validation successful with confidence {result.ValidationConfidence:.2f}")
                return result
            else:
                console.print(f"[yellow]⚠[/yellow] Header validation returned incomplete data")
                return None
                
        except Exception as e:
            console.print(f"[red]Error in header validation: {str(e)}[/red]")
            return None
    
    def _create_validation_messages(self) -> List[Dict]:
        """
        Create example messages for header validation.
        
        Returns:
            List of message dictionaries for the LLM
        """
        # This method now relies on self.config_manager being set before it's called
        if not self.config_manager:
             console.print("[yellow]⚠[/yellow] Cannot create header validation examples without config manager.")
             return []

        # Get header validation configuration
        validation_config = self.config_manager.get_validation_config()
        
        # Load system message from prompt file
        from ..utils.prompt_utils import load_prompt
        system_message = {
            "role": "system",
            "content": load_prompt("header_validation_system_v1.md") # Consider making prompt name configurable?
        }
        
        # Load header examples from configuration manager
        header_examples = self.config_manager.get_header_examples()
        
        # Create examples for the LLM - format them for validation
        examples = []
        for idx, example in enumerate(header_examples[:2], start=1):  # Add index with enumerate
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
    
    def __init__(self, llm): # Remove config_manager argument
        """
        Initialize the header detection agent.
        
        Args:
            llm: LLM instance to use for detection
        """
        self.llm = llm
        self.config_manager = None # Initialize as None
        self.messages = [] # Initialize as empty list
        
    def detect_headers(self, markdown_content: str) -> Optional[ContextModel]:
        # Ensure config_manager is set before creating messages
        if self.config_manager is None:
             console.print("[red]Error: Config manager not set for Header Detection Agent.[/red]")
             return None
        # Load messages here, using the dynamically set config_manager
        # Check if messages need reloading (if config_manager changed or messages are empty)
        if not self.messages: # Simple check, could be more robust if needed
            self.messages = self._create_example_messages() 
            if not self.messages: # If loading failed
                 console.print("[red]Error: Failed to load example messages for Header Detection.[/red]")
                 return None

        """
        Detect header positions in markdown content.
        
        Args:
            markdown_content: Markdown content to analyze
            
        Returns:
            ContextModel with header positions, or None if detection failed
        """
        console.print("[blue]Running header detection agent...[/blue]")
        
        # Display the input for the header agent
        console.print("[cyan]Header Agent Input:[/cyan]")
        console.print(markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content)
        
        try:
            # Use a dictionary to capture the raw response including header_detection_confidence
            raw_result = None
            
            # Extract the header information
            result = extract_section(
                markdown_content=markdown_content,
                section_name="HeaderDetection",
                model_class=ContextModel,
                messages=self.messages,
                llm=self.llm
            )
            
            if result and hasattr(result, 'HeaderStartLine') and result.HeaderStartLine is not None:
                # Store ValidationConfidence as an attribute for internal use
                if hasattr(result, 'ValidationConfidence'):
                    # Store the confidence value as an attribute for internal use
                    result._header_detection_confidence = result.ValidationConfidence
                    console.print(f"[green]✓[/green] Header detection successful with confidence {result.ValidationConfidence:.2f}")
                else:
                    console.print(f"[green]✓[/green] Header detection successful (no confidence score)")
                    
                return result
            else:
                console.print("[yellow]⚠[/yellow] Header detection returned incomplete data")
                return None
                
        except Exception as e:
            console.print(f"[red]Error in header detection: {str(e)}[/red]")
            return None
        
    def _create_example_messages(self) -> List[Dict]:
        """
        Create example messages for few-shot learning.
        
        Returns:
            List of message dictionaries for the LLM
        """
        # This method now relies on self.config_manager being set before it's called
        if not self.config_manager:
             console.print("[yellow]⚠[/yellow] Cannot create header examples without config manager.")
             return [] # Return empty list or handle error

        # Get header detection configuration
        header_config = self.config_manager.get_header_detection_config()
        
        # Load system message from prompt file
        from ..utils.prompt_utils import load_prompt
        system_message = {
            "role": "system",
            "content": load_prompt("header_detection_system_v1.md") # Consider making prompt name configurable?
        }
        
        # Load header examples from configuration manager
        header_examples = self.config_manager.get_header_examples()
        
        # Create examples for the LLM - format them for validation
        examples = []
        for idx, example in enumerate(header_examples[:2], start=1):  # Add index with enumerate
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
        console.print("[yellow]Using fallback header detection strategy...[/yellow]")
        
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
        
        console.print(f"[yellow]Fallback detection: headers at lines {HeaderStartLine}-{HeaderEndLine}, content starts at line {ContentStartLine}[/yellow]")
        
        # Create ContextModel without header_detection_confidence
        return ContextModel(
            HeaderStartLine=HeaderStartLine,
            HeaderEndLine=HeaderEndLine,
            ContentStartLine=ContentStartLine
        )


class DynamicExtractionAgent:
    """Agent for extracting structured data from Excel sheets using dynamic configuration."""
    
    def __init__(self, llm, app_config: Optional[AppConfig] = None): # Removed config_manager
        """
        Initialize the extraction agent.
        
        Args:
            llm: LLM instance to use for extraction
            app_config: Application configuration instance
        """
        self.llm = llm
        self.config_manager = None # Initialize as None
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
        console.print(f"[blue]Extracting {section_name} data...[/blue]")

        # Ensure config_manager is set
        if self.config_manager is None:
             console.print(f"[red]Error: Config manager not set for Extraction Agent ({section_name}).[/red]")
             return None

        try:
            # Focus on relevant rows based on header detection
            focused_content = self._focus_content(markdown_content, header_info)
            
            # Display the input for the extraction agent
            console.print(f"[cyan]Extraction Agent Input for {section_name}:[/cyan]")
            
            # Get or create example messages for this section, using the dynamically set config_manager
            if section_name not in self.section_messages or not self.section_messages[section_name]:
                self.section_messages[section_name] = self._create_section_messages(section_name, self.app_config)
                if not self.section_messages[section_name]: # Check if loading failed
                     console.print(f"[red]Error: Failed to load messages for {section_name} extraction.[/red]")
                     return None
                
            # Extract data
            result = extract_section(
                markdown_content=focused_content,
                section_name=section_name,
                model_class=model_class,
                messages=self.section_messages[section_name],
                llm=self.llm
            )
            
            if result:
                console.print(f"[green]✓[/green] {section_name} extraction successful")
                return result
            else:
                console.print(f"[yellow]⚠[/yellow] {section_name} extraction returned no data")
                return None
                
        except Exception as e:
            console.print(f"[red]Error in {section_name} extraction: {str(e)}[/red]")
            return None
        
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
             console.print(f"[yellow]⚠[/yellow] Cannot create section messages for {section_name} without config manager.")
             return []

        # Load system message template from prompt file, passing necessary context
        from ..utils.prompt_utils import load_template_prompt
        system_content = load_template_prompt(
            "section_extraction_system_template.md", 
            section_name=section_name,
            config_manager=self.config_manager, # Use the dynamically set manager
            app_config=app_config
        )
        system_message = {"role": "system", "content": system_content}
        
        # Format examples for extraction using the dynamic examples manager
        return format_extraction_messages(self.config_manager, section_name, system_message)


class DynamicValidationAgent:
    """Agent for validating and correcting extracted data using dynamic configuration."""
    
    def __init__(self, llm, app_config: Optional[AppConfig] = None): # Removed config_manager
        """
        Initialize the validation agent.
        
        Args:
            llm: LLM instance to use for validation
            app_config: Application configuration instance
        """
        self.llm = llm
        self.config_manager = None # Initialize as None
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
        console.print(f"[blue]Validating {section_name} data...[/blue]")

        # Ensure config_manager is set
        if self.config_manager is None:
             console.print(f"[red]Error: Config manager not set for Validation Agent ({section_name}).[/red]")
             return None

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
                if not self.section_messages[section_name]: # Check if loading failed
                     console.print(f"[red]Error: Failed to load messages for {section_name} validation.[/red]")
                     return None
            
            # Validate data using section-specific messages
            result = extract_section(
                markdown_content=json_input,
                section_name=f"{section_name}Validation",
                model_class=validation_model,
                messages=self.section_messages[section_name],
                llm=self.llm
            )
            
            if result and hasattr(result, 'ValidationConfidence'):
                console.print(f"[green]✓[/green] {section_name} validation successful with confidence {result.ValidationConfidence:.2f}")
                return result
            else:
                console.print(f"[yellow]⚠[/yellow] {section_name} validation returned incomplete data")
                return None
                
        except Exception as e:
            console.print(f"[red]Error in {section_name} validation: {str(e)}[/red]")
            return None
        
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
             console.print(f"[yellow]⚠[/yellow] Cannot create section validation messages for {section_name} without config manager.")
             return []

        # Load system message template from prompt file, passing necessary context
        from ..utils.prompt_utils import load_template_prompt
        system_content = load_template_prompt(
            "section_validation_system_template.md", 
            section_name=section_name,
            config_manager=self.config_manager, # Use the dynamically set manager
            app_config=app_config
        )
        system_message = {"role": "system", "content": system_content}
        
        # Format examples for validation using the dynamic examples manager
        # Pass the section_name to get only examples for this specific section
        return format_validation_messages(self.config_manager, system_message, section_name)


class DynamicAgentPipelineCoordinator:
    """Coordinator for the dynamic agent pipeline."""
    
    def __init__(self, config: AppConfig):
        """
        Initialize the agent pipeline coordinator.
        
        Args:
            config: Application configuration
        """
        self.config = config
        # Import the new LLM configuration function
        from .llm import configure_llm, configure_llm_classification 

        # Configure the two LLM instances
        self.main_llm = configure_llm(config)
        self.classification_llm = configure_llm_classification(config)
        
        # Initialize classification agents with classification LLM and app_config
        self.classification_agent = DynamicClassificationAgent(self.classification_llm, self.config)
        self.classification_validation_agent = DynamicClassificationValidationAgent(self.classification_llm, self.config)

        # Initialize other agents with main LLM
        # Config manager and models will be loaded dynamically in process_markdown
        self.header_agent = DynamicHeaderDetectionAgent(self.main_llm) 
        self.header_validation_agent = DynamicHeaderValidationAgent(self.main_llm) 
        self.extraction_agent = DynamicExtractionAgent(self.main_llm, app_config=self.config) 
        self.validation_agent = DynamicValidationAgent(self.main_llm, app_config=self.config) 
        self.deduplication_agent = DynamicDeduplicationAgent(self.main_llm, app_config=self.config) 
        
        # Reset config manager and models - they are loaded conditionally per run
        self.config_manager = None
        self.extraction_models = {}
        
    def process_markdown(self, markdown_content: str, source_file: str) -> Dict:
        """
        Process markdown content through the agent pipeline.
        
        Args:
            markdown_content: Markdown content to process
            source_file: Source file path (for output)
            
        Returns:
            Dictionary of extracted and validated data
        """
        # Initialize an ordered dictionary to maintain the order of sections
        from collections import OrderedDict
        import os
        ordered_results = OrderedDict()
        start_time = time.perf_counter()  # Start timer

        # --- NEW: Classification Steps ---
        doc_name = os.path.splitext(os.path.basename(source_file))[0] if source_file else "Unknown Document"
        
        classification_output = self.classification_agent.run(markdown_content, doc_name)
        if not classification_output:
            console.print("[red]Classification failed. Stopping pipeline.[/red]")
            return {"error": "Classification failed"}

        validation_output = self.classification_validation_agent.run(markdown_content, doc_name, classification_output)
        if not validation_output:
            console.print("[red]Classification validation failed. Stopping pipeline.[/red]")
            return {"error": "Classification validation failed"}

        console.print(f"[blue]Classification Validated:[/blue] Class='{validation_output.predicted_class}', Confidence='{validation_output.confidence}', Reason='{validation_output.validation_reason}'")
        
        # Store classification result
        ordered_results["Classification"] = validation_output.model_dump()
        
        # --- Conditional Logic ---
        validated_class = validation_output.predicted_class
        if validated_class == "Mutual Funds":
            # Construct the specific config path
            config_file_name = f"full_config_{validated_class.replace(' ', '_')}.json" # e.g., full_config_Mutual_Funds.json
            config_path = os.path.join("config", config_file_name)

            if not os.path.exists(config_path):
                console.print(f"[red]Error: Config file not found: {config_path}[/red]")
                # Add timing info before returning error
                end_time = time.perf_counter()
                processing_time = end_time - start_time
                ordered_results["ProcessingTimeSeconds"] = round(processing_time, 3)
                return {"error": f"Config file not found for {validated_class}", **ordered_results}

            # Initialize config_manager and models FOR THIS RUN
            try:
                # Import here to avoid circular dependency if models.py imports this file later
                from ..config_manager import get_configuration_manager 
                from ..dynamic_model_factory import create_extraction_models_dict
                
                current_run_config_manager = get_configuration_manager(config_path)
                current_run_extraction_models = create_extraction_models_dict(
                    current_run_config_manager,
                    include_examples=self.config.include_header_examples_in_prompt
                )
                # Store them for this run (needed by agents)
                self.config_manager = current_run_config_manager
                self.extraction_models = current_run_extraction_models

            except Exception as e:
                 console.print(f"[red]Error loading config or models from {config_path}: {e}[/red]")
                 # Add timing info
                 end_time = time.perf_counter()
                 processing_time = end_time - start_time
                 ordered_results["ProcessingTimeSeconds"] = round(processing_time, 3)
                 return {"error": f"Failed to load config/models for {validated_class}", **ordered_results}

            console.print(f"[green]Proceeding for '{validated_class}' using '{config_path}'[/green]")

            # --- Existing Pipeline Steps (Modified to use dynamically loaded config) ---
            
            # Step: Header Detection 
            # Pass config manager to agent instance for this run
            self.header_agent.config_manager = self.config_manager 
            self.header_agent.messages = self.header_agent._create_example_messages() # Reload messages based on new config
            header_info = self.header_agent.detect_headers(markdown_content)
            
            if not header_info:
                console.print("[red]Header detection failed[/red]")
                # Add timing info
                end_time = time.perf_counter()
                processing_time = end_time - start_time
                ordered_results["ProcessingTimeSeconds"] = round(processing_time, 3)
                return {"error": "header detection failed", **ordered_results}

            # Step: Header Validation
            self.header_validation_agent.config_manager = self.config_manager
            self.header_validation_agent.messages = self.header_validation_agent._create_validation_messages() # Reload messages
            validated_header_info = self.header_validation_agent.validate(header_info, markdown_content)
            
            validation_config = self.config_manager.get_validation_config()
            header_confidence_threshold = validation_config.get('confidence_threshold', 0.7)
            
            if not validated_header_info or (hasattr(validated_header_info, 'ValidationConfidence') and 
                                             validated_header_info.ValidationConfidence < header_confidence_threshold):
                console.print("[red]Header validation failed or has low confidence[/red]")
                # Add timing info
                end_time = time.perf_counter()
                processing_time = end_time - start_time
                ordered_results["ProcessingTimeSeconds"] = round(processing_time, 3)
                return {"error": "header validation failed", **ordered_results}
            
            header_info = validated_header_info
            header_confidence = validated_header_info.ValidationConfidence if hasattr(validated_header_info, 'ValidationConfidence') else 0.7
            
            # Context Section Setup
            FileName = doc_name # Already extracted
            file_ext = os.path.splitext(source_file)[1].lower() if source_file else None
            FileType = file_ext.lstrip('.') if file_ext else None
            context_data = {
                "ValidationConfidence": header_confidence, # Use header validation confidence
                "FileName": FileName,
                "HeaderStartLine": getattr(header_info, 'HeaderStartLine', None),
                "HeaderEndLine": getattr(header_info, 'HeaderEndLine', None),
                "ContentStartLine": getattr(header_info, 'ContentStartLine', None),
                "FileType": FileType
            }
            ordered_results["Context"] = context_data
            
            # Step: Section Extraction & Validation
            # Pass config manager to agents for this run
            self.extraction_agent.config_manager = self.config_manager
            self.validation_agent.config_manager = self.config_manager
            # Reset messages cache as config changed
            self.extraction_agent.section_messages = {} 
            self.validation_agent.section_messages = {} 

            results = {}
            extraction_confidence_threshold = validation_config.get('confidence_threshold', 0.8)
            for section_name, model_class in self.extraction_models.items():
                results[section_name] = {k: None for k in model_class.model_fields.keys() if k != 'ValidationConfidence'} # Init without confidence field
                
                extracted_data = self.extraction_agent.extract_data(
                    markdown_content, header_info, section_name, model_class
                )
                
                if not extracted_data:
                    console.print(f"[yellow]⚠[/yellow] No data extracted for {section_name}")
                    results[section_name]['ValidationConfidence'] = 0.0 # Mark as low confidence if extraction failed
                    continue
                    
                validated_result = self.validation_agent.validate(
                    extracted_data, markdown_content, header_info, section_name, model_class
                )
                
                current_section_confidence = 0.5 # Default if validation fails
                if validated_result and hasattr(validated_result, 'ValidatedData'):
                    ValidatedData = validated_result.ValidatedData.model_dump()
                    for field, value in ValidatedData.items():
                        if value is not None and field != 'ValidationConfidence':
                            results[section_name][field] = value

                    if hasattr(validated_result, 'ValidationConfidence'):
                        current_section_confidence = validated_result.ValidationConfidence
                        results[section_name]['ValidationConfidence'] = current_section_confidence
                    else:
                        results[section_name]['ValidationConfidence'] = 0.9 # Default high if validation worked but no score

                    console.print(f"[green]✓[/green] Used validated {section_name} (confidence {results[section_name]['ValidationConfidence']:.2f})")
                    if hasattr(validated_result, 'CorrectionsMade') and validated_result.CorrectionsMade:
                        console.print(f"[blue]Corrections made:[/blue]")
                        for correction in validated_result.CorrectionsMade: console.print(f"  • {correction}")
                else:
                    console.print(f"[yellow]⚠[/yellow] Validation failed for {section_name}, using original extraction")
                    result_data = extracted_data.model_dump()
                    for field, value in result_data.items():
                        if value is not None and field != 'ValidationConfidence':
                            results[section_name][field] = value
                    results[section_name]['ValidationConfidence'] = current_section_confidence # Keep default 0.5

            # Add extraction results to ordered results
            for section_name, section_data in results.items():
                ordered_results[section_name] = section_data
            
            # Step: Deduplication
            if self.config.enable_deduplication_agent:
                # Pass config manager for this run
                self.deduplication_agent.config_manager = self.config_manager 
                console.print("[blue]Running deduplication agent to resolve header conflicts...[/blue]")
                deduplication_result = self.deduplication_agent.deduplicate(
                    ordered_results, markdown_content
                )
                
                if deduplication_result and "DeduplicatedData" in deduplication_result:
                    deduplicated_data = deduplication_result["DeduplicatedData"]
                    for section_name, section_data in deduplicated_data.items():
                        if section_name in ordered_results:
                            ordered_results[section_name] = section_data
                    console.print(f"[green]✓[/green] Successfully deduplicated extraction results with confidence {deduplication_result.get('DeduplicationConfidence', 0.0):.2f}")
                    # Note: ConflictsResolved logging removed as per previous user request
                else:
                    console.print("[yellow]⚠[/yellow] Deduplication failed, using original extraction results")

        else: # Not Mutual Funds
            console.print(f"[yellow]Skipping extraction for class: '{validated_class}'[/yellow]")
            # Add timing info and return early
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            ordered_results["ProcessingTimeSeconds"] = round(processing_time, 3)
            console.print(f"[cyan]Total processing time (skipped extraction): {processing_time:.3f} seconds[/cyan]")
            return ordered_results 

        # --- Final processing for the successful Mutual Funds case ---
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # Ensure Context exists before adding time
        if "Context" not in ordered_results or ordered_results["Context"] is None:
             ordered_results["Context"] = {} # Create if missing
        ordered_results["Context"]["ProcessingTimeSeconds"] = round(processing_time, 3)
            
        console.print(f"[cyan]Total processing time for {source_file}: {processing_time:.3f} seconds[/cyan]")
        return ordered_results


class DynamicDeduplicationAgent:
    """Agent for resolving header conflicts between extraction models."""
    
    def __init__(self, llm, app_config: Optional[AppConfig] = None): # Removed config_manager
        """Initialize the deduplication agent."""
        self.llm = llm
        self.config_manager = None # Initialize as None
        self.app_config = app_config or AppConfig()
        self.messages = self._create_deduplication_messages() # Keep loading concise system prompt here
        
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
             console.print("[red]Error: Config manager not set for Deduplication Agent.[/red]")
             return None
             
        try:
            # Get the Pydantic model definitions from config (uses self.config_manager)
            pydantic_models = self._get_pydantic_models()
            if not pydantic_models: # Check if loading failed
                 console.print("[red]Error: Failed to get Pydantic models for deduplication.[/red]")
                 return None
            
            # Convert inputs to formatted strings for the LLM
            extraction_json = json.dumps(extraction_results, indent=2)
            models_json = json.dumps(pydantic_models, indent=2)
            
            # Load detailed instructions (ROLE, OBJECTIVE, etc.)
            from ..utils.prompt_utils import load_prompt
            detailed_instructions = load_prompt("deduplication_agent.md")

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
            from pydantic import create_model, Field
            DeduplicationResult = create_model(
                "DeduplicationResult",
                DeduplicatedData=(Dict, Field(..., description="Deduplicated data with resolved conflicts", json_schema_extra=detailed_schema)),
                DeduplicationConfidence=(float, Field(..., description="Confidence score for deduplication (0.0-1.0)"))
            )
            
            # Use the extract_section function for consistency
            from .llm import extract_section
            
            # Call extract_section with the structured messages
            # self.messages contains the concise system prompt loaded in __init__
            response = extract_section(
                markdown_content=user_message_content, # This is the detailed user message
                section_name="Deduplication",
                model_class=DeduplicationResult,
                messages=self.messages, # Contains concise system prompt
                llm=self.llm
            )
            
            # Return deduplicated results if successful
            if response and hasattr(response, "DeduplicatedData"):
                return response.model_dump()
            else:
                console.print("[yellow]⚠[/yellow] Deduplication failed to return valid data")
                return None
                
        except Exception as e:
            console.print(f"[red]Error in deduplication: {str(e)}[/red]")
            return None
            
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
             console.print("[yellow]⚠[/yellow] Cannot get pydantic models without config manager.")
             return {}

        # Get the extraction models configuration
        extraction_models_config = self.config_manager.get_extraction_models()
        
        # Extract the relevant information for each model
        models_info = {}
        for model_config in extraction_models_config:
            model_name = model_config.get("name")
            if model_name:
                models_info[model_name] = {
                    "description": model_config.get("description", ""),
                    "fields": model_config.get("fields", {}),
                    # Include examples if they exist
                    "examples": model_config.get("examples", [])
                }
                
        return models_info
        
    def _create_deduplication_messages(self) -> List[Dict]:
        """Create example messages for deduplication."""
        # Load the concise system message from the new prompt file
        from ..utils.prompt_utils import load_prompt
        system_content = load_prompt("deduplication_system.md") # Load from new file
        system_message = {"role": "system", "content": system_content}
        
        # For now, we don't include examples as this is a new agent
        # In the future, examples could be added to the config
        # Return only the system message for now
        return [system_message]


# Helper function to get extraction models from config manager
def create_extraction_models_dict(
    config_manager: ConfigurationManager, 
    include_examples: bool = True  # Add the argument here
) -> Dict[str, Type[BaseExtraction]]:
    """
    Create a dictionary mapping section names to model classes.
    
    Args:
        config_manager: Configuration manager instance
        include_examples: Whether to include examples in model descriptions
        
    Returns:
        Dictionary mapping section names to model classes
    """
    # Import the function from the factory
    from ..dynamic_model_factory import create_extraction_models_dict as factory_create_models
    
    # Call the factory function, passing the include_examples flag
    return factory_create_models(config_manager, include_examples=include_examples)
