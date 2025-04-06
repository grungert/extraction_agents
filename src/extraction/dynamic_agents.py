"""Dynamic LLM-based agents for Excel Header Mapper."""
import json
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


class DynamicHeaderValidationAgent:
    """Agent for validating header detection results using dynamic configuration."""
    
    def __init__(self, llm, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the header validation agent.
        
        Args:
            llm: LLM instance to use for validation
            config_manager: Configuration manager, or None to use default
        """
        self.llm = llm
        self.config_manager = config_manager or get_configuration_manager()
        self.messages = self._create_validation_messages()
        
    def validate(self, header_info: ContextModel, markdown_content: str) -> Optional[ContextModel]:
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
        # Get header validation configuration
        validation_config = self.config_manager.get_validation_config()
        
        # Load system message from prompt file
        from ..utils.prompt_utils import load_prompt
        system_message = {
            "role": "system",
            "content": load_prompt("header_validation_system.md")
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
    
    def __init__(self, llm, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the header detection agent.
        
        Args:
            llm: LLM instance to use for detection
            config_manager: Configuration manager, or None to use default
        """
        self.llm = llm
        self.config_manager = config_manager or get_configuration_manager()
        self.messages = self._create_example_messages()
        
    def detect_headers(self, markdown_content: str) -> Optional[ContextModel]:
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
        # Get header detection configuration
        header_config = self.config_manager.get_header_detection_config()
        
        # Load system message from prompt file
        from ..utils.prompt_utils import load_prompt
        system_message = {
            "role": "system",
            "content": load_prompt("header_detection_system.md")
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
    
    def __init__(self, llm, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the extraction agent.
        
        Args:
            llm: LLM instance to use for extraction
            config_manager: Configuration manager, or None to use default
        """
        self.llm = llm
        self.config_manager = config_manager or get_configuration_manager()
        self.section_messages = {}
        
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
        
        try:
            # Focus on relevant rows based on header detection
            focused_content = self._focus_content(markdown_content, header_info)
            
            # Display the input for the extraction agent
            console.print(f"[cyan]Extraction Agent Input for {section_name}:[/cyan]")
            console.print(focused_content[:500] + "..." if len(focused_content) > 500 else focused_content)
            
            # Get or create example messages for this section
            if section_name not in self.section_messages:
                self.section_messages[section_name] = self._create_section_messages(section_name)
                
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
        
    def _create_section_messages(self, section_name: str) -> List[Dict]:
        """
        Create example messages for a specific section.
        
        Args:
            section_name: Name of the section
            
        Returns:
            List of message dictionaries for the LLM
        """
        # Load system message template from prompt file
        from ..utils.prompt_utils import load_template_prompt
        system_message = {
            "role": "system",
            "content": load_template_prompt("section_extraction_system_template.md", section_name=section_name)
        }
        
        # Format examples for extraction using the dynamic examples manager
        return format_extraction_messages(self.config_manager, section_name, system_message)


class DynamicValidationAgent:
    """Agent for validating and correcting extracted data using dynamic configuration."""
    
    def __init__(self, llm, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the validation agent.
        
        Args:
            llm: LLM instance to use for validation
            config_manager: Configuration manager, or None to use default
        """
        self.llm = llm
        self.config_manager = config_manager or get_configuration_manager()
        self.section_messages = {}  # Initialize empty dictionary for section-specific messages
        
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
            
            # Get or create example messages for this section
            if section_name not in self.section_messages:
                self.section_messages[section_name] = self._create_section_validation_messages(section_name)
            
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
        
    def _create_section_validation_messages(self, section_name: str) -> List[Dict]:
        """
        Create example messages for validation of a specific section.
        
        Args:
            section_name: Name of the section
            
        Returns:
            List of message dictionaries for the LLM
        """
        # Load system message template from prompt file
        from ..utils.prompt_utils import load_template_prompt
        system_message = {
            "role": "system",
            "content": load_template_prompt("section_validation_system_template.md", section_name=section_name)
        }
        
        # Format examples for validation using the dynamic examples manager
        # Pass the section_name to get only examples for this specific section
        return format_validation_messages(self.config_manager, system_message, section_name)


class DynamicAgentPipelineCoordinator:
    """Coordinator for the dynamic agent pipeline."""
    
    def __init__(self, config: AppConfig, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the agent pipeline coordinator.
        
        Args:
            config: Application configuration
            config_manager: Configuration manager, or None to use default
        """
        self.config = config
        self.config_manager = config_manager or get_configuration_manager(config.config_path)
        self.llm = configure_llm(config)
        
        # Initialize agents with dynamic configuration
        self.header_agent = DynamicHeaderDetectionAgent(self.llm, self.config_manager)
        self.header_validation_agent = DynamicHeaderValidationAgent(self.llm, self.config_manager)
        self.extraction_agent = DynamicExtractionAgent(self.llm, self.config_manager)
        self.validation_agent = DynamicValidationAgent(self.llm, self.config_manager)
        
        # Get the dynamically configured extraction models
        self.extraction_models = create_extraction_models_dict(self.config_manager)
        
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
        
        # Step 1: Detect headers
        header_info = self.header_agent.detect_headers(markdown_content)
        
        # Check if header detection failed
        if not header_info:
            console.print("[red]Header detection failed[/red]")
            return {"error": "header detection failed"}
        
        # Step 2: Validate header detection
        validated_header_info = self.header_validation_agent.validate(header_info, markdown_content)
        
        # Get validation confidence threshold from config
        validation_config = self.config_manager.get_validation_config()
        header_confidence_threshold = validation_config.get('confidence_threshold', 0.7)
        
        # Check if header validation failed or has low confidence
        if not validated_header_info or (hasattr(validated_header_info, 'ValidationConfidence') and 
                                         validated_header_info.ValidationConfidence < header_confidence_threshold):
            console.print("[red]Header validation failed or has low confidence[/red]")
            return {"error": "header detection failed"}
        
        # Use validated header info
        header_info = validated_header_info
        header_confidence = validated_header_info.ValidationConfidence if hasattr(validated_header_info, 'ValidationConfidence') else 0.7
        
        # Extract file information from source_file
        console.print(f"[blue]Extracting file information from source file: {source_file}[/blue]")
        FileName = os.path.splitext(os.path.basename(source_file))[0] if source_file else None
        file_ext = os.path.splitext(source_file)[1].lower() if source_file else None
        FileType = file_ext.lstrip('.') if file_ext else None
        console.print(f"[blue]Extracted file info - name: {FileName}, extension: {file_ext}, type: {FileType}[/blue]")
        
        # Always create a Context section with header detection information
        context_data = {
            "ValidationConfidence": header_confidence,
            "FileName": FileName,
            "HeaderStartLine": None,
            "HeaderEndLine": None,
            "ContentStartLine": None,
            "FileType": FileType
        }
        
        # Copy header detection information to Context section
        if header_info:
            for field in ["HeaderStartLine", "HeaderEndLine", "ContentStartLine"]:
                if hasattr(header_info, field) and getattr(header_info, field) is not None:
                    context_data[field] = getattr(header_info, field)
        
        # Add Context section to ordered results first
        ordered_results["Context"] = context_data
        
        # Get validation confidence threshold from config
        validation_config = self.config_manager.get_validation_config()
        extraction_confidence_threshold = validation_config.get('confidence_threshold', 0.8)
        
        # Step 2: Extract each section
        results = {}  # Temporary dictionary for extraction results
        for section_name, model_class in self.extraction_models.items():
            # Initialize section results
            results[section_name] = {k: None for k in model_class.model_fields.keys()}
            
            # Extract data
            extracted_data = self.extraction_agent.extract_data(
                markdown_content, 
                header_info, 
                section_name, 
                model_class
            )
            
            if not extracted_data:
                console.print(f"[yellow]⚠[/yellow] No data extracted for {section_name}")
                continue
                
            # Step 3: Validate data
            validated_result = self.validation_agent.validate(
                extracted_data,
                markdown_content,
                header_info,
                section_name,
                model_class
            )
            
            # Check validation confidence
            if (validated_result and 
                hasattr(validated_result, 'ValidationConfidence') and 
                validated_result.ValidationConfidence >= extraction_confidence_threshold and
                hasattr(validated_result, 'ValidatedData')):
                
                # Update results with validated data
                ValidatedData = validated_result.ValidatedData.model_dump()
                for field, value in ValidatedData.items():
                    if value is not None:
                        results[section_name][field] = value
                
                # Add validation confidence to results
                results[section_name]['ValidationConfidence'] = validated_result.ValidationConfidence
                        
                console.print(f"[green]✓[/green] Validated {section_name} with confidence {validated_result.ValidationConfidence:.2f}")
                
                # Log corrections if any
                if hasattr(validated_result, 'CorrectionsMade') and validated_result.CorrectionsMade:
                    console.print(f"[blue]Corrections made:[/blue]")
                    for correction in validated_result.CorrectionsMade:
                        console.print(f"  • {correction}")
            else:
                # Use original extraction if validation fails
                console.print(f"[yellow]⚠[/yellow] Low validation confidence for {section_name}, using original extraction")
                result_data = extracted_data.model_dump()
                for field, value in result_data.items():
                    if value is not None:
                        results[section_name][field] = value
                
                # Add a lower validation confidence
                if validated_result and hasattr(validated_result, 'ValidationConfidence'):
                    results[section_name]['ValidationConfidence'] = validated_result.ValidationConfidence
                else:
                    results[section_name]['ValidationConfidence'] = 0.5  # Default medium confidence
        
        # Add extraction results to ordered results
        for section_name, section_data in results.items():
            ordered_results[section_name] = section_data
        
        return ordered_results


# Helper function to get extraction models from config manager
def create_extraction_models_dict(config_manager: ConfigurationManager) -> Dict[str, Type[BaseExtraction]]:
    """
    Create a dictionary mapping section names to model classes.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Dictionary mapping section names to model classes
    """
    from ..dynamic_model_factory import create_extraction_models_dict
    return create_extraction_models_dict(config_manager)
