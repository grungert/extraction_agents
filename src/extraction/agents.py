"""LLM-based agents for Excel Header Mapper."""
import json
from typing import Dict, List, Optional, Type, Any
import re

from ..utils.display import console
from ..models import (
    AppConfig,
    BaseExtraction,
    ContextModel,
    ValidationResult,
    EXTRACTION_MODELS
)
from .llm import extract_section, configure_llm
from ..examples.header_examples import _get_header_examples
from ..examples.extraction_examples import get_extraction_examples, format_extraction_messages


class HeaderValidationAgent:
    """Agent for validating header detection results."""
    
    def __init__(self, llm):
        """
        Initialize the header validation agent.
        
        Args:
            llm: LLM instance to use for validation
        """
        self.llm = llm
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
{markdown_content[:500]}

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
        # Create system message
        system_message = {
            "role": "system",
            "content": """You are an expert at validating header detection in Excel tables.
Your task is to validate and correct the detected header positions by comparing them with the original table.

Input Format:
The input contains two sections:
1. Original Table: The Excel table in markdown format
2. Detected Headers: The header positions detected by the header detection agent

Guidelines:
1. Compare the detected header positions with the original table to verify accuracy
2. Check if the HeaderStartLine, HeaderEndLine, and ContentStartLine make sense for the table
3. Use examples as guides for common header patterns
4. Correct any issues found in the header positions
5. Provide a confidence score (0.0-1.0) for your validation
6. Use examples to guide your validation process

Return your validation as a JSON object with these fields:
- HeaderStartLine: Line where headers start 
- HeaderEndLine: Line where headers end 
- ContentStartLine: Line where content starts 
- ValidationConfidence: Your confidence score (0.0-1.0)
"""
        }
        
        # Get shared examples
        header_examples = _get_header_examples()
        
        # Create examples for the LLM - format them for validation
        examples = []
        for idx, example in enumerate(header_examples[:2], start=1):  # Add index with enumerate
            validation_input = f"""
# Example {idx}
# Input Table 
{example["table"]}

# Output JSON
```json
{json.dumps(example["json"], indent=2)}
```"""

            examples.extend([
                {"role": "user", "content": validation_input},
                {"role": "assistant", "content": json.dumps(example["json"])}
            ])
    
        # Combine system message and examples
        return [system_message] + examples


class HeaderDetectionAgent:
    """Agent for detecting header positions in Excel sheets."""
    
    def __init__(self, llm):
        """
        Initialize the header detection agent.
        
        Args:
            llm: LLM instance to use for detection
        """
        self.llm = llm
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
        # Create system message
        system_message = {
            "role": "system",
            "content": """You are an expert at analyzing Excel tables converted to markdown format.
Your task is to identify the header rows and content rows in the table.

Guidelines:
1. Analyze the first 15 rows of the table to identify patterns
2. Use examples as guides for common header patterns
3. Determine where the header starts and ends
4. Determine where the actual content starts
5. Provide a confidence score (0.0-1.0) for your detection
6. Headers often have different formatting or contain column titles
7. Content rows typically contain actual data values
8. Use examples to guide your detection process

Return your analysis as a JSON object with these fields:
- HeaderStartLine: Line where headers start
- HeaderEndLine: Line where headers end 
- ContentStartLine: Line where content starts 
- ValidationConfidence: Your confidence score (0.0-1.0)
"""
        }
        
        # Get shared examples
        header_examples = _get_header_examples()
        
        # Create examples for the LLM - format them for validation
        examples = []
        for idx, example in enumerate(header_examples[:2], start=1):  # Add index with enumerate
            validation_input = f"""
# Example {idx}
# Input Table              
{example["table"]}

# Output JSON
```json
{json.dumps(example["json"], indent=2)}
```"""

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


class LLMExtractionAgent:
    """Agent for extracting structured data from Excel sheets."""
    
    def __init__(self, llm):
        """
        Initialize the extraction agent.
        
        Args:
            llm: LLM instance to use for extraction
        """
        self.llm = llm
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
        # Create system message based on section
        system_message = {
            "role": "system",
            "content": f"""You are an expert at extracting {section_name} information from Excel tables.
Your task is to identify the column headers that correspond to {section_name} fields.

Guidelines:
1. Analyze the table structure to identify relevant columns
2. Match column headers to the corresponding fields in the {section_name} section
3. Return only the headers, not the actual data values
4. If a field is not present in the table, return null for that field
5. Provide a confidence score (0.0-1.0) for your extraction
6. Use examples to guide your extraction process

Return your extraction as a JSON object with these fields:
- All the fields from the {section_name} model
- ValidationConfidence: Your confidence score (0.0-1.0)
"""
        }
        
        # Get examples for this section - use the model class if available
        if section_name in EXTRACTION_MODELS:
            model_class = EXTRACTION_MODELS[section_name]
            examples = get_extraction_examples(model_class)
        else:
            examples = get_extraction_examples(section_name)
        
        # Format examples for extraction
        return format_extraction_messages(examples, system_message)


class ValidationAgent:
    """Agent for validating and correcting extracted data."""
    
    def __init__(self, llm):
        """
        Initialize the validation agent.
        
        Args:
            llm: LLM instance to use for validation
        """
        self.llm = llm
        self.messages = self._create_validation_messages()
        
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
            
            # Validate data
            result = extract_section(
                markdown_content=json_input,
                section_name=f"{section_name}Validation",
                model_class=validation_model,
                messages=self.messages,
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
        
    def _create_validation_messages(self) -> List[Dict]:
        """
        Create example messages for validation.
        
        Returns:
            List of message dictionaries for the LLM
        """
        # Create system message
        system_message = {
            "role": "system",
            "content": """You are an expert at validating extracted data from Excel tables.
Your task is to validate and correct the extracted data by comparing it with the original table.

Input Format:
The input contains two sections:
1. Original Table: The Excel table in markdown format
2. Extracted Data: The data extracted from the table in JSON format

Guidelines:
1. Compare the extracted data with the original table to verify accuracy
2. Check if all relevant fields have been correctly identified
3. Correct any issues found (e.g., wrong mappings, formatting, capitalization, etc.)
4. Provide a confidence score (0.0-1.0) for your validation
5. List any corrections you made
6. Use examples to guide your validation process

Return your validation as a JSON object with these fields:
- ValidatedData: The corrected data
- ValidationConfidence: Your confidence score (0.0-1.0)
- CorrectionsMade: List of corrections you made
"""
        }
        
        # Get examples for validation
        from ..examples.extraction_examples import get_extraction_examples, format_validation_messages
        
        # Get examples for all model classes in EXTRACTION_MODELS
        all_examples = []
        for model_name, model_class in EXTRACTION_MODELS.items():
            examples = get_extraction_examples(model_class)
            all_examples.extend(examples)
        
        # Format examples for validation
        return format_validation_messages(all_examples, system_message)


class AgentPipelineCoordinator:
    """Coordinator for the agent pipeline."""
    
    def __init__(self, config: AppConfig):
        """
        Initialize the agent pipeline coordinator.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.llm = configure_llm(config)
        self.header_agent = HeaderDetectionAgent(self.llm)
        self.header_validation_agent = HeaderValidationAgent(self.llm)
        self.extraction_agent = LLMExtractionAgent(self.llm)
        self.validation_agent = ValidationAgent(self.llm)
        
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
        
        # Check if header validation failed or has low confidence
        if not validated_header_info or (hasattr(validated_header_info, 'ValidationConfidence') and 
                                         validated_header_info.ValidationConfidence < 0.7):
            console.print("[red]Header validation failed or has low confidence[/red]")
            return {"error": "header detection failed"}
        
        # Use validated header info
        header_info = validated_header_info
        header_confidence = validated_header_info.ValidationConfidence if hasattr(validated_header_info, 'ValidationConfidence') else 0.7
        
        # Extract file information from source_file
        FileName = os.path.splitext(os.path.basename(source_file))[0] if source_file else None
        file_ext = os.path.splitext(source_file)[1].lower() if source_file else None
        FileType = file_ext.lstrip('.') if file_ext else None
        
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
        
        # Step 2: Extract each section
        results = {}  # Temporary dictionary for extraction results
        for section_name, model_class in EXTRACTION_MODELS.items():
            # Initialize section results
            results[section_name] = {k: None for k in model_class().model_fields.keys()}
            
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
                validated_result.ValidationConfidence >= 0.8 and
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
