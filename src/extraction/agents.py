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
            
            if result and hasattr(result, 'header_start_line') and result.header_start_line is not None:
                console.print(f"[green]✓[/green] Header detection successful")
                
                # Store header_detection_confidence as an attribute for internal use
                # but don't include it in the returned model
                if hasattr(result, 'header_detection_confidence'):
                    # Store the confidence value as an attribute for internal use
                    result._header_detection_confidence = result.header_detection_confidence
                    
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
        # Example 1: Simple table with clear headers
        example1_input = """| Column1 | Column2 | Column3 |
|---------|---------|---------|
| data1   | data2   | data3   |
| data4   | data5   | data6   |"""
        
        # Create example output with header_detection_confidence
        example1_output_dict = {
            "header_start_line": 0,
            "header_end_line": 0,
            "content_start_line": 2,
            "header_detection_confidence": 0.95
        }
        
        # Create ContextModel without header_detection_confidence
        example1_output = ContextModel(
            header_start_line=0,
            header_end_line=0,
            content_start_line=2
        )
        
        # Example 2: Table with multi-row headers
        example2_input = """| Main Category | | |
|--------------|--------------|--------------|
| Subcategory1 | Subcategory2 | Subcategory3 |
| data1        | data2        | data3        |
| data4        | data5        | data6        |"""
        
        # Create example output with header_detection_confidence
        example2_output_dict = {
            "header_start_line": 0,
            "header_end_line": 2,
            "content_start_line": 3,
            "header_detection_confidence": 0.9
        }
        
        # Create ContextModel without header_detection_confidence
        example2_output = ContextModel(
            header_start_line=0,
            header_end_line=2,
            content_start_line=3
        )
        
        # Example 3: Table with no clear headers
        example3_input = """| data1 | data2 | data3 |
|-------|-------|-------|
| data4 | data5 | data6 |
| data7 | data8 | data9 |"""
        
        # Create example output with header_detection_confidence
        example3_output_dict = {
            "header_start_line": 0,
            "header_end_line": 0,
            "content_start_line": 1,
            "header_detection_confidence": 0.7
        }
        
        # Create ContextModel without header_detection_confidence
        example3_output = ContextModel(
            header_start_line=0,
            header_end_line=0,
            content_start_line=1
        )
        
        # Create system message
        system_message = {
            "role": "system",
            "content": """You are an expert at analyzing Excel tables converted to markdown format.
Your task is to identify the header rows and content rows in the table.

Guidelines:
1. Analyze the first 15 rows of the table to identify patterns
2. Determine where the header starts and ends
3. Determine where the actual content starts
4. Provide a confidence score (0.0-1.0) for your detection
5. Headers often have different formatting or contain column titles
6. Content rows typically contain actual data values

Return your analysis as a JSON object with these fields:
- header_start_line: Line where headers start (0-based index)
- header_end_line: Line where headers end (0-based index)
- content_start_line: Line where content starts (0-based index)
- header_detection_confidence: Your confidence score (0.0-1.0)
"""
        }
        
        # Create examples - use the dictionaries with header_detection_confidence for the examples
        examples = [
            {"role": "user", "content": example1_input},
            {"role": "assistant", "content": json.dumps(example1_output_dict)},
            {"role": "user", "content": example2_input},
            {"role": "assistant", "content": json.dumps(example2_output_dict)},
            {"role": "user", "content": example3_input},
            {"role": "assistant", "content": json.dumps(example3_output_dict)}
        ]
        
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
        header_start_line = 0
        header_end_line = 0
        content_start_line = 1
        
        # Look for separator row (contains mainly dashes or pipes)
        for i, line in enumerate(lines):
            if i > 0 and re.match(r'^[\|\-\+\s]+$', line.strip()):
                # Found separator row, headers are above it
                header_end_line = i - 1
                content_start_line = i + 1
                break
        
        # Look for first row with numeric data (likely content)
        for i, line in enumerate(lines):
            if i > header_end_line:
                # Check if line contains numeric data
                if re.search(r'\d+\.\d+|\d+', line):
                    content_start_line = i
                    break
        
        console.print(f"[yellow]Fallback detection: headers at lines {header_start_line}-{header_end_line}, content starts at line {content_start_line}[/yellow]")
        
        # Create ContextModel without header_detection_confidence
        return ContextModel(
            header_start_line=header_start_line,
            header_end_line=header_end_line,
            content_start_line=content_start_line
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
        header_start = header_info.header_start_line or 0
        header_end = header_info.header_end_line or 0
        content_start = header_info.content_start_line or (header_end + 1)
        
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
"""
        }
        
        # For now, return just the system message
        # In a real implementation, you would add examples specific to each section
        return [system_message]


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
        
    def validate(self, extracted_data: BaseExtraction, section_name: str, 
                model_class: Type[BaseExtraction]) -> Optional[Any]:
        """
        Validate extracted data for a specific section.
        
        Args:
            extracted_data: Data to validate
            section_name: Name of the section
            model_class: Model class for this section
            
        Returns:
            Validated data with confidence score, or None if validation failed
        """
        console.print(f"[blue]Validating {section_name} data...[/blue]")
        
        try:
            # Convert extracted data to JSON string
            json_input = json.dumps(extracted_data.model_dump())
            
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
            
            if result and hasattr(result, 'validation_confidence'):
                console.print(f"[green]✓[/green] {section_name} validation successful with confidence {result.validation_confidence:.2f}")
                return result
            else:
                console.print(f"[yellow]⚠[/yellow] {section_name} validation returned incomplete data")
                return None
                
        except Exception as e:
            console.print(f"[red]Error in {section_name} validation: {str(e)}[/red]")
            return None
        
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
            validated_data: model_class
            
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
Your task is to validate and correct the extracted data.

Guidelines:
1. Analyze the extracted data for inconsistencies or errors
2. Correct any issues found (e.g., formatting, capitalization, etc.)
3. Provide a confidence score (0.0-1.0) for your validation
4. List any corrections you made

Return your validation as a JSON object with these fields:
- validated_data: The corrected data
- validation_confidence: Your confidence score (0.0-1.0)
- corrections_made: List of corrections you made
"""
        }
        
        # For now, return just the system message
        # In a real implementation, you would add examples
        return [system_message]


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
        results = {}
        
        # Step 1: Detect headers
        header_info = self.header_agent.detect_headers(markdown_content)
        
        # Check confidence and use fallback if needed
        if (header_info and 
            hasattr(header_info, '_header_detection_confidence') and 
            header_info._header_detection_confidence >= 0.8):
            console.print(f"[green]✓[/green] Header detection successful with confidence {header_info._header_detection_confidence:.2f}")
        else:
            console.print("[yellow]⚠[/yellow] Low confidence in header detection, using fallback strategy")
            header_info = self.header_agent.fallback_header_detection(markdown_content)
        
        # Ensure we have the Context section initialized
        if "Context" in EXTRACTION_MODELS:
            if "Context" not in results:
                results["Context"] = {k: None for k in EXTRACTION_MODELS["Context"]().model_fields.keys()}
            
            # Copy header detection information to Context section (excluding header_detection_confidence)
            if header_info:
                for field in ["header_start_line", "header_end_line", "content_start_line"]:
                    if hasattr(header_info, field) and getattr(header_info, field) is not None:
                        results["Context"][field] = getattr(header_info, field)
        
        # Step 2: Extract each section
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
                section_name,
                model_class
            )
            
            # Check validation confidence
            if (validated_result and 
                hasattr(validated_result, 'validation_confidence') and 
                validated_result.validation_confidence >= 0.8 and
                hasattr(validated_result, 'validated_data')):
                
                # Update results with validated data
                validated_data = validated_result.validated_data.model_dump()
                for field, value in validated_data.items():
                    if value is not None:
                        results[section_name][field] = value
                
                # Add validation confidence to results
                results[section_name]['validation_confidence'] = validated_result.validation_confidence
                        
                console.print(f"[green]✓[/green] Validated {section_name} with confidence {validated_result.validation_confidence:.2f}")
                
                # Log corrections if any
                if hasattr(validated_result, 'corrections_made') and validated_result.corrections_made:
                    console.print(f"[blue]Corrections made:[/blue]")
                    for correction in validated_result.corrections_made:
                        console.print(f"  • {correction}")
            else:
                # Use original extraction if validation fails
                console.print(f"[yellow]⚠[/yellow] Low validation confidence for {section_name}, using original extraction")
                result_data = extracted_data.model_dump()
                for field, value in result_data.items():
                    if value is not None:
                        results[section_name][field] = value
                
                # Add a lower validation confidence
                if validated_result and hasattr(validated_result, 'validation_confidence'):
                    results[section_name]['validation_confidence'] = validated_result.validation_confidence
                else:
                    results[section_name]['validation_confidence'] = 0.5  # Default medium confidence
        
        return results
