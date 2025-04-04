# 💡 Enhanced System Extension Plan: LLM-Based Agents Pipeline

---

## 🎯 **Key Principle**
All intelligent components in the pipeline — **extraction**, **header detection**, **header validation**, and **data validation** — are implemented as **LLM Agents**. Each uses a well-structured prompt, and runs on a **local LLM (e.g., Gemma)** via your `GemmaLLMAgent`.

---

## Issues and steps:

1.  When trying to read Excel files, if there is a cell with formatting (for example: 1.2 $), I couldn't manage to read the currency sign. This affects the extraction side.

2.  Create a dynamic app to import new settings for agents.

3.  Modify the JSON to have a structure based on the Excel sheet (open to the possibility that data can be in multiple sheets).

4.  Modify the JSON to include a confidence level and processing time.

5.  Work on the agents' prompts.

## 🔄 Pipeline Architecture

```mermaid
flowchart TD
    Input[Excel/CSV Input] --> Convert[Convert to Markdown]
    Convert --> HeaderDetect[HeaderDetectionAgent]
    
    HeaderDetect --> HeaderValidate[HeaderValidationAgent]
    HeaderValidate -->|Confidence ≥ 0.7| Extract[LLMExtractionAgent]
    HeaderValidate -->|Confidence < 0.7| Error[Error: Header Detection Failed]
    
    Extract --> Validate[ValidationAgent]
    
    Validate -->|Confidence ≥ 0.8| Output[JSON Output]
    Validate -->|Confidence < 0.8| UseOriginal[Use Original Extraction]
    UseOriginal --> Output
    
    subgraph "Integration with Existing Code"
        HeaderDetect -.-> Process[processing.py]
        HeaderValidate -.-> Process
        Extract -.-> Process
        Validate -.-> Process
        Process -.-> Output
    end
```

---

## 🧠 Agent Specifications

### 🧩 1. **HeaderDetectionAgent (LLM Agent)**

#### ✅ Purpose:
- Analyze the **first 15 rows** of a sheet (provided in markdown format)
- Determine:
  - `HeaderStartLine`
  - `HeaderEndLine`
  - `ContentStartLine`
  - `ValidationConfidence` (0.0–1.0)

#### 🔗 Implementation Pattern:
Uses this format internally:
```python
def extract_section(markdown_content, section_name, model_class, messages, llm):
```

- `markdown_content`: First 15 rows of the sheet, converted to markdown
- `section_name`: `"HeaderDetection"`
- `model_class`: `ContextModel` (includes start/end lines + confidence)
- `messages`: Prompt messages for the LLM
- `llm`: Your local GemmaLLMAgent instance

#### 🎯 Output:
Populates `ContextModel`:
```json
{
  "HeaderStartLine": 3,
  "HeaderEndLine": 4,
  "ContentStartLine": 5,
  "ValidationConfidence": 0.94
}
```

#### 🔄 Class Implementation:
```python
class HeaderDetectionAgent:
    def __init__(self, llm):
        self.llm = llm
        self.messages = self._create_example_messages()
        
    def detect_headers(self, markdown_content):
        """Detect header positions in markdown content."""
        return extract_section(
            markdown_content=markdown_content,
            section_name="HeaderDetection",
            model_class=ContextModel,
            messages=self.messages,
            llm=self.llm
        )
        
    def _create_example_messages(self):
        # Create example messages for few-shot learning
        # Return formatted messages for the LLM
```

---

### 🧩 2. **HeaderValidationAgent (LLM Agent)**

#### ✅ Purpose:
- Validate the header detection results against the original table
- Correct any issues in the header positions
- Provide a confidence score for the validation

#### 🔗 Implementation Pattern:
```python
def extract_section(markdown_content, section_name, model_class, messages, llm)
```

#### 🎯 Input:
```
# Original Table
| Column1 | Column2 | Column3 |
|---------|---------|---------|
| data1   | data2   | data3   |
| data4   | data5   | data6   |

# Detected Headers
```json
{"HeaderStartLine": 0, "HeaderEndLine": 0, "ContentStartLine": 2}
```
```

#### 🎯 Output:
```json
{
  "HeaderStartLine": 0,
  "HeaderEndLine": 0,
  "ContentStartLine": 2,
  "ValidationConfidence": 0.95
}
```

#### 🔄 Class Implementation:
```python
class HeaderValidationAgent:
    def __init__(self, llm):
        self.llm = llm
        self.messages = self._create_validation_messages()
        
    def validate(self, header_info, markdown_content):
        """Validate header detection results against the original table."""
        # Convert header info to JSON string
        header_json = json.dumps(header_info.model_dump())
        
        # Create a combined input with both the table and header info
        combined_input = f"""
# Original Table
{markdown_content[:500]}

# Detected Headers
```json
{header_json}
```
"""
        
        return extract_section(
            markdown_content=combined_input,
            section_name="HeaderValidation",
            model_class=ContextModel,
            messages=self.messages,
            llm=self.llm
        )
        
    def _create_validation_messages(self):
        # Create example messages for validation
        # Return formatted messages for the LLM
```

---

### 🧩 3. **LLMExtractionAgent (Existing Agent)**

#### ✅ Purpose:
- Process structured markdown (headers + 10 rows of content)
- Extract data for each section (identifier, denomination, etc.)
- Uses few-shot prompting and validation inside the prompt design

#### 🔗 Implementation Pattern:
```python
def extract_section(markdown_content, section_name, model_class, messages, llm)
```

#### 🔄 Enhanced Implementation:
```python
class LLMExtractionAgent:
    def __init__(self, llm):
        self.llm = llm
        self.section_messages = {}
        
    def extract_data(self, markdown_content, header_info, section_name, model_class):
        """Extract data for a specific section."""
        # Use header_info to focus on relevant rows
        focused_content = self._focus_content(markdown_content, header_info)
        
        # Get or create example messages for this section
        if section_name not in self.section_messages:
            self.section_messages[section_name] = self._create_section_messages(section_name)
            
        return extract_section(
            markdown_content=focused_content,
            section_name=section_name,
            model_class=model_class,
            messages=self.section_messages[section_name],
            llm=self.llm
        )
        
    def _focus_content(self, markdown_content, header_info):
        """Focus on relevant rows based on header detection."""
        # Implementation to extract header + 10 rows of content
        
    def _create_section_messages(self, section_name):
        """Create example messages for a specific section."""
        # Implementation to create section-specific examples
```

---

### 🧩 4. **ValidationAgent (LLM Agent)**

#### ✅ Purpose:
- Receives initial extracted data per section and the original table
- Validates structure, corrects inconsistencies (via prompt)
- Returns cleaned/confirmed data + **confidence score (0.0–1.0)**

#### 🔗 Implementation Pattern:
```python
def extract_section(markdown_content, section_name, model_class, messages, llm)
```

#### 🎯 Input:
```
# Original Table
| Column1 | Column2 | Column3 |
|---------|---------|---------|
| data1   | data2   | data3   |
| data4   | data5   | data6   |

# Extracted Data
```json
{
  "code": "CODE ISIN",
  "code_type": "Isin",
  "currency": "Euro",
  "cic_code": null
}
```
```

#### 🎯 Output:
```json
{
  "ValidatedData": {
    "code": "CODE ISIN",
    "code_type": "Isin",
    "currency": "EUR",
    "cic_code": null
  },
  "ValidationConfidence": 0.92,
  "CorrectionsMade": ["Fixed currency format from 'Euro' to 'EUR'"]
}
```

#### 🔄 Class Implementation:
```python
class ValidationAgent:
    def __init__(self, llm):
        self.llm = llm
        self.messages = self._create_validation_messages()
        
    def validate(self, extracted_data, markdown_content, header_info, section_name, model_class):
        """Validate extracted data for a specific section against the original table."""
        # Focus on relevant rows based on header detection
        focused_content = self._focus_content(markdown_content, header_info)
        
        # Convert extracted data to JSON string
        extracted_json = json.dumps(extracted_data.model_dump())
        
        # Create a combined input with both the table and extracted data
        combined_input = f"""
# Original Table
{focused_content}

# Extracted Data
```json
{extracted_json}
```
"""
        
        # Create validation model class that extends the original
        validation_model = self._create_validation_model(model_class)
        
        return extract_section(
            markdown_content=combined_input,
            section_name=f"{section_name}Validation",
            model_class=validation_model,
            messages=self.messages,
            llm=self.llm
        )
        
    def _focus_content(self, markdown_content, header_info):
        """Focus on relevant rows based on header detection."""
        # Implementation to extract header + 10 rows of content
        
    def _create_validation_model(self, model_class):
        """Create a validation model that extends the original model."""
        # Dynamically create a model with ValidationConfidence and CorrectionsMade
        
    def _create_validation_messages(self):
        """Create example messages for validation."""
        # Implementation to create validation examples
```

---

## 🔄 Pipeline Coordinator

### ✅ Purpose:
- Orchestrate the entire extraction pipeline
- Handle agent interactions and error handling
- Integrate with existing processing.py workflow

### 🔄 Implementation:
```python
class AgentPipelineCoordinator:
    def __init__(self, config):
        self.config = config
        self.llm = configure_llm(config)
        self.header_agent = HeaderDetectionAgent(self.llm)
        self.header_validation_agent = HeaderValidationAgent(self.llm)
        self.extraction_agent = LLMExtractionAgent(self.llm)
        self.validation_agent = ValidationAgent(self.llm)
        
    def process_markdown(self, markdown_content, source_file):
        """Process markdown content through the agent pipeline."""
        # Initialize an ordered dictionary to maintain the order of sections
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
        
        # Step 3: Extract each section
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
                
            # Step 4: Validate data
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
```

---

## 🔌 Integration with Existing Code

### 1. **Modify processing.py**:

```python
# In processing.py

def extract_all_sections(markdown_content, source_file, config, llm_pipeline, messages):
    """
    Extract all sections from markdown content using the agent pipeline.
    """
    # Initialize the agent pipeline
    agent_pipeline = AgentPipelineCoordinator(config)
    
    # Process the markdown content
    results = agent_pipeline.process_markdown(markdown_content, source_file)
    
    # Check if header detection failed
    if "error" in results:
        console.print(f"[red]Error: {results['error']}[/red]")
        return results
    
    # Rest of the function remains the same (statistics, JSON output, etc.)
    # ...
    
    return results
```

### 2. **Add New Models**:

```python
# In models.py

# Add validation model base class
class ValidationResult(BaseModel):
    """Base model for validation results."""
    ValidationConfidence: float = Field(0.0, description="Confidence in validation (0.0-1.0)")
    CorrectionsMade: List[str] = Field([], description="List of corrections made during validation")

# Update ContextModel
class ContextModel(BaseExtraction):
    """Model for extracting context information."""
    FileName: Optional[str] = Field(None, description="File name of the Excel document")
    HeaderStartLine: Optional[int] = Field(None, description="Line where headers start (1-based)")
    HeaderEndLine: Optional[int] = Field(None, description="Line where headers end (1-based)")
    ContentStartLine: Optional[int] = Field(None, description="Line where content starts (1-based)")
    FileType: Optional[str] = Field(None, description="File type (xlsx, csv)")
    # ValidationConfidence is inherited from BaseExtraction
```

---

## 🧪 Testing & Evaluation

### 1. **Metrics**:
- Header detection accuracy (compared to manual labeling)
- Header validation effectiveness (% of errors caught and fixed)
- Extraction field accuracy (% of correctly extracted fields)
- Validation effectiveness (% of errors caught and fixed)
- End-to-end processing time

### 2. **Test Cases**:
- Simple sheets with clear headers
- Complex sheets with multi-row headers
- Sheets with missing or ambiguous headers
- Sheets with varying data formats

### 3. **Evaluation Process**:
```python
def evaluate_pipeline(test_files, ground_truth, config):
    """Evaluate the agent pipeline against ground truth data."""
    metrics = {
        "header_detection": {"correct": 0, "total": 0},
        "header_validation": {"fixed": 0, "missed": 0},
        "extraction": {"correct": 0, "total": 0},
        "validation": {"fixed": 0, "missed": 0},
        "processing_time": []
    }
    
    # Implementation of evaluation logic
    # ...
    
    return metrics
```

---

## 🚀 Implementation Plan

1. **Phase 1: Core Agent Implementation**
   - Implement HeaderDetectionAgent
   - Implement HeaderValidationAgent
   - Integrate with existing extraction logic

2. **Phase 2: Validation & Refinement**
   - Enhance ValidationAgent to use original table
   - Add confidence scoring
   - Implement error handling

3. **Phase 3: Pipeline Coordination**
   - Implement AgentPipelineCoordinator
   - Integrate with processing.py
   - Add performance optimizations

4. **Phase 4: Testing & Tuning**
   - Create evaluation framework
   - Test with diverse datasets
   - Tune prompts and confidence thresholds
