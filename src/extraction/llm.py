"""LLM extraction utilities for Excel Header Mapper."""
import os
import uuid
from typing import List, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    AIMessage, 
    BaseMessage, 
    HumanMessage, 
    SystemMessage, 
    ToolMessage
)
from ..utils.display import console
from ..models import (
    AppConfig, 
    Data, 
    Example, 
    IdentifierModel,
    EXTRACTION_MODELS
)

def configure_llm(config: AppConfig):
    """
    Configure the LLM based on application config.
    
    Args:
        config (AppConfig): Application configuration
        
    Returns:
        ChatOpenAI: Configured LLM instance with optional Langfuse monitoring
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model_name=config.model.model_name,
        base_url=config.model.base_url,
        api_key=config.model.api_key,
        temperature=config.model.temperature,
        max_retries=config.model.max_retries
    )
    
    # Add Langfuse monitoring if enabled in .env
    if os.getenv("LANGFUSE_ENABLED", "false").lower() == "true":
        try:
            from langfuse.callback import CallbackHandler
            
            # Create Langfuse callback handler (will use env vars automatically)
            langfuse_handler = CallbackHandler()
            
            # Test connection (optional, can be removed in production)
            try:
                langfuse_handler.auth_check()
                console.print("[green]✓[/green] Langfuse authentication successful")
            except Exception as auth_error:
                console.print(f"[yellow]⚠[/yellow] Langfuse authentication failed: {str(auth_error)}")
            
            # Add callback handler to LLM
            llm.callbacks = [langfuse_handler]
            
            console.print("[green]✓[/green] Langfuse monitoring enabled")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed to initialize Langfuse: {str(e)}")
    
    return llm

def create_extraction_prompt(section_name=None):
    """
    Create a prompt template for extraction with optional section focus.
    
    Args:
        section_name (str, optional): If provided, focuses extraction on this specific section
        
    Returns:
        ChatPromptTemplate: Configured prompt template
    """
    # Customize based on whether we're extracting a specific section
    section_text = f"the {section_name} section" if section_name else "a structured JSON output"
    section_focus = f"that correspond to the {section_name} section" if section_name else "in the given input table"
    
    return ChatPromptTemplate.from_messages([
        ("system", 
         f"""You are an expert data extraction algorithm. Your task is to map the column headers from an Excel table to the corresponding fields in {section_text}.  

### Instructions:  
- Identify the column headers {section_focus}.
- Match each header to the corresponding attribute in the JSON output.
- Extract only relevant data under each header and assign it to the correct JSON field.
- If a required value is missing or unavailable, return `null` for that field.
- Ensure accuracy by strictly following the column-to-attribute mapping.
"""
        ),
        MessagesPlaceholder("examples"),
        ("human", "{text}"),
    ])

def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """
    Convert an example into a list of messages that can be fed into an LLM.
    
    Args:
        example (Example): Example with input and expected tool calls
        
    Returns:
        List[BaseMessage]: List of messages for LLM
    """
    try:
        console.print("[dim]Converting example to messages...[/dim]")
        messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
        
        tool_calls = []
        for tool_call in example["tool_calls"]:
            tool_calls.append(
                {
                    "id": str(uuid.uuid4()),
                    "args": tool_call.model_dump(),
                    "name": tool_call.__class__.__name__,
                },
            )
        
        messages.append(AIMessage(content="", tool_calls=tool_calls))
        
        tool_outputs = example.get("tool_outputs") or [
            "You have correctly called this tool."
        ] * len(tool_calls)
        
        for output, tool_call in zip(tool_outputs, tool_calls):
            messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
            
        console.print(f"[green]Converted example to {len(messages)} messages[/green]")
        return messages
        
    except Exception as e:
        console.print(f"[red]Error converting example to messages: {str(e)}[/red]")
        raise

def initialize_llm_pipeline(config: AppConfig):
    """
    Initialize the LLM and extraction pipeline.
    
    Args:
        config (AppConfig): Application configuration
        
    Returns:
        tuple: (runnable, messages) - The extraction pipeline and example messages
    """
    # Configure LLM with the provided configuration
    llm = configure_llm(config)
    
    # Create general extraction prompt template (no section focus)
    prompt = create_extraction_prompt()
    
    # Define example data
    example_data = """CODE ISIN FCP	DATE	COUPON	CREDIT D'IMPÔT	VL APRES DETACH.	Coefficient	Ouvrant à PL	NET du PL
FR0007436969	UFF AVENIR SECURITE	12/27/89	5,53	0,03	75,42	1,0732752522	NR	5,52627687485613 €
FR0007437124	UFF AVENIR DIVERSIFIE	12/27/89	1,30	0,65	90,64	1,0143469851	NR	1,30039011703511 €
FR0010180786	UFF AVENIR FRANCE	12/27/89	1,27	0,41	106,70	1,0118588370	NR	1,26532684307051 €
FR0007437124	UFF AVENIR DIVERSIFIE	12/26/90	0,04	0,02	74,65	1,0005310024	NR	0,0396367444817267 €
FR0007436969	UFF AVENIR SECURITE	12/28/90	7,65	0,04	77,38	1,0988623769	NR	7,65043 €
FR0010180786	UFF AVENIR FRANCE	12/15/91	1,13	0,56	90,24	1,0125183721	NR	1,12964721772921 €
FR0007436969	UFF AVENIR SECURITE	12/23/91	4,73	0,01	78,86	1,0600243577	NR	4,73354198522159 €"""

    # Create a proper example with the consolidated model
    example_tool_call = IdentifierModel(
        code="CODE ISIN",
        code_type="Isin",
        currency="EUR",
        cic_code=None
    )

    # Create examples in the format expected by tool_example_to_messages
    examples = [
        {
            "input": example_data,
            "tool_calls": [Data(data_extracted=[example_tool_call])]
        }
    ]

    # Convert examples to messages format
    messages = []
    for example in examples:
        messages.extend(tool_example_to_messages(example))

    # Create extraction pipeline
    runnable = prompt | llm.with_structured_output(
        schema=Data,
        include_raw=False,
    )
    
    return runnable, messages

def extract_section(markdown_content, section_name, model_class, messages, llm):
    """
    Extract a specific section using a specific model.
    
    Args:
        markdown_content (str): Input markdown content to extract from
        section_name (str): Name of the section to extract
        model_class (BaseModel): Pydantic model class for this section
        messages (List): Example messages for the LLM
        llm (ChatOpenAI): Configured LLM instance
        
    Returns:
        BaseModel or None: Extracted data or None if extraction failed
    """
    # Create a trace for this extraction if Langfuse is enabled
    trace_id = None
    if hasattr(llm, 'callbacks') and llm.callbacks:
        for callback in llm.callbacks:
            if hasattr(callback, 'create_trace'):
                try:
                    trace = callback.create_trace(
                        name=f"extract_{section_name}",
                        metadata={
                            "section": section_name,
                            "model_class": model_class.__name__
                        }
                    )
                    trace_id = trace.id
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Failed to create Langfuse trace: {str(e)}")
    
    try:
        # Create a section-specific prompt
        section_prompt = create_extraction_prompt(section_name)
        
        # Create a runnable specifically for this model
        section_runnable = section_prompt | llm.with_structured_output(
            schema=model_class,
            include_raw=False,
        )
        
        # Run the extraction
        result = section_runnable.invoke(
            {
                "text": markdown_content,
                "examples": messages,
            }
        )
        
        # Validate the result structure
        if not hasattr(result, 'model_dump'):
            # End trace with error if it exists
            if trace_id:
                for callback in llm.callbacks:
                    if hasattr(callback, 'update_trace'):
                        try:
                            callback.update_trace(
                                trace_id=trace_id,
                                status="error",
                                metadata={"error": "Invalid result structure"}
                            )
                        except Exception:
                            pass
            return None
        
        # End trace with success if it exists
        if trace_id:
            for callback in llm.callbacks:
                if hasattr(callback, 'update_trace'):
                    try:
                        callback.update_trace(
                            trace_id=trace_id,
                            status="success"
                        )
                    except Exception:
                        pass
            
        return result
        
    except Exception as e:
        # Log error to trace if it exists
        if trace_id:
            for callback in llm.callbacks:
                if hasattr(callback, 'update_trace'):
                    try:
                        callback.update_trace(
                            trace_id=trace_id,
                            status="error",
                            metadata={"error": str(e)}
                        )
                    except Exception:
                        pass
        
        console.print(f"[red]Error extracting {section_name}: {str(e)}[/red]")
        return None
