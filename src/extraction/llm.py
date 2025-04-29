"""LLM extraction utilities for Excel Header Mapper."""
import os
import uuid
import json # Added json import
from typing import List, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage
)
from ..utils.display import console, logger # Ensure logger is imported
from ..models import (
    AppConfig,
    Example
    # EXTRACTION_MODELS removed as it's loaded dynamically now
)
# Import PipelineException and LLMInteractionError
from ..exceptions import PipelineException, LLMInteractionError
from ..config_manager import get_configuration_manager
from ..dynamic_model_factory import create_extraction_models_dict
from .agent_utils import get_tokenizer, count_tokens # Ensure token helpers are imported

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
            # Log warning instead of printing directly
            logger.warning(f"Failed to initialize Langfuse: {str(e)}")

    return llm

def configure_llm_classification(config: AppConfig):
    """
    Configure the LLM specifically for classification tasks.

    Args:
        config (AppConfig): Application configuration containing classification_model settings

    Returns:
        ChatOpenAI: Configured LLM instance for classification
    """
    # Load environment variables from .env file
    load_dotenv()

    # Use classification-specific model config
    class_config = config.classification_model

    # Initialize LLM
    llm = ChatOpenAI(
        model_name=class_config.model_name,
        base_url=class_config.base_url,
        api_key=class_config.api_key,
        temperature=class_config.temperature,
        max_retries=class_config.max_retries
    )

    # Add Langfuse monitoring if enabled (using the same global setting)
    if os.getenv("LANGFUSE_ENABLED", "false").lower() == "true":
        try:
            from langfuse.callback import CallbackHandler
            langfuse_handler = CallbackHandler()
            # No need to auth_check again if done for main LLM
            llm.callbacks = [langfuse_handler]
            # logger.info("Langfuse monitoring enabled for Classification LLM") # Optional log
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse for Classification LLM: {str(e)}")

    return llm

def create_extraction_prompt(section_name=None):
    """
    Create a prompt template for extraction with optional section focus.

    Args:
        section_name (str, optional): If provided, focuses extraction on this specific section

    Returns:
        ChatPromptTemplate: Configured prompt template
    """
    # Create a template structure with placeholders for system, examples, and human messages
    return ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="system_message"), # Placeholder for system message
        MessagesPlaceholder(variable_name="examples"),      # Placeholder for examples
        ("human", "{text}")                                # Placeholder for user text (human input)
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
        # Use logger instead of console.print
        logger.debug("Converting example to messages...")
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

        logger.debug(f"Converted example to {len(messages)} messages")
        return messages

    except Exception as e:
        # Wrap unexpected error in a PipelineException
        raise PipelineException(
            message=f"Error converting tool example to messages: {e}",
            error_code="TOOL_EXAMPLE_CONVERSION_FAILED",
            context={"exception_type": e.__class__.__name__}
        )

# initialize_llm_pipeline function removed - not used by the dynamic pipeline

def call_llm_with_json_response(prompt, model_class, llm):
    """
    Call the LLM directly with a prompt and parse the JSON response.

    Args:
        prompt (str): The complete prompt to send to the LLM
        model_class (Type): Pydantic model class for the expected response
        llm (ChatOpenAI): Configured LLM instance

    Returns:
        model_class instance or None: Parsed response or None if parsing failed
    """
    # Create a trace for this call if Langfuse is enabled
    trace_id = None
    if hasattr(llm, 'callbacks') and llm.callbacks:
        for callback in llm.callbacks:
            if hasattr(callback, 'create_trace'):
                try:
                    trace = callback.create_trace(
                        name="direct_llm_call",
                        metadata={
                            "model_class": model_class.__name__
                        }
                    )
                    trace_id = trace.id
                except Exception as e:
                    logger.warning(f"Failed to create Langfuse trace: {str(e)}")

    try:
        # Create a direct message to the LLM
        message = HumanMessage(content=prompt)

        # Create a runnable specifically for this model
        direct_runnable = llm.with_structured_output(
            schema=model_class,
            include_raw=False,
        )

        # Run the extraction
        result = direct_runnable.invoke([message])

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
            # Enhanced Exception for invalid structure
            raise LLMInteractionError(
                message="LLM returned invalid result structure in direct call.",
                error_code="LLM_INVALID_RESULT_STRUCTURE",
                context={"model_class": model_class.__name__}
            )

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
        # Enhanced Exception for runtime errors during LLM call
        # Wrap original exception
        raise LLMInteractionError(
            message=f"Error during direct LLM call: {e}",
            error_code="LLM_DIRECT_CALL_FAILED",
            context={"model_class": model_class.__name__, "exception_type": e.__class__.__name__}
        )


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
                    logger.warning(f"Failed to create Langfuse trace: {str(e)}") # Use logger.warning

    try:
        # Create a section-specific prompt
        section_prompt = create_extraction_prompt(section_name)

        # Create a runnable specifically for this model
        section_runnable = section_prompt | llm.with_structured_output(
            schema=model_class,
            include_raw=False,
        )

        # Extract system message and examples from the input 'messages' list
        system_prompt_obj = None
        example_messages = []
        if messages:
            if messages[0]['role'] == 'system':
                # Import SystemMessage if not already imported
                from langchain_core.messages import SystemMessage
                system_prompt_obj = SystemMessage(content=messages[0]['content'])
                example_messages = messages[1:] # The rest are examples
            else: # Assume no system message provided, all are examples
                example_messages = messages

        # Prepare invocation dictionary
        invoke_input = {
            "text": markdown_content, # This is the user/human message content
            "examples": example_messages
        }
        # Add system message to input if it exists
        if system_prompt_obj:
            invoke_input["system_message"] = [system_prompt_obj]
        else:
            # Provide a default minimal system message if none was in messages
            from langchain_core.messages import SystemMessage
            invoke_input["system_message"] = [SystemMessage(content="You are a helpful assistant.")]

        # --- NEW: Accurate Input Token Counting ---
        # Reconstruct the full prompt text to count tokens accurately
        full_prompt_text = ""
        if system_prompt_obj:
            full_prompt_text += system_prompt_obj.content + "\n" # Add system message content
        for msg in example_messages:
             # Assuming example messages are HumanMessage and AIMessage
             if isinstance(msg, HumanMessage):
                  full_prompt_text += msg.content + "\n"
             elif isinstance(msg, AIMessage):
                  full_prompt_text += msg.content + "\n" # Or format as needed
             elif isinstance(msg, dict) and 'content' in msg: # Handle dict format if necessary
                  full_prompt_text += msg['content'] + "\n"

        full_prompt_text += markdown_content # Add the main human input

        input_tokens = count_tokens(get_tokenizer(), full_prompt_text)
        logger.info(f"LLM Call Input Tokens ({section_name}): {input_tokens}")
        # --- END NEW ---


        # Run the extraction
        result = section_runnable.invoke(invoke_input)

        # --- NEW: Accurate Output Token Counting ---
        output_tokens = count_tokens(get_tokenizer(), json.dumps(result.model_dump()) if result else "")
        logger.info(f"LLM Call Output Tokens ({section_name}): {output_tokens}")
        # --- END NEW ---


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
            # Enhanced Exception for invalid structure
            raise LLMInteractionError(
                message=f"LLM returned invalid result structure for section '{section_name}'.",
                error_code="LLM_INVALID_RESULT_STRUCTURE",
                context={"section": section_name, "model_class": model_class.__name__}
            )

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

        # Enhanced Exception for runtime errors during section extraction
        # Wrap original exception
        # Note: logger.exception removed as ErrorManager will handle logging
        raise LLMInteractionError(
            message=f"Error extracting section '{section_name}': {e}",
            error_code="LLM_SECTION_EXTRACTION_FAILED",
            context={
                "section": section_name,
                "model_class": model_class.__name__,
                "exception_type": e.__class__.__name__
            }
        )
