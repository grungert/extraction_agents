"""Dynamic LLM-based pipeline coordinator for Excel Header Mapper."""
import json
import time
from typing import Dict, List, Optional, Type, Any, Union
import os
from collections import OrderedDict

from ..utils.display import logger
from src.models import (
    AppConfig,
    BaseExtraction,
    ContextModel,
    ValidationConfig,
)
from src.exceptions import (
    PipelineException,
    ConfigurationError,
    ExtractionError,
    ValidationError,
)
from .llm import configure_llm, configure_llm_classification
from ..config_manager import ConfigurationManager, get_configuration_manager

# Import agents from their new locations
from .agents.classification import DynamicClassificationAgent, DynamicClassificationValidationAgent
from .agents.header import DynamicHeaderDetectionAgent, DynamicHeaderValidationAgent
from .agents.extraction import DynamicExtractionAgent
from .agents.validation import DynamicValidationAgent
from .agents.deduplication import DynamicDeduplicationAgent

# Import utility function from agent_utils
from .agent_utils import create_extraction_models_dict


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
        # from .llm import configure_llm, configure_llm_classification # Already imported above

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
        self.config_manager: Optional[ConfigurationManager] = None # Type hint
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
        ordered_results = OrderedDict()
        start_time = time.perf_counter()  # Start timer

        # --- Step Tracking ---
        current_step = 0
        total_steps = 0 # Will be calculated later

        # --- Classification Steps ---
        doc_name = os.path.splitext(os.path.basename(source_file))[0] if source_file else "Unknown Document"

        try:
            # Step 1: Classification
            current_step += 1
            logger.info(f"--- Starting Pipeline Step {current_step}: Document Classification ---")
            classification_output = self.classification_agent.run(markdown_content, doc_name)
            if not classification_output:
                # Enhanced Exception
                raise ExtractionError(
                    message="Classification agent failed to return output.",
                    error_code="CLASSIFICATION_AGENT_FAILED",
                    context={"doc_name": doc_name}
                )

            # Step 2: Classification Validation
            current_step += 1
            logger.info(f"--- Starting Pipeline Step {current_step}: Classification Validation ---")
            validation_output = self.classification_validation_agent.run(markdown_content, doc_name, classification_output)
            if not validation_output:
                # Enhanced Exception
                raise ValidationError(
                    message="Classification validation agent failed to return output.",
                    error_code="CLASSIFICATION_VALIDATION_FAILED",
                    context={"doc_name": doc_name, "initial_class": classification_output.predicted_class}
                )

            logger.info(f"Classification Validated: Class='{validation_output.predicted_class}', Confidence='{validation_output.confidence}'") # Removed Reason as it might be long

            # Store classification result
            ordered_results["Classification"] = validation_output.model_dump()

            # --- Conditional Logic ---
            validated_class = validation_output.predicted_class

            # Handle cases where classification is uncertain or unsupported
            if validated_class == "None of those":
                logger.warning(f"Skipping extraction because classification result is '{validated_class}'")
                # Add timing info and return early
                end_time = time.perf_counter()
                processing_time = end_time - start_time
                # Ensure Context key exists before adding time
                if "Context" not in ordered_results: ordered_results["Context"] = {}
                ordered_results["Context"]["ProcessingTimeSeconds"] = round(processing_time, 3) # Add time directly if no other context
                logger.info(f"Total processing time (skipped extraction): {processing_time:.3f} seconds")
                return ordered_results
            else:
                # Proceed with dynamic config loading for the validated class
                logger.info(f"Attempting to load configuration for class: '{validated_class}'")
                # Construct the specific config path dynamically
                config_file_name = f"full_config_{validated_class.replace(' ', '_')}.json"
                config_path = os.path.join("config", config_file_name)

                # Check if config file exists (handled by get_configuration_manager now)
                # get_configuration_manager now raises ConfigurationError if not found

                # Initialize config_manager and models FOR THIS RUN
                try:
                    # Import here to avoid circular dependency if models.py imports this file later
                    # from ..config_manager import get_configuration_manager # Already imported above
                    # from ..dynamic_model_factory import create_extraction_models_dict # Imported from agent_utils

                    current_run_config_manager = get_configuration_manager(config_path)
                    current_run_extraction_models = create_extraction_models_dict(
                        current_run_config_manager,
                        include_examples=self.config.include_header_examples_in_prompt
                    )
                    # Store them for this run (needed by agents)
                    self.config_manager = current_run_config_manager
                    self.extraction_models = current_run_extraction_models

                except ConfigurationError:
                    # Re-raise custom exception
                    raise
                except Exception as e:
                     # Enhanced Exception (wrapping original)
                     # Keep original ConfigurationError from get_configuration_manager if possible,
                     # but wrap other exceptions.
                     if isinstance(e, ConfigurationError):
                         # Re-raise if it's already a ConfigurationError with context
                         raise
                     else:
                         logger.exception(f"Unexpected error loading config or models from {config_path}: {e}")
                         raise ConfigurationError(
                             message=f"Failed to load config/models for class '{validated_class}': {e}",
                             error_code="CONFIG_LOAD_FAILED_FOR_CLASS",
                             context={
                                 "doc_name": doc_name,
                                 "classified_class": validated_class,
                                 "config_path": config_path,
                                 "exception_type": e.__class__.__name__
                             }
                         )

                logger.info(f"Proceeding with extraction for class '{validated_class}' using '{config_path}'")

                # --- Calculate Total Steps ---
                num_extraction_sections = len(self.extraction_models)
                deduplication_step = 1 if self.config.enable_deduplication_agent else 0
                total_steps = 2 + 2 + (2 * num_extraction_sections) + deduplication_step # Class+Val, Head+Val, Extract+Val pairs, Dedupe
                logger.info(f"Calculated Total Steps for '{validated_class}': {total_steps}")

                # --- Existing Pipeline Steps (Now run for any valid class with a config file) ---

                # Step 3: Header Detection
                current_step += 1
                logger.info(f"--- Starting Pipeline Step {current_step} of {total_steps}: Header Detection ---")
                # Pass config manager to agent instance for this run
                self.header_agent.config_manager = self.config_manager
                # Reload messages based on new config - use dot notation for prompt file
                self.header_agent.messages = self.header_agent._create_example_messages()
                header_info = self.header_agent.detect_headers(markdown_content)

                if not header_info:
                    # Enhanced Exception
                    raise ExtractionError(
                        message="Header detection agent failed to return header info.",
                        error_code="HEADER_DETECTION_FAILED",
                        context={"doc_name": doc_name, "classified_class": validated_class}
                    )

                # Step 4: Header Validation
                current_step += 1
                logger.info(f"--- Starting Pipeline Step {current_step} of {total_steps}: Header Validation ---")
                self.header_validation_agent.config_manager = self.config_manager
                # Reload messages - use dot notation for prompt file
                self.header_validation_agent.messages = self.header_validation_agent._create_validation_messages()
                validated_header_info = self.header_validation_agent.validate(header_info, markdown_content)

                # Use dot notation for validation config and confidence threshold
                validation_config: ValidationConfig = self.config_manager.get_validation_config()
                header_confidence_threshold = validation_config.confidence_threshold

                header_confidence = getattr(validated_header_info, 'ValidationConfidence', None)

                if not validated_header_info or (header_confidence is not None and header_confidence < header_confidence_threshold):
                    # Enhanced Exception
                    raise ValidationError(
                        message="Header validation failed or confidence below threshold.",
                        error_code="HEADER_VALIDATION_LOW_CONFIDENCE",
                        context={
                            "doc_name": doc_name,
                            "classified_class": validated_class,
                            "confidence_score": header_confidence,
                            "threshold": header_confidence_threshold
                        }
                    )

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

                # Step 5 onwards: Section Extraction & Validation
                logger.info(f"--- Starting Section Processing Phase (Steps {current_step + 1} to {current_step + (2 * num_extraction_sections)}) ---")
                # Pass config manager to agents for this run
                self.extraction_agent.config_manager = self.config_manager
                self.validation_agent.config_manager = self.config_manager
                # Reset messages cache as config changed
                self.extraction_agent.section_messages = {}
                self.validation_agent.section_messages = {}

                results = {}
                # Use dot notation for extraction confidence threshold
                extraction_confidence_threshold = validation_config.confidence_threshold # Assuming same threshold for extraction validation
                section_index = 0 # To track progress within this phase
                for section_name, model_class in self.extraction_models.items():
                    section_index += 1
                    results[section_name] = {k: None for k in model_class.model_fields.keys() if k != 'ValidationConfidence'} # Init without confidence field

                    # Step 5.1, 5.3, etc.: Extraction
                    current_step += 1
                    logger.info(f"--- Starting Step {current_step} of {total_steps}: Extracting Section {section_index}/{num_extraction_sections} ('{section_name}') ---")
                    extracted_data = self.extraction_agent.extract_data(
                        markdown_content, header_info, section_name, model_class
                    )

                    if not extracted_data:
                        logger.warning(f"No data extracted for {section_name}")
                        results[section_name]['ValidationConfidence'] = 0.0 # Mark as low confidence if extraction failed
                        # Skip validation for this section if extraction failed
                        logger.warning(f"Skipping validation for section '{section_name}' due to extraction failure.")
                        # Need to increment step counter even if skipping validation part
                        current_step += 1
                        continue

                    # Step 5.2, 5.4, etc.: Validation
                    current_step += 1
                    logger.info(f"--- Starting Step {current_step} of {total_steps}: Validating Section {section_index}/{num_extraction_sections} ('{section_name}') ---")
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

                        logger.info(f"Used validated {section_name} (confidence {results[section_name]['ValidationConfidence']:.2f})")
                        if hasattr(validated_result, 'CorrectionsMade') and validated_result.CorrectionsMade:
                            logger.info("Corrections made:")
                            for correction in validated_result.CorrectionsMade: logger.info(f"  • {correction}")
                    else:
                        logger.warning(f"Validation failed for {section_name}, using original extraction")
                        result_data = extracted_data.model_dump()
                        for field, value in result_data.items():
                            if value is not None and field != 'ValidationConfidence':
                                results[section_name][field] = value
                        results[section_name]['ValidationConfidence'] = current_section_confidence # Keep default 0.5

                # Add extraction results to ordered results
                for section_name, section_data in results.items():
                    ordered_results[section_name] = section_data

                # Step: Deduplication (Final step if enabled)
                if self.config.enable_deduplication_agent:
                    current_step += 1
                    logger.info(f"--- Starting Pipeline Step {current_step} of {total_steps}: Deduplication ---")
                    # Pass config manager for this run
                    self.deduplication_agent.config_manager = self.config_manager
                    # logger.info("Running deduplication agent to resolve header conflicts...") # Redundant with step log
                    deduplication_result = self.deduplication_agent.deduplicate(
                        ordered_results, markdown_content
                    )

                    if deduplication_result and "DeduplicatedData" in deduplication_result:
                        deduplicated_data = deduplication_result["DeduplicatedData"]
                        for section_name, section_data in deduplicated_data.items():
                            if section_name in ordered_results:
                                ordered_results[section_name] = section_data
                        logger.info(f"Successfully deduplicated extraction results with confidence {deduplication_result.get('DeduplicationConfidence', 0.0):.2f})")
                        # Note: ConflictsResolved logging removed as per previous user request
                    else:
                        logger.warning("Deduplication failed, using original extraction results")

        except PipelineException:
            # Catch custom pipeline exceptions and re-raise
            raise
        except Exception as e:
            # Catch any other unexpected exceptions during the pipeline run
            logger.exception(f"An unexpected error occurred during pipeline execution for {doc_name}: {e}")
            # Raise a generic PipelineException for unhandled errors, adding context
            raise PipelineException(
                message=f"An unexpected error occurred during pipeline execution: {e}",
                error_code="PIPELINE_UNEXPECTED_ERROR",
                context={
                    "doc_name": doc_name,
                    "classified_class": validated_class if 'validated_class' in locals() else "Unknown",
                    "exception_type": e.__class__.__name__
                }
            )

        # --- Final processing (runs only if class was not "None of those" and config was found) ---
        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Ensure Context exists before adding time
        if "Context" not in ordered_results or ordered_results["Context"] is None:
             ordered_results["Context"] = {} # Create if missing
        ordered_results["Context"]["ProcessingTimeSeconds"] = round(processing_time, 3)

        logger.info(f"Total processing time for {source_file}: {processing_time:.3f} seconds")
        return ordered_results
