from langchain.chains import SequentialChain
from typing import Dict, Any
from classification_agent import ClassificationAgentInput, ClassificationAgentOutput, ClassificationAgent
from validation_classification_agent import Validation_ClassifAgentInput, Validation_ClassifAgentOutput, ValidationClassifAgent
import json

with open('config/classifier_test_config.json', 'r') as f:
    config = json.load(f)

# Configuration for LLM
LLM_MODEL_NAME = config["llm_choosen4classification"]
LM_STUDIO_URL = config['lm_studio_url']
LLM_MAX_TOKENS = config['llm_max_tokens'] 
MAX_PROMPT_TOKEN_PERCENTAGE = config['max_prompt_token_percentage']
LLM_CONFIG = {
    "temperature": config["llm_temperature"],
    "max_tokens": config["llm_max_tokens"],
}
#Configuration for Prompt
BEST_PROMPT = config["prompt_choosen4classification"]
BEST_PROMPT_VALID = config["prompt_choosen4valid_classif"]
PROMPT_FOLDER = 'src/prompts'
#Labels for classification
LABELS = config['labels']

def create_classification_workflow() -> SequentialChain:
    """Creates the classification workflow using Langchain."""
    # Instantiate the agents, passing in the LLM and prompts
    classification_agent = ClassificationAgent(
        model_name = LLM_MODEL_NAME,
        prompt_file_name = BEST_PROMPT,
        prompt_folder = PROMPT_FOLDER,
        config_llm = LLM_CONFIG,
        max_tokens = LLM_MAX_TOKENS,
        percent_tokens_context_window = MAX_PROMPT_TOKEN_PERCENTAGE,
        labels = LABELS,
    )
    validation_agent = ValidationClassifAgent(
        model_name = LLM_MODEL_NAME,
        prompt_folder=PROMPT_FOLDER,
        prompt_file=BEST_PROMPT_VALID,
        config_llm=LLM_CONFIG,
        max_tokens=LLM_MAX_TOKENS,
        percent_tokens_context_window=MAX_PROMPT_TOKEN_PERCENTAGE,
        labels=LABELS
    )

    steps = [
        {"agent": classification_agent, "input_keys": ["doc_name","markdown_text"], "output_keys": ["classification_result"]},
        {"agent": validation_agent, "input_keys": ["doc_name","markdown_text", "classification_result"], "output_keys": ["validation_result"]},
    ]

    # Create the SequentialChain
    workflow = SequentialChain(
        steps=steps,
        input_variables=["doc_name", "markdown_text"],  # Initial input to the workflow
        output_variables=["classification_result", "validation_result"],  # Final outputs
    )
    return workflow

if __name__ == "__main__":
    workflow = create_classification_workflow()

    # Sample input
    document_name = "8208700_PIMCO Global Investor Series PLC Re Investment Notice January 2025.2025.01.28"
    document_content = """ |     Adviser | Fund Name                              |   SSB Fund # | Inception Date        | Denomination   | ISIN         |   Per Share Rate |   Re-Invest Net Asset Value Per Share |
                           |------------:|:---------------------------------------|-------------:|:----------------------|:---------------|:-------------|---------------:|--------------------------------------:|
                           |         nan | PIMCO Capital Securities Fund          |         nan  | nan                   | nan            | nan          |            nan |                                   nan |
                           |       14603 | Class M Retail Income II               | PNE3H       | 2013-12-23 00:00:00   | USD            | IE00BH3X8443 |          0.0431 |                                  9.22 |
                           |       14607 | Class M Retail (Hedged) Income II      | PNE3J       | 2013-12-23 00:00:00   | SGD            | IE00BH3X8559 |          0.0408 |                                  8.78 |
                           |       14603 | Class M HKD (Unhedged) Income          | PNE310      | 2017-07-28 00:00:00   | HKD            | IE00BF0F6D43 |          0.0352 |                                 10.2  |
                           |       14603 | Class M Retail                         | PNE324      | 2024-05-09 00:00:00   | GBP            | IE0002Z7IO91 |          0.0357 |                                 10.32 |"""

    result: Dict[str, Any] = workflow({
        "doc_name": document_name,
        "markdown_text": document_content,
    })

    classification_result: ClassificationAgentOutput = result["classification_result"]
    validation_result: Validation_ClassifAgentOutput = result["validation_result"]

    print(f"Document: {document_name}")
    print(f"Classification Agent: Class={classification_result.predicted_class}, Confidence={classification_result.confidence}")
    print(f"Validation Agent: Class={validation_result.predicted_class}, Confidence={validation_result.confidence}, Reasoning={validation_result.validation_reason}")

   
    if classification_result.predicted_class == validation_result.predicted_class:
        print(f"Both agents agree on class '{classification_result.predicted_class}'. This flow is ready for the next analysis.")
    else:
        print(f"Agents disagree. Agent 1 predicted '{classification_result.predicted_class}', Agent 2 predicted '{validation_result.predicted_class}'. This flow needs to be checked by a human.")