from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
from classification_agent import ClassificationAgentInput, ClassificationAgentOutput, ClassificationAgent, get_tokenizer, count_tokens
from langchain.prompts import ChatPromptTemplate
import re
import os
import json
import requests

with open('config/classifier_test_config.json', 'r') as f:
    config = json.load(f)

# Configuration for LLM
LLM_MODEL_NAME = config["llm_choosen4classification"]
LM_STUDIO_URL = config['lm_studio_url']
LLM_MAX_TOKENS = config['llm_max_tokens'] 
MAX_PROMPT_TOKEN_PERCENTAGE = 0.45
LLM_CONFIG = {
    "temperature": config["llm_temperature"],
    "max_tokens": config["llm_max_tokens"],
}
#Configuration for Prompt
BEST_PROMPT_VALID = config["prompt_choosen4valid_classif"]
PROMPT_FOLDER = 'src/prompts'
#Labels for classification
LABELS = config['labels']

class Validation_ClassifAgentInput(BaseModel):
    """Input model for the Validation Agent.
    Receives the prediction from the Classification Agent and
    an associated confidence score."""
    doc_name: str = Field(..., description="The name of the file that we want to analyse.")
    markdown_text: str = Field(..., description="The biginning of the document that we want to analyse.")
    classification_result: ClassificationAgentOutput = Field(..., description="The class and the validation score output from the Classification Agent.")

class Validation_ClassifAgentOutput(BaseModel):
    """Output model for the Validation Agent.
    Indicates if the classification is valid and provides a reason if not."""
    predicted_class: str = Field(..., description="The class predicted by the LLM (including None of those).")
    confidence: str = Field(..., description="The confidence score of the LLM (=0.0 when it returns None).")
    validation_reason: str = Field(None, description="Reason why the classification is not valid.")

class ValidationClassifAgent():
    """Agent responsible for validating the output of the Classification Agent."""
    def __init__(self, model_name: str, prompt_folder, prompt_file, config_llm: dict, max_tokens, percent_tokens_context_window, labels: List[str]):
        self.model_name = model_name
        self.model_name = model_name
        self.folder_name_prompt = prompt_folder
        self.file_name_prompt = prompt_file
        self.llm_config = config_llm
        self.max_tokens = max_tokens
        self.percent_context = percent_tokens_context_window
        self.labels = labels
        self.encoding = get_tokenizer("cl100k_base")

    def _load_prompt_template(self, prompt_file_name, prompt_folder):
        prompt_path = os.path.join(prompt_folder, prompt_file_name)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read()
                return ChatPromptTemplate.from_template(prompt_text)
        except FileNotFoundError:
            print(f"Error: Prompt file '{prompt_file_name}' not found in '{prompt_folder}'.")
            return None

    def _call_lm_studio(self, prompt: str, llm_config: dict) -> str:
        """Calls the local LLM server in LM Studio."""
        api_endpoint = f"{LM_STUDIO_URL}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model_name,
            **llm_config, 
            }
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, stream=False)
            response.raise_for_status()
            llm_output = response.json()
            if 'choices' in llm_output and llm_output['choices']:
                return llm_output['choices'][0]['message']['content'].strip() if 'message' in llm_output['choices'][0] else llm_output['choices'][0]['text'].strip()
            return f"Error: Could not extract response: {llm_output}"
        except requests.exceptions.RequestException as e:
            return f"Error communicating with LM Studio: {e}"
        except json.JSONDecodeError:
            return f"Error decoding JSON: {e}"

    def _calculating_nb_char(self, prompt_llm: ChatPromptTemplate, name_doc, tokenizer, max_percent, data_as_markdown, previous_val):
        base_prompt_messages = prompt_llm.format_messages(doc_name=name_doc, previous_class= previous_val, text ="")
        base_prompt_text = "".join([msg.content for msg in base_prompt_messages])
        base_prompt_tokens = count_tokens(tokenizer, base_prompt_text)
        available_tokens = int(LLM_MAX_TOKENS * max_percent) - base_prompt_tokens
        nb_charac_to_use = max(min(len(data_as_markdown), available_tokens * 4 // 3), 0)
        return nb_charac_to_use

    def _extract_class_confidence_reason(self, content: str):
        predicted_class = None
        confidence = None
        reasoning = None

        class_match = re.search(r"<class>(.*?)</class>", content, re.IGNORECASE)
        if class_match:
            predicted_class = class_match.group(1).strip()
        else:
            for element in self.labels:
                if element.lower() in content.lower():
                    predicted_class = element
                    break

        confidence_match = re.search(r"<confidence>(.*?)</confidence>", content, re.IGNORECASE)
        if confidence_match:
            confidence = confidence_match.group(1).strip()
        else:
            confidence = 'Low'

        reason_match = 'reason' in content
        if reason_match:
            reasoning = content.split('<reason>')[1].split('</reason>')[0]
        else:
            reasoning = " "
        if predicted_class is not None:
            return {"predicted_class": predicted_class, "confidence": confidence, "reason": reasoning}
        else:
            return {"error": f"Could not find <class> tag in LLM output: {content}"}


    def run(self, input_data: Validation_ClassifAgentInput) -> Validation_ClassifAgentOutput:
        try:
            text_to_classify = input_data.markdown_text
            doc_name = input_data.doc_name

            loaded_prompt = self._load_prompt_template(self.file_name_prompt, self.folder_name_prompt)

            nb_charac_to_use = self._calculating_nb_char(loaded_prompt, doc_name, get_tokenizer(), self.percent_context, text_to_classify, input_data.classification_result.predicted_class)
            if nb_charac_to_use == 0:
                print(f"Warning: Base prompt is already too long for {self.percent_context:.2f}. Skipping document {doc_name}.") 

            text_to_classify_truncated = text_to_classify[:nb_charac_to_use]

            prompt_value = loaded_prompt.format_messages(text= text_to_classify_truncated, doc_name= doc_name, previous_class= input_data.classification_result.predicted_class)
            final_prompt_text = "".join([msg.content for msg in prompt_value])

            llm_response = self._call_lm_studio(final_prompt_text, self.llm_config)
            extracted_data = self._extract_class_confidence_reason(llm_response)

            if extracted_data:
                return Validation_ClassifAgentOutput(
                    predicted_class = extracted_data["predicted_class"],
                    confidence = extracted_data["confidence"],
                    validation_reason = extracted_data["reason"],
                )
            else:
                raise ValueError("Could not extract class from LLM response.")

        except ValidationError as e:
            print(f"Input validation error: {e}")
            raise
        except ValueError as e:
            print(f"Error processing the LLM's response: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error in the classification agent: {e}")
            raise
    
    
# Example of integration (assuming you have an instance of ClassificationAgent and some input)
if __name__ == "__main__":
    
    doc_name_extracted = "8208700_PIMCO Global Investor Series PLC Re Investment Notice January 2025.2025.01.28"

    markdown_input = """ |     Adviser | Fund Name                              |   SSB Fund # | Inception Date        | Denomination   | ISIN         |   Per Share Rate |   Re-Invest Net Asset Value Per Share |
                         |------------:|:---------------------------------------|-------------:|:----------------------|:---------------|:-------------|---------------:|--------------------------------------:|
                         |         nan | PIMCO Capital Securities Fund          |         nan  | nan                   | nan            | nan          |            nan |                                   nan |
                         |       14603 | Class M Retail Income II               | PNE3H        | 2013-12-23 00:00:00   | USD            | IE00BH3X8443 |         0.0431 |                                  9.22 |
                         |       14607 | Class M Retail (Hedged) Income II      | PNE3J        | 2013-12-23 00:00:00   | SGD            | IE00BH3X8559 |         0.0408 |                                  8.78 |
                         |       14603 | Class M HKD (Unhedged) Income          | PNE310       | 2017-07-28 00:00:00   | HKD            | IE00BF0F6D43 |         0.0352 |                                 10.2  |
                         |       14603 | Class M Retail                         | PNE324       | 2024-05-09 00:00:00   | GBP            | IE0002Z7IO91 |         0.0357 |                                 10.32 |"""
    
    classif_outp = ClassificationAgentOutput(predicted_class='Bonds', confidence='Medium')
    
    
    valid_agent = ValidationClassifAgent(
        model_name = LLM_MODEL_NAME,
        prompt_folder=PROMPT_FOLDER,
        prompt_file=BEST_PROMPT_VALID,
        config_llm=LLM_CONFIG,
        max_tokens=LLM_MAX_TOKENS,
        percent_tokens_context_window=MAX_PROMPT_TOKEN_PERCENTAGE,
        labels=LABELS
    )

    agent_input = Validation_ClassifAgentInput(markdown_text=markdown_input, doc_name=doc_name_extracted, classification_result= classif_outp)

    try:
        output = valid_agent.run(agent_input)
        print("---------Validation Agent--------- ")
        print(f"Predicted Class: {output.predicted_class}")
        print(f"Confidence Score: {output.confidence}")
        print(f"Reasoning: {output.validation_reason}")
    except Exception as e:
        print(f"An error occurred while running the agent with LM Studio: {e}")