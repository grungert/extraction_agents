from pydantic import BaseModel, ValidationError, Field
from typing import Optional, Literal, Dict, List
from langchain.prompts import ChatPromptTemplate
import re
import json
import requests
import os
import tiktoken

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
BEST_PROMPT = config["prompt_choosen4classification"]
PROMPT_FOLDER = 'src/prompts'
#Labels for classification
LABELS = config['labels']

class ClassificationAgentInput(BaseModel):
    doc_name: str = Field(..., description="The name of the file that we want to analyse.")
    markdown_text: str = Field(..., description="The biginning of the document that we want to analyse.")

class ClassificationAgentOutput(BaseModel):
    predicted_class: str = Field(..., description="The class predicted by the LLM (including None of those).")
    confidence: str = Field(..., description="The confidence score of the LLM (=0.0 when it returns None).")

def get_tokenizer(
    tokenizer_name: Literal["o200k_base", "cl100k_base"] = "cl100k_base",
):
    try:
        enc = tiktoken.get_encoding(tokenizer_name)
        return enc
    except Exception as e:
        print(f"Error loading tiktoken encoding '{tokenizer_name}': {e}")
        return None

def count_tokens(encoding, text):
    if encoding:
        return len(encoding.encode(text))
    return 0

class ClassificationAgent:
    def __init__(self, model_name: str, prompt_folder, prompt_file_name, config_llm: dict, max_tokens, percent_tokens_context_window, labels: List[str]):
        self.model_name = model_name
        self.folder_name_prompt = prompt_folder
        self.file_name_prompt = prompt_file_name
        self.llm_config = config_llm
        self.max_tokens = max_tokens
        self.percent_context = percent_tokens_context_window
        self.labels = labels
        self.encoding = get_tokenizer("cl100k_base") # Using default 'cl100k_base' for tokenization estimation.

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

    def _extract_class_confidence(self, content: str):
        predicted_class = None
        confidence = None

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

        if predicted_class is not None:
            return {"predicted_class": predicted_class, "confidence": confidence}
        else:
            return {"error": f"Could not find <class> tag in LLM output: {content}"}
    
    def _calculating_nb_char(self, prompt_llm: ChatPromptTemplate, name_doc, tokenizer, data_as_markdown):
        base_prompt_messages = prompt_llm.format_messages(doc_name=name_doc, question="")
        base_prompt_text = "".join([msg.content for msg in base_prompt_messages])
        base_prompt_tokens = count_tokens(tokenizer, base_prompt_text)
        available_tokens = int(self.max_tokens * self.percent_context) - base_prompt_tokens
        nb_charac_to_use = max(min(len(data_as_markdown), available_tokens * 4 // 3), 0)
        return nb_charac_to_use

    def run(self, input_data: ClassificationAgentInput) -> ClassificationAgentOutput:
        try:
            text_to_classify = input_data.markdown_text
            doc_name = input_data.doc_name

            loaded_prompt = self._load_prompt_template(self.file_name_prompt, self.folder_name_prompt)

            nb_charac_to_use = self._calculating_nb_char(loaded_prompt, doc_name, get_tokenizer(), text_to_classify)
            if nb_charac_to_use == 0:
                print(f"Warning: Base prompt is already too long for {self.percent_context:.2f}. Skipping document {doc_name}.") 

            text_to_classify_truncated = text_to_classify[:nb_charac_to_use]

            prompt_value = loaded_prompt.format_messages(question=text_to_classify_truncated, doc_name=doc_name)
            final_prompt_text = "".join([msg.content for msg in prompt_value])

            llm_response = self._call_lm_studio(final_prompt_text, self.llm_config)
            extracted_data = self._extract_class_confidence(llm_response)

            if extracted_data:
                return ClassificationAgentOutput(
                    predicted_class=extracted_data["predicted_class"],
                    confidence=extracted_data["confidence"]
                )
            else:
                raise ValueError("Could not extract class and confidence from LLM response.")

        except ValidationError as e:
            print(f"Input validation error: {e}")
            raise
        except ValueError as e:
            print(f"Error processing the LLM's response: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error in the classification agent: {e}")
            raise

# Example 
if __name__ == "__main__":
    doc_name_extracted = "8208700_PIMCO Global Investor Series PLC Re Investment Notice January 2025.2025.01.28"

    markdown_input = """ |     Adviser | Fund Name                              |   SSB Fund # | Inception Date        | Denomination   | ISIN         |   Per Share Rate |   Re-Invest Net Asset Value Per Share |
|------------:|:---------------------------------------|-------------:|:----------------------|:---------------|:-------------|---------------:|--------------------------------------:|
|         nan | PIMCO Capital Securities Fund          |         nan  | nan                   | nan            | nan          |            nan |                                   nan |
|       14603 | Class M Retail Income II               | PNE3H       | 2013-12-23 00:00:00   | USD            | IE00BH3X8443 |          0.0431 |                                  9.22 |
|       14607 | Class M Retail (Hedged) Income II      | PNE3J       | 2013-12-23 00:00:00   | SGD            | IE00BH3X8559 |          0.0408 |                                  8.78 |
|       14603 | Class M HKD (Unhedged) Income          | PNE310      | 2017-07-28 00:00:00   | HKD            | IE00BF0F6D43 |          0.0352 |                                 10.2  |
|       14603 | Class M Retail                         | PNE324      | 2024-05-09 00:00:00   | GBP            | IE0002Z7IO91 |          0.0357 |                                 10.32 |"""


    classifier_agent = ClassificationAgent(
        model_name = LLM_MODEL_NAME,
        prompt_file_name = BEST_PROMPT,
        prompt_folder = PROMPT_FOLDER,
        config_llm = LLM_CONFIG,
        max_tokens = LLM_MAX_TOKENS,
        percent_tokens_context_window = MAX_PROMPT_TOKEN_PERCENTAGE,
        labels = LABELS,
    )

    agent_input = ClassificationAgentInput(markdown_text=markdown_input, doc_name=doc_name_extracted)

    try:
        output = classifier_agent.run(agent_input)
        print("---------Classification Agent--------- ")
        print(f"Predicted Class: {output.predicted_class}")
        print(f"Confidence: {output.confidence}")
    except Exception as e:
        print(f"An error occurred while running the agent with LM Studio: {e}")