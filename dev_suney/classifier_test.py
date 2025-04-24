import requests
import json
import numpy as np
import re
import os
import time
from dataclasses import dataclass
from typing import Dict, Literal, List
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from langchain_core.prompts import ChatPromptTemplate
from create_examples_extraction import loading_data_as_df
from classification_agent import get_tokenizer, count_tokens
from utils_data import load_vars, save_vars
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

with open('config/classifier_test_config.json', 'r') as f:
    config = json.load(f)

LLMS_TO_CONSIDER = config['llms_to_test']
LIST_ALGO = config['list_algo']
GLICLASS_MODEL_NAME = config['gliclass_model']
LM_STUDIO_URL = config['lm_studio_url']
LLM_TEMPERATURE = config['llm_temperature']
LLM_MAX_TOKENS = config['llm_max_tokens']
LLM_MODEL_NAME = config["llm_choosen4classification"]

BASE_DOCUMENT_PATH = config['base_document_path']
BASE_LABEL_PATH = config['base_label_path']

PREDICTION_FILE = config['prediction_file']
OPTIMIZATION_FILE = config['optimizing_percent_file']

LABELS = config['labels']

F1_AVERAGE = config['f1_average']

PROMPT_FILES = config['prompt_file']
PROMPT_FOLDER = 'src/prompts'
BEST_PROMPT = config["prompt_choosen4classification"]

MAX_PROMPT_TOKEN_PERCENTAGE = 0.45  # Aim to use less than this percentage of max tokens for the context windows

def load_prompt_template(prompt_file_name, prompt_folder):
    prompt_path = os.path.join(prompt_folder, prompt_file_name)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
            return ChatPromptTemplate.from_template(prompt_text)
    except FileNotFoundError:
        print(f"Error: Prompt file '{prompt_file_name}' not found in '{prompt_folder}'.")
        return None

def extract_class_confidence(llm_output, list_labels: List[str]):
    predicted_class = None
    confidence_score = None
    if 'choices' in llm_output and llm_output['choices']:
        content = llm_output['choices'][0]['message']['content'].strip() if 'message' in llm_output['choices'][0] else llm_output['choices'][0]['text'].strip()

        class_match = re.search(r"<class>(.*?)</class>", content, re.IGNORECASE)
        if class_match:
            predicted_class = class_match.group(1).strip()
        else:
            for element in list_labels:
                if element.lower() in content.lower():
                    predicted_class = element
                    break

        confidence_match = re.search(r"<confidence>(.*?)</confidence>", content, re.IGNORECASE)
        if confidence_match:
            confidence_score = float(confidence_match.group(1).strip())
        else:
            confidence_score = 0.0
            
    return {"Class": predicted_class, "ValidationConfidence": confidence_score}

def call_lm_studio(model_name, prompt: ChatPromptTemplate, lm_studio_url, tokenizer, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS):
    api_endpoint = f"{lm_studio_url}/v1/chat/completions"
    messages = prompt.format_messages()
    payload_messages = [{"role": "user", "content": msg.content} for msg in messages]
    combined_prompt_text = "".join([msg.content for msg in messages])
    num_prompt_tokens = count_tokens(tokenizer, combined_prompt_text)

    if num_prompt_tokens > int(max_tokens * MAX_PROMPT_TOKEN_PERCENTAGE):
        print(f"Warning: Prompt for {model_name} exceeds {MAX_PROMPT_TOKEN_PERCENTAGE*100}% of max tokens ({num_prompt_tokens} > {int(max_tokens * MAX_PROMPT_TOKEN_PERCENTAGE)}). Skipping...")
        return "Error: Prompt too long"

    payload = {"messages": payload_messages, "model": model_name, "temperature": temperature, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_endpoint, headers=headers, json=payload, stream=False)
        response.raise_for_status()
        llm_output = response.json()
        answer = extract_class_confidence(llm_output, list_labels=LABELS)
        return answer
    except requests.exceptions.RequestException as e:
        return {"error": f"Error communicating with LM Studio: {e}"}
    except json.JSONDecodeError:
        return {"error": f"Error decoding LM Studio JSON response: {e}"}

def run_huggingface_model(model_name, question, labels, tokenizer, device='cuda:0', threshold=0.5):
    model = GLiClassModel.from_pretrained(model_name)
    pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device=device)

    question_tokens = tokenizer.encode(question)
    label_tokens = tokenizer.encode(" ".join(labels))
    combined_length = len(question_tokens) + len(label_tokens)
    max_model_input_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 2048 # Default if not found

    if combined_length > int(max_model_input_length * MAX_PROMPT_TOKEN_PERCENTAGE):
        print(f"Warning: Input length for {model_name} might exceed {MAX_PROMPT_TOKEN_PERCENTAGE*100}% of max input ({combined_length} > {int(max_model_input_length * MAX_PROMPT_TOKEN_PERCENTAGE)}).")

    start_time = time.time()
    results = pipeline(question, labels, threshold=threshold)[0]
    best_result_data = max(results, key=lambda x: x['score']) if results else None
    best_result_label = best_result_data['label'] if best_result_data else ''
    confidence_score = best_result_data['score'] if best_result_data else None
    end_time = time.time()
    return encode_class(best_result_label), confidence_score, (end_time - start_time)

@dataclass
class solution_pred:
    pred: Dict[str, Dict]

@dataclass
class solution_opt:
    opt: Dict[str, Dict]

#Encoder
def encode_class(value: str):
    mapping = {
        "Mutual Funds": 0, "ETF": 1, "Stocks": 2, "Bonds": 3, "Real Estate": 4,
        "Crypto Asset": 5, "Commodities": 6, "Private Equities": 7, "Index": 8,
        "Currencies Pairs": 9, "None of those": 10
    }
    return mapping.get(value, 10) # Default to 10 if not found

def creating_ground_truth():
    ground_truth = np.array([])
    for json_ in os.listdir(BASE_LABEL_PATH):
        with open(os.path.join(BASE_LABEL_PATH, json_), 'r') as f:
            meta_data = json.load(f)
            ground_truth = np.concatenate((ground_truth, [encode_class(meta_data["Type"])]))
    return ground_truth

def calculating_nb_char(prompt_llm: ChatPromptTemplate, name_doc, tokenizer, max_percent, data_as_markdown):
    base_prompt_messages = prompt_llm.format_messages(doc_name=name_doc, question="")
    base_prompt_text = "".join([msg.content for msg in base_prompt_messages])
    base_prompt_tokens = count_tokens(tokenizer, base_prompt_text)
    available_tokens = int(LLM_MAX_TOKENS * max_percent) - base_prompt_tokens
    nb_charac_to_use = max(min(len(data_as_markdown), available_tokens * 4 // 3), 0)
    return nb_charac_to_use

def test_prompts_vs_llm_45percent_context_window():
    ground_truth = creating_ground_truth()
    file_path_with_pred = PREDICTION_FILE
    pred = load_vars(file_path_with_pred) if os.path.exists(file_path_with_pred) else solution_pred(pred={})
    for algorithm in LIST_ALGO:
        tokenizer = None
        if algorithm in LLMS_TO_CONSIDER:
            tokenizer = get_tokenizer() # Use the default 'cl100k_base' or specify if needed
            if tokenizer is None:
                continue # Skip if tokenizer couldn't be loaded for LM Studio model
        elif algorithm == "gli":
            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(GLICLASS_MODEL_NAME)
            except Exception as e:
                print(f"Error loading tokenizer for GLiClass model {GLICLASS_MODEL_NAME}: {e}")
                continue # Skip if tokenizer couldn't be loaded for GLiClass

        if algorithm not in pred.pred:
            pred.pred[algorithm] = {}
        for prompt_index, prompt_ in enumerate(PROMPT_FILES):
            prompt_llm = load_prompt_template(PROMPT_FILES[prompt_index],PROMPT_FOLDER)
            if prompt_llm is None:
                exit()

            prompt_key = str(prompt_index)
            if prompt_key not in pred.pred[algorithm]:
                pred.pred[algorithm][prompt_key] = [[], [], []]  # Initialize with empty lists

            predictions = np.array(pred.pred[algorithm][prompt_key][0])
            time_algo = np.array(pred.pred[algorithm][prompt_key][1])
            valid_conf = np.array(pred.pred[algorithm][prompt_key][2])

            for index, document_ in enumerate(os.listdir(BASE_DOCUMENT_PATH)):
                if index < len(predictions):
                    continue

                name_doc, extension = os.path.splitext(document_)
                document_path = os.path.join(BASE_DOCUMENT_PATH, document_)
                data_as_df = loading_data_as_df(extension, document_path)
                data_as_markdown = data_as_df.to_markdown(index=False)

                nb_charac_to_use = calculating_nb_char(prompt_llm, name_doc, tokenizer, MAX_PROMPT_TOKEN_PERCENTAGE, data_as_markdown)
                if nb_charac_to_use == 0:
                    print(f"Warning: Base prompt is already too long for {MAX_PROMPT_TOKEN_PERCENTAGE:.2f}. Skipping document {name_doc}.")
                question_ = data_as_markdown[:nb_charac_to_use]
                custom_prompt = prompt_llm.partial(doc_name=name_doc, question=question_)

                if algorithm in LLMS_TO_CONSIDER:
                    start_time = time.time()
                    response = call_lm_studio(algorithm, custom_prompt, LM_STUDIO_URL, tokenizer, LLM_TEMPERATURE, LLM_MAX_TOKENS)
                    end_time = time.time()
                    if "error" in response:
                        print(f"Error with {algorithm}: {response['error']}")
                        continue # Or handle the error as needed

                    identified_class_str = response.get("Class")
                    confidence_score = response.get("ValidationConfidence")

                    if identified_class_str is not None:
                        encd = encode_class(identified_class_str)
                        duration = end_time - start_time
                    else:
                        print(f"Warning: Could not extract predicted class from LLM response for {algorithm}.")
                    
                elif algorithm == "gli":
                    encd, confidence_score, duration = run_huggingface_model(GLICLASS_MODEL_NAME, question_, LABELS, tokenizer)
                else:
                    continue # Handle unknown algorithms

                time_algo = np.concatenate((time_algo, [duration]))
                predictions = np.concatenate((predictions, [encd]))
                valid_conf = np.concatenate((valid_conf, [confidence_score]))
                print(f"Processed {algorithm} - prompt {prompt_index=} - Document {index} (Chars: {len(question_)})")
                pred.pred[algorithm][prompt_key] = [predictions.tolist(), time_algo.tolist(), valid_conf.tolist()]
                save_vars(pred, PREDICTION_FILE)

    for algorithm in pred.pred.keys():
        for prompt_index in pred.pred[algorithm].keys():
            predictions = np.array(pred.pred[algorithm][prompt_index][0])
            f1 = f1_score(ground_truth[:len(predictions)], predictions, average=F1_AVERAGE)
            mean_time_algo = np.mean(pred.pred[algorithm][str(prompt_index)][1])
            print('----------------------------')
            print(f"{algorithm} - Prompt {prompt_index}: F1 Score = {f1:.4f}, Mean Time = {mean_time_algo:.4f} seconds")

def test_optimizing_windows_context(model_name: str, prompt_file: str, max_tokens_context_window: List[float]):
    """Iterates over a list of max_token_percentages with a fixed prompt and model
    to evaluate classification performance.
    Args:
        model_name (str): The name of the LLM to use (must be in LLMS_TO_CONSIDER or "gli").
        prompt_file (str): The filename of the prompt template to use.
        max_token_percentages (List[float]): A list of percentages (e.g., [0.3, 0.4, 0.5]) to test as the maximum prompt token ratio.
    """
    f1_list = []
    mean_time_list = []
    min_confidence_list = []
    max_confidence_list = []
    mean_confidence_list = []
    ground_truth = creating_ground_truth()
    file_path4optim = OPTIMIZATION_FILE
    opt_percent = load_vars(file_path4optim) if os.path.exists(file_path4optim) else solution_opt(opt={})
    prompt_llm = load_prompt_template(prompt_file,PROMPT_FOLDER)
    if prompt_llm is None:
        print(f"Error: Could not load prompt file '{prompt_file}'.")
        return

    if model_name in LLMS_TO_CONSIDER:
        tokenizer = get_tokenizer()
        if tokenizer is None:
            print(f"Error: Could not load tokenizer for LM Studio model '{model_name}'.")
            return
    else:
        print(f"Error: Model '{model_name}' not recognized.")
        return
    
    for max_percent in max_tokens_context_window:
        print(f"\n--- Testing MAX_PROMPT_TOKEN_PERCENTAGE = {max_percent:.2f} ---")
        if max_percent not in opt_percent.opt:
                opt_percent.opt[max_percent] = [[], [], []]  # Initialize with empty lists

        all_predictions = np.array(opt_percent.opt[max_percent][0])
        all_durations = np.array(opt_percent.opt[max_percent][1])
        all_confidences = np.array(opt_percent.opt[max_percent][2])

        for index, document_ in enumerate(os.listdir(BASE_DOCUMENT_PATH)):
            if index < len(all_predictions):
                    continue
            else:
                name_doc, extension = os.path.splitext(document_)
                document_path = os.path.join(BASE_DOCUMENT_PATH, document_)
                data_as_df = loading_data_as_df(extension, document_path)
                data_as_markdown = data_as_df.to_markdown(index=False)
                nb_charac_to_use = calculating_nb_char(prompt_llm, name_doc, tokenizer, MAX_PROMPT_TOKEN_PERCENTAGE, data_as_markdown)
                if nb_charac_to_use == 0:
                    print(f"Warning: Base prompt is already too long for {MAX_PROMPT_TOKEN_PERCENTAGE:.2f}. Skipping document {name_doc}.")
                question_ = data_as_markdown[:nb_charac_to_use]
                custom_prompt = prompt_llm.partial(doc_name=name_doc, question=question_)
                
                if model_name in LLMS_TO_CONSIDER:
                    start_time = time.time()
                    response = call_lm_studio(model_name, custom_prompt, LM_STUDIO_URL, tokenizer, LLM_TEMPERATURE, LLM_MAX_TOKENS)
                    end_time = time.time()
                    if "error" in response:
                        print(f"Error with {model_name} at {max_percent:.2f}: {response['error']}")
                        continue
                    identified_class_str = response.get("Class")
                    confidence_score = response.get("ValidationConfidence")
                    if identified_class_str is not None:
                        prediction = encode_class(identified_class_str)
                        duration = end_time - start_time
                    else:
                        print(f"Warning: Could not extract predicted class for {model_name} at {max_percent:.2f}.")
                        continue
                else:
                    continue
                all_predictions = np.concatenate((all_predictions, [prediction]))
                all_durations = np.concatenate((all_durations, [duration]))
                all_confidences = np.concatenate((all_confidences, [confidence_score]))
                opt_percent.opt[max_percent]  = [all_predictions, all_durations, all_confidences]
                print(f"Processed: Percent {max_percent} - Document {index} - Class {identified_class_str} - Score {confidence_score}")
                save_vars(opt_percent, OPTIMIZATION_FILE)

        if len(all_confidences) == len(os.listdir(BASE_DOCUMENT_PATH)):
            f1 = f1_score(ground_truth, all_predictions, average=F1_AVERAGE)
            f1_list.append(f1)
            mean_time = np.mean(all_durations)
            mean_time_list.append(mean_time)
            min_confidence = np.min(all_confidences)
            min_confidence_list.append(min_confidence)
            max_confidence = np.max(all_confidences)
            max_confidence_list.append(max_confidence)
            mean_confidence = np.mean(all_confidences)
            mean_confidence_list.append(mean_confidence)
            print(f"MAX_PROMPT_TOKEN_PERCENTAGE: {max_percent:.2f}, F1 Score: {f1:.4f}, Mean Time: {mean_time:.4f} seconds, Min Confidence: {min_confidence:.4f}")
        else:
            print(f"No predictions made for MAX_PROMPT_TOKEN_PERCENTAGE: {max_percent:.2f}.")
    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(max_tokens_context_window, f1_list, marker='o')
    plt.title('F1 Score vs. Max Percentage of Tokens for Context Windows')
    plt.xlabel('Max Percentage of Tokens for Context Windows')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(max_tokens_context_window, mean_confidence_list, marker='o', label='Mean Confidence')
    plt.plot(max_tokens_context_window, min_confidence_list, marker='o', linestyle='--', label='Min Confidence')
    plt.plot(max_tokens_context_window, max_confidence_list, marker='o', linestyle='--', label='Max Confidence')
    plt.title('Confidence vs. Max Percentage of Tokens for Context Windows')
    plt.xlabel('Max Percentage of Tokens for Context Windows')
    plt.ylabel('Confidence')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    percent = np.linspace(0.25, 0.75, 10)
    test_optimizing_windows_context(model_name = LLM_MODEL_NAME, prompt_file = BEST_PROMPT, max_tokens_context_window= percent)