# coding=UTF-8
import requests
import json
import re
from tqdm import tqdm
import numpy as np
import time
import random
import os
import traceback

# --- New: Import OpenAI library ---
import openai

# --- MODIFIED: Configuration now points to DeepSeek API from your first script ---
API_KEY = "" 
BASE_URL = ""
API_MODEL = ""

# --- MODIFIED: Initialize OpenAI Client for DeepSeek ---
# This client will be used to make requests in the style of your first script.
client = openai.OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

# --- Retry Configuration ---
MAX_RETRIES = 2  # Maximum number of retries
RETRY_DELAY = 10 # Delay between retries in seconds

# --- Save Configuration ---
SAVE_INTERVAL = 10  # Save progress every N items

# --- Helper functions from Script 2 (Unchanged) ---

def load_data(file_path):
    """从指定路径加载 JSON 文件。"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def read_text_file(file_path: str) -> str:
    """读取文本文件并返回其内容字符串。"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_data(data, file_path):
    """将数据保存为 JSON 文件。"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def append_save_data(data, file_path):
    """追加保存数据到 JSON 文件。"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list) and isinstance(data, list):
                existing_data.extend(data)
                save_data(existing_data, file_path)
            else:
                 save_data(data, file_path) # Overwrite if not a list
        except (json.JSONDecodeError, FileNotFoundError):
            save_data(data, file_path)
    else:
        save_data(data, file_path)


def load_progress(progress_file):
    """加载进度文件，返回已处理的数据索引。"""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                progress = json.load(f)
                return set(progress.get("processed_indices", []))
        except (json.JSONDecodeError, FileNotFoundError):
            return set()
    return set()


def save_progress(progress_file, processed_indices, current_stats):
    """保存进度文件。"""
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    progress = {
        "processed_indices": list(processed_indices),
        "timestamp": time.time(),
        "stats": current_stats
    }
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=4)


# --- MODIFIED: evaluate_response now uses the OpenAI/DeepSeek request method ---
def evaluate_response(prompt: str, response_text: str, reference_text: str, template: str) -> int:
    """
    使用 OpenAI Python SDK (configured for DeepSeek) 发送请求来评估分数。
    包含重试逻辑。
    """
    prompt = prompt.split("You are a helpful assistant.\nuser\n")[1].split("\n")[0]
    
    messages = [
        {"role": "system", "content": template},
        {"role": "user", "content": f"\nPrompt: [{prompt}]\nResponse: [{response_text}]\nReference Answer:[{reference_text}]\n Score:"},
    ]
    # print(messages)

    for attempt in range(MAX_RETRIES + 1):
        try:
            # Using the openai client to make the request, as per your first script
            completion = client.chat.completions.create(
                model=API_MODEL,
                messages=messages,
                # timeout=60,
            )
            
            content = completion.choices[0].message.content.strip()
            # print(f"\nJudge Response: {content}")

            # Parse score from the judge's response
            try:
                score = int(content)
            except (ValueError, TypeError):
                match = re.search(r"(\d+)", content)
                score = int(match.group(1)) if match else None
            
            return score
            
        except openai.APIError as e:
            print(f"  API Error (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                print("  Max retries reached due to API error.")
                return None
        except Exception as e:
            print(f"  An unexpected error occurred (Attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}")
            traceback.print_exc()
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                print("  Max retries reached due to an unexpected error.")
                return None
    
    return None


# --- Main test loop from Script 2 (Unchanged) ---
def test(base_test_dir, start_dataset_idx=0, start_eval_idx=0, start_data_idx=0):
    """
    执行测试，支持断点续传。
    """
    dataset_names = [
        "Capybara-Preferences",
        "HC3",
        "hh-rlhf-harmless-base",
        "hh-rlhf-helpful-base",
        "safe-rlhf",
        "TruthfulQA",
    ]

    for i, dataset_name in enumerate(dataset_names):
        if i < start_dataset_idx:
            continue
            
        print(f"\n====================\n当前训练任务: {dataset_name}\n====================")
        
        for j in range(i + 1): # Evaluate on all datasets
            if i == start_dataset_idx and j < start_eval_idx:
                continue
                
            eval_dataset_name = dataset_names[j]
            
            prompt_template_path = f"llm_judge/{eval_dataset_name}.txt"
            if not os.path.exists(prompt_template_path):
                print(f"  Warning: Prompt template not found, skipping - {prompt_template_path}")
                continue
            prompt_template = read_text_file(prompt_template_path)
            
            test_path = f"{base_test_dir}/{dataset_name}/{eval_dataset_name}/generated_predictions.json"
            metric_path = f"{base_test_dir}/{dataset_name}/{eval_dataset_name}/predict_results.json"
            new_pred_path = f"{base_test_dir}/{dataset_name}/{eval_dataset_name}/generated_predictions-dpsk-judged.json"
            progress_path = f"{base_test_dir}/{dataset_name}/{eval_dataset_name}/progressed.json"
            
            print(f"  > 正在评估任务: {eval_dataset_name}")
            print(f"    读取测试文件: {test_path}")

            try:
                metric = load_data(metric_path)
                predict_data = load_data(test_path)
            except FileNotFoundError:
                print(f"    警告: 文件未找到，跳过此路径 - {test_path}")
                continue

            processed_indices = load_progress(progress_path)
            
            if i == start_dataset_idx and j == start_eval_idx:
                processed_indices = set(range(start_data_idx))
            
            avg_score, bad_attemp = [], []
            batch_predictions = []
            
            # Load existing results to correctly calculate average score
            if os.path.exists(new_pred_path):
                try:
                    existing_predictions = load_data(new_pred_path)
                    if isinstance(existing_predictions, list):
                        for pred in existing_predictions:
                            if "score" in pred and pred["score"] is not None:
                                avg_score.append(pred["score"])
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            
            total_items = len(predict_data)
            processed_count = len(processed_indices)
            
            print(f"    总数据量: {total_items}, 已处理: {processed_count}, 剩余: {total_items - processed_count}")
            
            for k, predict in enumerate(tqdm(predict_data, desc=f"Eval {dataset_name}/{eval_dataset_name}")):
                if k in processed_indices:
                    continue
                
                score = evaluate_response(predict["prompt"], predict["predict"], predict["label"], prompt_template)
                
                if score is None:
                    bad_attemp.append(k)
                else:
                    predict["score"] = score
                    avg_score.append(score)
                    batch_predictions.append(predict)
                
                processed_indices.add(k)
                
                if len(batch_predictions) >= SAVE_INTERVAL or k == total_items - 1:
                    if batch_predictions:
                        append_save_data(batch_predictions, new_pred_path)
                        batch_predictions = []
                    
                    current_stats = {
                        "processed": len(processed_indices),
                        "total": total_items,
                        "avg_score": 10 * float(np.mean(avg_score)) if avg_score else None,
                        "errors": len(bad_attemp)
                    }
                    save_progress(progress_path, processed_indices, current_stats)
                    
            # Final metric calculation and save
            metric['LLM-judge'] = 10 * float(np.mean(avg_score)) if avg_score else None
            metric["error"] = len(bad_attemp)
            metric["processed"] = len(processed_indices)
            metric["total"] = total_items
            save_data(metric, metric_path)
            
            print(f"    任务完成: {eval_dataset_name}")
            print(f"    平均分数: {metric['LLM-judge']:.2f}" if avg_score else "    平均分数: None")
            print(f"    错误数量: {len(bad_attemp)}")

    print("所有处理完成。")


if __name__ == "__main__":
    model_name = "Qwen2.5-7B-Instruct"
    methods = ["DPO+CPPO-w-replay-order1"]
    
    # Set parameters to resume from a breakpoint
    START_DATASET_IDX = 0
    START_EVAL_IDX = 0
    START_DATA_IDX = 0
    
    for method in methods:
        test_dir = f"save_test/{model_name}/lora/{method}"
        print(f"\n{'='*50}")
        print(f"开始处理方法: {method}")
        print(f"{'='*50}")
        test(test_dir, START_DATASET_IDX, START_EVAL_IDX, START_DATA_IDX)