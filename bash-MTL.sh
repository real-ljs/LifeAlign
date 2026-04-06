#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -e
# 执行时打印出相应的命令，方便调试
# set -x

# ==================== Configuration / 配置区 ====================
GPU_ID=6

MODEL_PATH=/mnt/workspace2/models/Qwen2.5-7B-Instruct

BASE_ADAPTER_PATH=saves/Qwen2.5-7B-Instruct/lora/DPO+MTL-0618

BASE_OUTPUT_DIR=save_test/Qwen2.5-7B-Instruct/lora/DPO+MTL-0618

declare -a TASK_NAMES=(
    "Capybara-Preferences"
    "HC3"
    "hh-rlhf-harmless-base"
    "hh-rlhf-helpful-base"
    "safe-rlhf"
    "TruthfulQA"
)
# ==================== Continual Training Loop / 持续学习训练循环 ====================
TRAIN_DATASETS_STRING="CRL/Capybara-Preferences,CRL/HC3,CRL/hh-rlhf-harmless-base,CRL/hh-rlhf-helpful-base,CRL/safe-rlhf,CRL/TruthfulQA"
EVAL_DATASETS_STRING="CRL/Capybara-Preferences-test-sft, CRL/HC3-test-sft, CRL/hh-rlhf-harmless-base-test-sft, CRL/hh-rlhf-helpful-base-test-sft, CRL/safe-rlhf-test-sft, CRL/TruthfulQA-test-sft"
mkdir -p "${BASE_ADAPTER_PATH}"

TRAIN_LOG_FILE="${BASE_ADAPTER_PATH}/run_train.log"

CUDA_VISIBLE_DEVICES=${GPU_ID} llamafactory-cli train \
        --model_name_or_path ${MODEL_PATH} \
        --stage dpo \
        --do_train \
        --finetuning_type lora \
        --lora_target q_proj,v_proj \
        --pref_beta 0.1 \
        --pref_loss sigmoid \
        --CL_method MTL \
        --dataset ${TRAIN_DATASETS_STRING} \
        --template qwen \
        --cutoff_len 2048 \
        --overwrite_cache \
        --preprocessing_num_workers 4 \
        --output_dir ${BASE_ADAPTER_PATH} \
        --logging_steps 10 \
        --save_steps 1000 \
        --plot_loss \
        --overwrite_output_dir \
        --report_to none \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5.0e-6 \
        --num_train_epochs 3.0 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --bf16 \
        --ddp_timeout 180000000 > "${TRAIN_LOG_FILE}" 2>&1

# ==================== Continual Evaluation Loop / 持续学习评测循环 ====================
mkdir -p "${BASE_OUTPUT_DIR}"

EVAL_LOG_FILE="${BASE_OUTPUT_DIR}/run_predict.log"

CUDA_VISIBLE_DEVICES=${GPU_ID} llamafactory-cli train \
        --model_name_or_path ${MODEL_PATH} \
        --adapter_name_or_path ${BASE_ADAPTER_PATH} \
        --stage sft \
        --do_predict \
        --finetuning_type lora \
        --eval_dataset ${EVAL_DATASETS_STRING} \
        --template qwen \
        --cutoff_len 2048 \
        --overwrite_cache \
        --preprocessing_num_workers 4 \
        --max_new_tokens 1024 \
        --output_dir ${OUTPUT_DIR} \
        --overwrite_output_dir \
        --report_to none \
        --per_device_eval_batch_size 4 \
        --predict_with_generate \
        --ddp_timeout 180000000 > "${EVAL_LOG_FILE}" 2>&1

echo "All continual learning evaluations have been successfully completed!"