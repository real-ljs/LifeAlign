#!/bin/bash

# 如果任何命令执行失败，立即退出脚本
set -e
# 执行时打印出相应的命令，方便调试
# set -x

# ==================== Configuration / 配置区 ====================
# 指定使用的GPU ID
GPU_ID=5

# 基础模型路径
MODEL_PATH=/mnt/workspace2/models/Qwen2.5-7B-Instruct

# LoRA适配器和评测结果的基础保存目录
BASE_ADAPTER_DIR=saves/Qwen2.5-7B-Instruct/lora/single_task
BASE_OUTPUT_DIR=save_test/Qwen2.5-7B-Instruct/lora/single_task

# 定义所有需要处理的数据集任务名称
declare -a TASK_NAMES=(
    "Capybara-Preferences"
    "HC3"
    "hh-rlhf-harmless-base"
    "hh-rlhf-helpful-base"
    "safe-rlhf"
    "TruthfulQA"
)

# ==================== Train and Evaluate Loop / 训练与评测循环 ====================

# 遍历所有任务
for TASK_NAME in "${TASK_NAMES[@]}"; do
    
    echo "================================================================="
    echo " P R O C E S S I N G   T A S K : ${TASK_NAME}"
    echo "================================================================="

    # --- 1. 训练阶段 ---
    
    # 为当前任务设置特定的适配器保存路径
    mkdir -p "${BASE_ADAPTER_DIR}"
    TRAIN_LOG_FILE="${BASE_ADAPTER_DIR}/run_train.log"

    echo "--> Step 1: Training model on dataset '${TASK_NAME}'"
    echo "--> Adapter will be saved to: ${BASE_ADAPTER_DIR}"
    echo "--> Log file: ${TRAIN_LOG_FILE}"

    # 执行DPO训练命令
    CUDA_VISIBLE_DEVICES=${GPU_ID} llamafactory-cli train \
        --model_name_or_path ${MODEL_PATH} \
        --stage dpo \
        --do_train \
        --finetuning_type lora \
        --lora_target q_proj,v_proj \
        --pref_beta 0.1 \
        --pref_loss sigmoid \
        --dataset "${TASK_NAME}" \
        --template qwen \
        --cutoff_len 2048 \
        --overwrite_cache \
        --preprocessing_num_workers 4 \
        --output_dir ${BASE_ADAPTER_DIR} \
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

    echo "--> Training for task '${TASK_NAME}' finished."

    # --- 2. 评测阶段 ---

    # 为当前任务设置特定的评测输出路径
    ADAPTER_PATH="${BASE_ADAPTER_PATH}/${TASK_NAME}"
    EVAL_OUTPUT_PATH="${BASE_OUTPUT_DIR}/${TASK_NAME}"
    mkdir -p "${EVAL_OUTPUT_PATH}"
    EVAL_LOG_FILE="${EVAL_OUTPUT_PATH}/run_predict.log"
    
    # 定义评测数据集
    EVAL_DATASET="${TASK_NAME}-test-sft"

    echo "--> Step 2: Evaluating model on dataset '${EVAL_DATASET}'"
    echo "--> Predictions will be saved to: ${EVAL_OUTPUT_PATH}"
    echo "--> Log file: ${EVAL_LOG_FILE}"

    # 执行评测命令
    CUDA_VISIBLE_DEVICES=${GPU_ID} llamafactory-cli train \
        --model_name_or_path ${MODEL_PATH} \
        --adapter_name_or_path ${ADAPTER_PATH} \
        --stage sft \
        --do_predict \
        --finetuning_type lora \
        --eval_dataset ${EVAL_DATASET} \
        --template qwen \
        --cutoff_len 2048 \
        --overwrite_cache \
        --preprocessing_num_workers 4 \
        --max_new_tokens 1024 \
        --output_dir ${EVAL_OUTPUT_PATH} \
        --overwrite_output_dir \
        --report_to none \
        --per_device_eval_batch_size 4 \
        --predict_with_generate \
        --ddp_timeout 180000000 > "${EVAL_LOG_FILE}" 2>&1

    echo "--> Evaluation for task '${TASK_NAME}' finished."
    echo ""

done

echo "================================================================="
echo " All tasks have been successfully trained and evaluated! 🎉"
echo "================================================================="