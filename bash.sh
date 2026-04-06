#!/bin/bash
set -euo pipefail
set -x

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 默认使用 order1
ORDER="order1"
case "${ORDER}" in
  order1)
    TASK_NAMES=(
      "Capybara-Preferences"
      "HC3"
      "hh-rlhf-harmless-base"
      "hh-rlhf-helpful-base"
      "safe-rlhf"
      "TruthfulQA"
    )
    ;;
  order2)
    TASK_NAMES=(
      "TruthfulQA"
      "safe-rlhf"
      "hh-rlhf-helpful-base"
      "hh-rlhf-harmless-base"
      "HC3"
      "Capybara-Preferences"
    )
    ;;
  order3)
    TASK_NAMES=(
      "hh-rlhf-harmless-base"
      "Capybara-Preferences"
      "TruthfulQA"
      "hh-rlhf-helpful-base"
      "HC3"
      "safe-rlhf"
    )
    ;;
  *)
    echo "Unknown order: ${ORDER}"
    echo "Usage: $0 [order1|order2|order3]"
    exit 1
    ;;
esac

# 拼接训练用的数据集字符串：<task1>,<task2>,...
TRAIN_DATASETS_STRING=$(IFS=,; echo "${TASK_NAMES[*]/#/}")

GPU_ID=1,2
MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="/mnt/workspace2/models/${MODEL_NAME}"
BASE_ADAPTER_PATH="saves/${MODEL_NAME}/lora/DPO+CPPO-w-replay-${ORDER}"
BASE_OUTPUT_DIR="save_test/${MODEL_NAME}/lora/DPO+CPPO-w-replay-${ORDER}"
TEMPLATE="qwen"

mkdir -p "${BASE_ADAPTER_PATH}"
TRAIN_LOG_FILE="${BASE_ADAPTER_PATH}/run_train.log"

# 第一阶段：DPO 训练
CUDA_VISIBLE_DEVICES="${GPU_ID}" llamafactory-cli train \
  --model_name_or_path "${MODEL_PATH}" \
  --stage dpo \
  --do_train \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --pref_beta 0.1 \
  --pref_loss sigmoid \
  --CL_method CPPO \
  --use_replay \
  --loss_func DPO \
  --denoising_threshold 0.9 \
  --projection_gamma 0.5 \
  --dataset "${TRAIN_DATASETS_STRING}" \
  --template ${TEMPLATE} \
  --cutoff_len 2048 \
  --overwrite_cache \
  --preprocessing_num_workers 4 \
  --output_dir "${BASE_ADAPTER_PATH}" \
  --logging_steps 10 \
  --save_steps 1000 \
  --plot_loss \
  --overwrite_output_dir \
  --report_to none \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5.0e-6 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  >> "${TRAIN_LOG_FILE}" 2>&1

TOTAL_TASKS=${#TASK_NAMES[@]}
for (( i=TOTAL_TASKS-1; i>=0; i-- )); do
  CURRENT_TASK="${TASK_NAMES[i]}"

  # 构建评估数据集列表（包含到当前任务的所有前序任务）
  tasks_to_evaluate=( "${TASK_NAMES[@]:0:i+1}" )
  eval_list=()
  for tname in "${tasks_to_evaluate[@]}"; do
    eval_list+=( "${tname}-test-sft" )
  done
  EVAL_DATASETS_STRING=$(IFS=,; echo "${eval_list[*]}")

  ADAPTER_PATH="${BASE_ADAPTER_PATH}/${CURRENT_TASK}"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CURRENT_TASK}"
  mkdir -p "${OUTPUT_DIR}"
  EVAL_LOG_FILE="${OUTPUT_DIR}/run_predict.log"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" llamafactory-cli train \
    --model_name_or_path "${MODEL_PATH}" \
    --adapter_name_or_path "${ADAPTER_PATH}" \
    --stage sft \
    --do_predict \
    --finetuning_type lora \
    --eval_dataset "${EVAL_DATASETS_STRING}" \
    --template ${TEMPLATE} \
    --cutoff_len 2048 \
    --overwrite_cache \
    --preprocessing_num_workers 4 \
    --max_new_tokens 1024 \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --report_to none \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    >> "${EVAL_LOG_FILE}" 2>&1
done
