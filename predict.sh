#!/bin/bash

set -e
set -x

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

GPU_ID=5

MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="/mnt/workspace2/models/${MODEL_NAME}"
BASE_ADAPTER_PATH="saves/${MODEL_NAME}/lora/DPO+CPPO-w-replay-order1"
BASE_OUTPUT_DIR="save_test/${MODEL_NAME}/lora/DPO+CPPO-w-replay-order1"
TEMPLATE="qwen"

ORDER="order1"
case "${ORDER}" in
  order1)
    TASK_NAMES=(
      "Capybara-Preferences"
      "HC3"
      "hh-rlhf-harmless-base"
      "hh-rlhf-helpful-base"
      # "safe-rlhf"
      # "TruthfulQA"
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

echo "Predicting process run on the GPU:${GPU_ID}. The results will be saved on ${BASE_OUTPUT_DIR}"

TOTAL_TASKS=${#TASK_NAMES[@]}

for (( i=$TOTAL_TASKS-1; i>=0; i-- )); do
    
    CURRENT_TRAIN_TASK_NAME=${TASK_NAMES[i]}
    
    echo "================================================================="
    echo " L O A D I N G   M O D E L (Reversed Order)"
    echo " Trained on Task #${i}: ${CURRENT_TRAIN_TASK_NAME}"
    echo "================================================================="

    declare -a tasks_to_evaluate=("${TASK_NAMES[@]:0:i+1}")
    
    declare -a eval_list=()
    for task_name in "${tasks_to_evaluate[@]}"; do
        eval_list+=("${task_name}-test-sft")
    done

    EVAL_DATASETS_STRING=$(IFS=,; echo "${eval_list[*]}")

    ADAPTER_PATH="${BASE_ADAPTER_PATH}/${CURRENT_TRAIN_TASK_NAME}"

    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CURRENT_TRAIN_TASK_NAME}"

    mkdir -p "${OUTPUT_DIR}"

    LOG_FILE="${OUTPUT_DIR}/run_predict.log"

    echo "--> This model will be evaluated on the following datasets:"
    echo "    ${EVAL_DATASETS_STRING}"
    echo "--> Output will be saved to: ${OUTPUT_DIR}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} llamafactory-cli train \
        --model_name_or_path ${MODEL_PATH} \
        --adapter_name_or_path ${ADAPTER_PATH} \
        --stage sft \
        --do_predict \
        --finetuning_type lora \
        --eval_dataset ${EVAL_DATASETS_STRING} \
        --template ${TEMPLATE} \
        --cutoff_len 2048 \
        --overwrite_cache \
        --preprocessing_num_workers 4 \
        --max_new_tokens 1024 \
        --output_dir ${OUTPUT_DIR} \
        --overwrite_output_dir \
        --report_to none \
        --per_device_eval_batch_size 4 \
        --predict_with_generate \
        --ddp_timeout 180000000 >> "${LOG_FILE}" 2>&1
    echo "Evaluation for model trained on '${CURRENT_TRAIN_TASK_NAME}' finished."

done

echo "All continual learning evaluations have been successfully completed!"