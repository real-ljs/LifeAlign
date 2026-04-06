# LifeAlign

![](Images/framework.png)

本仓库是论文 [LifeAlign: Lifelong Alignment for Large Language Models with Memory-Augmented Focalized Preference Optimization](https://arxiv.org/abs/2509.17183) 的代码实现。

项目基于 LlamaFactory 扩展，实现了面向大语言模型持续对齐的训练流程。按照论文思路，方法核心包含两部分：

- Focalized Preference Optimization, FPO：在偏好优化过程中根据样本难度自适应分配学习力度。
- Short-to-Long Memory Consolidation, SLMC：对任务更新进行去噪，并投影到稳定的长期记忆子空间中，以缓解灾难性遗忘。

代码中与方法最相关的模块包括：

- `src/llamafactory/train/dpo/MyDPOTrainer.py`：FPO 风格的偏好学习损失。
- `src/llamafactory/train/dpo/CL_methods/CLManager.py`：记忆巩固、SVD 去噪与子空间投影。
- `src/llamafactory/train/dpo/BaseTrainerCL.py`：持续学习主流程，以及内嵌的 SFT 初始化阶段。

## 1. 环境配置

由于本项目建立在 LlamaFactory 之上，因此环境准备基本可以直接沿用 LlamaFactory 的方式。

### 1.1 创建 Python 环境

```bash
conda create -n lifealign python=3.10 -y
conda activate lifealign
pip install --upgrade pip
```

### 1.2 安装 PyTorch

请先根据本机 CUDA 版本安装对应的 PyTorch。下面以 CUDA 12.1 为例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果你的 CUDA 版本不同，请替换为对应的官方 wheel 源。

### 1.3 安装本项目

```bash
pip install -e ".[metrics]"
```

如果你需要使用 DeepSpeed，可以额外安装：

```bash
pip install -e ".[deepspeed]"
```

安装完成后，主要训练入口为：

```bash
llamafactory-cli train ...
```

## 2. 数据组织

本仓库的数据集注册信息位于 `data/dataset_info.json`。当前持续对齐任务主要包含：

- `Capybara-Preferences`
- `HC3`
- `hh-rlhf-harmless-base`
- `hh-rlhf-helpful-base`
- `safe-rlhf`
- `TruthfulQA`

同时，每个任务还提供：

- `-sft`：用于 SFT 初始化
- `-test-sft`：用于生成式评测

## 3. 启动方式

本项目支持三种常用启动方式。

### 3.1 使用 `bash.sh` 全自动启动

`bash.sh` 是一个全自动脚本，包含：

- 任务顺序构建
- 持续对齐训练
- 对当前任务及历史任务的自动评测

启动方式：

```bash
bash bash.sh
```

运行前建议先按需修改 `bash.sh` 中的以下变量：

- `ORDER`：任务顺序，例如 `order1`、`order2`、`order3`
- `GPU_ID`
- `MODEL_NAME` / `MODEL_PATH`
- `BASE_ADAPTER_PATH`
- `BASE_OUTPUT_DIR`

仓库中还提供了其他相关脚本：

- `bash-MTL.sh`：多任务训练基线
- `bash-Sep.sh`：单任务/分任务训练基线
- `predict.sh`：仅执行评测

说明：

- 当前仓库中的 `bash.sh` 是一个可直接运行的自动化模板。
- 如果你希望更明确地按论文中的 LifeAlign 设定运行，请重点关注 `--CL_method`、`--loss_func`、`--use_replay`、`--denoising_threshold`、`--projection_gamma` 这些参数。

### 3.2 使用 `llamafactory-cli train + 显式参数`

下面给出一个 LifeAlign 风格的持续对齐训练示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train \
  --model_name_or_path /path/to/Qwen2.5-7B-Instruct \
  --stage dpo \
  --do_train \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --pref_beta 0.1 \
  --pref_loss sigmoid \
  --CL_method my_method \
  --loss_func FPO \
  --use_replay \
  --denoising_threshold 0.9 \
  --projection_gamma 0.5 \
  --dataset Capybara-Preferences,HC3,hh-rlhf-harmless-base,hh-rlhf-helpful-base,safe-rlhf,TruthfulQA \
  --template qwen \
  --cutoff_len 2048 \
  --overwrite_cache \
  --preprocessing_num_workers 4 \
  --output_dir saves/Qwen2.5-7B-Instruct/lora/LifeAlign-order1 \
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
  --bf16
```

评测示例：

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  --model_name_or_path /path/to/Qwen2.5-7B-Instruct \
  --adapter_name_or_path saves/Qwen2.5-7B-Instruct/lora/LifeAlign-order1/TruthfulQA \
  --stage sft \
  --do_predict \
  --finetuning_type lora \
  --eval_dataset Capybara-Preferences-test-sft,HC3-test-sft,hh-rlhf-harmless-base-test-sft,hh-rlhf-helpful-base-test-sft,safe-rlhf-test-sft,TruthfulQA-test-sft \
  --template qwen \
  --cutoff_len 2048 \
  --overwrite_cache \
  --preprocessing_num_workers 4 \
  --max_new_tokens 1024 \
  --output_dir save_test/Qwen2.5-7B-Instruct/lora/LifeAlign-order1/TruthfulQA \
  --overwrite_output_dir \
  --report_to none \
  --per_device_eval_batch_size 4 \
  --predict_with_generate
```

### 3.3 使用 YAML 配置文件启动

你也可以直接通过 YAML 配置文件启动训练：

```bash
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml
```

或者使用你自己的配置文件：

```bash
llamafactory-cli train /path/to/your_config.yaml
```

仓库中已经提供了一些示例配置：

- `examples/train_lora/llama3_lora_dpo.yaml`
- `examples/train_lora/llama3_lora_sft.yaml`
- `examples/train_lora/llama3_lora_sft_initialize.yaml`

## 4. SFT 初始化说明

`examples/train_lora/llama3_lora_sft_initialize.yaml` 是 SFT 初始化阶段的配置文件。

在当前代码实现中，这一步不是可选示例，而是内嵌在训练流程中的固定步骤：

- 每个持续对齐训练任务都会先执行一次 SFT 初始化
- 然后再进入后续的对齐训练阶段

因此，如果你需要修改以下内容：

- 模型基座
- template
- LoRA 目标模块
- 初始化阶段使用的数据集
- 输出目录
- batch size、learning rate 等超参数

都需要手动修改：

- `examples/train_lora/llama3_lora_sft_initialize.yaml`

再启动训练。

## 5. 输出目录

训练和评测结果通常保存在：

- `saves/...`：checkpoint 与 LoRA adapter
- `save_test/...`：生成结果与评测结果

对于持续学习任务，一般会在主输出目录下按任务拆分子目录保存。

## 6. 额外说明

- 本仓库大部分训练接口和使用习惯与 LlamaFactory 保持一致。
- 当检测到多张可见 GPU 时，`llamafactory-cli train` 会自动走分布式训练入口。
- 在正式跑大规模实验前，建议先确认模型路径、数据集注册名、template 名称以及输出目录是否正确。

## 7. 引用

如果本项目对你的研究有帮助，欢迎引用论文：

```bibtex
@article{li2025lifealign,
  title={LifeAlign: Lifelong Alignment for Large Language Models with Memory-Augmented Focalized Preference Optimization},
  author={Li, Junsong and Zhou, Jie and Zhan, Bihao and Yang, Yutao and Pan, Qianjun and Chen, Shilian and Huai, Tianyu and Li, Xin and Chen, Qin and He, Liang},
  journal={arXiv preprint arXiv:2509.17183},
  year={2025}
}
```
