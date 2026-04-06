# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
from typing_extensions import override

from .vanillaDPOTrainer import vanillaDPOTrainer
from .CL_methods.L2P import L2P
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..trainer_utils import get_batch_logps


if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)


class DPOwithL2PTrainer(vanillaDPOTrainer):
    """
    DPOTrainer integrated with Learning to Prompt (L2P) for continual learning.
    This version is compatible with LoRA (PeftModel).
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        **kwargs,
    ):
        super().__init__(finetuning_args=finetuning_args, **kwargs)
        
        logger.info("Initializing DPOwithL2PTrainer.")
        
        # 正确处理LoRA模型，获取底层模型以初始化L2P
        if hasattr(self.model, "base_model"):
            logger.info("LoRA model detected. Using `model.base_model` for L2P Manager.")
            base_model = self.model.base_model
        else:
            logger.info("Base model detected. Using `model` for L2P Manager.")
            base_model = self.model

        self.l2p_manager = L2P(model=base_model)
        if not self.l2p_manager.prompt_pool.is_leaf:
            logger.warning("Re-creating L2P prompt_pool as a leaf tensor to make it optimizable.")
            prompt_pool_data = self.l2p_manager.prompt_pool.detach().clone()
            self.l2p_manager.prompt_pool = torch.nn.Parameter(prompt_pool_data, requires_grad=True)
            
        logger.info("Freezing all model parameters (including LoRA adapters) for L2P training.")
        for param in self.model.parameters():
            param.requires_grad = False
        
        self._loss_context = None # 用于在concatenated_forward和get_batch_loss_metrics之间传递上下文

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        重写此方法以在模型前向传播之前注入L2P的prompt。
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # 1. 使用L2P管理器准备带有prompt的输入
        final_input_embeds, final_attention_mask, loss_context = self.l2p_manager._prepare_inputs(
            input_ids, attention_mask
        )
        # 存储loss_context以便后续使用
        self._loss_context = loss_context
        
        # 2. 准备新的labels以匹配增加了prompt的输入长度
        prefix_length = final_input_embeds.shape[1] - labels.shape[1]
        prefix_labels = torch.full(
            (labels.shape[0], prefix_length),
            IGNORE_INDEX,
            device=self.model.device,
            dtype=labels.dtype,
        )
        final_labels = torch.cat([prefix_labels, labels], dim=1)

        # 3. 使用修改后的输入调用模型
        all_logits: "torch.Tensor" = model(
            inputs_embeds=final_input_embeds,
            attention_mask=final_attention_mask,
            return_dict=True,
            use_cache=False,
        ).logits.to(torch.float32)
        
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=final_labels)

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """
        重写此方法以加入L2P的多样性损失。
        """
        dpo_loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)

        if self._loss_context is not None and train_eval == "train":
            diversity_loss = self.l2p_manager.calculate_diversity_loss(self._loss_context)
            total_loss = dpo_loss.mean() + diversity_loss
            
            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}diversity_loss"] = diversity_loss.detach().cpu()
            return total_loss, metrics
        
        return dpo_loss.mean(), metrics

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        super()._save(output_dir, state_dict)
        if self.l2p_manager:
            self.l2p_manager.save_prompt_pool(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        if self.l2p_manager:
            self.l2p_manager.load_prompt_pool(resume_from_checkpoint)

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            optimizer_grouped_parameters = [
                {
                    "params": self.l2p_manager.get_trainable_parameters(),
                    "weight_decay": self.args.weight_decay,
                }
            ]
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def train_single_dataset(self, dataset, args, task_name=None, output_dir=None, previous_task_output_dir=None):
        if previous_task_output_dir and os.path.isdir(previous_task_output_dir):
            logger.info(f"Continual learning (DPO): Loading model state from previous task at {previous_task_output_dir}")
            self._load_from_checkpoint(previous_task_output_dir)
        else:
            logger.info("Continual learning (DPO): Starting from scratch or first task, no previous state loaded.")

        self.train_dataset = dataset
        self.args = args
        self.args.output_dir = output_dir 

        self.train()

        logger.info(f"Continual learning (DPO): Saving final model state for task '{task_name}' to {output_dir}")
        self.save_model(output_dir)
