# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
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

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override
from peft import PeftModel, LoraConfig
from peft.tuners.lora import LoraLayer
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .vanillaDPOTrainer import vanillaDPOTrainer
from .CL_methods.CLManager import ContinualLearningManager
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput
    from transformers.modeling_utils import PreTrainedModel
    from transformers import TrainerControl
    from transformers import TrainingArguments
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)

class MyDPOTrainer(vanillaDPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        **kwargs,
    ):
        super().__init__(
            model=model,
            ref_model=ref_model,
            finetuning_args=finetuning_args,
            **kwargs
        )
    
    def mcpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Memory Consolidation-based Preference Learning (MCPL) Loss
        
        基于记忆巩固理论的偏好学习损失函数，用统一的巩固强度处理所有样本类型
        
        核心思想：
        - 不确定样本(logits≈0): 编码阶段，高巩固强度
        - 错误样本(logits<0): 重新巩固阶段，最高巩固强度  
        - 正确样本(logits>0): 维护阶段，递减巩固强度
        
        Args:
            policy_chosen_logps: Policy model log probabilities for chosen responses
            policy_rejected_logps: Policy model log probabilities for rejected responses  
            reference_chosen_logps: Reference model log probabilities for chosen responses
            reference_rejected_logps: Reference model log probabilities for rejected responses
            
        Returns:
            Tuple of (losses, chosen_rewards, rejected_rewards, mcpl_metrics)
        """
        # 确保tensor在正确的device上
        device = self.accelerator.device
        policy_chosen_logps = policy_chosen_logps.to(device)
        policy_rejected_logps = policy_rejected_logps.to(device) 
        reference_chosen_logps = reference_chosen_logps.to(device)
        reference_rejected_logps = reference_rejected_logps.to(device)
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = self.beta * (pi_logratios - ref_logratios)
        
        error_penalty = 2
        
        consolidation_weight = torch.where(
            logits < 0,
            # 情况3: 错误样本 - 权重 = 1 + λ × |X|
            1.0 + error_penalty * torch.abs(logits),
            # 情况1&2: 不确定&正确样本 - 权重 = exp(-X)
            torch.exp(-logits)
        )
        
        base_dpo_loss = -F.logsigmoid(logits)
        
        mcpl_losses = base_dpo_loss * consolidation_weight
        
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return mcpl_losses, chosen_rewards, rejected_rewards

    def fpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the Self-Correcting Focal Preference Optimization (SC-FPO) loss.
        This version includes robust device placement for use with Hugging Face Accelerate.
        """
        policy_chosen_logps = policy_chosen_logps.to(self.accelerator.device)
        policy_rejected_logps = policy_rejected_logps.to(self.accelerator.device)
        reference_chosen_logps = reference_chosen_logps.to(self.accelerator.device)
        reference_rejected_logps = reference_rejected_logps.to(self.accelerator.device)
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = pi_logratios - ref_logratios

        p = torch.sigmoid(self.beta * logits)
        log_p = F.logsigmoid(self.beta * logits)

        focal_weight = torch.pow(1 - p, 2)
        
        sc_fpo_loss = -focal_weight * log_p
        
        losses = sc_fpo_loss
        
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError("Unknown loss type: {}.".format(self.loss_type))

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            if self.finetuning_args.loss_func == "FPO":
                losses, chosen_rewards, rejected_rewards = self.fpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            else:
                losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
                )

        return losses, chosen_rewards, rejected_rewards
    
    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        primary_task_loss = losses.mean()
        logger.info(f"RL Loss: {primary_task_loss:.4f}")
        # --- Continual Learning Regularization ---
        final_loss = primary_task_loss
        # logger.info(f"FPO Loss: {final_loss:.4f}")
        # if self.cl_manager and self.cl_manager.task_index > 0:
        #     reg_loss = self.cl_manager.calculate_regularization_loss()
        #     final_loss = self.cl_manager.combine_losses(primary_task_loss, reg_loss)

        #     metrics[f"{'eval_' if train_eval == 'eval' else ''}cl_reg_loss"] = reg_loss.detach().cpu().item()
        #     metrics[f"{'eval_' if train_eval == 'eval' else ''}dpo_loss_pre_cl"] = primary_task_loss.detach().cpu().item()

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.mean().cpu()
        metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.mean().cpu()
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
        metrics["{}rewards/margins".format(prefix)] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics["{}logps/rejected".format(prefix)] = policy_rejected_logps.detach().mean().cpu()
        metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().mean().cpu()
        metrics["{}logits/rejected".format(prefix)] = policy_rejected_logits.detach().mean().cpu()
        metrics["{}logits/chosen".format(prefix)] = policy_chosen_logits.detach().mean().cpu()
        if self.loss_type == "orpo":
            metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean().cpu()
            metrics["{}odds_ratio_loss".format(prefix)] = ((losses - sft_loss) / self.beta).detach().mean().cpu()
            if not (self.cl_state and self.task_index > 0 and hasattr(model, "cl_alpha_logit")):
                 metrics[f"{prefix}loss"] = primary_task_loss.detach().cpu() # Total loss if no CL
            else: # If CL is active, final_loss is the one returned, dpo_loss_pre_cl is logged
                 metrics[f"{prefix}loss"] = final_loss.detach().cpu()

        return final_loss, metrics