# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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
import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import importlib.metadata
import numpy as np
import torch
import gc
from transformers import Seq2SeqTrainer, TrainerCallback, PreTrainedModel
import torch.nn as nn
from typing_extensions import override
from peft import PeftModel
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from deepspeed.runtime.zero.partition_parameters import GatheredParameters
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from .SFTtrainer import SFTTrainer
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from transformers.utils import is_peft_available
from packaging import version
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)

class SFTwithEWCTrainer(SFTTrainer):
    """
    继承 Seq2SeqTrainer 并实现 EWC 正则化，防止灾难性遗忘。
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        **kwargs,
    ) -> None:
        super().__init__(finetuning_args, **kwargs)
        self.finetuning_args = finetuning_args
        self.ewc_manager = None

    def get_batch_loss_metrics(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        train_eval: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the base SFT loss for a batch of inputs.
        This method provides a unified interface for the EWCManager and
        returns the pure task loss, without the EWC penalty.
        """
        # We use super().compute_loss() to get the original loss calculation
        # from the parent transformers.Trainer.
        loss = super().compute_loss(model, inputs, return_outputs=False)
        
        # DPO trainer's method returns (loss, metrics_dict). We mimic that.
        # For SFT, there are no extra metrics to return here.
        return loss, {}

    @override
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overrides the compute_loss method to add the EWC regularization term.
        """
        # Get the standard SFT loss from the parent class
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        # Add EWC loss if the manager is ready
        if self.ewc_manager and self.ewc_manager._is_ready:
            ewc_loss = self.ewc_manager.compute_loss()
            total_loss = loss + ewc_loss
            
            if self.state.global_step % 10 == 0:
                self.log({"loss": loss.item(), "ewc_loss": ewc_loss.item(), "total_loss": total_loss.item()})
            
            return (total_loss, outputs) if return_outputs else total_loss
        
        return (loss, outputs) if return_outputs else loss