import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union, Any
from ...data import get_dataset
from ...hparams import get_train_args
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override
from ...model import load_model, load_tokenizer
from deepspeed.runtime.zero.partition_parameters import GatheredParameters
from .vanillaDPOTrainer import vanillaDPOTrainer
from ...extras.constants import IGNORE_INDEX
import gc
import torch.nn as nn
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import (
    create_custom_optimizer,
    create_custom_scheduler,
    get_batch_logps,
)
from ...data.collator import MyCollator, DataCollatorForSeq2Seq
from ...extras.logging import get_logger
from datasets import concatenate_datasets
import os

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from ...hparams import FinetuningArguments


class DPOwithEWCTrainer(vanillaDPOTrainer):
    """
    DPO Trainer with Elastic Weight Consolidation (EWC) for continual learning.
    EWC helps prevent catastrophic forgetting by penalizing changes to parameters
    that were important for previous tasks.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            ref_model=ref_model,
            finetuning_args=finetuning_args,
            processor=processor,
            **kwargs
        )
        self.ewc_manager = None

    @override
    def compute_loss(
        self,
        model: Union["PreTrainedModel", nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Overrides the compute_loss method to add the EWC regularization term.
        """
        # Get the standard DPO loss and metrics
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")
        
        # Add EWC loss if the manager is ready
        if self.ewc_manager and self.ewc_manager._is_ready:
            ewc_loss = self.ewc_manager.compute_loss()
            total_loss = loss + ewc_loss
            
            if self.state.global_step % 10 == 0:
                metrics["ewc_loss"] = ewc_loss.item()
        else:
            total_loss = loss

        # Store metrics and return
        self.store_metrics(metrics, train_eval="train")
        
        if return_outputs:
            return (total_loss, metrics)
        return total_loss
    
