# SFTwithL2P.py
import os
import torch
import json
import numpy as np
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from .SFTtrainer import SFTTrainer
from typing_extensions import override
from .CL_methods.L2P import L2P
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)


class SFTwithL2PTrainer(SFTTrainer):
    """
    SFTTrainer integrated with Learning to Prompt (L2P) for continual learning.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        logger.info("Initializing SFTwithL2PTrainer.")
        
        # L2P-specific arguments from finetuning_args
        if hasattr(self.model, "base_model"):
            logger.info("LoRA model detected. Using `model.base_model` for L2P Manager.")
            base_model = self.model.base_model
        else:
            logger.info("Base model detected. Using `model` for L2P Manager.")
            base_model = self.model
        
        # Initialize L2P manager
        self.l2p_manager = L2P(model=base_model)
        if not self.l2p_manager.prompt_pool.is_leaf:
            logger.warning("Re-creating L2P prompt_pool as a leaf tensor to make it optimizable.")
            prompt_pool_data = self.l2p_manager.prompt_pool.detach().clone()
            self.l2p_manager.prompt_pool = torch.nn.Parameter(prompt_pool_data, requires_grad=True)
        # Freeze the base model parameters, only train the prompt pool
        logger.info("Freezing base model parameters for L2P.")
        for name, param in self.model.named_parameters():
            if "prompt" not in name: # A simple check, assuming L2P params won't have 'prompt' in name
                param.requires_grad = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Overrides the default compute_loss to inject L2P logic.
        """
        # 1. Prepare inputs with prompts using the L2P manager
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs.get("labels")

        final_input_embeds, final_attention_mask, loss_context = self.l2p_manager._prepare_inputs(
            input_ids, attention_mask
        )

        # 2. Prepare new labels for the combined input
        if labels is not None:
            prefix_length = final_input_embeds.shape[1] - labels.shape[1]
            prefix_labels = torch.full(
                (labels.shape[0], prefix_length),
                IGNORE_INDEX,
                device=self.model.device,
                dtype=labels.dtype,
            )
            final_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            final_labels = None

        # 3. Get the base model's output (and loss)
        outputs = model(
            inputs_embeds=final_input_embeds,
            attention_mask=final_attention_mask,
            labels=final_labels,
            return_dict=True,
        )
        base_loss = outputs.loss

        # 4. Calculate L2P's diversity loss
        diversity_loss = self.l2p_manager.calculate_diversity_loss(loss_context)

        # 5. Combine losses
        total_loss = base_loss + diversity_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Saves the model and the L2P prompt pool.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        super()._save(output_dir, state_dict)
        if self.l2p_manager:
            self.l2p_manager.save_prompt_pool(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        """
        Loads the model and the L2P prompt pool from a checkpoint.
        """
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        if self.l2p_manager:
            self.l2p_manager.load_prompt_pool(resume_from_checkpoint)

    # We need to tell the trainer to use L2P's parameters for optimization
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
            logger.info(f"Continual learning: Loading model state from previous task at {previous_task_output_dir}")
            self._load_from_checkpoint(previous_task_output_dir)
        else:
            logger.info("Continual learning: Starting from scratch or first task, no previous state loaded.")

        self.train_dataset = dataset
        self.args = args
        self.args.output_dir = output_dir 

        self.train()
        logger.info(f"Continual learning: Saving final model state for task '{task_name}' to {output_dir}")
        self.save_model(output_dir)