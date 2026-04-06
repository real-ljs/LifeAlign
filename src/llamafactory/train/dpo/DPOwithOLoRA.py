import os
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union
import torch
from typing_extensions import override

from ...extras.logging import get_logger
from .vanillaDPOTrainer import vanillaDPOTrainer  # 假设你的 DPO trainer 在这个路径
from .CL_methods.Olora import OLoRA

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from ...hparams import FinetuningArguments

logger = get_logger(__name__)


class DPOwithOLoRATrainer(vanillaDPOTrainer):
    """
    Extends vanillaDPOTrainer to integrate O-LoRA orthogonal and L2 regularization losses,
    and manage adapter merging across continual tasks using output directories.
    """
    
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        olora_lambda: float = 0.1,
        l2_lambda: float = 0.01,
        **kwargs,
    ) -> None:
        # Initialize base DPO trainer
        super().__init__(
            model=model,
            ref_model=ref_model,
            finetuning_args=finetuning_args,
            processor=processor,
            **kwargs
        )
        
        # Initialize O-LoRA manager
        self.olora = OLoRA(
            model=self.model,
            orthogonal_lambda=olora_lambda,
            l2_lambda=l2_lambda,
            output_dir=self.args.output_dir,
            device=next(self.model.parameters()).device
        )
        
        logger.info(f"Initialized DPOwithOLoRA with orthogonal_lambda={olora_lambda}, l2_lambda={l2_lambda}")

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """
        Computes the DPO loss with O-LoRA regularization and other metrics for the given batch.
        """
        # Get base DPO loss and metrics
        dpo_loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)
        
        # Add O-LoRA regularization only during training
        if train_eval == "train":
            # Compute O-LoRA penalties
            orthogonal_loss = self.olora.compute_orthogonal_loss()
            l2_loss = self.olora.compute_l2_loss()
            
            # Add to total loss
            total_loss = dpo_loss + orthogonal_loss + l2_loss
            
            # Add O-LoRA metrics
            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}olora/orthogonal_loss"] = orthogonal_loss.detach().cpu()
            metrics[f"{prefix}olora/l2_loss"] = l2_loss.detach().cpu()
            metrics[f"{prefix}olora/total_penalty"] = (orthogonal_loss + l2_loss).detach().cpu()
            metrics[f"{prefix}dpo_loss"] = dpo_loss.detach().cpu()
            metrics[f"{prefix}total_loss"] = total_loss.detach().cpu()
            
            # Log detailed O-LoRA info periodically
            if hasattr(self, 'state') and self.state.global_step % 100 == 0:
                has_historical = self.olora.merged_historical_weights is not None
                logger.info(f"Step {self.state.global_step}: DPO={dpo_loss:.4f}, "
                           f"Orthogonal={orthogonal_loss:.4f}, L2={l2_loss:.4f}, "
                           f"HasHistorical={has_historical}")
            
            return total_loss, metrics
        else:
            # During evaluation, only return DPO loss without O-LoRA penalties
            return dpo_loss, metrics

    def train_single_dataset(
        self,
        dataset,
        args,
        task_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        previous_task_output_dir: Optional[str] = None,
    ):
        """
        Performs DPO training on a single dataset for continual learning,
        then saves model, merged O-LoRA adapter, and logs accordingly.
        
        Args:
            dataset: Training dataset for DPO
            args: Training arguments
            task_name: Name of current task (for logging)
            output_dir: Directory to save current task outputs
            previous_task_output_dir: Directory containing previous task's outputs
        """
        # Setup dataset and args
        self.train_dataset = dataset
        self.args = args
        
        if output_dir:
            self.args.output_dir = output_dir
            self.olora.set_output_dir(output_dir)
        
        # Load previous task's merged weights if provided
        if previous_task_output_dir:
            if self.olora.load_merged_weights(previous_task_output_dir):
                logger.info(f"Successfully loaded merged weights from '{previous_task_output_dir}' for DPO task '{task_name or 'unnamed'}'")
            else:
                logger.warning(f"Failed to load merged weights from '{previous_task_output_dir}'. "
                             "This might be okay if it's the first task.")
        else:
            logger.info(f"No previous task output directory provided for DPO task '{task_name or 'unnamed'}'. Treating as first task.")
        
        # Log training setup
        logger.info(f"Starting DPO training for task '{task_name or 'unnamed'}' in directory '{self.args.output_dir}'")
        if self.olora.merged_historical_weights is not None:
            num_historical = len(self.olora.merged_historical_weights)
            logger.info(f"Using {num_historical} historical weight modules for orthogonal regularization")
        
        # Run DPO training loop
        self.train()
        
        # Save final model state
        if output_dir:
            logger.info(f"Saving final DPO model state for task '{task_name or 'unnamed'}' to {output_dir}")
            self.save_model(output_dir)
        
        # Save merged adapter for continual learning
        if self.olora.save_merged_adapter(previous_task_output_dir):
            logger.info(f"Successfully saved merged O-LoRA adapter for DPO task '{task_name or 'unnamed'}'")
        else:
            logger.error(f"Failed to save merged O-LoRA adapter for DPO task '{task_name or 'unnamed'}'")