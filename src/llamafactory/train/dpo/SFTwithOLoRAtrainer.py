import os
from transformers import Seq2SeqTrainer
from typing import Optional
from ...extras.logging import get_logger
from .SFTtrainer import SFTTrainer
from .CL_methods.Olora import OLoRA

logger = get_logger(__name__)

class SFTwithOloraTrainer(SFTTrainer):
    """
    Extends SFTTrainer to integrate O-LoRA orthogonal and L2 regularization losses,
    and manage adapter merging across continual tasks using output directories.
    """
    def __init__(
        self,
        finetuning_args,
        processor=None,
        olora_lambda: float = 0.1,
        l2_lambda: float = 0.01,
        **kwargs,
    ) -> None:
        # Initialize base SFTTrainer
        super().__init__(finetuning_args, processor, **kwargs)
        
        # Initialize simplified O-LoRA manager
        self.olora = OLoRA(
            model=self.model,
            orthogonal_lambda=olora_lambda,
            l2_lambda=l2_lambda,
            output_dir=self.args.output_dir,
            device=next(self.model.parameters()).device
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute base SFT loss and add O-LoRA orthogonal and L2 penalties.
        """
        # Base loss and outputs
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # O-LoRA penalties
        orth_loss = self.olora.compute_orthogonal_loss()
        l2_loss = self.olora.compute_l2_loss()
        total = loss + orth_loss + l2_loss
        
        # Optional: log the individual loss components
        if hasattr(self, 'log') and self.state.global_step % 100 == 0:
            self.log({
                'sft_loss': loss.item(),
                'orthogonal_loss': orth_loss.item(),
                'l2_loss': l2_loss.item(),
                'total_loss': total.item()
            })
        
        return (total, outputs) if return_outputs else total
    
    def train_single_dataset(
        self,
        dataset,
        args,
        task_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        previous_task_output_dir: Optional[str] = None,
    ):
        """
        Performs training on a single dataset for continual learning,
        then saves model, merged O-LoRA adapter, and logs accordingly.
        
        Args:
            dataset: Training dataset
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
                logger.info(f"Successfully loaded merged weights from '{previous_task_output_dir}'")
            else:
                logger.warning(f"Failed to load merged weights from '{previous_task_output_dir}'. "
                             "This might be okay if it's the first task.")
        else:
            logger.info("No previous task output directory provided. Treating as first task.")
        
        # Run standard training loop
        logger.info(f"Starting training for task '{task_name or 'unnamed'}' in directory '{self.args.output_dir}'")
        state = super().train()
        
        # Save final model state
        if output_dir:
            logger.info(f"Saving final model state for task '{task_name or 'unnamed'}' to {output_dir}")
            self.save_model(output_dir)
        
        # Save merged adapter
        if self.olora.save_merged_adapter(previous_task_output_dir):
            logger.info(f"Successfully saved merged adapter for task '{task_name or 'unnamed'}'")
        else:
            logger.error(f"Failed to save merged adapter for task '{task_name or 'unnamed'}'")
        
        return state

    def prepare_for_next_task(self, current_output_dir: str):
        """
        Prepare the trainer for the next task by ensuring merged weights are saved.
        
        Args:
            current_output_dir: Current task's output directory
        """
        if not self.olora.has_merged_weights(current_output_dir):
            logger.warning(f"No merged weights found in {current_output_dir}. "
                         "This might cause issues for the next task.")
        else:
            logger.info(f"Merged weights are ready in {current_output_dir} for next task.")