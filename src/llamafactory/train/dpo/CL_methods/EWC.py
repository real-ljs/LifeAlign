# ewc.py

import torch
import torch.nn as nn
from tqdm import tqdm, trange
from transformers import Trainer
from datasets import Dataset
from ....extras.logging import get_logger
import gc

logger = get_logger(__name__)

class EWCManager:
    """
    Manages the EWC state, including Fisher information and previous model parameters.
    This manager is instantiated by the BaseCLTrainer and passed to the specific
    SFT/DPO trainers to compute the EWC loss during training.
    """
    def __init__(self, model: nn.Module, ewc_lambda: float):
        """
        Initializes the EWC Manager.

        Args:
            model (nn.Module): The model to be trained.
            ewc_lambda (float): The EWC regularization strength.
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        
        # Dictionaries to store Fisher information and optimal parameters for each parameter name
        self.fisher_info = {}
        self.previous_params = {}
        
        # A flag to indicate if the EWC parameters are ready for loss calculation
        self._is_ready = False

    def compute_fisher_matrix(self, trainer: "Trainer", dataset: "Dataset"):
        """
        Computes the Fisher Information Matrix for the current model on a given dataset.
        
        Args:
            trainer (Trainer): The trainer instance (SFT or DPO) to get the dataloader.
            dataset (Dataset): The dataset to compute the Fisher matrix on.
        """
        logger.info("Computing Fisher Information Matrix...")
        
        # Store a reference to the original dataset and replace it for Fisher computation
        original_dataset = trainer.train_dataset
        trainer.train_dataset = dataset
        # original_batch_size = trainer.args.per_device_train_batch_size
        # trainer.args.per_device_train_batch_size = 1
        dataloader = trainer.get_train_dataloader()
        # trainer.args.per_device_train_batch_size = original_batch_size

        # Initialize Fisher matrix with zeros
        self.fisher_info = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()  # Set model to evaluation mode

        num_samples = 0
        for inputs in tqdm(dataloader, desc="Computing Fisher Matrix..."):
            num_samples += 1
            self.model.zero_grad()
            loss, _ = trainer.get_batch_loss_metrics(self.model, inputs, train_eval="train")
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_info[name] += param.grad.data.pow(2)

        # Normalize by the number of samples
        if num_samples > 0:
            for name in self.fisher_info:
                self.fisher_info[name] /= num_samples
        
        logger.info(f"Fisher Information Matrix computed based on {len(dataset)} samples.")
        
        # Restore the original dataset and set model back to train mode
        trainer.train_dataset = original_dataset
        self.model.train()

    def free_memory(self, trainer):
        if hasattr(self, "_train_dataloader"):
            trainer._train_dataloader = None

        trainer.train_dataset = None

        gc.collect()
        torch.cuda.empty_cache()

    def register_ewc_params(self, trainer: "Trainer", dataset: "Dataset"):
        """
        Public method to be called after a task is finished. It computes the Fisher
        matrix and saves the current model parameters.
        
        Args:
            trainer (Trainer): The trainer instance used for the task.
            dataset (Dataset): The dataset from the completed task.
        """
        # subset_size = int(len(dataset) * 0.2)
        # ewc_samples = dataset.select(range(subset_size))
        self.free_memory(trainer)
        self.compute_fisher_matrix(trainer, dataset)
        
        # Store deep copies of the current parameters as the "previous" parameters
        self.previous_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        self._is_ready = True
        logger.info("EWC parameters (Fisher matrix and model weights) have been registered.")

    def compute_loss(self) -> torch.Tensor:
        """
        Computes the EWC regularization loss. Should be called within the training
        loop of the subsequent task.
        
        Returns:
            torch.Tensor: The EWC loss term.
        """
        if not self._is_ready:
            return torch.tensor(0.0, device=self.model.device)

        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_info:
                delta = param - self.previous_params[name]
                ewc_loss += (self.fisher_info[name] * delta.pow(2)).sum()
        
        return (self.ewc_lambda / 2.0) * ewc_loss