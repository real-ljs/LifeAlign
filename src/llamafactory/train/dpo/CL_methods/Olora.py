import os
import json
import torch
import torch.nn as nn
import re
import logging
from dataclasses import dataclass
from safetensors.torch import load_file as safe_load_file

logger = logging.getLogger(__name__)

@dataclass
class AdapterInfo:
    task_id: str
    path: str
    config: dict
    is_first_task: bool = False

class OLoRA:
    """
    Simplified O-LoRA without any distributed or DeepSpeed/Accelerate logic.
    Modified to work with output directories instead of task IDs.
    """
    def __init__(
        self,
        model: nn.Module,
        orthogonal_lambda: float = 0.1,
        l2_lambda: float = 0.01,
        output_dir: str = "model_output",
        device: str = "cpu"
    ):
        self.model = model
        self.orthogonal_lambda = orthogonal_lambda
        self.l2_lambda = l2_lambda
        self.output_dir = os.path.abspath(output_dir)
        self.device = device
        self.merged_historical_weights = None

    def set_output_dir(self, output_dir):
        """Set the current output directory."""
        self.output_dir = os.path.abspath(output_dir)

    def _validate_adapter_path(self, path: str) -> str:
        """Validate that the adapter path contains required files and compatible config."""
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise ValueError(f"Adapter path not found: {path}")
        
        config_path = os.path.join(path, "adapter_config.json")
        if not os.path.isfile(config_path):
            raise ValueError(f"Missing adapter_config.json in {path}")
        
        safetensors_path = os.path.join(path, "adapter_model.safetensors")
        if not os.path.isfile(safetensors_path):
            raise ValueError(f"Missing adapter_model.safetensors in {path}")
        
        # Validate config compatibility
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Log config info for debugging
            logger.debug(f"Adapter config from {path}: "
                        f"r={config.get('r')}, alpha={config.get('lora_alpha')}, "
                        f"target_modules={config.get('target_modules')}")
            
            # Check for critical incompatibilities
            if config.get('peft_type') != 'LORA':
                logger.warning(f"Non-LoRA adapter found in {path}: {config.get('peft_type')}")
            
        except Exception as e:
            logger.warning(f"Could not validate adapter config in {path}: {e}")
        
        return path

    def load_adapter_weights(self, adapter_path: str) -> dict:
        """
        Load adapter weights from a given path.
        
        Args:
            adapter_path: Path to the adapter directory containing adapter_model.safetensors
            
        Returns:
            Dictionary of adapter weights
        """
        path = self._validate_adapter_path(adapter_path)
        # Load state dict
        state = safe_load_file(os.path.join(path, "adapter_model.safetensors"), device="cpu")
        weights = {}
        for k, v in state.items():
            if ".lora_A." in k or ".lora_B." in k:
                module_path = k.split('.lora_')[0]
                typ = 'merged_A' if '.lora_A.' in k else 'merged_B'
                weights[f"{module_path}.{typ}"] = v
        return weights

    def _validate_weight_compatibility(self, weights: dict, source_path: str) -> bool:
        """
        Validate that the loaded weights are compatible with current model.
        
        Args:
            weights: Loaded weights dictionary
            source_path: Path where weights were loaded from
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            # Check if we have current model LoRA modules to compare against
            current_modules = {}
            for name, module in self.model.named_modules():
                if hasattr(module, 'lora_A') and 'default' in getattr(module.lora_A, 'keys', lambda: [])():
                    current_modules[f"{name}.merged_A"] = module.lora_A['default'].weight.shape
                    if hasattr(module, 'lora_B') and 'default' in getattr(module.lora_B, 'keys', lambda: [])():
                        current_modules[f"{name}.merged_B"] = module.lora_B['default'].weight.shape
            
            # Check compatibility of each weight
            for key, weight in weights.items():
                if key in current_modules:
                    expected_shape = current_modules[key]
                    # For A matrices: new weights should have same input dim (dim 1)
                    # For B matrices: new weights should have same output dim (dim 0)
                    if key.endswith('.merged_A'):
                        if weight.shape[1] != expected_shape[1]:
                            logger.error(f"Incompatible A matrix shape for {key}: "
                                       f"historical={weight.shape}, current={expected_shape}")
                            return False
                    elif key.endswith('.merged_B'):
                        if weight.shape[0] != expected_shape[0]:
                            logger.error(f"Incompatible B matrix shape for {key}: "
                                       f"historical={weight.shape}, current={expected_shape}")
                            return False
            
            logger.info(f"Weight compatibility validated for {len(weights)} modules from {source_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating weight compatibility: {e}")
            return False

    def load_merged_weights(self, previous_output_dir: str) -> bool:
        """
        Load merged weights from previous task output directory with compatibility validation.
        
        Args:
            previous_output_dir: Path to previous task's output directory
            
        Returns:
            True if successfully loaded, False otherwise
        """
        merged_path = os.path.join(previous_output_dir, "merged_adapter.pt")
        weights = None
        
        # Try to load merged weights first
        if os.path.isfile(merged_path):
            try:
                weights = torch.load(merged_path, map_location=self.device)
                logger.info(f"Loaded merged weights from {merged_path}")
            except Exception as e:
                logger.error(f"Failed to load merged weights from {merged_path}: {e}")
        
        # Fallback: try to load current adapter weights
        if weights is None:
            try:
                weights = self.load_adapter_weights(previous_output_dir)
                # Move weights to correct device
                weights = {k: v.to(self.device) for k, v in weights.items()}
                logger.info(f"Loaded adapter weights as merged weights from {previous_output_dir}")
            except Exception as e:
                logger.error(f"Failed to load adapter weights from {previous_output_dir}: {e}")
                return False
        
        # Validate compatibility
        if not self._validate_weight_compatibility(weights, previous_output_dir):
            logger.error(f"Weight compatibility validation failed for {previous_output_dir}")
            return False
        
        self.merged_historical_weights = weights
        return True

    def save_merged_adapter(self, previous_output_dir: str = None) -> bool:
        """
        Save merged adapter weights to current output directory.
        
        Args:
            previous_output_dir: Path to previous task's output directory (None for first task)
            
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            # Load current task's adapter weights
            current_weights = self.load_adapter_weights(self.output_dir)
        except Exception as e:
            logger.error(f"Failed to load current adapter weights: {e}")
            return False

        # Determine merged weights
        if previous_output_dir is None:
            # First task: merged weights = current weights
            merged_weights = current_weights
            logger.info("First task: saving current weights as merged weights")
        else:
            # Subsequent tasks: merge with previous weights
            if self.merged_historical_weights is None:
                logger.error("Historical weights not loaded. Call load_merged_weights() first.")
                return False
            
            merged_weights = {}
            for k, v in current_weights.items():
                if k in self.merged_historical_weights:
                    # Ensure both tensors are on the same device
                    historical_weight = self.merged_historical_weights[k].to(v.device)
                    current_weight = v.to(v.device)  # Ensure current weight is on its own device
                    
                    if k.endswith('merged_A'):
                        # Concatenate along dimension 0 for A matrices
                        merged_weights[k] = torch.cat([historical_weight, current_weight], dim=0)
                    else:  # merged_B
                        # Concatenate along dimension 1 for B matrices
                        merged_weights[k] = torch.cat([historical_weight, current_weight], dim=1)
                else:
                    # New module, just use current weights
                    merged_weights[k] = v
            
            logger.info(f"Merged current weights with historical weights from {previous_output_dir}")

        # Save merged weights to current output directory
        merged_path = os.path.join(self.output_dir, "merged_adapter.pt")
        try:
            # Move weights to CPU for saving to reduce memory usage
            cpu_merged_weights = {k: v.cpu() for k, v in merged_weights.items()}
            torch.save(cpu_merged_weights, merged_path)
            logger.info(f"Saved merged adapter to {merged_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save merged adapter to {merged_path}: {e}")
            return False

    def compute_orthogonal_loss(self) -> torch.Tensor:
        """
        Compute orthogonal loss between current and historical LoRA weights.
        
        Returns:
            Orthogonal loss tensor
        """
        if self.merged_historical_weights is None:
            return torch.tensor(0.0, device=self.device)
        
        loss = torch.tensor(0.0, device=self.device)
        count = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and 'default' in getattr(module.lora_A, 'keys', lambda: [])():
                new_w = module.lora_A['default'].weight
                key = f"{name}.merged_A"
                
                if key in self.merged_historical_weights:
                    old_w = self.merged_historical_weights[key].to(new_w.device)
                    if new_w.shape[1] == old_w.shape[1]:
                        # Compute dot product for orthogonality constraint
                        dp = torch.mm(new_w, old_w.T)
                        loss = loss + dp.abs().sum()
                        count += 1
        
        if count > 0:
            logger.debug(f"Computed orthogonal loss over {count} modules")
        
        return self.orthogonal_lambda * loss

    def compute_l2_loss(self) -> torch.Tensor:
        """
        Compute L2 regularization loss on current LoRA weights.
        
        Returns:
            L2 loss tensor
        """
        loss = torch.tensor(0.0, device=self.device)
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and 'default' in getattr(module.lora_A, 'keys', lambda: [])():
                a = module.lora_A['default'].weight
                loss = loss + a.pow(2).sum()
                
                if hasattr(module, 'lora_B') and 'default' in getattr(module.lora_B, 'keys', lambda: [])():
                    b = module.lora_B['default'].weight
                    loss = loss + b.pow(2).sum()
        
        return self.l2_lambda * loss

    def get_merged_weights_path(self, output_dir: str = None) -> str:
        """
        Get the path to merged weights file.
        
        Args:
            output_dir: Output directory (defaults to current output_dir)
            
        Returns:
            Path to merged_adapter.pt file
        """
        if output_dir is None:
            output_dir = self.output_dir
        return os.path.join(output_dir, "merged_adapter.pt")

    def has_merged_weights(self, output_dir: str = None) -> bool:
        """
        Check if merged weights file exists.
        
        Args:
            output_dir: Output directory (defaults to current output_dir)
            
        Returns:
            True if merged weights file exists
        """
        return os.path.isfile(self.get_merged_weights_path(output_dir))