import torch
import torch.nn as nn
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments, DataCollator
from transformers import Seq2SeqTrainer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import numpy as np
import os
from dataclasses import dataclass, field
import torch
from typing import Dict, List, Optional
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)

@dataclass
class ContinuationLearningState:
    deltaw_prev: Dict[str, torch.Tensor] = field(default_factory=dict)
    lora_weights_prev: Dict[str, torch.Tensor] = field(default_factory=dict)
    historical_updates: Dict[str, List[torch.Tensor]] = field(default_factory=dict)

class ContinualLearningManager:
    def __init__(
            self,
            model: PreTrainedModel, 
            denoising_threshold: float = 0.9, 
            projection_gamma: float = 0.1,
            stage: str = "SFT"
        ):
        if not isinstance(model, PeftModel):
            raise ValueError("The ContinualLearningManager currently only supports PEFT models.")
            
        self.model = model
        self.denoising_threshold = denoising_threshold
        self.projection_gamma = projection_gamma
        self.state = ContinuationLearningState()
        self.task_index = -1
        self.device_deltaw_prev = {}
        self.lora_layer_names = self._infer_lora_layer_names()
        self.stage = stage
        
        if not hasattr(self.model, "cl_alpha_logit"):
            initial_alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.model.register_parameter("cl_alpha_logit", initial_alpha)
        
        logger.info(f"ContinualLearningManager initialized.")

    def on_task_begin(self, task_index: int):
        self.task_index = task_index
        if self.task_index == 0 and self.stage == "SFT":
            logger.info(f"CL Manager for {self.stage}: Starting first task (Task {self.task_index + 1}). State is fresh.")
        else:
            logger.info(f"CL Manager for {self.stage}: Starting new task (Task {self.task_index + 1}).")
            logger.info(f"CL Manager for {self.stage}: Pre-loading state tensors to GPU for efficient access.")
            self.device_deltaw_prev = {}
            self.device_directional_consistency = {}
            device = next(self.model.parameters()).device
            for name in self.lora_layer_names:
                if name in self.state.deltaw_prev and isinstance(self.state.deltaw_prev[name], torch.Tensor):
                    self.device_deltaw_prev[name] = self.state.deltaw_prev[name].to(device)
            logger.info(f"CL Manager for {self.stage}: State tensors cached on GPU.")

        self.state.lora_weights_prev = {}
        for module_name in self.lora_layer_names:
            module = self.model.get_submodule(module_name)
            param_name_A = f"{module_name}.lora_A.default.weight"
            param_name_B = f"{module_name}.lora_B.default.weight"
            self.state.lora_weights_prev[param_name_A] = module.lora_A["default"].weight.detach().clone().cpu()
            self.state.lora_weights_prev[param_name_B] = module.lora_B["default"].weight.detach().clone().cpu()
        logger.info(f"CL Manager for {self.stage}: Saved initial LoRA weights for Task {self.task_index + 1}.")

    def on_task_end(self, trainer: Seq2SeqTrainer, importance_data: Optional[Dataset]):
        logger.info(f"CL Manager for {self.stage}: Finalizing Task {self.task_index + 1}...")
        self._refine_and_apply_task_vector()
        self._update_deltaw_prev()
        logger.info(f"CL Manager for {self.stage}: Task {self.task_index + 1} finalized successfully.")

    def _denoise_update_vector(self, update_vector: torch.Tensor) -> torch.Tensor:
        try:
            U, S, Vh = torch.linalg.svd(update_vector, full_matrices=False)
            
            total_energy = torch.sum(S.pow(2))
            if total_energy <= 1e-9: 
                return update_vector

            cumulative_energy = torch.cumsum(S.pow(2), dim=0)
            k_candidates = torch.where(cumulative_energy / total_energy >= self.denoising_threshold)[0]
            
            k_alpha = k_candidates[0].item() + 1 if len(k_candidates) > 0 else S.numel()

            S_truncated = torch.zeros_like(S)
            S_truncated[:k_alpha] = S[:k_alpha]
            
            denoised_vector = U @ torch.diag(S_truncated) @ Vh
            logger.info(f"[Denoising] Denoised vector from rank {S.numel()} to {k_alpha}.")
            return denoised_vector
        except Exception as e:
            logger.warning(f"SVD denoising failed: {e}. Returning original vector.")
            return update_vector

    def _project_onto_forgetting_subspace(self, vector_to_project: torch.Tensor, history_list: list, param_name: str) -> torch.Tensor:
        """
        修改说明:
        1. 增加了 param_name 参数，用于生成唯一的文件名。
        2. 增加了 os.makedirs 来创建工件目录。
        3. 在计算出 H, delta_new_flat, 和 beta_flat 后，使用 torch.save 将它们保存。
        4. 对文件名中的'.'进行了替换，以避免路径问题。
        """
        if not history_list:
            return vector_to_project
            
        artifact_dir = "cl_artifacts"
        os.makedirs(artifact_dir, exist_ok=True)
        sanitized_name = param_name.replace('.', '_')

        device = vector_to_project.device
        try:
            flattened_history = torch.stack([h.flatten().to(device) for h in history_list])
            _, _, Vh_h = torch.linalg.svd(flattened_history, full_matrices=False)
            basis = Vh_h

            original_shape = vector_to_project.shape
            delta_flat = vector_to_project.flatten()

            coords_in_subspace = basis @ delta_flat
            delta_forget_flat = basis.T @ coords_in_subspace
            
            delta_new_flat = delta_flat - delta_forget_flat
            beta_flat = delta_new_flat + self.projection_gamma * delta_forget_flat
            
            if self.task_index == 2 and self.stage == "RL": 
                logger.info(f"[Saving Artifacts] Saving tensors for task {self.task_index + 1}, param {param_name}")
                
                h_filename = f"{artifact_dir}/H_task{self.task_index + 1}_{sanitized_name}.pt"
                torch.save(flattened_history.cpu(), h_filename)

                delta_new_filename = f"{artifact_dir}/delta_new_flat_task{self.task_index + 1}_{sanitized_name}.pt"
                torch.save(delta_new_flat.cpu(), delta_new_filename)

                beta_flat_filename = f"{artifact_dir}/beta_flat_task{self.task_index + 1}_{sanitized_name}.pt"
                torch.save(beta_flat.cpu(), beta_flat_filename)

            final_update = beta_flat.reshape(original_shape)
            logger.info(f"[Projection] Projection applied using {len(history_list)} historical vectors.")
            logger.info(f"[Projection] Constructed a basis with {basis.shape[0]} vectors.")
            return final_update
        except Exception as e:
            logger.warning(f"Subspace projection failed: {e}. Returning original vector.")
            return vector_to_project

    def _project_onto_forgetting_subspace_bkup(self, vector_to_project: torch.Tensor, history_list: list) -> torch.Tensor:
        if not history_list:
            return vector_to_project
            
        device = vector_to_project.device
        try:
            flattened_history = torch.stack([h.flatten().to(device) for h in history_list])
            _, _, Vh_h = torch.linalg.svd(flattened_history, full_matrices=False)
            basis = Vh_h

            original_shape = vector_to_project.shape
            delta_flat = vector_to_project.flatten()

            coords_in_subspace = basis @ delta_flat
            delta_forget_flat = basis.T @ coords_in_subspace
            
            delta_new_flat = delta_flat - delta_forget_flat
            beta_flat = delta_new_flat + self.projection_gamma * delta_forget_flat
            
            final_update = beta_flat.reshape(original_shape)
            logger.info(f"[Projection] Projection applied using {len(history_list)} historical vectors.")
            logger.info(f"[Projection] Constructed a basis with {basis.shape[0]} vectors.")
            return final_update
        except Exception as e:
            logger.warning(f"Subspace projection failed: {e}. Returning original vector.")
            return vector_to_project

    def _refine_and_apply_task_vector(self):

        logger.info(f"CL Manager for {self.stage}: Starting memory consolidation process (denoising_threshold={self.denoising_threshold}, projection_gamma={self.projection_gamma})...")
        device = next(self.model.parameters()).device

        optimizer_update_vector = self._get_raw_task_vector()
        
        final_refined_vector = {}

        for name, alpha_original in optimizer_update_vector.items():
            alpha_denoised = self._denoise_update_vector(alpha_original)
            
            history = self.state.historical_updates.get(name, [])
            beta_final = self._project_onto_forgetting_subspace(alpha_denoised, history, name)
            
            final_refined_vector[name] = beta_final

        logger.info(f"CL Manager for {self.stage}: Applying consolidated update to model weights...")
        with torch.no_grad():
            for module_name in self.lora_layer_names:
                module = self.model.get_submodule(module_name)
                for matrix_type in ["lora_A", "lora_B"]:
                    param_name = f"{module_name}.{matrix_type}.default.weight"
                    if param_name in final_refined_vector:
                        current_param = getattr(module, matrix_type)["default"].weight
                        prev_param = self.state.lora_weights_prev[param_name].to(device)
                        final_weight = prev_param + final_refined_vector[param_name]
                        current_param.copy_(final_weight)
        
        self._update_historical_dynamics(final_refined_vector)
        logger.info(f"CL Manager for {self.stage}: Memory consolidation and application complete.")

    def _get_final_deltaw(self) -> Dict[str, torch.Tensor]:
        final_deltaw = {}
        with torch.no_grad():
            for module_name in self.lora_layer_names:
                module = self.model.get_submodule(module_name)
                lora_A = module.lora_A["default"].weight
                lora_B = module.lora_B["default"].weight
                final_deltaw[module_name] = (lora_B @ lora_A).detach().clone().cpu()
        return final_deltaw

    def _update_deltaw_prev(self):
        logger.info(f"CL Manager for {self.stage}: Updating deltaW_prev for the next task's regularization.")
        self.state.deltaw_prev = self._get_final_deltaw()

    def _get_raw_task_vector(self) -> Dict[str, torch.Tensor]:
        alpha_vector = {}
        device = next(self.model.parameters()).device
        for module_name in self.lora_layer_names:
            module = self.model.get_submodule(module_name)
            for matrix_type in ["lora_A", "lora_B"]:
                param_name = f"{module_name}.{matrix_type}.default.weight"
                if param_name in self.state.lora_weights_prev:
                    current_param = getattr(module, matrix_type)["default"].weight
                    prev_param = self.state.lora_weights_prev[param_name].to(device)
                    alpha_vector[param_name] = current_param.detach() - prev_param
        return alpha_vector

    def _update_historical_dynamics(self, beta_vector: Dict[str, torch.Tensor]):
        for name, beta_param in beta_vector.items():
            beta_cpu = beta_param.detach().clone().cpu()
            if name not in self.state.historical_updates:
                self.state.historical_updates[name] = [torch.zeros_like(beta_cpu) for _ in range(self.task_index)] if self.task_index > 0 else []
            self.state.historical_updates[name].append(beta_cpu)
            hist_list = self.state.historical_updates[name]
    
    def _infer_lora_layer_names(self) -> List[str]:
        if not isinstance(self.model, PeftModel):
            return []
        lora_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                if name not in lora_names:
                    lora_names.append(name)
        return lora_names