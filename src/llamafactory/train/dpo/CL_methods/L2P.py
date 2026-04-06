# L2P.py

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional, Dict, Tuple, Any
from copy import deepcopy
from transformers import PreTrainedModel
from ....extras.logging import get_logger

logger = get_logger(__name__)

def l2_normalize(x, dim=None, epsilon=1e-12):
    """L2范数归一化"""
    square_norm = torch.sum(x**2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(torch.maximum(square_norm, torch.tensor(epsilon, device=x.device)))
    return x * x_inv_norm

class L2P:
    """
    一个独立的、可组合的 'Learning to Prompt' (L2P) 逻辑管理器。
    
    这个类不包装模型，而是作为一个辅助工具，在训练的 `compute_loss` 步骤中被调用，
    以实现 L2P 的功能。
    """
    def __init__(self,
        model: PreTrainedModel,
        pool_size: int = 10,
        prompt_length: int = 5,
        top_k: int = 3,
        diversity_loss_weight: float = 0.5,
        prompt_init: str = 'random',
        embedding_key: str = 'mean',
        batchwise_prompt: bool = False
    ):
        """
        初始化L2P管理器。

        Args:
            model (nn.Module): 需要应用L2P的Hugging Face模型。
            pool_size (int): prompt池的大小。
            prompt_length (int): 每个prompt的长度。
            top_k (int): 每个输入选择top-k个最相似的prompt。
            diversity_loss_weight (float): 多样性损失的权重。
            prompt_init (str): prompt的初始化方法 (目前仅支持 'random')。
            embedding_key (str): 从输入计算query embedding的方法 ('mean', 'max', 'mean_max')。
            batchwise_prompt (bool): 是否为整个batch选择一组统一的prompt。
        """
        logger.info("Initializing L2P Manager...")
        self.model = model
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.top_k = top_k
        self.diversity_loss_weight = diversity_loss_weight
        self.embedding_key = embedding_key
        self.batchwise_prompt = batchwise_prompt
        self.device = next(model.parameters()).device

        self.input_embeddings = self.model.get_input_embeddings()
        embedding_dim = self.input_embeddings.weight.shape[1]

        prompt_weights = self._create_prompt_weights(prompt_init, embedding_dim)
        prompt_weights_on_device = prompt_weights.to(self.device)
        self.prompt_pool = nn.Parameter(prompt_weights_on_device)
        logger.info(f"L2P prompt pool created on device: {self.prompt_pool.device} with shape: {self.prompt_pool.shape}")

    def _create_prompt_weights(self, prompt_init: str, embedding_dim: int) -> torch.Tensor:
        """创建prompt池的初始权重。"""
        if prompt_init == 'random':
            embedding_weights = self.input_embeddings.weight.detach().cpu()
            prompt_pool_size = (self.pool_size, self.prompt_length, embedding_dim)
            
            prompt_weights = torch.zeros(prompt_pool_size)
            for i in range(self.pool_size):
                for j in range(self.prompt_length):
                    rand_idx = np.random.randint(embedding_weights.shape[0])
                    prompt_weights[i, j] = deepcopy(embedding_weights[rand_idx])
            return prompt_weights
        else:
            raise ValueError(f"Unsupported prompt initialization: {prompt_init}")

    def get_trainable_parameters(self) -> list:
        """返回L2P的可训练参数，用于优化器。"""
        return [self.prompt_pool]

    def save_prompt_pool(self, path: str):
        """保存prompt池的权重。"""
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path, "l2p_prompt_pool.pt")
        torch.save({"prompt_pool": self.prompt_pool}, save_path)
        logger.info(f"L2P prompt pool saved to {save_path}")

    def load_prompt_pool(self, path: str):
        """加载prompt池的权重。"""
        load_path = os.path.join(path, "l2p_prompt_pool.pt")
        if not os.path.exists(load_path):
            logger.warning(f"Prompt pool file not found at {load_path}, skipping.")
            return
        state_dict = torch.load(load_path, map_location=self.device)
        with torch.no_grad():
            self.prompt_pool.copy_(state_dict["prompt_pool"])
        logger.info(f"L2P prompt pool loaded from {load_path}")
        
    def _prepare_inputs(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        准备模型的输入，将选择的prompt注入到input_embeds中。

        Returns:
            - torch.Tensor: 修改后的 inputs_embeds
            - torch.Tensor: 修改后的 attention_mask
            - dict: 用于计算损失的上下文信息
        """
        input_embeds = self.input_embeddings(input_ids)
        
        if self.embedding_key == 'mean':
            query_embeds = torch.mean(input_embeds, dim=1)
        elif self.embedding_key == 'max':
            query_embeds = torch.max(input_embeds, dim=1)[0]
        else:
            raise NotImplementedError(f"Unsupported embedding key: {self.embedding_key}")

        prompt_keys = torch.mean(self.prompt_pool, dim=1)
        prompt_keys_norm = l2_normalize(prompt_keys, dim=1)
        query_embeds_norm = l2_normalize(query_embeds, dim=1)

        similarity = torch.matmul(query_embeds_norm.to(prompt_keys_norm.dtype), prompt_keys_norm.t())
        
        _, top_k_indices = torch.topk(similarity, k=self.top_k, dim=1)

        batch_size = input_embeds.shape[0]
        selected_prompts = self.prompt_pool[top_k_indices]
        selected_prompts = selected_prompts.view(batch_size, self.top_k * self.prompt_length, -1)
        
        final_input_embeds = torch.cat([selected_prompts, input_embeds], dim=1)

        prefix_length = selected_prompts.shape[1]
        prefix_attention_mask = torch.ones(batch_size, prefix_length, device=self.device, dtype=attention_mask.dtype)
        final_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        loss_context = {
            "prompt_keys_norm": prompt_keys_norm,
            "query_embeds_norm": query_embeds_norm,
            "top_k_indices": top_k_indices,
            "batch_size": batch_size
        }
        
        return final_input_embeds, final_attention_mask, loss_context

    def calculate_diversity_loss(self, loss_context: Dict[str, Any]) -> torch.Tensor:
        """
        计算多样性损失。
        """
        prompt_keys_norm = loss_context["prompt_keys_norm"]
        query_embeds_norm = loss_context["query_embeds_norm"]
        top_k_indices = loss_context["top_k_indices"]
        batch_size = loss_context["batch_size"]

        batched_key_norm = prompt_keys_norm[top_k_indices]
        query_embeds_norm_unsqueezed = query_embeds_norm.unsqueeze(1)
        
        sim_to_selected_keys = torch.sum(batched_key_norm * query_embeds_norm_unsqueezed) / batch_size
        
        diversity_loss = -sim_to_selected_keys * self.diversity_loss_weight
        
        return diversity_loss
