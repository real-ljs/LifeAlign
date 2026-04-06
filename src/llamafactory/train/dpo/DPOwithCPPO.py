import copy
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import Trainer
from trl import DPOTrainer
from collections import defaultdict
from .vanillaDPOTrainer import vanillaDPOTrainer
from typing_extensions import override
from typing import Literal, Dict, Tuple, Optional, TYPE_CHECKING
import torch
if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

class DPOWithCPPOTrainer(vanillaDPOTrainer):
    """
    修正版本：DPOTrainer enhanced with CPPO-style sample-level weights.
    
    主要修正：
    1. 正确的样本分类逻辑
    2. 合理的损失函数组合
    3. 动态旧策略更新
    4. 完整的监控指标
    """

    def __init__(
        self,
        *args,
        weight_scheme: str = "heuristic",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.weight_scheme = weight_scheme
        self.std_mult = getattr(self.finetuning_args, "cp_std_mult", 1.0)
        self.kr_coef = getattr(self.finetuning_args, "cp_kr_coef", 0.1)
        self.alpha_lb = getattr(self.finetuning_args, "cp_alpha_lb", 0.5)
        self.alpha_ub = getattr(self.finetuning_args, "cp_alpha_ub", 2.0)
        self.beta_lb = getattr(self.finetuning_args, "cp_beta_lb", 0.5) 
        self.beta_ub = getattr(self.finetuning_args, "cp_beta_ub", 2.0)
        
        self.old_policy = None

    def snapshot_old_policy(self):
        """创建当前策略的冻结副本用于知识保持"""
        if self.old_policy is not None:
            del self.old_policy  # 释放内存
        
        self.old_policy = copy.deepcopy(self.model)
        self.old_policy.eval()
        
        for param in self.old_policy.parameters():
            param.requires_grad = False

    def compute_sample_weights(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        修正版：正确实现CPPO样本权重计算
        """
        # 使用奖励边际和chosen样本的生成概率（符合CPPO论文）
        reward_margins = chosen_rewards - rejected_rewards
        generation_probs = policy_chosen_logps  # 使用chosen的绝对概率
        
        # 计算动态阈值
        reward_mean, reward_std = reward_margins.mean(), reward_margins.std()
        prob_mean, prob_std = generation_probs.mean(), generation_probs.std()
        
        # 避免除零
        reward_std = torch.clamp(reward_std, min=1e-8)
        prob_std = torch.clamp(prob_std, min=1e-8)
        
        reward_high_thr = reward_mean + self.std_mult * reward_std
        reward_low_thr = reward_mean - self.std_mult * reward_std
        prob_high_thr = prob_mean + self.std_mult * prob_std
        prob_low_thr = prob_mean - self.std_mult * prob_std
        
        # 初始化权重
        alpha = torch.ones_like(reward_margins)
        beta = torch.ones_like(reward_margins)
        
        # CPPO样本分类（修正版）
        # High-performance: 高概率 + 高奖励 → 强化学习和保持
        high_performance = (generation_probs >= prob_high_thr) & (reward_margins >= reward_high_thr)
        alpha[high_performance] = self.alpha_ub
        beta[high_performance] = self.beta_ub
        
        # Overfitting: 高概率 + 低奖励 → 强化学习，减少保持
        overfitting = (generation_probs >= prob_high_thr) & (reward_margins <= reward_low_thr)
        alpha[overfitting] = self.alpha_ub
        beta[overfitting] = self.beta_lb
        
        # High-variance: 低概率 + 高奖励 → 强化学习，减少保持
        high_variance = (generation_probs <= prob_low_thr) & (reward_margins >= reward_high_thr)
        alpha[high_variance] = self.alpha_ub
        beta[high_variance] = self.beta_lb
        
        # Noisy: 低概率 + 低奖励 → 减少学习和保持
        noisy = (generation_probs <= prob_low_thr) & (reward_margins <= reward_low_thr)
        alpha[noisy] = self.alpha_lb
        beta[noisy] = self.beta_lb
        
        return alpha, beta

    def compute_preference_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor],
        reference_rejected_logps: Optional[torch.Tensor],
        old_chosen_logps: torch.Tensor,
        old_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        修正版：正确组合DPO损失和知识保持损失
        """
        # 计算基础DPO损失和隐式奖励
        dpo_loss, chosen_rewards, rejected_rewards = super().compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        
        # 计算样本权重
        alpha, beta = self.compute_sample_weights(
            chosen_rewards, 
            rejected_rewards,
            policy_chosen_logps, 
            policy_rejected_logps
        )
        
        # 应用alpha权重到DPO损失
        weighted_dpo_loss = alpha * dpo_loss
        
        # 计算知识保持损失（L2距离）
        kr_chosen = (policy_chosen_logps - old_chosen_logps.detach()).pow(2)
        kr_rejected = (policy_rejected_logps - old_rejected_logps.detach()).pow(2)
        kr_loss = kr_chosen + kr_rejected
        
        # 应用beta权重到知识保持损失
        weighted_kr_loss = beta * kr_loss
        
        # 总损失：加权DPO + 系数*加权知识保持
        total_loss = weighted_dpo_loss + self.kr_coef * weighted_kr_loss
        
        return total_loss, chosen_rewards, rejected_rewards

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """修正版：完整的指标监控"""
        metrics = {}
        # 确保旧策略存在
        if self.old_policy is None:
            self.snapshot_old_policy()

        # 前向传播
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)

        with torch.no_grad():
            old_chosen_logps, old_rejected_logps, *_ = self.concatenated_forward(self.old_policy, batch)

        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            old_chosen_logps,
            old_rejected_logps,
        )
        
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        alpha_weights, beta_weights = self.compute_sample_weights(
            chosen_rewards, rejected_rewards,
            policy_chosen_logps, policy_rejected_logps
        )

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
        
        metrics["{}cppo/alpha_mean".format(prefix)] = alpha_weights.mean().cpu()
        metrics["{}cppo/alpha_std".format(prefix)] = alpha_weights.std().cpu()
        metrics["{}cppo/beta_mean".format(prefix)] = beta_weights.mean().cpu()
        metrics["{}cppo/beta_std".format(prefix)] = beta_weights.std().cpu()
        
        reward_margins = chosen_rewards - rejected_rewards
        generation_probs = policy_chosen_logps
        reward_mean, reward_std = reward_margins.mean(), reward_margins.std()
        prob_mean, prob_std = generation_probs.mean(), generation_probs.std()
        
        reward_high_thr = reward_mean + self.std_mult * reward_std
        reward_low_thr = reward_mean - self.std_mult * reward_std
        prob_high_thr = prob_mean + self.std_mult * prob_std
        prob_low_thr = prob_mean - self.std_mult * prob_std
        
        high_perf = ((generation_probs >= prob_high_thr) & (reward_margins >= reward_high_thr)).float().mean()
        overfitting = ((generation_probs >= prob_high_thr) & (reward_margins <= reward_low_thr)).float().mean()
        high_var = ((generation_probs <= prob_low_thr) & (reward_margins >= reward_high_thr)).float().mean()
        noisy = ((generation_probs <= prob_low_thr) & (reward_margins <= reward_low_thr)).float().mean()
        
        metrics["{}cppo/high_performance_ratio".format(prefix)] = high_perf.cpu()
        metrics["{}cppo/overfitting_ratio".format(prefix)] = overfitting.cpu()
        metrics["{}cppo/high_variance_ratio".format(prefix)] = high_var.cpu()
        metrics["{}cppo/noisy_ratio".format(prefix)] = noisy.cpu()
        
        if self.loss_type == "orpo":
            metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean().cpu()
            metrics["{}odds_ratio_loss".format(prefix)] = ((losses - sft_loss) / self.beta).detach().mean().cpu()

        return losses.mean(), metrics

    def train(self, *args, **kwargs):
        if self.old_policy is None:
            self.snapshot_old_policy()
        return super().train(*args, **kwargs)

    def train_single_dataset(
        self,
        dataset,
        args,
        task_name=None,
        output_dir=None,
        previous_task_output_dir=None,
    ):
        if self.old_policy is None:
            self.snapshot_old_policy()
        return super().train_single_dataset(
            dataset, args, task_name, output_dir, previous_task_output_dir
        )