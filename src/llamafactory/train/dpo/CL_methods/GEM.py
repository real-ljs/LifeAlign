# CL_methods/GEM.py

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from qpth.qp import QPFunction
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union, Any
from ....extras.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from ....hparams import FinetuningArguments

logger = get_logger(__name__)

class TaskGradientCallback(TrainerCallback):
    """
    GEM的回调实现。
    这个回调实例将在所有任务之间共享，通过更新task_name来切换当前任务。
    """
    def __init__(
        self,
        model: torch.nn.Module,
        initial_task_name: str,
        margin: float = 0.1,
        eps: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.task_name = initial_task_name
        self.margin = margin
        self.eps = eps
        self.grads_dict: Dict[str, Dict[str, torch.Tensor]] = {}
        self._initialize_grads_dict()
        logger.info(f"Initialized shared GEM callback for initial task: {initial_task_name}")

    def _initialize_grads_dict(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.grads_dict.setdefault(name, {})

    def on_optimizer_step(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue

            current_grad = p.grad.view(-1)
            historical_grads = [
                avg_vec.to(current_grad.device)
                for old_task, avg_vec in self.grads_dict[name].items()
                if old_task != self.task_name
            ]

            if historical_grads:
                G = torch.stack(historical_grads, dim=0)
                dot_product = torch.mm(current_grad.unsqueeze(0), G.t())
                if (dot_product < 0).any():
                    # QP 投影逻辑
                    A = G @ G.t()
                    P = 0.5 * (A + A.t())
                    eigenvalues = torch.linalg.eigvals(P)
                    if not (eigenvalues.real > 0).all():
                        P += torch.eye(P.shape[0], device=P.device) * self.eps
                    q = -(G @ current_grad)
                    Gm = -torch.eye(G.shape[0], device=G.device)
                    hm = torch.zeros(G.shape[0], device=G.device) - self.margin
                    e = torch.empty(0, device=P.device)
                    v = QPFunction(verbose=False)(P.float(), q.float(), Gm, hm, e, e)[0]
                    updated_grad = (G.t() @ v + current_grad).view_as(p.grad)
                    p.grad.copy_(updated_grad)
            
            detached_grad = p.grad.view(-1).detach().clone()
            old_avg = self.grads_dict[name].get(self.task_name)
            
            if old_avg is not None:
                new_avg = 0.9 * old_avg + 0.1 * detached_grad
            else:
                new_avg = detached_grad

            self.grads_dict[name][self.task_name] = new_avg

class GEMManager:
    """
    管理GEM状态，主要是统一的TaskGradientCallback实例。
    """
    def __init__(self, model: torch.nn.Module, finetuning_args: "FinetuningArguments"):
        self.model = model
        self.finetuning_args = finetuning_args
        self.gem_callback = None
        logger.info("GEMManager initialized.")

    def prepare_for_task(self, task_name: str, trainer: "Trainer"):
        """
        为一个新任务做准备。如果需要，创建回调；如果已存在，则更新其任务名称。
        """
        if self.gem_callback is None:
            # 第一次调用时，创建并添加到trainer中
            logger.info(f"First task '{task_name}'. Creating shared GEM callback.")
            self.gem_callback = TaskGradientCallback(
                model=self.model,
                initial_task_name=task_name,
                margin=getattr(self.finetuning_args, 'gem_margin', 0.1),
                eps=getattr(self.finetuning_args, 'gem_eps', 1.0),
            )
            trainer.add_callback(self.gem_callback)
        else:
            # 后续任务，只需更新task_name
            logger.info(f"Switching to task '{task_name}'. Updating task name in shared GEM callback.")
            self.gem_callback.task_name = task_name