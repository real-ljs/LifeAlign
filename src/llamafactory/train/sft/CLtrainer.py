# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

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


class CLTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        train_dataset=None,
        eval_dataset=None,
        gen_kwargs=None,
        l2p_manager=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version
 
            self.accelerator.clip_grad_norm_ = MethodType(
                clip_grad_norm_old_version, self.accelerator
            )
            self.add_callback(BAdamCallback)
        self.continual_eval_datasets = eval_dataset
        self.gen_kwargs = gen_kwargs
        self.l2p_manager = l2p_manager

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(
                self.model, self.args, self.finetuning_args
            )
        return super().create_optimizer()

    @override
    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional["torch.optim.Optimizer"] = None,
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.
        Modified to support L2P inference.

        Subclass and override to inject custom behavior.
        """
        # Apply L2P if available
        if self.l2p_manager is not None:
            inputs = self._prepare_inputs_with_l2p(inputs)
        
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert (
                self.tokenizer.padding_side == "left"
            ), "This method only accepts left-padded tensor."
            labels = (
                labels.detach().clone() if labels is not None else None
            )  # backup labels
            
            # Handle input length calculation for L2P case
            if "inputs_embeds" in inputs:
                prompt_len = inputs["inputs_embeds"].size(-2)  # sequence length dimension
            else:
                prompt_len = inputs["input_ids"].size(-1)
            
            label_len = inputs["labels"].size(-1)
            
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(
                    inputs["labels"], inputs.get("input_ids", torch.zeros(inputs["labels"].shape[0], prompt_len, device=inputs["labels"].device, dtype=torch.long))
                )
            if (label_len > prompt_len):  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        (
            loss,
            generated_tokens,
            _,
        ) = super().prediction_step(  # ignore the returned labels (may be truncated)
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )
        
        if generated_tokens is not None and self.args.predict_with_generate:
            # Handle prompt length for L2P case
            if "inputs_embeds" in inputs:
                prompt_len = inputs["inputs_embeds"].size(-2)
            else:
                prompt_len = inputs.get("input_ids", torch.tensor([])).size(-1)
            
            if prompt_len > 0 and generated_tokens.size(-1) >= prompt_len:
                generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
                generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(
        self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor"
    ) -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_data(self, data, filepath):
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", dataset_name=None
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return
        output_dir = os.path.join(self.args.output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        output_prediction_file = os.path.join(
            output_dir, f"generated_predictions.json"
        )
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX,
            predict_results.label_ids,
            self.tokenizer.pad_token_id,
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.tokenizer.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )

        decoded_inputs = self.tokenizer.batch_decode(
            dataset["input_ids"], skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # with open(output_prediction_file, "w", encoding="utf-8") as writer:
        res: List[str] = []
        for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
            res.append({"prompt": text, "label": label, "predict": pred})
        self.save_data(res, output_prediction_file)

    def save_metrics(self, split, metrics, combined=True, dataset_name=None):
        """
        Save metrics into a json file for that split, e.g. `train_results.json`.

        Under distributed environment this is done only for a process with rank 0.

        Args:
            split (`str`):
                Mode/split name: one of `train`, `eval`, `test`, `all`
            metrics (`Dict[str, float]`):
                The metrics returned from train/evaluate/predict
            combined (`bool`, *optional*, defaults to `True`):
                Creates combined metrics by updating `all_results.json` with metrics of this call

        To understand the metrics please read the docstring of [`~Trainer.log_metrics`]. The only difference is that raw
        unformatted numbers are saved in the current method.

        """
        if not self.is_world_process_zero():
            return
        output_dir = os.path.join(self.args.output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

        if combined:
            path = os.path.join(self.args.output_dir, "all_results.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            all_metrics.update(metrics)
            with open(path, "w") as f:
                json.dump(all_metrics, f, indent=4, sort_keys=True)
    
    def _prepare_inputs_with_l2p(
        self, inputs: Dict[str, Union["torch.Tensor", Any]]
    ) -> Dict[str, Union["torch.Tensor", Any]]:
        """
        Prepare inputs with L2P prompts if L2P manager is available.
        """
        if self.l2p_manager is None:
            return inputs
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Use L2P manager to prepare inputs with prompts
        final_input_embeds, final_attention_mask, _ = self.l2p_manager._prepare_inputs(
            input_ids, attention_mask
        )
        
        model_dtype = next(self.model.parameters()).dtype
        if final_input_embeds.dtype != model_dtype:
            final_input_embeds = final_input_embeds.to(dtype=model_dtype)
            
        # Replace input_ids with inputs_embeds and update attention_mask
        modified_inputs = inputs.copy()
        modified_inputs.pop("input_ids", None)  # Remove input_ids since we're using inputs_embeds
        modified_inputs["inputs_embeds"] = final_input_embeds
        modified_inputs["attention_mask"] = final_attention_mask
        
        return modified_inputs

    def load_l2p_prompt_pool_for_task(self, task_checkpoint_dir: str):
        """
        Load L2P prompt pool for a specific task.
        
        Args:
            task_checkpoint_dir (str): Directory containing the task-specific checkpoint with L2P prompt pool
        """
        if self.l2p_manager is None:
            logger.warning("L2P manager not initialized, cannot load prompt pool.")
            return
        print(task_checkpoint_dir)
        prompt_pool_path = os.path.join(task_checkpoint_dir, "l2p_prompt_pool.pt")
        if not os.path.exists(prompt_pool_path):
            logger.warning(f"L2P prompt pool not found at {prompt_pool_path}, using default initialization.")
            return
        
        try:
            state_dict = torch.load(prompt_pool_path, map_location=self.l2p_manager.device)
            with torch.no_grad():
                self.l2p_manager.prompt_pool.copy_(state_dict["prompt_pool"])
            logger.info(f"Successfully loaded L2P prompt pool from {prompt_pool_path}")
        except Exception as e:
            logger.error(f"Failed to load L2P prompt pool from {prompt_pool_path}: {e}")

    def continual_eval(self, task_checkpoint_dir: Optional[str] = None):
        if task_checkpoint_dir and self.l2p_manager is not None:
            logger.info(f"Loading L2P prompt pool from {task_checkpoint_dir}")
            self.load_l2p_prompt_pool_for_task(task_checkpoint_dir[0])

        for id, eval_dataset in enumerate(self.continual_eval_datasets):
            datatset_info = eval_dataset.info.download_checksums.keys()
            dataset_name = (
                list(datatset_info)[0].split("")[-1].split("/test.json")[0]
            )
            predict_results = self.predict(
                eval_dataset, metric_key_prefix="predict", **self.gen_kwargs
            )
            predict_results.metrics.pop("predict_loss", None)
            self.log_metrics("predict", predict_results.metrics)
            self.save_metrics(
                "predict", predict_results.metrics, dataset_name=dataset_name
            )
            self.save_predictions(
                eval_dataset, predict_results, dataset_name=dataset_name
            )