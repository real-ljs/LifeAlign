# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
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

from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union
from ...data import get_dataset
from ...hparams import get_train_args
import torch
import os
from datasets import Dataset
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from .CL_methods.EWC import EWCManager
from .CL_methods.GEM import GEMManager
from .CL_methods.CLManager import ContinualLearningManager
from .SFTwithEWCtrainer import SFTwithEWCTrainer
from .SFTtrainer import SFTTrainer
from .SFTwithGEMtrainer import SFTwithGEMTrainer
from .SFTwithL2Ptrainer import SFTwithL2PTrainer
from .SFTwithOLoRAtrainer import SFTwithOloraTrainer
from .MySFTTrainer import MySFTTrainer
from ...extras.constants import IGNORE_INDEX
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import (
    create_custom_optimizer,
    create_custom_scheduler,
    get_batch_logps,
)
from ...data import (
    get_dataset,
    get_template_and_fix_tokenizer,
)
from ...data.collator import MyCollator, DataCollatorForSeq2Seq
from ...extras.logging import get_logger
from datasets import concatenate_datasets

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class BaseCLTrainer:
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        finetuning_args: "FinetuningArguments",
        training_args=None,
        train_dataset=None,
        template=None,
        callbacks=None,
        trainer=None,
        CL_method="vanilla",
        **kwargs,
    ):
        self.finetuning_args = finetuning_args
        self.training_args = training_args
        self.trainer = trainer
        self.template = template
        self.continual_training_dataset = train_dataset
        self.method_name = CL_method
        self.replay_buffer_percentage = 0.2
        self.max_replay_samples = self.finetuning_args.replay_buffer_size
        ### SFT initialize
        (
            self.sft_model_args,
            self.sft_data_args,
            self.sft_training_args,
            self.sft_finetuning_args,
            self.sft_generating_args,
        ) = get_train_args(
            yaml_path="examples/train_lora/llama3_lora_sft_initialize.yaml"
        )
        self.sft_tokenizer_module = load_tokenizer(self.sft_model_args)
        self.sft_tokenizer = self.sft_tokenizer_module["tokenizer"]
        sft_template = get_template_and_fix_tokenizer(self.sft_tokenizer, self.sft_data_args)
        self.sft_collator = DataCollatorForSeq2Seq(
            tokenizer=self.sft_tokenizer,
            pad_to_multiple_of=(
                8 if self.sft_tokenizer.padding_side == "right" else None
            ),  # for shift short attention
            label_pad_token_id=(
                IGNORE_INDEX
                if self.sft_data_args.ignore_pad_token_for_loss
                else self.sft_tokenizer.pad_token_id
            ),
        )
        merge_type = self.finetuning_args.CL_method == "MTL"
        self.sft_dataset_module = get_dataset(
            sft_template,
            self.sft_model_args,
            self.sft_data_args,
            self.sft_training_args,
            stage="sft",
            **self.sft_tokenizer_module,
            merge=merge_type,
        )
        self.ewc_manager = None
        self.gem_manager = None
        self.sft_trainer = SFTTrainer(
            model=model,
            args=self.sft_training_args,
            finetuning_args=self.sft_finetuning_args,
            data_collator=self.sft_collator,
            callbacks=callbacks,
            **self.sft_tokenizer_module,
        )
        if CL_method == 'EWC':
            self.ewc_manager = EWCManager(model, self.finetuning_args.ewc_lambda)
            self.sft_trainer = SFTwithEWCTrainer(
                model=model,
                args=self.sft_training_args,
                finetuning_args=self.sft_finetuning_args,
                data_collator=self.sft_collator,
                callbacks=callbacks,
                **self.sft_tokenizer_module,
            )
            self.sft_trainer.ewc_manager = self.ewc_manager
            self.trainer.ewc_manager = self.ewc_manager
        elif CL_method == 'GEM':
            self.gem_manager = GEMManager(model, self.finetuning_args)
            self.sft_trainer = SFTwithGEMTrainer(
                model=model,
                args=self.sft_training_args,
                finetuning_args=self.sft_finetuning_args,
                data_collator=self.sft_collator,
                callbacks=callbacks,
                **self.sft_tokenizer_module,
            )
        elif CL_method == 'L2P':
            self.sft_trainer = SFTwithL2PTrainer(
                model=model,
                args=self.sft_training_args,
                finetuning_args=self.sft_finetuning_args,
                data_collator=self.sft_collator,
                callbacks=callbacks,
                **self.sft_tokenizer_module,
            )
        elif CL_method == 'OLoRA':
            self.sft_trainer = SFTwithOloraTrainer(
                model=model,
                args=self.sft_training_args,
                finetuning_args=self.sft_finetuning_args,
                data_collator=self.sft_collator,
                callbacks=callbacks,
                **self.sft_tokenizer_module,
            )
        elif CL_method == 'my_method':
            self.cl_manager = ContinualLearningManager(
                model=model,
                denoising_threshold=self.finetuning_args.denoising_threshold,
                projection_gamma=self.finetuning_args.projection_gamma,
            )
            self.sft_trainer = MySFTTrainer(
                model=model,
                args=self.sft_training_args,
                finetuning_args=self.sft_finetuning_args,
                data_collator=self.sft_collator,
                callbacks=callbacks,
                **self.sft_tokenizer_module
            )

    def _manage_replay_buffer(
        self,
        current_task_dataset: "Dataset",
        replay_buffer: Optional["Dataset"],
        max_replay_samples: int,
        replay_sample_percentage: float,
        buffer_name: str = "ReplayBuffer",
    ) -> Tuple["Dataset", Optional["Dataset"]]:
        """
        Manages the replay buffer and prepares the dataset for training.

        Args:
            current_task_dataset: The dataset for the current task.
            replay_buffer: The existing replay buffer (can be None).
            max_replay_samples: Maximum number of samples to store in the replay buffer.
            replay_sample_percentage: Percentage of current_task_dataset to add to the buffer.
            buffer_name: Name of the buffer for logging purposes.

        Returns:
            A tuple containing:
                - merged_dataset_for_training: Dataset ready for training (current task + buffer).
                - updated_replay_buffer: The replay buffer after adding new samples and truncation.
        """
        logger.info(
            f"Managing {buffer_name}: "
            f"max_samples={max_replay_samples}, new_sample_percentage={replay_sample_percentage}"
        )
        logger.info(
            f"Current task dataset for {buffer_name} phase has {len(current_task_dataset)} samples."
        )

        # 1. Create merged dataset for training
        if replay_buffer is not None and len(replay_buffer) > 0:
            logger.info(
                f"Mixing {len(replay_buffer)} samples from {buffer_name} "
                f"with {len(current_task_dataset)} current task samples."
            )
            merged_dataset_for_training = concatenate_datasets(
                [replay_buffer, current_task_dataset]
            )
            merged_dataset_for_training = merged_dataset_for_training.shuffle()
        else:
            logger.info(
                f"No samples in {buffer_name} to mix. Using {len(current_task_dataset)} current task samples directly for training."
            )
            merged_dataset_for_training = current_task_dataset

        logger.info(
            f"Total samples for training ({buffer_name} phase): {len(merged_dataset_for_training)}."
        )

        # 2. Prepare new samples for the replay buffer from the current task dataset
        num_new_replay_samples = int(
            replay_sample_percentage * len(current_task_dataset)
        )

        # Ensure select gets a valid range; range(0) is fine for empty selection.
        new_samples_for_buffer = current_task_dataset.select(
            range(num_new_replay_samples)
        )

        if len(new_samples_for_buffer) > 0:
            logger.info(
                f"Selected {len(new_samples_for_buffer)} new samples for {buffer_name}."
            )
        else:
            logger.info(
                f"No new samples selected for {buffer_name} "
                f"(calculated num: {num_new_replay_samples}, actual selected: {len(new_samples_for_buffer)})."
            )

        # 3. Update the replay buffer
        if replay_buffer is None or len(replay_buffer) == 0:
            updated_replay_buffer = new_samples_for_buffer
            if updated_replay_buffer and len(updated_replay_buffer) > 0:
                logger.info(
                    f"Initialized {buffer_name} with {len(updated_replay_buffer)} samples."
                )
            else:
                logger.info(
                    f"{buffer_name} initialized as empty or with no new samples."
                )
        else:  # replay_buffer exists and has items
            if new_samples_for_buffer and len(new_samples_for_buffer) > 0:
                # Append new samples to the end of the existing buffer
                updated_replay_buffer = concatenate_datasets(
                    [replay_buffer, new_samples_for_buffer]
                )
                logger.info(
                    f"Appended {len(new_samples_for_buffer)} new samples to {buffer_name} "
                    f"(previous size: {len(replay_buffer)}, new potential size: {len(updated_replay_buffer)})."
                )
            else:  # No new samples to add, or new_samples_for_buffer is empty
                updated_replay_buffer = replay_buffer
                logger.info(
                    f"No new valid samples to append. {buffer_name} remains at size {len(updated_replay_buffer)}."
                )

        # 4. Enforce max_replay_samples limit (FIFO-like truncation)
        if (
            updated_replay_buffer is not None
            and len(updated_replay_buffer) > max_replay_samples
        ):
            logger.info(
                f"{buffer_name} size {len(updated_replay_buffer)} exceeds max {max_replay_samples}. "
                f"Truncating to keep the first {max_replay_samples} samples."
            )
            updated_replay_buffer = updated_replay_buffer.select(
                range(max_replay_samples)
            )

        if updated_replay_buffer is not None and len(updated_replay_buffer) > 0:
            logger.info(f"Final {buffer_name} size: {len(updated_replay_buffer)}.")
        else:
            logger.info(f"Final {buffer_name} is None or empty.")

        return merged_dataset_for_training, updated_replay_buffer

    def save_state(self, output_dir):
        """
        Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model

        Under distributed environment this is done only for a process with rank 0.
        """
        path = os.path.join(output_dir, "trainer_state.json")
        self.trainer.state.save_to_json(path)

    def train_for_MTL(self):
        sft_dataset, rl_dataset = self.sft_dataset_module["train_dataset"], self.continual_training_dataset
        output_dir = f"{self.training_args.output_dir}"
        sft_output_dir = f"{self.sft_training_args.output_dir}"
        logger.info(f"Start SFT training for MTL...")
        self.sft_trainer.train_single_dataset(
            dataset=sft_dataset,
            args=self.sft_training_args,
            output_dir=sft_output_dir
        )
        logger.info("SFT training finish.")
        logger.info(f"Start RL training for MTL...")
        self.trainer.train_single_dataset(
            dataset=rl_dataset,
            args=self.training_args,
            output_dir=output_dir
        )
        logger.info("RL training finish, models are saved by train_single_dataset.")

    def continual_learning(self):
        logger.info("Initializing custom Continual Learning modules with the following hyperparameters:")
        logger.info(f"  [Method Name] : {self.method_name}")
        logger.info(f"  [SC-FPO Loss] gamma_base: {self.finetuning_args.gamma_base}")
        logger.info(f"  [Refinement Step] denoising_threshold: {self.finetuning_args.denoising_threshold}")
        logger.info(f"  [Refinement Step] projection_gamma: {self.finetuning_args.projection_gamma}")
        logger.info(f"  [Loss Function] : {self.finetuning_args.loss_func}")
        rl_replay_buffer, sft_replay_buffer = None, None
        previous_output_dir = None
        base_rl_output_dir = self.training_args.output_dir
        base_sft_output_dir = self.sft_training_args.output_dir
        for id, (sft_dataset, rl_dataset) in enumerate(
            zip(
                self.sft_dataset_module["train_dataset"],
                self.continual_training_dataset,
            )
        ):
            original_sft_dataset = sft_dataset
            original_rl_dataset = rl_dataset
            # print(self.finetuning_args.CL_method)

            datatset_info = rl_dataset.info.download_checksums.keys()
            dataset_name = (
                list(datatset_info)[0].split("")[-1].split("/train.json")[0]
            )
            output_dir = f"{base_rl_output_dir}/{dataset_name}"
            sft_output_dir = f"{base_sft_output_dir}/{dataset_name}"
            logger.info(f"SFT for {dataset_name} Results will be saved on {sft_output_dir}.")
            logger.info(f"RL for {dataset_name} Results will be saved on {output_dir}.")
            if self.finetuning_args.use_replay:
                sft_dataset, sft_replay_buffer = self._manage_replay_buffer(
                    current_task_dataset=sft_dataset,
                    replay_buffer=sft_replay_buffer,
                    max_replay_samples=self.max_replay_samples,
                    replay_sample_percentage=self.replay_buffer_percentage,
                    buffer_name="SFT Replay Buffer",
                )
                rl_dataset, rl_replay_buffer = self._manage_replay_buffer(
                    current_task_dataset=rl_dataset,
                    replay_buffer=rl_replay_buffer,
                    max_replay_samples=self.max_replay_samples,
                    replay_sample_percentage=self.replay_buffer_percentage,
                    buffer_name="RL Replay Buffer",
                )
            if self.method_name == "my_method":
                logger.info(f"[my_method] Start SFT training on [{dataset_name}]...")
                self.cl_manager.stage = "SFT" 
                self.cl_manager.on_task_begin(task_index=id)
                self.sft_trainer.train_single_dataset(
                    dataset=sft_dataset,
                    args=self.sft_training_args,
                    task_name=dataset_name,
                    output_dir=sft_output_dir,
                    previous_task_output_dir=previous_output_dir,
                )
                self.cl_manager.on_task_end(trainer=self.sft_trainer, importance_data=original_sft_dataset)
                logger.info("[my_method] SFT training finish.")

                logger.info(f"[my_method] Start RL training on [{dataset_name}]...")
                self.cl_manager.stage = "RL"
                self.cl_manager.on_task_begin(task_index=id)
                self.trainer.train_single_dataset(
                    dataset=rl_dataset,
                    args=self.training_args,
                    task_name=dataset_name,
                    output_dir=output_dir,
                    previous_task_output_dir=sft_output_dir,
                )
                self.cl_manager.on_task_end(trainer=self.trainer, importance_data=original_rl_dataset)
                logger.info("RL training finish, saving models...")
            else:
                logger.info(f"Start SFT training on [{dataset_name}]...")
                if self.method_name == "GEM":
                    self.gem_manager.prepare_for_task(
                        task_name=f"SFT_{dataset_name}", 
                        trainer=self.sft_trainer
                    )
                self.sft_trainer.train_single_dataset(
                    dataset=sft_dataset,
                    args=self.sft_training_args,
                    task_name=dataset_name,
                    output_dir=sft_output_dir,
                    previous_task_output_dir=previous_output_dir,
                )
                logger.info("SFT training finish.")

                if self.method_name == "EWC":
                    logger.info("Registering EWC parameters after SFT stage...")
                    # Use the original SFT dataset for Fisher computation
                    self.ewc_manager.register_ewc_params(self.sft_trainer, original_sft_dataset)

                logger.info(f"Start RL training on [{dataset_name}]...")
                if self.method_name == "GEM":
                    if self.gem_manager.gem_callback not in self.trainer.callback_handler.callbacks:
                        self.trainer.add_callback(self.gem_manager.gem_callback)

                    self.gem_manager.prepare_for_task(
                        task_name=f"DPO_{dataset_name}", 
                        trainer=self.trainer
                    )
                self.trainer.train_single_dataset(
                    dataset=rl_dataset,
                    args=self.training_args,
                    task_name=dataset_name,
                    output_dir=output_dir,
                    previous_task_output_dir=sft_output_dir,
                )
                logger.info("RL training finish, models are saved by train_single_dataset.")

                if self.method_name == "EWC":
                    logger.info("Registering EWC parameters after DPO stage for the next task...")
                    # Use the original DPO dataset for Fisher computation
                    self.ewc_manager.register_ewc_params(self.trainer, original_rl_dataset)

                previous_output_dir = output_dir
