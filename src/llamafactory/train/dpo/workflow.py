# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
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

from typing import TYPE_CHECKING, List, Optional

from ...data import (
    PairwiseDataCollatorWithPadding,
    get_dataset,
    get_template_and_fix_tokenizer,
)
from ...extras.constants import IGNORE_INDEX
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomDPOTrainer
from .DPOwithEWC import DPOwithEWCTrainer
from .vanillaDPOTrainer import vanillaDPOTrainer
from .BaseTrainerCL import BaseCLTrainer
from .MyDPOTrainer import MyDPOTrainer
from .DPOwithGEM import DPOwithGEMTrainer
from .DPOwithL2P import DPOwithL2PTrainer
from .DPOwithCPPO import DPOWithCPPOTrainer
from .DPOwithOLoRA import DPOwithOLoRATrainer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments


def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    merge_type = False
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="rm",
        **tokenizer_module,
        merge=merge_type,
    )
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        pad_to_multiple_of=8,
        label_pad_token_id=(
            IGNORE_INDEX
            if data_args.ignore_pad_token_for_loss
            else tokenizer.pad_token_id
        ),
        **tokenizer_module,
    )

    # Create reference model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (
            not training_args.do_train
        ):  # use the model itself
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        ref_model = None

    # Update arguments
    training_args.remove_unused_columns = (
        False  # important for multimodal and pairwise dataset
    )

    # Initialize our Trainer
    cl_method = finetuning_args.CL_method
    if cl_method == "EWC":
        RLTrainerClass = DPOwithEWCTrainer
    elif cl_method == "GEM":
        RLTrainerClass = DPOwithGEMTrainer
    elif cl_method == "my_method":
        RLTrainerClass = MyDPOTrainer
    elif cl_method == "L2P":
        RLTrainerClass = DPOwithL2PTrainer
    elif cl_method == "CPPO":
        RLTrainerClass = DPOWithCPPOTrainer
    elif cl_method == "OLoRA":
        RLTrainerClass = DPOwithOLoRATrainer
    else:
        RLTrainerClass = vanillaDPOTrainer

    RL_trainer = RLTrainerClass(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **tokenizer_module,
    )
    trainer = BaseCLTrainer(
        model=model,
        finetuning_args=finetuning_args,
        training_args=training_args,
        template=template,
        callbacks=callbacks,
        trainer=RL_trainer,
        CL_method=cl_method,
        **dataset_module,
    )

    # Training
    if training_args.do_train:
        # if cl_method == "MTL":
        #     trainer.train_for_MTL()
        # else:
        trainer.continual_learning()
        # train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

        # trainer.log_metrics("train", train_result.metrics)
        # trainer.save_metrics("train", train_result.metrics)
        # if trainer.is_world_process_zero() and finetuning_args.plot_loss:
        #     plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(
            ref_model
        ):  # unable to compute rewards if reference model is the model itself
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
