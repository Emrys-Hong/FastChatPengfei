# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
import os
import pathlib
import sys
import typing
from dataclasses import dataclass, field

import torch
import transformers
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    make_supervised_data_module,
    make_supervised_data_module_contrastive,
    make_supervised_data_module_nut,
    make_supervised_data_module_pos_neg,
)
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, Trainer

from trainer_module import CustomLlamaForCausalLM, CustomTrainer


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (model_args, data_args, training_args) = parser.parse_args_into_dataclasses()

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    print("rishabh:", device_map)

    model = CustomLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        load_in_8bit=False,  # rishabh changed from true
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module_contrastive = make_supervised_data_module_contrastive(
        tokenizer=tokenizer, data_args=data_args
    )
    train_dataset, eval_dataset = (
        data_module_contrastive["train_dataset"],
        data_module_contrastive["eval_dataset"],
    )
    # data_module_pos_neg = make_supervised_data_module_pos_neg(
    #     tokenizer=tokenizer, data_args=data_args
    # )

    # train_dataset_pos = data_module_pos_neg["train_dataset_pos"]
    # train_dataset_pos.add_data_tag("pos")

    # train_dataset_neg = data_module_pos_neg["train_dataset_neg"]
    # train_dataset_neg.add_data_tag("neg")

    # eval_dataset_pos = data_module_pos_neg["eval_dataset_pos"]
    # eval_dataset_pos.add_data_tag("pos")

    # eval_dataset_neg = data_module_pos_neg["eval_dataset_neg"]
    # eval_dataset_neg.add_data_tag("neg")

    # train_dataset = copy.deepcopy(train_dataset_pos)
    # train_dataset.make_contrastive_data(train_dataset_pos, train_dataset_neg)
    train_dataset = copy.deepcopy(train_dataset)
    train_dataset.make_contrastive_data()

    eval_dataset = copy.deepcopy(eval_dataset)
    eval_dataset.make_contrastive_data()

    # excluding negative data from eval (for now)
    # eval_dataset.make_contrastive_data(eval_dataset_pos)

    # if you want to include negative data as well, please uncomment the following line and comment the above
    # eval_dataset.make_contrastive_data(eval_dataset_pos, eval_dataset_neg, eval_dataset_nut)

    data_module = {"train_dataset": train_dataset, "eval_dataset": eval_dataset}

    def collator(inputs):
        input_positive = [o.positive for o in inputs]
        input_negative = [o.negative for o in inputs]
        input_neutral = [o.neutral for o in inputs]

        # The first half is positive and the second half is negative
        inputs = input_positive + input_negative
        # device = model.device
        inputs = dict(
            input_ids=torch.cat([o["input_ids"] for o in inputs], dim=0).contiguous(),
            # .to(device),
            attention_mask=torch.cat(
                [o["attention_mask"] for o in inputs], dim=0
            ).contiguous(),
            # .to(device),
            labels=torch.cat([o["labels"] for o in inputs], dim=0).contiguous(),
            # .to(device),
        )
        inputs["input_ids"][:, 0] = 1

        return inputs

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    model.config.use_cache = False

    print(
        f"\n\n#Number trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        **data_module,
    )

    # print(f"\n\n Device Map: {model.hf_device_map}")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        # trainer.evaluate()

    trainer.save_state()

    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    with torch.autocast("cuda"):
        train()


"""
deepspeed --num_gpus=1 fastchat/train/train_ft_contrastive.py     --ddp_timeout=360000     --output_dir "/data/emrys"     --deepspeed "deepspeed_configs/fp16_ft_7b.json" --fp16 True     --model_name_or_path "lmsys/vicuna-7b-v1.3"     --data_path_pos "data/data_h1_finetune.json"     --data_path_neg "data/data_lh1_finetune.json"     --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8     --num_train_epochs 3 --evaluation_strategy "steps" --eval_steps 20    --save_strategy "steps" --save_steps 100 --save_total_limit 20     --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.0001 --lr_scheduler_type "cosine" --logging_steps 10     --model_max_length 1280     --gradient_checkpointing True --lazy_preprocess True --disable_tqdm False
deepspeed --include localhost:1 fastchat/train/train_ft_contrastive.py     --ddp_timeout=360000     --output_dir "/data/emrys"     --deepspeed "deepspeed_configs/fp16_ft_7b.json" --fp16 True     --model_name_or_path "lmsys/vicuna-7b-v1.3"     --data_path_pos "data/data_h1_finetune.json"     --data_path_neg "data/data_lh1_finetune.json"     --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8     --num_train_epochs 3 --evaluation_strategy "steps" --eval_steps 20    --save_strategy "steps" --save_steps 100 --save_total_limit 20     --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.0001 --lr_scheduler_type "cosine" --logging_steps 10     --model_max_length 1280     --gradient_checkpointing True --lazy_preprocess True --disable_tqdm False
"""
