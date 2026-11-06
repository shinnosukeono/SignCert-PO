"""Optimize a policy against a proxy reward model with SignCert-PO."""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import PartialState
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, HfArgumentParser

from reward_model import ProxyRewardModel, ScalarModel
from signcert_po import SignCertPOConfig, SignCertPOTrainer

torch.set_float32_matmul_precision("high")


@dataclass
class ScriptArguments:
    sft_model: str = field(metadata={"help": "SFT policy, also the KL reference."})
    reward_model: str = field(metadata={"help": "Proxy reward model (ScalarModel)."})
    dataset: str = field(
        metadata={
            "help": "Prompt dataset with a `query` column and a `train` split "
            "(Hub id or `save_to_disk` directory)."
        }
    )
    reward_batch_size: Optional[int] = field(
        default=None, metadata={"help": "Batch size of reward-model forward passes."}
    )


def main():
    script_args, training_args = HfArgumentParser(
        (ScriptArguments, SignCertPOConfig)
    ).parse_args_into_dataclasses()

    if os.path.isdir(script_args.dataset):
        dataset = load_from_disk(script_args.dataset)["train"]
    else:
        dataset = load_dataset(script_args.dataset, split="train")
    dataset = dataset.rename_column("query", "prompt")

    tokenizer = AutoTokenizer.from_pretrained(script_args.reward_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = ScalarModel.from_pretrained(script_args.reward_model)
    model = model.to(torch.bfloat16).to(PartialState().device).eval()
    reward_model = ProxyRewardModel(
        model, tokenizer.eos_token_id, script_args.reward_batch_size
    )

    trainer = SignCertPOTrainer(
        model=script_args.sft_model,
        reward_funcs=reward_model,
        reward_processing_classes=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
