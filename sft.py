"""Supervised fine-tuning on the reference responses."""

import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from accelerate import Accelerator, PartialState
from accelerate.utils import set_seed
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    get_scheduler,
)


@dataclass
class Arguments:
    base_model: str = field(metadata={"help": "Pretrained model to fine-tune."})
    dataset: str = field(
        metadata={
            "help": "Dataset with a `train` split and a `query_reference_response_token` "
            "column (Hub id or `save_to_disk` directory)."
        }
    )
    output_dir: str = field(metadata={"help": "Where to save the SFT policy."})
    seed: int = 1
    lr: float = 3e-6
    batch_size: int = field(default=128, metadata={"help": "Total batch size."})
    per_device_batch_size: int = 16


def main():
    (args,) = HfArgumentParser(Arguments).parse_args_into_dataclasses()
    num_processes = PartialState().num_processes
    accelerator = Accelerator(
        gradient_accumulation_steps=args.batch_size
        // (args.per_device_batch_size * num_processes)
    )

    if os.path.isdir(args.dataset):
        dataset = load_from_disk(args.dataset)["train"]
    else:
        dataset = load_dataset(args.dataset, split="train")
    dataset = dataset.with_format("torch", columns=["query_reference_response_token"])
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # The datasets are padded with an added [PAD] token.
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataset) // args.batch_size,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.per_device_batch_size, shuffle=True
    )
    set_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    model.train()
    for step, batch in enumerate(dataloader):
        ids = batch["query_reference_response_token"]
        mask = ids != tokenizer.pad_token_id
        with accelerator.accumulate(model):
            logits = model(
                input_ids=ids.masked_fill(~mask, 0), attention_mask=mask
            ).logits
            # The loss covers the query and the reference response.
            labels = ids.masked_fill(~mask, -100)
            loss = F.cross_entropy(
                logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten()
            )
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        if accelerator.sync_gradients:
            scheduler.step()
            if step % 100 == 0:
                accelerator.print(f"step {step}: loss {loss.item():.4f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)
    accelerator.unwrap_model(model).save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )


if __name__ == "__main__":
    main()
