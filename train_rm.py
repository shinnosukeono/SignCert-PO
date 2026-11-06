"""Train the proxy reward model on the labeled response pairs (Bradley-Terry loss)."""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from accelerate import Accelerator, PartialState
from accelerate.utils import set_seed
from datasets import load_from_disk
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    get_scheduler,
)

from reward_model import ScalarModel, ScalarModelConfig


@dataclass
class Arguments:
    sft_model: str = field(metadata={"help": "SFT policy, the RM's backbone."})
    dataset: str = field(metadata={"help": "Output directory of label.py."})
    output_dir: str = field(metadata={"help": "Where to save the reward model."})
    seed: int = 1
    lr: float = 3e-6
    num_train_epochs: int = 1
    batch_size: int = field(default=128, metadata={"help": "Total batch size."})
    per_device_batch_size: int = 16


def main():
    (args,) = HfArgumentParser(Arguments).parse_args_into_dataclasses()
    num_processes = PartialState().num_processes
    accelerator = Accelerator(
        gradient_accumulation_steps=args.batch_size
        // (args.per_device_batch_size * num_processes)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    pad = tokenizer.pad_token_id
    dataset = load_from_disk(args.dataset)
    columns = ["query_token", "response1_token", "response2_token"]
    # vLLM can sample token ids that the tokenizer does not have.
    dataset = dataset.filter(lambda x: max(max(x[c]) for c in columns) < len(tokenizer))
    # Each process shuffles with its own seed, as in the paper's pipeline.
    dataset = dataset.shuffle(seed=args.seed + accelerator.process_index * 100003)
    dataset = dataset.with_format("torch", columns=columns + ["response1_preferred"])
    dataloader = DataLoader(dataset, batch_size=args.per_device_batch_size)

    set_seed(args.seed)
    base_config = AutoConfig.from_pretrained(args.sft_model)
    model = ScalarModel(
        ScalarModelConfig(
            base_model=args.sft_model,
            base_config=base_config.to_dict(),
            hidden_size=base_config.hidden_size,
        )
    )
    model.lm_backbone = AutoModel.from_pretrained(args.sft_model)
    nn.init.normal_(
        model.scalar_head.weight, std=1 / math.sqrt(base_config.hidden_size + 1)
    )
    nn.init.zeros_(model.scalar_head.bias)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = get_scheduler(
        "cosine",
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataset) // args.batch_size * args.num_train_epochs,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    model.train()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(dataloader):
            first = batch["response1_preferred"][:, None]
            r1, r2 = batch["response1_token"], batch["response2_token"]
            query = batch["query_token"]
            ids = torch.cat(
                [
                    torch.cat([query, torch.where(first, r1, r2)], dim=1),
                    torch.cat([query, torch.where(first, r2, r1)], dim=1),
                ]
            )
            mask = ids != pad
            with accelerator.accumulate(model):
                rewards = model(
                    input_ids=ids.masked_fill(~mask, 0), attention_mask=mask
                )
                # The reward is read at the token before the first [PAD]. The query is
                # left-padded, so this is the last token of the response, which keeps
                # the tokens generated after EOS (as in the paper's pipeline).
                idx = (ids == pad).int().argmax(dim=1) - 1
                rewards = rewards[torch.arange(len(ids), device=ids.device), idx, 0]
                chosen, rejected = rewards.chunk(2)
                loss = -F.logsigmoid(chosen - rejected).mean()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                scheduler.step()
                if step % 100 == 0:
                    accuracy = (chosen > rejected).float().mean().item()
                    accelerator.print(
                        f"epoch {epoch} step {step}: loss {loss.item():.4f}, "
                        f"accuracy {accuracy:.3f}"
                    )

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
