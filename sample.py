"""Sample two responses per prompt from the SFT policy for preference labeling."""

import os
from dataclasses import dataclass, field

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams


@dataclass
class Arguments:
    sft_model: str = field(metadata={"help": "SFT policy."})
    dataset: str = field(
        metadata={
            "help": "Prompt dataset with a `train` split and `query` and `query_token` "
            "columns (Hub id or `save_to_disk` directory)."
        }
    )
    output: str = field(metadata={"help": "Directory of the output dataset."})
    num_epochs: int = field(
        default=1, metadata={"help": "Number of response pairs per prompt."}
    )
    max_new_tokens: int = 53
    temperature: float = 1.0
    seed: int = 1


def main():
    (args,) = HfArgumentParser(Arguments).parse_args_into_dataclasses()

    if os.path.isdir(args.dataset):
        dataset = load_from_disk(args.dataset)["train"]
    else:
        dataset = load_dataset(args.dataset, split="train")
    query_token = dataset.with_format("numpy")["query_token"]

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    llm = LLM(
        model=args.sft_model,
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=query_token.shape[1] + args.max_new_tokens,
    )
    params = SamplingParams(
        n=2,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
    )

    shards = []
    for epoch in range(args.num_epochs):
        outputs = llm.generate(dataset["query"], params)
        shard = {
            "query": dataset["query"],
            "query_token": query_token,
            "epoch": [epoch] * len(dataset),
        }
        for k in (1, 2):
            ids = [list(out.outputs[k - 1].token_ids) for out in outputs]
            # The text ends at the first EOS. The tokens keep everything generated
            # after it, as in the paper's pipeline.
            shard[f"response{k}"] = tokenizer.batch_decode(
                [x[: x.index(eos) + 1] if eos in x else x for x in ids],
                skip_special_tokens=True,
            )
            shard[f"response{k}_token"] = [
                x + [pad] * (args.max_new_tokens - len(x)) for x in ids
            ]
        shards.append(Dataset.from_dict(shard))
    concatenate_datasets(shards).save_to_disk(args.output)


if __name__ == "__main__":
    main()
