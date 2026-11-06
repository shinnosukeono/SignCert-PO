"""Gold-RM win rate of a policy against the reference responses.

The policy decodes greedily on the first `num_prompts` validation prompts. The gold RM
scores the policy's and the reference response to each prompt, and the win rate is the
fraction of prompts on which the policy's response scores higher.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams

from gold_rm import GOLD_MODEL, gold_scores, load_gold_model


@dataclass
class Arguments:
    model: str = field(metadata={"help": "Policy to evaluate."})
    dataset: str = field(
        metadata={
            "help": "Prompt dataset with a `validation` split and `query`, `query_token` "
            "and `reference_response_token` columns."
        }
    )
    output: str = field(metadata={"help": "Path of the output JSON file."})
    gold_model: str = GOLD_MODEL
    num_prompts: int = 512
    max_new_tokens: int = 53
    batch_size: int = field(default=32, metadata={"help": "Gold-RM batch size."})


def main():
    (args,) = HfArgumentParser(Arguments).parse_args_into_dataclasses()

    if os.path.isdir(args.dataset):
        dataset = load_from_disk(args.dataset)["validation"]
    else:
        dataset = load_dataset(args.dataset, split="validation")
    dataset = dataset.select(range(args.num_prompts))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # bfloat16 as in the vLLM engine used during training.
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.3)
    params = SamplingParams(
        temperature=0.0, max_tokens=args.max_new_tokens, ignore_eos=True
    )
    outputs = llm.generate(dataset["query"], params)
    eos = tokenizer.eos_token_id
    response_ids = [list(o.outputs[0].token_ids) for o in outputs]
    response_ids = [
        ids[: ids.index(eos) + 1] if eos in ids else ids for ids in response_ids
    ]
    responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    prompts = tokenizer.batch_decode(dataset["query_token"], skip_special_tokens=True)
    references = tokenizer.batch_decode(
        dataset["reference_response_token"], skip_special_tokens=True
    )

    gold_model, gold_tokenizer = load_gold_model(args.gold_model)
    scores = gold_scores(
        gold_model, gold_tokenizer, prompts, responses, args.batch_size
    )
    ref_scores = gold_scores(
        gold_model, gold_tokenizer, prompts, references, args.batch_size
    )

    win_rate = float(np.mean(np.array(scores) > np.array(ref_scores)))
    print(f"win rate: {win_rate:.4f}")
    with open(args.output, "w") as f:
        json.dump(
            {
                "win_rate": win_rate,
                "gold_score": float(np.mean(scores)),
                "reference_gold_score": float(np.mean(ref_scores)),
                "samples": [
                    {
                        "prompt": p,
                        "response": r,
                        "reference": ref,
                        "score": s,
                        "reference_score": rs,
                    }
                    for p, r, ref, s, rs in zip(
                        prompts, responses, references, scores, ref_scores
                    )
                ],
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
