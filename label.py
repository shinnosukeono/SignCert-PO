"""Label which of the two sampled responses is preferred.

The labeler is either the gold RM (Skywork-Reward-Llama-3.1-8B-v0.2) or GPT-4.1 Nano
(through the OpenAI API, which reads the key from OPENAI_API_KEY).
"""

import asyncio
import random
from dataclasses import dataclass, field

from datasets import load_from_disk
from openai import AsyncOpenAI
from transformers import HfArgumentParser

from gold_rm import gold_scores, load_gold_model

PROMPTS = {
    "tldr": """
You will be shown a Reddit post and two summaries. You will be asked to rate which one captures the content of the post better.

<POST>
{post}
</POST>

<SUMMARY A>
{response_a}
</SUMMARY A>

<SUMMARY B>
{response_b}
</SUMMARY B>

Please reply only with the letter "A" or "B", for which summary you think is better. Pick one, even if neither is good!
""",
    "alpaca": """
You will be shown an instruction and two responses. You will be asked to rate which one is the better response.

<INSTRUCTION>
{post}
</INSTRUCTION>

<RESPONSE A>
{response_a}
</RESPONSE A>

<RESPONSE B>
{response_b}
</RESPONSE B>

Please reply only with the letter "A" or "B", for which summary you think is better. You have to pick one, even if neither is good!
""",
}


@dataclass
class Arguments:
    input: str = field(metadata={"help": "Output directory of sample.py."})
    output: str = field(metadata={"help": "Directory of the labeled dataset."})
    labeler: str = field(
        default="skywork", metadata={"help": "`skywork` or `gpt-4.1-nano`."}
    )
    task: str = field(
        default="tldr", metadata={"help": "`tldr` or `alpaca` (GPT prompt)."}
    )
    batch_size: int = 32
    max_concurrency: int = 64


async def gpt_labels(queries, responses1, responses2, args):
    client = AsyncOpenAI(max_retries=8)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def label(query, a, b):
        async with semaphore:
            completion = await client.chat.completions.create(
                model=args.labeler,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {
                        "role": "user",
                        "content": PROMPTS[args.task].format(
                            post=query, response_a=a, response_b=b
                        ),
                    },
                ],
                max_tokens=24,
                temperature=0.7,
            )
        reply = completion.choices[0].message.content or ""
        # A reply that names neither or both responses is decided by a coin flip.
        if ("A" in reply) != ("B" in reply):
            return "A" in reply
        return random.random() < 0.5

    return await asyncio.gather(
        *(label(*x) for x in zip(queries, responses1, responses2))
    )


def main():
    (args,) = HfArgumentParser(Arguments).parse_args_into_dataclasses()
    dataset = load_from_disk(args.input)
    queries, responses1, responses2 = (
        dataset["query"],
        dataset["response1"],
        dataset["response2"],
    )

    if args.labeler == "skywork":
        model, tokenizer = load_gold_model()
        scores1 = gold_scores(model, tokenizer, queries, responses1, args.batch_size)
        scores2 = gold_scores(model, tokenizer, queries, responses2, args.batch_size)
        labels = [s1 >= s2 for s1, s2 in zip(scores1, scores2)]
    else:
        labels = asyncio.run(gpt_labels(queries, responses1, responses2, args))

    print(f"response1 preferred in {sum(labels)} of {len(labels)} pairs")
    dataset.add_column("response1_preferred", labels).save_to_disk(args.output)


if __name__ == "__main__":
    main()
