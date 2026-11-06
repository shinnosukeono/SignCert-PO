"""Gold reward model that scores (prompt, response) pairs as a chat."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

GOLD_MODEL = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


def load_gold_model(name=GOLD_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    return model.to("cuda").eval(), tokenizer


def gold_scores(model, tokenizer, prompts, responses, batch_size=32):
    scores = []
    for i in range(0, len(prompts), batch_size):
        convs = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
            )
            for prompt, response in zip(
                prompts[i : i + batch_size], responses[i : i + batch_size]
            )
        ]
        inputs = tokenizer(convs, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            scores += model(**inputs).logits[:, 0].float().tolist()
    return scores
