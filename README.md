# SignCert-PO

Code for *Mitigating Reward Hacking in Reinforcement Learning from Human Feedback via Advantage Sign Robustness* (Ono et al., TMLR 2026, [OpenReview](https://openreview.net/forum?id=yu3lbJi5sh)).

SignCert-PO optimizes a policy against a proxy reward model $r(x, y) = w^\top h(x, y) + b$, where $h(x, y) \in \mathbb{R}^d$ is the reward model's last-layer feature of prompt $x$ and response $y$. For each prompt, the policy samples $K$ completions $y^{(1)}, \dots, y^{(K)}$, and Dr.GRPO (Liu et al., 2025) gives completion $j$ the advantage $A_j = r_j - \frac{1}{K}\sum_k r_k$. SignCert-PO replaces $A_j$ with its worst case when the head $w$ is perturbed by a vector of norm at most $\epsilon$:

$$\tilde A_j = \left(1 - \frac{\epsilon}{\Delta_j}\right) A_j, \qquad \Delta_j = \frac{|A_j|}{\Vert h_j - \bar h \Vert_2},$$

where $h_j = h(x, y^{(j)})$, $\bar h$ is the mean of $h_1, \dots, h_K$, and $\Delta_j$ is the largest perturbation norm under which the sign of $A_j$ cannot flip. We set $\epsilon$ to the $q_t$-quantile of $\Delta_1, \dots, \Delta_K$ within each prompt's group. The function `signcert_advantages` in `signcert_po.py` implements this.

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions we used. `signcert_po.py` subclasses `GRPOTrainer` of TRL 0.18.1 and may not work with other TRL versions.

## Usage

`scripts/run.sh` runs the full pipeline for one setting of Table 1 on a single node with 8 GPUs:

```bash
bash scripts/run.sh tldr_pythia1b        # optional second argument: seed
```

The pipeline has the following steps. Each step is a script that can also be run alone; `--help` lists its options.

| Step | Script | Output |
|------|--------|--------|
| 1. Fine-tune the base model on reference responses | `sft.py` | SFT policy |
| 2. Sample two responses per prompt from the SFT policy | `sample.py` | response pairs |
| 3. Label the preferred response with Skywork-Reward-Llama-3.1-8B-v0.2 or GPT-4.1 Nano | `label.py` | preference data |
| 4. Train the proxy reward model with the Bradley–Terry loss | `train_rm.py` | proxy reward model |
| 5. Optimize the SFT policy against the proxy reward model with SignCert-PO | `train.py` | policy |
| 6. Compute the win rate against the reference responses with the gold reward model | `evaluate.py` | JSON file |

Labeling with GPT-4.1 Nano reads the API key from `OPENAI_API_KEY`. The gold reward model of step 6 is Skywork-Reward-Llama-3.1-8B-v0.2 in every setting. The win rate is the fraction of the first 512 validation prompts on which the policy's greedy response scores higher than the reference response.

## Settings

The following table lists the hyperparameters of the six settings in `scripts/run.sh`. "Pairs per prompt" is the number of response pairs sampled per prompt in step 2, and "RM passes" is the number of passes over them in step 4.

| Setting | Base model | Labeler | Pairs per prompt | RM passes | Max new tokens | KL coefficient $\beta$ | $q_t$ | Steps |
|---------|------------|---------|------------------|-----------|----------------|------------------------|-------|-------|
| `tldr_pythia1b` | Pythia-1B | Skywork | 1 | 1 | 53 | 0.01 | 0.4 | 2,000 |
| `tldr_pythia2.8b` | Pythia-2.8B | GPT-4.1 Nano | 1 | 1 | 53 | 0.005 | 0.2 | 2,000 |
| `tldr_qwen1.5b` | Qwen2.5-1.5B | Skywork | 1 | 1 | 53 | 0.1 | TODO | 500 |
| `tldr_qwen3b` | Qwen2.5-3B | GPT-4.1 Nano | 1 | 1 | 53 | 0.01 | 0.25 | 500 |
| `alpaca_qwen1.5b` | Qwen2.5-1.5B | Skywork | 3 | 3 | 128 | 0.01 | 0.5 | 500 |
| `alpaca_qwen3b` | Qwen2.5-3B | GPT-4.1 Nano | 3 | 3 | 128 | 0.1 | 0.25 | 500 |

All settings share the following hyperparameters.

- SFT: learning rate $3 \times 10^{-6}$, cosine schedule, one epoch, total batch size 128.
- Response pairs: sampling temperature 1.0.
- Proxy reward model: learning rate $3 \times 10^{-6}$, cosine schedule without warmup, AdamW with $\epsilon = 10^{-5}$ and weight decay 0.01, total batch size 128.
- SignCert-PO: learning rate $10^{-6}$, constant schedule, $K = 8$ completions per prompt, 256 completions per step, sampling temperature 0.7, clip range 0.2.

The prompt datasets are [TL;DR for Pythia](https://huggingface.co/datasets/vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144) (Huang et al., 2024), [TL;DR for Qwen2.5](https://huggingface.co/datasets/johannesack/summarize_from_feedback_tldr_3_filtered_oai_preprocessingQWEN_1748920372), and [AlpacaFarm for Qwen2.5](https://huggingface.co/datasets/shinnosukeono/alpaca_farm_preprocessing_1771395192) (Dubois et al., 2023).

## Implementation details

The following details reproduce the pipeline that produced the paper's results.

- **Position of the reward.** Step 2 samples exactly "max new tokens" tokens per response and keeps the tokens generated after the end-of-sequence (EOS) token. Step 4 therefore reads the reward at the last sampled token. Step 5 reads the reward at an EOS token written after each completion, including completions cut at the maximum length. The pipeline of Ackermann et al. (2025), on which ours is based, reads the reward in the same way.
- **Preference pairs used in step 4.** Each process shuffles the preference data with its own seed and trains on every $N$-th batch of its own order, where $N$ is the number of processes (Huang et al., 2024; Ackermann et al., 2025). With 8 processes, about a third of the pairs are not used in a pass.
- **Feature position.** Step 5 takes the feature $h_j$ at the last token of the completion, which is one token before the EOS token where the reward is read.
- **Pythia-2.8B.** vLLM 0.8.4 has no FlashAttention-2 kernel for Pythia-2.8B's head size of 80, so `scripts/run.sh` sets `VLLM_ATTENTION_BACKEND=FLEX_ATTENTION` for this setting.

## Acknowledgments

`reward_model.py`, `sft.py`, `sample.py`, `label.py` and `train_rm.py` are adapted from the code of Ackermann et al. (2025) ([OffPolicyCorrectedRewardModeling](https://github.com/JohannesAck/OffPolicyCorrectedRewardModeling)), which builds on the code of Huang et al. (2024) ([summarize_from_feedback_details](https://github.com/vwxyzjn/summarize_from_feedback_details)). Both are released under the MIT License, with the copyright notices "Copyright (c) 2025 Johannes Ackermann" and "Copyright (c) 2024 Costa Huang".

## Citation

```bibtex
@article{ono2026mitigating,
  title   = {Mitigating Reward Hacking in Reinforcement Learning from Human Feedback via Advantage Sign Robustness},
  author  = {Ono, Shinnosuke and Ackermann, Johannes and Nishimori, Soichiro and Ishida, Takashi and Sugiyama, Masashi},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=yu3lbJi5sh}
}
```

## References

- Ackermann, J., Ishida, T., and Sugiyama, M. Off-policy corrected reward modeling for reinforcement learning from human feedback. COLM, 2025.
- Dubois, Y., et al. AlpacaFarm: A simulation framework for methods that learn from human feedback. NeurIPS, 2023.
- Huang, S., et al. The N+ implementation details of RLHF with PPO: A case study on TL;DR summarization. COLM, 2024.
- Liu, Z., et al. Understanding R1-Zero-like training: A critical perspective. COLM, 2025.
