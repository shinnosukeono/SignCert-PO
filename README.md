# Mitigating Reward Hacking in RLHF via Advantage Sign Robustness

Code for our paper, *Mitigating Reward Hacking in Reinforcement Learning from Human Feedback via Advantage Sign Robustness* (Ono et al., TMLR 2026, [OpenReview](https://openreview.net/forum?id=yu3lbJi5sh)).

SignCert-PO optimizes a policy against a proxy reward model $r(x, y) = w^\top h(x, y) + b$, where $h(x, y) \in \mathbb{R}^d$ is the reward model's last-layer feature of prompt $x$ and response $y$. For each prompt, the policy samples $K$ completions $y^{(1)}, \dots, y^{(K)}$, and Dr.GRPO (Liu et al., 2025) gives completion $j$ the advantage $A_j = r_j - \frac{1}{K}\sum_k r_k$. SignCert-PO replaces $A_j$ with its worst case when the head $w$ is perturbed by a vector of norm at most $\epsilon$:

$$\tilde A_j = \left(1 - \frac{\epsilon}{\Delta_j}\right) A_j, \qquad \Delta_j = \frac{|A_j|}{\Vert h_j - \bar h \Vert_2},$$

where $h_j = h(x, y^{(j)})$, $\bar h$ is the mean of $h_1, \dots, h_K$, and $\Delta_j$ is the largest perturbation norm under which the sign of $A_j$ cannot flip.

## Installation

Our experiments ran on a shared cluster with NVIDIA GH200 nodes.
We also used an apptainer container to set up the environment.

```bash
pip install -r requirements.txt
```

`signcert_po.py` overwrites `GRPOTrainer` of TRL 0.18.1 and may not work with other TRL versions.

## Usage

`scripts/run.sh` runs the full RLHF pipeline in our main experiments:

```bash
bash scripts/run.sh tldr_pythia1b        # optional second argument: seed
```

The pipeline has the following steps.

| Step | Script | Output |
|------|--------|--------|
| 1. SFT on reference responses | `sft.py` | SFT policy |
| 2. Sample two responses per prompt from the SFT policy | `sample.py` | response pairs |
| 3. Label the preferred response with a gold RM | `label.py` | preference data |
| 4. Train the proxy RM | `train_rm.py` | proxy reward model |
| 5. RL training with SignCert-PO | `train.py` | policy |
| 6. Compute the win rate against the reference responses with the gold RM | `evaluate.py` | JSON file |

## Acknowledgments

`reward_model.py`, `sft.py`, `sample.py`, `label.py` and `train_rm.py` are adapted from the code of Ackermann et al. (2025) ([OffPolicyCorrectedRewardModeling](https://github.com/JohannesAck/OffPolicyCorrectedRewardModeling)), which builds on the code of Huang et al. (2024) ([summarize_from_feedback_details](https://github.com/vwxyzjn/summarize_from_feedback_details)).

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
