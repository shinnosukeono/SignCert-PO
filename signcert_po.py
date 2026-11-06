"""SignCert-PO (Algorithm 1) on top of TRL's `GRPOTrainer` (trl==0.18.1)."""

from dataclasses import dataclass, field

import torch
from accelerate.utils import gather
from trl import GRPOConfig, GRPOTrainer


def signcert_advantages(
    advantages: torch.Tensor, features: torch.Tensor, quantile: float
) -> torch.Tensor:
    """Worst-case advantages rho_j * A_j (Theorem 4.2).

    Args:
        advantages: Dr.GRPO advantages A_j = r_j - mean_k r_k, shape (B, K).
        features: RM features h(x, y^(j)), shape (B, K, d).
        quantile: q_t; epsilon is the q_t-quantile of the radii within each group.
    """
    dev = torch.norm(features - features.mean(dim=1, keepdim=True), dim=2)
    radius = advantages.abs() / (dev + 1e-8)  # Delta_j (Theorem 4.1)
    eps = torch.quantile(radius, quantile, dim=1, keepdim=True)
    return advantages - eps * dev * torch.sign(advantages)  # (1 - eps/Delta_j) A_j


@dataclass
class SignCertPOConfig(GRPOConfig):
    quantile: float = field(
        default=0.4,
        metadata={"help": "Quantile q_t of the certified radii that sets epsilon."},
    )


class SignCertPOTrainer(GRPOTrainer):
    """`GRPOTrainer` with SignCert-PO advantages.

    Use with Dr.GRPO advantages (`scale_rewards=False`, `loss_type="dr_grpo"`) and a
    single reward function that stores the features h(x, y) of its latest call in
    `features` (`reward_model.ProxyRewardModel`).
    """

    def _generate_and_score_completions(self, inputs):
        outputs = super()._generate_and_score_completions(inputs)
        # A group of K completions can span several processes.
        K = self.num_generations
        A = gather(outputs["advantages"]).view(-1, K)
        h = gather(self.reward_funcs[0].features)
        A = signcert_advantages(A, h.view(-1, K, h.shape[-1]), self.args.quantile)
        n = len(outputs["advantages"])
        start = self.accelerator.process_index * n
        outputs["advantages"] = A.view(-1)[start : start + n]
        return outputs
