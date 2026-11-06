"""Proxy reward model r(x, y) = w^T h(x, y) + b."""

from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "",
        base_config: Optional[dict] = None,
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    """Language-model backbone h with a linear head (w, b)."""

    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        base_config = config.base_config
        if isinstance(base_config, dict):
            base_config = AutoConfig.for_model(**base_config)
        self.lm_backbone = AutoModel.from_config(base_config)
        self.scalar_head = nn.Linear(config.hidden_size, 1)

    def forward(self, **kwargs):
        hidden = self.lm_backbone(**kwargs).last_hidden_state
        return self.scalar_head(hidden) - self.config.bias


class ProxyRewardModel(nn.Module):
    """Reward function for `GRPOTrainer`.

    Keeps the features h(x, y) of the latest call in `self.features` for SignCert-PO.
    """

    def __init__(
        self, model: ScalarModel, eos_token_id: int, batch_size: Optional[int] = None
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.eos_token_id = eos_token_id
        self.batch_size = batch_size
        self.features = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # GRPOTrainer decodes completions without special tokens. The RM reads its
        # reward at the EOS token, so we write EOS after the last token (or over it
        # when the row has no padding left).
        lengths = attention_mask.sum(dim=1)
        rows = (~(input_ids == self.eos_token_id).any(dim=1)).nonzero(as_tuple=True)[0]
        cols = lengths[rows].clamp(max=input_ids.shape[1] - 1)
        input_ids = input_ids.clone()
        input_ids[rows, cols] = self.eos_token_id
        eos_mask = attention_mask.clone()
        eos_mask[rows, cols] = 1

        batch_size = self.batch_size or len(input_ids)
        rewards, features = [], []
        for i in range(0, len(input_ids), batch_size):
            ids, mask = input_ids[i : i + batch_size], eos_mask[i : i + batch_size]
            hidden = self.model.lm_backbone(
                input_ids=ids, attention_mask=mask
            ).last_hidden_state
            idx = torch.arange(len(ids), device=ids.device)
            last = hidden[idx, mask.sum(dim=1) - 1]
            rewards.append(self.model.scalar_head(last) - self.config.bias)
            # The features h(x, y) are read at the last token before the EOS.
            features.append(hidden[idx, lengths[i : i + batch_size] - 1])
        self.features = torch.cat(features)
        return SequenceClassifierOutput(logits=torch.cat(rewards))
