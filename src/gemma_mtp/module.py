from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import AttentionSpec, MtpDrafterConfig


@dataclass
class DrafterOutput:
    logits: Tensor
    projected_activations: Tensor
    hidden_states: Tensor


class ExternalAttentionAdapter(nn.Module):
    def __init__(self, spec: AttentionSpec, model_dim: int) -> None:
        super().__init__()
        self.spec = spec
        self.q_out_dim = spec.query_heads * spec.query_head_dim
        self.q_proj = nn.Linear(model_dim, self.q_out_dim, bias=False)
        self.o_proj = nn.Linear(self.q_out_dim, model_dim, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        mask: Tensor | None = None,
        base_kv_cache: dict[str, Tensor] | None = None,
        input_pos: Tensor | None = None,
        param_tensor: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError


class ZeroAttentionAdapter(ExternalAttentionAdapter):
    def forward(
        self,
        hidden_states: Tensor,
        *,
        mask: Tensor | None = None,
        base_kv_cache: dict[str, Tensor] | None = None,
        input_pos: Tensor | None = None,
        param_tensor: Tensor | None = None,
    ) -> Tensor:
        del mask, base_kv_cache, input_pos, param_tensor
        # Keep the linear modules present for weight loading, but return a
        # no-op attention contribution until cache semantics are reconstructed.
        return torch.zeros_like(hidden_states)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states * self.weight


class GatedMLP(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(model_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(model_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, model_dim, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gated = F.gelu(self.gate_proj(hidden_states), approximate="tanh")
        value = self.up_proj(hidden_states)
        return self.down_proj(gated * value)


class MtpDrafterBlock(nn.Module):
    def __init__(self, spec: AttentionSpec, model_dim: int, mlp_hidden_dim: int) -> None:
        super().__init__()
        self.pre_attn_norm = RMSNorm(model_dim)
        self.post_attn_norm = RMSNorm(model_dim)
        self.pre_ffw_norm = RMSNorm(model_dim)
        self.post_ffw_norm = RMSNorm(model_dim)
        self.attention = ZeroAttentionAdapter(spec, model_dim)
        self.mlp = GatedMLP(model_dim=model_dim, hidden_dim=mlp_hidden_dim)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        mask: Tensor | None = None,
        base_kv_cache: dict[str, Tensor] | None = None,
        input_pos: Tensor | None = None,
        param_tensor: Tensor | None = None,
    ) -> Tensor:
        attn_input = self.pre_attn_norm(hidden_states)
        hidden_states = hidden_states + self.attention(
            attn_input,
            mask=mask,
            base_kv_cache=base_kv_cache,
            input_pos=input_pos,
            param_tensor=param_tensor,
        )
        hidden_states = self.post_attn_norm(hidden_states)

        mlp_input = self.pre_ffw_norm(hidden_states)
        hidden_states = hidden_states + self.mlp(mlp_input)
        hidden_states = self.post_ffw_norm(hidden_states)
        return hidden_states


class GemmaMtpDrafter(nn.Module):
    def __init__(self, config: MtpDrafterConfig) -> None:
        super().__init__()
        self.config = config
        self.pre_project = nn.Linear(
            config.input_activation_dim, config.model_dim, bias=False
        )
        self.blocks = nn.ModuleList(
            MtpDrafterBlock(spec, config.model_dim, config.mlp_hidden_dim)
            for spec in config.attention_specs
        )
        self.final_norm = RMSNorm(config.model_dim)
        self.logits_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.post_project = nn.Linear(
            config.model_dim, config.projected_activation_dim, bias=False
        )

    def forward(
        self,
        activations: Tensor,
        *,
        mask: Tensor | None = None,
        base_kv_cache: dict[str, Tensor] | None = None,
        input_pos: Tensor | None = None,
        param_tensor: Tensor | None = None,
    ) -> DrafterOutput:
        hidden_states = self.pre_project(activations)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                mask=mask,
                base_kv_cache=base_kv_cache,
                input_pos=input_pos,
                param_tensor=param_tensor,
            )

        hidden_states = self.final_norm(hidden_states)
        logits = self.logits_head(hidden_states)
        projected_activations = self.post_project(hidden_states)
        return DrafterOutput(
            logits=logits,
            projected_activations=projected_activations,
            hidden_states=hidden_states,
        )
