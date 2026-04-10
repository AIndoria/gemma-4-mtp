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
        self.query_norm = RMSNorm(spec.query_head_dim)
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


class GroupedQueryAttentionAdapter(ExternalAttentionAdapter):
    def _end_index(self, param_tensor: Tensor | None, context_size: int) -> int:
        if param_tensor is None:
            return context_size
        flat = param_tensor.reshape(-1)
        if flat.numel() < 3:
            return context_size
        return max(0, min(int(flat[2].item()), context_size))

    def _prepare_value_cache(self, value_cache: Tensor, window_start: int, window_end: int) -> Tensor:
        if value_cache.dim() != 4:
            raise ValueError(f"Expected 4D value cache, got shape {tuple(value_cache.shape)}")
        # TFLite layout: [B, H, D, C]. Common PyTorch layout: [B, H, C, D].
        if value_cache.shape[-1] >= window_end and value_cache.shape[-2] == self.spec.query_head_dim:
            value_cache = value_cache[..., window_start:window_end].transpose(-1, -2)
        else:
            value_cache = value_cache[..., window_start:window_end, :]
        return value_cache

    def _prepare_key_cache(self, key_cache: Tensor, window_start: int, window_end: int) -> Tensor:
        if key_cache.dim() != 4:
            raise ValueError(f"Expected 4D key cache, got shape {tuple(key_cache.shape)}")
        return key_cache[..., window_start:window_end, :]

    def forward(
        self,
        hidden_states: Tensor,
        *,
        mask: Tensor | None = None,
        base_kv_cache: dict[str, Tensor] | None = None,
        input_pos: Tensor | None = None,
        param_tensor: Tensor | None = None,
    ) -> Tensor:
        del input_pos
        if not base_kv_cache:
            return torch.zeros_like(hidden_states)
        if self.spec.key_cache_name is None or self.spec.value_cache_name is None:
            return torch.zeros_like(hidden_states)
        if self.spec.key_cache_name not in base_kv_cache or self.spec.value_cache_name not in base_kv_cache:
            return torch.zeros_like(hidden_states)

        key_cache = base_kv_cache[self.spec.key_cache_name]
        value_cache = base_kv_cache[self.spec.value_cache_name]
        if not key_cache.is_floating_point() or not value_cache.is_floating_point():
            raise ValueError(
                "GroupedQueryAttentionAdapter currently expects dequantized float KV caches."
            )

        batch_size, steps, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        q = q.view(
            batch_size,
            steps,
            self.spec.query_heads,
            self.spec.query_head_dim,
        )
        q = self.query_norm(q)
        q = q.view(
            batch_size,
            steps,
            self.spec.kv_heads,
            self.spec.queries_per_kv,
            self.spec.query_head_dim,
        )
        q = q.permute(0, 2, 1, 3, 4).reshape(
            batch_size,
            self.spec.kv_heads,
            steps * self.spec.queries_per_kv,
            self.spec.query_head_dim,
        )

        context_size = key_cache.shape[-2]
        end_index = self._end_index(param_tensor, context_size)
        if self.spec.local_window_size is not None:
            window_start = max(0, end_index - self.spec.local_window_size)
        else:
            window_start = 0
        window_end = max(window_start, end_index)
        if window_end <= window_start:
            return torch.zeros_like(hidden_states)

        k = self._prepare_key_cache(key_cache, window_start, window_end).to(q.dtype)
        v = self._prepare_value_cache(value_cache, window_start, window_end).to(q.dtype)

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
        if mask is not None:
            mask_slice = mask[..., window_start:window_end]
            mask_slice = mask_slice.to(dtype=torch.bool)
            scores = scores.masked_fill(~mask_slice, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        context = torch.einsum("bhqk,bhkd->bhqd", probs, v)
        context = context.reshape(
            batch_size,
            self.spec.kv_heads,
            steps,
            self.spec.queries_per_kv,
            self.spec.query_head_dim,
        )
        context = context.permute(0, 2, 1, 3, 4).reshape(batch_size, steps, -1)
        return self.o_proj(context)


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
        self.attention = GroupedQueryAttentionAdapter(spec, model_dim)
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
