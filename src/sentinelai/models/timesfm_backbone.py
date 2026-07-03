"""TimesFM 2.0 PatchedTimeSeriesDecoder (v1 PyTorch, Apache-2.0, Google Research)."""

from __future__ import annotations

import dataclasses
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def create_quantiles() -> list[float]:
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclasses.dataclass
class TimesFMConfig:
    num_layers: int = 50
    num_heads: int = 16
    num_kv_heads: int = 16
    hidden_size: int = 1280
    intermediate_size: int = 1280
    head_dim: int = 80
    rms_norm_eps: float = 1e-6
    patch_len: int = 32
    horizon_len: int = 128
    quantiles: List[float] = dataclasses.field(default_factory=create_quantiles)
    pad_val: float = 1123581321.0
    tolerance: float = 1e-6
    use_positional_embedding: bool = False


def _masked_mean_std(
    inputs: torch.Tensor,
    padding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_sum = torch.sum(1 - padding, dim=2)

    def _get_patch_index(arr: torch.Tensor) -> torch.Tensor:
        indices = torch.argmax((arr >= 3).to(torch.int32), dim=1)
        row_sum = (arr >= 3).to(torch.int32).sum(dim=1)
        return torch.where(row_sum == 0, arr.shape[1] - 1, indices)

    patch_indices = _get_patch_index(pad_sum)
    bidxs = torch.arange(inputs.shape[0], device=inputs.device)
    arr = inputs[bidxs, patch_indices, :]
    pad = padding[bidxs, patch_indices, :]
    mask = 1 - pad
    num_valid_elements = torch.clamp(torch.sum(mask, dim=1), min=1.0)
    masked_sum = torch.sum(arr * mask, dim=1)
    masked_mean = masked_sum / num_valid_elements
    masked_centered_arr = (arr - masked_mean.unsqueeze(-1)) * mask
    masked_var = torch.sum(masked_centered_arr**2, dim=1) / num_valid_elements
    masked_std = torch.sqrt(torch.clamp(masked_var, min=0.0))
    return masked_mean, masked_std


def _shift_padded_seq(mask: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    batch_size, num_seq, feature_dim = seq.shape
    new_mask = mask == 0
    indices = new_mask.to(torch.int32).argmax(dim=1)
    indices[~new_mask.any(dim=1)] = -1
    idx_range = (
        torch.arange(num_seq, device=seq.device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch_size, -1, feature_dim)
    )
    shifted_idx = (idx_range - indices[:, None, None]) % num_seq
    return seq.gather(1, shifted_idx)


def get_large_negative_number(dtype: torch.dtype) -> torch.Tensor:
    if dtype.is_floating_point:
        dtype_max = torch.finfo(dtype).max
    else:
        dtype_max = torch.iinfo(dtype).max
    return torch.tensor(-0.7 * dtype_max, dtype=dtype)


def convert_paddings_to_mask(
    paddings: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    attention_mask = paddings.detach().clone()
    attention_mask = attention_mask[:, None, None, :]
    attention_mask *= get_large_negative_number(dtype)
    return attention_mask


def causal_mask(input_t: torch.Tensor) -> torch.Tensor:
    large_negative_number = get_large_negative_number(input_t.dtype)
    t = input_t.shape[1]
    col_idx = torch.arange(t, device=input_t.device).unsqueeze(0).repeat(t, 1)
    row_idx = torch.arange(t, device=input_t.device).unsqueeze(1).repeat(1, t)
    mask = (row_idx < col_idx).to(input_t.dtype) * large_negative_number
    return mask.unsqueeze(0).unsqueeze(0)


def merge_masks(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def expand_t(key_mask: torch.Tensor) -> torch.Tensor:
        query_mask = key_mask.transpose(-1, -2)
        return torch.minimum(query_mask, key_mask)

    if a.shape[2] != b.shape[2]:
        if a.shape[2] == 1:
            a = expand_t(a)
        else:
            b = expand_t(b)
    return torch.minimum(a, b)


class ResidualBlock(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int):
        super().__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.SiLU(),
        )
        self.output_layer = nn.Linear(hidden_dims, output_dims)
        self.residual_layer = nn.Linear(input_dims, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden_layer(x)
        output = self.output_layer(hidden)
        residual = self.residual_layer(x)
        return output + residual


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = False):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output.float()
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


class TransformerMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-6)

    def forward(self, x: torch.Tensor, paddings: torch.Tensor | None = None) -> torch.Tensor:
        gate_inp = self.layer_norm(x)
        gate = F.relu(self.gate_proj(gate_inp))
        outputs = self.down_proj(gate)
        if paddings is not None:
            outputs = outputs * (1.0 - paddings[:, :, None])
        return outputs + x


class TimesFMAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = nn.Parameter(torch.empty((head_dim,), dtype=torch.float32))
        self.qkv_proj = nn.Linear(
            hidden_size,
            (num_heads + 2 * num_kv_heads) * head_dim,
        )
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)

    def _per_dim_scaling(self, query: torch.Tensor) -> torch.Tensor:
        scale = 1.442695041 / math.sqrt(self.head_dim)
        scale = scale * F.softplus(self.scaling)
        return query * scale[None, None, None, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, input_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xq = self._per_dim_scaling(xq)
        if self.num_kv_heads != self.num_heads:
            xk = torch.repeat_interleave(xk, self.num_queries_per_kv, dim=2)
            xv = torch.repeat_interleave(xv, self.num_queries_per_kv, dim=2)
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
        return scores, self.o_proj(output)


class TimesFMDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.self_attn = TimesFMAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = TransformerMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        paddings: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        _, hidden_states = self.self_attn(hidden_states, mask)
        hidden_states = residual + hidden_states
        return self.mlp(hidden_states, paddings=paddings)


class StackedDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TimesFMDecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, paddings: torch.Tensor) -> torch.Tensor:
        padding_mask = convert_paddings_to_mask(paddings, hidden_states.dtype)
        atten_mask = causal_mask(hidden_states)
        mask = merge_masks(padding_mask, atten_mask)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, paddings)
        return hidden_states


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10_000,
    ):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(self, seq_length: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(seq_length, dtype=torch.float32, device=device).unsqueeze(0)
        num_timescales = self.embedding_dims // 2
        log_timescale_increment = math.log(float(self.max_timescale) / float(self.min_timescale)) / max(
            num_timescales - 1, 1
        )
        inv_timescales = self.min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32, device=device) * -log_timescale_increment
        )
        scaled_time = position.unsqueeze(2) * inv_timescales.unsqueeze(0).unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return F.pad(signal, (0, 0, 0, self.embedding_dims % 2))


class PatchedTimeSeriesDecoder(nn.Module):
    """Patched time-series decoder used as a frozen feature extractor."""

    def __init__(self, config: TimesFMConfig):
        super().__init__()
        self.config = config
        self.input_ff_layer = ResidualBlock(
            input_dims=2 * config.patch_len,
            output_dims=config.hidden_size,
            hidden_dims=config.intermediate_size,
        )
        self.freq_emb = nn.Embedding(num_embeddings=3, embedding_dim=config.hidden_size)
        self.horizon_ff_layer = ResidualBlock(
            input_dims=config.hidden_size,
            output_dims=config.horizon_len * (1 + len(config.quantiles)),
            hidden_dims=config.intermediate_size,
        )
        self.stacked_transformer = StackedDecoder(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            num_layers=config.num_layers,
            rms_norm_eps=config.rms_norm_eps,
        )
        if config.use_positional_embedding:
            self.position_emb = PositionalEmbedding(config.hidden_size)

    def _forward_transform(
        self,
        inputs: torch.Tensor,
        patched_pads: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mu, sigma = _masked_mean_std(inputs, patched_pads)
        sigma = torch.clamp(sigma, min=self.config.tolerance)
        outputs = (inputs - mu[:, None, None]) / sigma[:, None, None]
        outputs = torch.where(
            torch.abs(inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(self.config.pad_val, dtype=outputs.dtype, device=outputs.device),
            outputs,
        )
        return outputs, (mu, sigma)

    def _preprocess_input(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsize = input_ts.shape[0]
        patched_inputs = input_ts.view(bsize, -1, self.config.patch_len)
        patched_pads = input_padding.view(bsize, -1, self.config.patch_len)
        patched_inputs = torch.where(
            torch.abs(patched_pads - 1.0) < self.config.tolerance,
            torch.tensor(0.0, dtype=patched_inputs.dtype, device=patched_inputs.device),
            patched_inputs,
        )
        patched_pads = torch.where(
            torch.abs(patched_inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(1.0, dtype=patched_pads.dtype, device=patched_pads.device),
            patched_pads,
        )
        patched_inputs, _ = self._forward_transform(patched_inputs, patched_pads)
        patched_inputs = patched_inputs * (1.0 - patched_pads)
        concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
        model_input = self.input_ff_layer(concat_inputs)
        patched_padding = torch.min(patched_pads, dim=-1)[0]
        if self.config.use_positional_embedding:
            pos_emb = self.position_emb(model_input.shape[1], model_input.device)
            pos_emb = torch.concat([pos_emb] * model_input.shape[0], dim=0)
            pos_emb = _shift_padded_seq(patched_padding, pos_emb)
            model_input += pos_emb
        return model_input, patched_padding

    def encode(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.Tensor,
        freq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return transformer patch embeddings and patch-level padding mask."""
        model_input, patched_padding = self._preprocess_input(input_ts, input_padding)
        f_emb = self.freq_emb(freq).unsqueeze(1)
        model_input = model_input + f_emb
        model_output = self.stacked_transformer(model_input, patched_padding)
        return model_output, patched_padding
