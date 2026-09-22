"""LoRA modules for Linear, fused-QKV MultiheadAttention and Conv2d layers."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen linear layer plus (alpha / rank) * B @ A."""

    def __init__(self, original_layer: nn.Linear, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.original_layer = original_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(rank, original_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for parameter in self.original_layer.parameters():
            parameter.requires_grad_(False)

    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    @property
    def in_features(self):
        return self.original_layer.in_features

    @property
    def out_features(self):
        return self.original_layer.out_features

    def forward(self, inputs):
        base = self.original_layer(inputs)
        update = (self.dropout(inputs) @ self.lora_A.transpose(0, 1))
        update = update @ self.lora_B.transpose(0, 1)
        return base + update * self.scaling


class LoRAConv2d(nn.Module):
    """Kernel-aware convolutional LoRA.

    A has shape [rank, in_channels, kh, kw] and uses the original stride,
    padding and dilation. B is a 1x1 projection [out_channels, rank, 1, 1].
    This is exactly the flattened-kernel factorization Delta W = B @ A for
    groups=1 convolutions and preserves spatial alignment.
    """

    def __init__(self, original_layer: nn.Conv2d, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if original_layer.groups != 1:
            raise ValueError("LoRAConv2d currently supports groups=1 only")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.original_layer = original_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        kernel_h, kernel_w = original_layer.kernel_size
        self.lora_A = nn.Parameter(
            torch.empty(rank, original_layer.in_channels, kernel_h, kernel_w)
        )
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_channels, rank, 1, 1))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        for parameter in self.original_layer.parameters():
            parameter.requires_grad_(False)

    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    @property
    def in_channels(self):
        return self.original_layer.in_channels

    @property
    def out_channels(self):
        return self.original_layer.out_channels

    @property
    def kernel_size(self):
        return self.original_layer.kernel_size

    @property
    def stride(self):
        return self.original_layer.stride

    @property
    def padding(self):
        return self.original_layer.padding

    @property
    def dilation(self):
        return self.original_layer.dilation

    @property
    def groups(self):
        return self.original_layer.groups

    def _adapter_input(self, inputs):
        inputs = self.dropout(inputs)
        if self.original_layer.padding_mode != "zeros":
            inputs = F.pad(
                inputs,
                self.original_layer._reversed_padding_repeated_twice,
                mode=self.original_layer.padding_mode,
            )
            padding = 0
        else:
            padding = self.original_layer.padding
        return inputs, padding

    def forward(self, inputs):
        base = self.original_layer(inputs)
        adapter_inputs, padding = self._adapter_input(inputs)
        down = F.conv2d(
            adapter_inputs,
            self.lora_A,
            bias=None,
            stride=self.original_layer.stride,
            padding=padding,
            dilation=self.original_layer.dilation,
            groups=1,
        )
        update = F.conv2d(down, self.lora_B, bias=None, stride=1, padding=0)
        if update.shape != base.shape:
            raise RuntimeError(
                f"Conv-LoRA geometry mismatch: base={tuple(base.shape)}, "
                f"adapter={tuple(update.shape)}"
            )
        return base + update * self.scaling


class LoRAMultiheadAttention(nn.Module):
    """LoRA on the fused self-attention Q/K/V projection.

    Ultralytics RT-DETR uses nn.MultiheadAttention.in_proj_weight instead of
    child q_proj/k_proj/v_proj Linear modules. This wrapper supplies three
    independent low-rank updates to that fused projection and delegates the
    remaining attention operation to PyTorch.
    """

    def __init__(self, original_layer: nn.MultiheadAttention, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not getattr(original_layer, "_qkv_same_embed_dim", False):
            raise ValueError("Only equal-dimension fused Q/K/V attention is supported")
        if original_layer.in_proj_weight is None:
            raise ValueError("MultiheadAttention has no fused in_proj_weight")
        if dropout != 0:
            raise ValueError(
                "Q/K/V LoRA dropout is intentionally unsupported by the fused-weight "
                "wrapper; use --lora_dropout 0 for the documented experiment"
            )
        self.original_layer = original_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        embed_dim = original_layer.embed_dim
        self.lora_A = nn.Parameter(torch.empty(3, rank, embed_dim))
        self.lora_B = nn.Parameter(torch.zeros(3, embed_dim, rank))
        for index in range(3):
            nn.init.kaiming_uniform_(self.lora_A[index], a=math.sqrt(5))
        for parameter in self.original_layer.parameters():
            parameter.requires_grad_(False)

    @property
    def embed_dim(self):
        return self.original_layer.embed_dim

    @property
    def num_heads(self):
        return self.original_layer.num_heads

    @property
    def batch_first(self):
        return self.original_layer.batch_first

    def _qkv_weights(self):
        delta = torch.bmm(self.lora_B, self.lora_A) * self.scaling
        base_q, base_k, base_v = self.original_layer.in_proj_weight.chunk(3, dim=0)
        return base_q + delta[0], base_k + delta[1], base_v + delta[2]

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True,
                is_causal=False):
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = (tensor.transpose(0, 1) for tensor in (query, key, value))
        q_weight, k_weight, v_weight = self._qkv_weights()
        kwargs = dict(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.original_layer.embed_dim,
            num_heads=self.original_layer.num_heads,
            in_proj_weight=None,
            in_proj_bias=self.original_layer.in_proj_bias,
            bias_k=self.original_layer.bias_k,
            bias_v=self.original_layer.bias_v,
            add_zero_attn=self.original_layer.add_zero_attn,
            dropout_p=self.original_layer.dropout,
            out_proj_weight=self.original_layer.out_proj.weight,
            out_proj_bias=self.original_layer.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=q_weight,
            k_proj_weight=k_weight,
            v_proj_weight=v_weight,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        try:
            output, weights = F.multi_head_attention_forward(**kwargs)
        except TypeError as error:
            # PyTorch 2.0 compatibility: is_causal was not yet accepted.
            if "is_causal" not in str(error):
                raise
            kwargs.pop("is_causal")
            output, weights = F.multi_head_attention_forward(**kwargs)
        if self.batch_first and is_batched:
            output = output.transpose(0, 1)
        return output, weights


LORA_TYPES = (LoRALinear, LoRAConv2d, LoRAMultiheadAttention)


def _get_parent_and_attr(model: nn.Module, name: str):
    parent_name, separator, attr_name = name.rpartition(".")
    parent = dict(model.named_modules())[parent_name] if separator else model
    return parent, attr_name if separator else name


def inject_lora_linear(model: nn.Module, exact_names: Iterable[str], rank: int = 8,
                       alpha: float = 16.0, dropout: float = 0.0) -> list:
    exact_names = set(exact_names)
    candidates = [
        (name, module) for name, module in model.named_modules()
        if name in exact_names and isinstance(module, nn.Linear)
    ]
    replaced = []
    for name, module in candidates:
        parent, attr_name = _get_parent_and_attr(model, name)
        adapter = LoRALinear(module, rank, alpha, dropout)
        setattr(parent, attr_name, adapter)
        replaced.append((name, adapter))
    missing = exact_names - {name for name, _ in replaced}
    if missing:
        raise RuntimeError(f"Linear LoRA target(s) not found or wrong type: {sorted(missing)}")
    return replaced


def inject_lora_conv2d(model: nn.Module, exact_names: Iterable[str], rank: int = 8,
                       alpha: float = 16.0, dropout: float = 0.0,
                       min_channels: int = 64) -> list:
    exact_names = set(exact_names)
    modules = dict(model.named_modules())
    missing = exact_names - set(modules)
    if missing:
        raise RuntimeError(f"Conv-LoRA target(s) not found: {sorted(missing)}")
    candidates = []
    for name in sorted(exact_names):
        module = modules[name]
        if not isinstance(module, nn.Conv2d):
            raise RuntimeError(f"Conv-LoRA target is not Conv2d: {name}")
        if module.groups != 1:
            raise RuntimeError(f"Conv-LoRA target must use groups=1: {name}")
        if module.in_channels < min_channels or module.out_channels < min_channels:
            raise RuntimeError(
                f"Conv-LoRA target {name} is below min_channels={min_channels}: "
                f"{module.in_channels}->{module.out_channels}"
            )
        candidates.append((name, module))
    replaced = []
    for name, module in candidates:
        parent, attr_name = _get_parent_and_attr(model, name)
        adapter = LoRAConv2d(module, rank, alpha, dropout)
        setattr(parent, attr_name, adapter)
        replaced.append((name, adapter))
    return replaced


def inject_lora_mha(model: nn.Module, exact_names: Iterable[str], rank: int = 8,
                    alpha: float = 16.0, dropout: float = 0.0) -> list:
    exact_names = set(exact_names)
    candidates = [
        (name, module) for name, module in model.named_modules()
        if name in exact_names and isinstance(module, nn.MultiheadAttention)
    ]
    replaced = []
    for name, module in candidates:
        parent, attr_name = _get_parent_and_attr(model, name)
        adapter = LoRAMultiheadAttention(module, rank, alpha, dropout)
        setattr(parent, attr_name, adapter)
        replaced.append((name, adapter))
    missing = exact_names - {name for name, _ in replaced}
    if missing:
        raise RuntimeError(f"MHA LoRA target(s) not found or wrong type: {sorted(missing)}")
    return replaced


def _adapter_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, LORA_TYPES):
            yield name, module


def _extract_lora_state(model: nn.Module, include_a: bool, include_b: bool) -> dict:
    state = {}
    for name, module in _adapter_modules(model):
        if include_a:
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu().clone()
        if include_b:
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu().clone()
    return state


def get_lora_A_state_dict(model: nn.Module) -> dict:
    return _extract_lora_state(model, include_a=True, include_b=False)


def get_lora_B_state_dict(model: nn.Module) -> dict:
    return _extract_lora_state(model, include_a=False, include_b=True)


def get_lora_AB_state_dict(model: nn.Module) -> dict:
    return _extract_lora_state(model, include_a=True, include_b=True)


def _set_lora_state(model: nn.Module, state: dict, suffix: str):
    modules = dict(model.named_modules())
    expected = {f"{name}.{suffix}" for name, _ in _adapter_modules(model)}
    supplied = set(state)
    missing = expected - supplied
    unexpected = supplied - expected
    if missing or unexpected:
        raise RuntimeError(
            f"LoRA {suffix} state mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for key in sorted(supplied):
        module_name = key[: -(len(suffix) + 1)]
        target = getattr(modules[module_name], suffix)
        value = state[key]
        if tuple(target.shape) != tuple(value.shape):
            raise RuntimeError(
                f"Shape mismatch for {key}: expected {tuple(target.shape)}, got {tuple(value.shape)}"
            )
        with torch.no_grad():
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def set_lora_A_state_dict(model: nn.Module, state: dict):
    _set_lora_state(model, state, "lora_A")


def set_lora_B_state_dict(model: nn.Module, state: dict):
    _set_lora_state(model, state, "lora_B")


def set_lora_AB_state_dict(model: nn.Module, state: dict):
    expected = {
        f"{name}.{suffix}"
        for name, _ in _adapter_modules(model)
        for suffix in ("lora_A", "lora_B")
    }
    supplied = set(state)
    missing = expected - supplied
    unexpected = supplied - expected
    if missing or unexpected:
        raise RuntimeError(
            f"LoRA A/B state mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    set_lora_A_state_dict(
        model, {key: value for key, value in state.items() if key.endswith(".lora_A")}
    )
    set_lora_B_state_dict(
        model, {key: value for key, value in state.items() if key.endswith(".lora_B")}
    )


def count_lora_params(model: nn.Module) -> dict:
    a_params = sum(module.lora_A.numel() for _, module in _adapter_modules(model))
    b_params = sum(module.lora_B.numel() for _, module in _adapter_modules(model))
    return {
        "lora_A_params": int(a_params),
        "lora_B_params": int(b_params),
        "total_lora_params": int(a_params + b_params),
        "num_lora_modules": sum(1 for _ in _adapter_modules(model)),
    }
