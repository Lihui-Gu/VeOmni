# Copyright 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Memory-bounded PyTorch reference for compact Qwen sparse attention."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable


def _get_query_chunk_size(
    query: torch.Tensor,
    value: torch.Tensor,
    selected_width: int,
    max_gathered_elements: int,
) -> int:
    batch_size, query_heads, query_len, head_dim = query.shape
    elements_per_query = max(
        1,
        batch_size * query_heads * selected_width * (head_dim + value.shape[-1]),
    )
    return max(1, min(query_len, max_gathered_elements // elements_per_query))


def _qsa_attention_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_indices: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    selected_key, selected_value, valid, _ = _gather_qsa_rows(query, key, value, selected_indices)
    return _qsa_attention_from_selected(query, selected_key, selected_value, valid, scaling)


def _gather_qsa_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, query_heads = query.shape[:2]
    kv_heads, kv_len = key.shape[1:3]
    kv_head_for_query = torch.div(
        torch.arange(query_heads, device=query.device), query_heads // kv_heads, rounding_mode="floor"
    )
    batch_index = torch.arange(batch_size, device=query.device)[:, None, None, None]
    head_index = kv_head_for_query[None, :, None, None]
    valid = (selected_indices >= 0) & (selected_indices < kv_len)
    safe_indices = selected_indices.clamp(min=0, max=kv_len - 1).long()
    gather_index = safe_indices[:, None]
    return (
        key[batch_index, head_index, gather_index],
        value[batch_index, head_index, gather_index],
        valid,
        safe_indices,
    )


def _qsa_attention_from_selected(
    query: torch.Tensor,
    selected_key: torch.Tensor,
    selected_value: torch.Tensor,
    valid: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    logits = torch.einsum("bhqd,bhqkd->bhqk", query, selected_key) * scaling
    logits = logits.masked_fill(~valid[:, None], torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = probabilities.masked_fill(~valid[:, None], 0)
    denominator = probabilities.sum(dim=-1, keepdim=True)
    probabilities = probabilities / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return torch.einsum("bhqk,bhqkd->bhqd", probabilities, selected_value)


def _scatter_selected_grad(
    output: torch.Tensor,
    selected_grad: torch.Tensor,
    selected_indices: torch.Tensor,
) -> None:
    batch_size, query_heads, query_len, selected_width = selected_grad.shape[:4]
    kv_heads, kv_len, head_dim = output.shape[1:]
    kv_head_for_query = torch.div(
        torch.arange(query_heads, device=output.device),
        query_heads // kv_heads,
        rounding_mode="floor",
    )
    valid = (selected_indices >= 0) & (selected_indices < kv_len)
    valid = valid[:, None].expand(batch_size, query_heads, query_len, selected_width)
    safe_indices = selected_indices.clamp(min=0, max=kv_len - 1).long()
    safe_indices = safe_indices[:, None].expand(batch_size, query_heads, query_len, selected_width)
    batch_index = torch.arange(batch_size, device=output.device)[:, None, None, None].expand_as(safe_indices)
    head_index = kv_head_for_query[None, :, None, None].expand_as(safe_indices)
    flat_index = (batch_index * kv_heads + head_index) * kv_len + safe_indices
    output.view(-1, head_dim).index_add_(0, flat_index[valid], selected_grad[valid])


class _QSAAttention(torch.autograd.Function):
    """Recompute gathered K/V rows per chunk instead of saving them for backward."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        selected_indices: torch.Tensor,
        scaling: float,
        max_gathered_elements: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(query, key, value, selected_indices)
        ctx.scaling = scaling
        ctx.max_gathered_elements = max_gathered_elements
        ctx.device_type = query.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(ctx.device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device_type)

        selected_width = selected_indices.shape[-1]
        if selected_width == 0:
            return query.new_zeros(query.shape[0], query.shape[2], query.shape[1], value.shape[-1])

        chunk_size = _get_query_chunk_size(query, value, selected_width, max_gathered_elements)
        outputs = [
            _qsa_attention_chunk(
                query[:, :, start : start + chunk_size],
                key,
                value,
                selected_indices[:, start : start + chunk_size],
                scaling,
            )
            for start in range(0, query.shape[2], chunk_size)
        ]
        return torch.cat(outputs, dim=2).transpose(1, 2).contiguous()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        query, key, value, selected_indices = ctx.saved_tensors
        needs_query, needs_key, needs_value = ctx.needs_input_grad[:3]
        query_grad = torch.zeros_like(query) if needs_query else None
        key_grad = torch.zeros_like(key, memory_format=torch.contiguous_format) if needs_key else None
        value_grad = torch.zeros_like(value, memory_format=torch.contiguous_format) if needs_value else None
        selected_width = selected_indices.shape[-1]
        if selected_width == 0:
            return query_grad, key_grad, value_grad, None, None, None

        chunk_size = _get_query_chunk_size(query, value, selected_width, ctx.max_gathered_elements)
        grad_output = grad_output.transpose(1, 2)
        for start in range(0, query.shape[2], chunk_size):
            end = min(start + chunk_size, query.shape[2])
            with (
                torch.enable_grad(),
                torch.autocast(
                    device_type=ctx.device_type,
                    enabled=ctx.autocast_enabled,
                    dtype=ctx.autocast_dtype,
                ),
            ):
                chunk_query = query[:, :, start:end].detach().requires_grad_(needs_query)
                selected_key, selected_value, valid, _ = _gather_qsa_rows(
                    chunk_query,
                    key.detach(),
                    value.detach(),
                    selected_indices[:, start:end],
                )
                selected_key = selected_key.detach().requires_grad_(needs_key)
                selected_value = selected_value.detach().requires_grad_(needs_value)
                chunk_output = _qsa_attention_from_selected(
                    chunk_query,
                    selected_key,
                    selected_value,
                    valid,
                    ctx.scaling,
                )
                grad_inputs = tuple(
                    tensor
                    for tensor, needed in (
                        (chunk_query, needs_query),
                        (selected_key, needs_key),
                        (selected_value, needs_value),
                    )
                    if needed
                )
                chunk_grads = torch.autograd.grad(
                    chunk_output,
                    grad_inputs,
                    grad_output[:, :, start:end],
                )

            grad_index = 0
            if needs_query:
                query_grad[:, :, start:end] = chunk_grads[grad_index]
                grad_index += 1
            if needs_key:
                _scatter_selected_grad(key_grad, chunk_grads[grad_index], selected_indices[:, start:end])
                grad_index += 1
            if needs_value:
                _scatter_selected_grad(value_grad, chunk_grads[grad_index], selected_indices[:, start:end])

        return query_grad, key_grad, value_grad, None, None, None


def qsa_attention_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_indices: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    training: bool = False,
    *,
    max_gathered_elements: int = 16 * 1024 * 1024,
) -> torch.Tensor:
    """Attend only to ``selected_indices`` without materializing an ``S x S`` mask.

    Q/K/V use ``[B, H, S, D]`` and indices use ``[B, S, K]``. Invalid slots
    are ``-1``. Query heads may be a multiple of KV heads (GQA/MQA).

    This implementation deliberately chunks query rows and recomputes each
    chunk during backward. Advanced indexing therefore materializes selected
    K/V rows only for the active chunk instead of retaining every chunk in the
    autograd graph. It is the portable correctness backend and the contract for
    future fused CUDA/NPU kernels.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or selected_indices.ndim != 3:
        raise ValueError("QSA expects query/key/value [B,H,S,D] and selected_indices [B,S,K].")
    if dropout != 0:
        raise ValueError("Compact QSA currently requires dropout=0.")
    batch_size, query_heads, query_len, head_dim = query.shape
    if key.shape[:1] != (batch_size,) or value.shape[:1] != (batch_size,):
        raise ValueError("QSA query, key, and value batch sizes must match.")
    if key.shape != value.shape:
        raise ValueError(f"QSA key/value shapes must match; got key={key.shape}, value={value.shape}.")
    if selected_indices.shape[:2] != (batch_size, query_len):
        raise ValueError(
            "QSA selected indices must match the query batch and sequence dimensions; "
            f"got indices={selected_indices.shape}, query={query.shape}."
        )
    if selected_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"QSA selected indices must be int32 or int64; got {selected_indices.dtype}.")
    kv_heads, kv_len = key.shape[1:3]
    if kv_len == 0:
        raise ValueError("QSA requires at least one KV token.")
    if query_heads % kv_heads != 0:
        raise ValueError(f"QSA query heads ({query_heads}) must be divisible by KV heads ({kv_heads}).")
    if key.shape[-1] != head_dim:
        raise ValueError(f"QSA query/key head dimensions must match; got {head_dim} and {key.shape[-1]}.")
    invalid_indices = (selected_indices < -1) | (selected_indices >= kv_len)
    if bool(invalid_indices.any()):
        raise ValueError("QSA selected indices must be -1 or valid global KV token indices.")

    del training
    return _QSAAttention.apply(query, key, value, selected_indices, scaling, max_gathered_elements)


__all__ = ["qsa_attention_eager"]
