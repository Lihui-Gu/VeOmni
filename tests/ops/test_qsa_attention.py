# Copyright 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from veomni.ops.kernels.qsa_attention import qsa_attention_eager


def _dense_reference(query, key, value, indices, scaling):
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    logits = torch.matmul(query, key.transpose(-1, -2)) * scaling
    allowed = torch.zeros(indices.shape[0], indices.shape[1], key.shape[-2] + 1, dtype=torch.bool, device=query.device)
    safe = torch.where(indices >= 0, indices, key.shape[-2])
    allowed.scatter_(-1, safe.long(), True)
    logits = logits.masked_fill(~allowed[..., : key.shape[-2]].unsqueeze(1), torch.finfo(query.dtype).min)
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(probabilities, value).transpose(1, 2).contiguous()


def _compact_reference(query, key, value, indices, scaling):
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    valid = indices >= 0
    safe_indices = indices.clamp(min=0).long()
    batch_index = torch.arange(query.shape[0])[:, None, None, None]
    head_index = torch.arange(query.shape[1])[None, :, None, None]
    gather_index = safe_indices[:, None]
    selected_key = key[batch_index, head_index, gather_index]
    selected_value = value[batch_index, head_index, gather_index]
    logits = torch.einsum("bhqd,bhqkd->bhqk", query, selected_key) * scaling
    logits = logits.masked_fill(~valid[:, None], torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = probabilities.masked_fill(~valid[:, None], 0)
    denominator = probabilities.sum(dim=-1, keepdim=True)
    probabilities = probabilities / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return torch.einsum("bhqk,bhqkd->bhqd", probabilities, selected_value).transpose(1, 2).contiguous()


def test_qsa_compact_pytorch_matches_dense_gqa_forward_backward():
    torch.manual_seed(17)
    query = torch.randn(2, 7, 4, 8, dtype=torch.float64).transpose(1, 2).detach().requires_grad_(True)
    key = torch.randn(2, 7, 2, 8, dtype=torch.float64).transpose(1, 2).detach().requires_grad_(True)
    value = torch.randn(2, 7, 2, 8, dtype=torch.float64).transpose(1, 2).detach().requires_grad_(True)
    indices = torch.tensor(
        [
            [[0, -1, -1], [0, 1, -1], [2, 0, 1], [3, 1, 0], [4, 2, 0], [5, 3, 1], [6, 4, 2]],
            [[0, -1, -1], [1, 0, -1], [2, 1, 0], [3, 2, 1], [4, 3, 2], [5, 4, 3], [6, 5, 4]],
        ],
        dtype=torch.int32,
    )
    scaling = query.shape[-1] ** -0.5

    expected = _dense_reference(query, key, value, indices, scaling)
    expected_grads = torch.autograd.grad(expected.square().sum(), (query, key, value))
    actual = qsa_attention_eager(query, key, value, indices, scaling, max_gathered_elements=96)
    actual_grads = torch.autograd.grad(actual.square().sum(), (query, key, value))

    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


def test_qsa_compact_pytorch_rejects_incompatible_gqa_heads():
    query = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 2, 2, 4)
    indices = torch.zeros(1, 2, 1, dtype=torch.int32)

    try:
        qsa_attention_eager(query, key, key, indices, 0.5)
    except ValueError as error:
        assert "must be divisible" in str(error)
    else:
        raise AssertionError("Expected incompatible Q/KV head counts to be rejected")


def test_qsa_compact_pytorch_rejects_invalid_indices():
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 1, 3, 4)

    for indices in (
        torch.tensor([[[-2], [0], [1]]], dtype=torch.int32),
        torch.tensor([[[3], [0], [1]]], dtype=torch.int64),
        torch.zeros(1, 3, 1, dtype=torch.float32),
    ):
        try:
            qsa_attention_eager(query, key, key, indices, 0.5)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("Expected invalid compact QSA indices to be rejected")


def test_qsa_compact_pytorch_returns_zero_for_an_empty_selection():
    query = torch.randn(1, 2, 3, 4, requires_grad=True)
    key = torch.randn(1, 1, 3, 4, requires_grad=True)
    indices = torch.full((1, 3, 2), -1, dtype=torch.int32)

    output = qsa_attention_eager(query, key, key, indices, 0.5)

    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    torch.testing.assert_close(query.grad, torch.zeros_like(query))
    torch.testing.assert_close(key.grad, torch.zeros_like(key))


def test_qsa_compact_pytorch_accumulates_duplicate_index_gradients():
    torch.manual_seed(19)
    query = torch.randn(1, 4, 5, 8, dtype=torch.float64, requires_grad=True)
    key = torch.randn(1, 2, 5, 8, dtype=torch.float64, requires_grad=True)
    value = torch.randn(1, 2, 5, 8, dtype=torch.float64, requires_grad=True)
    indices = torch.tensor([[[0, 0, -1], [1, 0, 0], [2, 2, 1], [3, 1, 1], [4, 4, 2]]], dtype=torch.int32)
    scaling = query.shape[-1] ** -0.5

    expected = _compact_reference(query, key, value, indices, scaling)
    expected_grads = torch.autograd.grad(expected.square().sum(), (query, key, value))
    actual = qsa_attention_eager(query, key, value, indices, scaling, max_gathered_elements=96)
    actual_grads = torch.autograd.grad(actual.square().sum(), (query, key, value))

    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


def test_qsa_compact_pytorch_does_not_save_gathered_kv_for_backward():
    torch.manual_seed(23)
    batch_size, query_heads, kv_heads = 1, 8, 2
    seq_len, selected_width, head_dim = 128, 64, 16
    query = torch.randn(batch_size, query_heads, seq_len, head_dim, requires_grad=True)
    key = torch.randn(batch_size, kv_heads, seq_len, head_dim, requires_grad=True)
    value = torch.randn(batch_size, kv_heads, seq_len, head_dim, requires_grad=True)
    indices = torch.randint(0, seq_len, (batch_size, seq_len, selected_width), dtype=torch.int32)
    saved_tensors = []

    def pack(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = qsa_attention_eager(
            query,
            key,
            value,
            indices,
            head_dim**-0.5,
            max_gathered_elements=2 * query_heads * selected_width * head_dim,
        )

    assert [tensor.shape for tensor in saved_tensors] == [query.shape, key.shape, value.shape, indices.shape]
    selected_kv_bytes = 2 * batch_size * query_heads * seq_len * selected_width * head_dim * query.element_size()
    saved_bytes = sum(tensor.numel() * tensor.element_size() for tensor in saved_tensors)
    assert saved_bytes * 10 < selected_kv_bytes
    output.square().mean().backward()


def test_qsa_compact_pytorch_backward_under_non_reentrant_checkpoint():
    torch.manual_seed(29)
    query = torch.randn(1, 4, 16, 8, dtype=torch.float64, requires_grad=True)
    key = torch.randn(1, 2, 16, 8, dtype=torch.float64, requires_grad=True)
    value = torch.randn(1, 2, 16, 8, dtype=torch.float64, requires_grad=True)
    indices = torch.rand(1, 16, 16).argsort(dim=-1)[..., :7].to(torch.int32)
    scaling = query.shape[-1] ** -0.5

    expected = _dense_reference(query, key, value, indices, scaling)
    expected_grads = torch.autograd.grad(expected.square().sum(), (query, key, value))
    checkpoint_device = DefaultDeviceType.get_device_type()
    try:
        DefaultDeviceType.set_device_type("cpu")
        actual = checkpoint(
            lambda q, k, v: qsa_attention_eager(q, k, v, indices, scaling, max_gathered_elements=448),
            query,
            key,
            value,
            use_reentrant=False,
        )
        actual_grads = torch.autograd.grad(actual.square().sum(), (query, key, value))
    finally:
        DefaultDeviceType.set_device_type(checkpoint_device)

    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


def test_qsa_compact_pytorch_backward_preserves_autocast_state():
    torch.manual_seed(31)
    query = torch.randn(1, 4, 8, 8, requires_grad=True)
    key = torch.randn(1, 2, 8, 8, requires_grad=True)
    value = torch.randn(1, 2, 8, 8, requires_grad=True)
    indices = torch.rand(1, 8, 8).argsort(dim=-1)[..., :5].to(torch.int32)
    scaling = query.shape[-1] ** -0.5

    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = _compact_reference(query, key, value, indices, scaling)
        actual = qsa_attention_eager(query, key, value, indices, scaling, max_gathered_elements=160)
    expected_grads = torch.autograd.grad(expected.float().square().sum(), (query, key, value))
    actual_grads = torch.autograd.grad(actual.float().square().sum(), (query, key, value))

    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-2, atol=1e-3)
