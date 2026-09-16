# Copyright 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import eager_attention_forward as hf_eager_attention_forward

from veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu import (
    eager_attention_forward,
)


def test_qwen4_exp_eager_attention_preserves_standard_fallback():
    torch.manual_seed(0)
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    query = torch.randn(2, 4, 3, 5)
    key = torch.randn(2, 2, 3, 5)
    value = torch.randn(2, 2, 3, 5)
    attention_mask = torch.triu(torch.full((2, 1, 3, 3), float("-inf")), diagonal=1)

    expected = hf_eager_attention_forward(module, query, key, value, attention_mask, scaling=5**-0.5)
    actual = eager_attention_forward(module, query, key, value, attention_mask, scaling=5**-0.5)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_qwen4_exp_eager_attention_accepts_compact_selection():
    torch.manual_seed(0)
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    query = torch.randn(1, 4, 3, 5)
    key = torch.randn(1, 2, 3, 5)
    value = torch.randn(1, 2, 3, 5)
    selected_indices = torch.tensor([[[0, -1], [0, 1], [1, 2]]], dtype=torch.int32)
    attention_mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    attention_mask[:, :, 1, 0] = False

    output, attention_weights = eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=5**-0.5,
        selected_indices=selected_indices,
    )

    repeated_key = key.repeat_interleave(2, dim=1)
    repeated_value = value.repeat_interleave(2, dim=1)
    allowed = torch.zeros(1, 3, 4, dtype=torch.bool)
    allowed.scatter_(-1, torch.where(selected_indices >= 0, selected_indices, 3).long(), True)
    allowed = allowed[..., :3][:, None] & attention_mask
    scores = torch.matmul(query, repeated_key.transpose(2, 3)) * 5**-0.5
    probabilities = torch.softmax(scores.masked_fill(~allowed, torch.finfo(scores.dtype).min), dim=-1)
    expected = torch.matmul(probabilities.masked_fill(~allowed, 0), repeated_value).transpose(1, 2)

    torch.testing.assert_close(output, expected)
    assert attention_weights is None
