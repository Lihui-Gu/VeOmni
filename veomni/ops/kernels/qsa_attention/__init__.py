# Copyright 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compact Qwen sparse-attention implementation."""

from .pytorch import qsa_attention_eager


__all__ = ["qsa_attention_eager"]
