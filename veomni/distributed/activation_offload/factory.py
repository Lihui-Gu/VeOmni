# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Factory for building the appropriate activation offload runtime."""

from typing import Any, Optional

import torch.nn as nn

from ..module_selection import ResolvedModuleSelection
from .runtime import (
    BaseActivationOffloadRuntime,
    NullActivationOffloadRuntime,
    SelectiveAsyncActivationOffloadRuntime,
    ThresholdActivationOffloadRuntime,
)


def build_activation_offload_runtime(
    model: Optional[nn.Module],
    offload_config: Any,
    enable_gradient_checkpointing: bool = False,
    enable_selective_gradient_checkpointing: bool = False,
    enable_compile: bool = False,
    resolved_selection: Optional[tuple[ResolvedModuleSelection, ...]] = None,
) -> BaseActivationOffloadRuntime:
    """Build an activation offload runtime matching the training configuration.

    Args:
        model: The parallelized model. Required when ``offload_config.selection``
            is configured.
        offload_config: Parsed ``train.accelerator.offload_config``.
        enable_gradient_checkpointing: Whether any gradient checkpointing is enabled.
        enable_selective_gradient_checkpointing: Whether a selective checkpoint
            plan was resolved and installed. Selective offload may coexist with
            this mode.
        enable_compile: Whether ``torch.compile`` is enabled. Incompatible with
            selective offload.

    Returns:
        A runtime exposing ``forward_context``, ``backward_context``,
        ``log_summary`` and ``close``.
    """
    if not offload_config.enable_activation:
        return NullActivationOffloadRuntime()

    if offload_config.selection is None:
        return ThresholdActivationOffloadRuntime(
            activation_gpu_limit=offload_config.activation_gpu_limit,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
        )

    # Selection is configured. Validation (GC replacement / compile rejection)
    # has already been performed by VeOmniArguments, but we keep defensive checks
    # here because the factory may be called independently of args parsing.
    if enable_compile:
        raise ValueError(
            "Selective activation offload is not supported with torch.compile. "
            "Disable torch.compile or remove offload_config.selection."
        )

    if model is None:
        raise ValueError("model is required when activation offload module selection is configured.")

    checkpoint_plan_owner = model
    if not hasattr(checkpoint_plan_owner, "_veomni_selective_checkpoint_wrappers"):
        wrapped_model = getattr(model, "module", None)
        if isinstance(wrapped_model, nn.Module):
            checkpoint_plan_owner = wrapped_model
    checkpoint_plan_installed = hasattr(checkpoint_plan_owner, "_veomni_selective_checkpoint_wrappers")

    if enable_gradient_checkpointing and (
        not enable_selective_gradient_checkpointing or not checkpoint_plan_installed
    ):
        raise ValueError(
            "Selective activation offload with gradient checkpointing requires an installed checkpoint-replacement "
            "plan. Resolve the activation memory plan and install its checkpoint wrappers before building the runtime."
        )

    return SelectiveAsyncActivationOffloadRuntime(
        model=model,
        offload_config=offload_config,
        resolved_selection=resolved_selection,
        use_checkpoint_recompute_prefetch=(enable_gradient_checkpointing and enable_selective_gradient_checkpointing),
    )
