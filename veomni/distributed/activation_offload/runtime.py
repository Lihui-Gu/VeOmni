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

"""Selective asynchronous activation offload runtime."""

from contextlib import nullcontext
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks

from ..offloading import (
    OffloadPolicy,
    PackedThresholdActivation,
    _ActivationOffloadThresholdPolicy,
    build_activation_offloading_context,
)
from .config import ResolvedModuleSelection, resolve_module_class_selection
from .stats import ActivationOffloadStats


class BaseActivationOffloadRuntime:
    """Unified interface for activation offload runtimes."""

    @property
    def forward_context(self) -> Union["saved_tensors_hooks", nullcontext]:
        raise NotImplementedError

    @property
    def backward_context(self) -> Union["saved_tensors_hooks", nullcontext]:
        raise NotImplementedError

    def log_summary(self) -> None:
        """Log runtime counters at the end of training."""
        pass

    def close(self) -> None:
        """Release resources and remove hooks installed on the model."""
        pass


class NullActivationOffloadRuntime(BaseActivationOffloadRuntime):
    """No-op runtime used when activation offload is disabled."""

    @property
    def forward_context(self) -> nullcontext:
        return nullcontext()

    @property
    def backward_context(self) -> nullcontext:
        return nullcontext()


class ThresholdActivationOffloadRuntime(BaseActivationOffloadRuntime):
    """Legacy threshold-policy runtime without module selection."""

    def __init__(
        self,
        activation_gpu_limit: float,
        enable_gradient_checkpointing: bool,
    ) -> None:
        self.fwd_context, self.bwd_context = build_activation_offloading_context(
            enable_activation=True,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            activation_gpu_limit=activation_gpu_limit,
        )

    @property
    def forward_context(self) -> Union["saved_tensors_hooks", nullcontext]:
        return self.fwd_context

    @property
    def backward_context(self) -> Union["saved_tensors_hooks", nullcontext]:
        return self.bwd_context


class _PackedSelectedTensor:
    """Placeholder object returned by the selective pack hook.

    Carries enough metadata for the unpack hook to restore the tensor on the
    original device. In the synchronous first stage this simply stores the CPU
    copy; later it will hold an async offload handle.
    """

    __slots__ = ("device", "cpu_tensor", "call_id")

    def __init__(self, device: torch.device, cpu_tensor: torch.Tensor, call_id: int) -> None:
        self.device = device
        self.cpu_tensor = cpu_tensor
        self.call_id = call_id


class SelectiveAsyncActivationOffloadRuntime(BaseActivationOffloadRuntime):
    """Module-class-selective activation offload runtime.

    First-stage implementation: synchronous D2H/H2D via ``tensor.cpu()``.
    This gives correct numerical behaviour and lets us validate module
    selection, prefetch scheduling and threshold fallback before introducing
    dedicated streams and events.
    """

    def __init__(
        self,
        model: nn.Module,
        offload_config: Any,
    ) -> None:
        self.config = offload_config
        self.prefetch = bool(offload_config.prefetch)
        self.threshold_policy = _ActivationOffloadThresholdPolicy(
            gpu_limit_in_gb=offload_config.activation_gpu_limit,
        )
        self.stats = ActivationOffloadStats()

        self._call_counter = 0
        self._call_stack: List[int] = []
        self._forward_order: List[int] = []
        self._module_hooks: List[Tuple[nn.Module, Any, Any]] = []

        selections = resolve_module_class_selection(
            model,
            offload_config.selection.module_classes,
        )
        self._install_module_hooks(selections)
        self.stats.num_matched_module_calls = len(selections)

    # ------------------------------------------------------------------
    # Module hooks
    # ------------------------------------------------------------------
    def _install_module_hooks(self, selections: Tuple[ResolvedModuleSelection, ...]) -> None:
        """Attach forward pre/post hooks to every selected module instance."""
        for selection in selections:
            module = selection.module
            pre_handle = module.register_forward_pre_hook(self._make_forward_pre_hook())
            post_handle = module.register_forward_hook(self._make_forward_post_hook())
            self._module_hooks.append((module, pre_handle, post_handle))

    def _make_forward_pre_hook(self):
        def hook(module, inputs):
            self._call_counter += 1
            call_id = self._call_counter
            self._call_stack.append(call_id)
            self._forward_order.append(call_id)

        return hook

    def _make_forward_post_hook(self):
        def hook(module, inputs, outputs):
            if not self._call_stack:
                return
            call_id = self._call_stack.pop()
            if not self.prefetch:
                return
            output_tensor = self._get_grad_requiring_output(outputs)
            if output_tensor is not None:
                output_tensor.register_hook(self._make_output_grad_hook(call_id))

        return hook

    @staticmethod
    def _get_grad_requiring_output(outputs: Any) -> Optional[torch.Tensor]:
        if isinstance(outputs, torch.Tensor) and outputs.requires_grad:
            return outputs
        if isinstance(outputs, (tuple, list)) and outputs:
            for item in outputs:
                if isinstance(item, torch.Tensor) and item.requires_grad:
                    return item
        return None

    def _make_output_grad_hook(self, call_id: int):
        def hook(grad):
            try:
                idx = self._forward_order.index(call_id)
            except ValueError:
                return
            if idx > 0:
                prev_call_id = self._forward_order[idx - 1]
                self._prefetch_call(prev_call_id)

        return hook

    # ------------------------------------------------------------------
    # Saved-tensor hooks
    # ------------------------------------------------------------------
    def pack_hook(self, tensor: torch.Tensor) -> Union[_PackedSelectedTensor, PackedThresholdActivation]:
        """Pack a saved tensor for activation offload.

        Selected modules are always offloaded; non-selected tensors fall back
        to the legacy threshold policy.
        """
        call_id = self._current_module_call_id()
        if call_id is not None:
            return self._offload_selected(tensor, call_id)

        policy = self.threshold_policy.decide(tensor)
        if policy == OffloadPolicy.OFFLOAD:
            self.stats.num_threshold_fallback_offloads += 1
            return (policy, tensor.device, tensor.cpu())
        if policy == OffloadPolicy.KEEP_ON_GPU:
            self.stats.num_threshold_keep_on_gpu += 1
            return (policy, tensor.device, tensor)
        self.stats.num_ignored_tensors += 1
        return (policy, tensor.device, tensor)

    def unpack_hook(self, packed: Union[_PackedSelectedTensor, PackedThresholdActivation]) -> torch.Tensor:
        """Restore a tensor that was packed by :meth:`pack_hook`."""
        if isinstance(packed, _PackedSelectedTensor):
            return self._restore_selected(packed)

        policy, device, tensor = packed
        self.threshold_policy.release(policy, tensor)
        if policy in (OffloadPolicy.IGNORE, OffloadPolicy.KEEP_ON_GPU):
            return tensor
        return tensor.to(device, non_blocking=False)

    # ------------------------------------------------------------------
    # Core offload / restore (synchronous first stage)
    # ------------------------------------------------------------------
    def _current_module_call_id(self) -> Optional[int]:
        if self._call_stack:
            return self._call_stack[-1]
        return None

    def _offload_selected(self, tensor: torch.Tensor, call_id: int) -> _PackedSelectedTensor:
        self.stats.num_offloaded_tensors += 1
        self.stats.offloaded_bytes += tensor.numel() * tensor.element_size()
        return _PackedSelectedTensor(
            device=tensor.device,
            cpu_tensor=tensor.cpu(),
            call_id=call_id,
        )

    def _restore_selected(self, packed: _PackedSelectedTensor) -> torch.Tensor:
        self.stats.num_ondemand_restores += 1
        self.stats.restored_bytes += packed.cpu_tensor.numel() * packed.cpu_tensor.element_size()
        return packed.cpu_tensor.to(packed.device, non_blocking=False)

    def _prefetch_call(self, call_id: int) -> None:
        """Prefetch the activations belonging to ``call_id``.

        In the synchronous first stage this is a no-op because the tensor is
        already on the CPU after forward. It becomes meaningful once async
        D2H/H2D is introduced.
        """
        self.stats.num_prefetch_hits += 1

    # ------------------------------------------------------------------
    # Runtime interface
    # ------------------------------------------------------------------
    @property
    def forward_context(self) -> Any:
        from .context import _SelectiveForwardContext

        return _SelectiveForwardContext(self)

    @property
    def backward_context(self) -> Any:
        from .context import _SelectiveBackwardContext

        return _SelectiveBackwardContext(self)

    def log_summary(self) -> None:
        self.stats.log()

    def close(self) -> None:
        for _module, pre_handle, post_handle in self._module_hooks:
            pre_handle.remove()
            post_handle.remove()
        self._module_hooks.clear()
        self._call_stack.clear()
        self._forward_order.clear()
