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

import weakref
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks
from torch.utils.checkpoint import noop_context_fn

from ..module_selection import ResolvedModuleSelection, resolve_module_selection
from ..offloading import (
    OffloadPolicy,
    PackedThresholdActivation,
    _ActivationOffloadThresholdPolicy,
    build_activation_offloading_context,
)
from .handle import ActivationOffloadHandle
from .stats import ActivationOffloadStats
from .utils import _StreamCache


@dataclass
class _PackedResidentActivation:
    """An activation retained under the accelerator-residency budget."""

    tensor: torch.Tensor
    budget_finalizer: Any = field(default=None, repr=False)


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


class SelectiveAsyncActivationOffloadRuntime(BaseActivationOffloadRuntime):
    """Module-selective activation offload runtime.

    Selected saved tensors participate in the accelerator-residency budget.
    Tensors over budget are copied asynchronously to CPU on a dedicated offload
    stream and restored on a dedicated prefetch stream. Non-selected tensors
    fall back to the legacy threshold execution path.
    """

    def __init__(
        self,
        model: nn.Module,
        offload_config: Any,
        resolved_selection: Optional[Tuple[ResolvedModuleSelection, ...]] = None,
        use_checkpoint_recompute_prefetch: bool = False,
    ) -> None:
        self.config = offload_config
        self.prefetch = bool(offload_config.prefetch)
        self.use_checkpoint_recompute_prefetch = use_checkpoint_recompute_prefetch
        self.exclude_parameter_views = bool(getattr(offload_config, "exclude_parameter_views", False))
        self.threshold_policy = _ActivationOffloadThresholdPolicy(
            gpu_limit_in_gb=offload_config.activation_gpu_limit,
        )
        self.stats = ActivationOffloadStats()

        self._call_counter = 0
        self._call_stack: List[int] = []
        self._forward_order: List[int] = []
        self._backward_started_call_ids: set[int] = set()
        self._handles_by_call_id: Dict[int, List[ActivationOffloadHandle]] = {}
        self._checkpoint_group_cursor = 0
        self._checkpoint_boundary_counter = 0
        self._checkpoint_prefetch_groups: List[Tuple[int, Tuple[int, ...]]] = []
        self._active_checkpoint_prefetch_call_ids: set[int] = set()
        self._active_checkpoint_prefetch_boundary: Optional[int] = None
        self._prefetch_barrier_events: List[torch.Event] = []
        self._checkpoint_wrappers: List[nn.Module] = []
        self._parameter_storage_keys_by_call_id: Dict[int, set[tuple[str, Optional[int], int]]] = {}
        self._live_handles: weakref.WeakSet[ActivationOffloadHandle] = weakref.WeakSet()
        self._module_hooks: List[Tuple[nn.Module, Any, Any]] = []
        self._stream_cache = _StreamCache()
        self._current_pinned_bytes = 0

        selections = resolved_selection
        if selections is None:
            selections = resolve_module_selection(model, offload_config.selection)
        if self.prefetch:
            self._validate_prefetch_selection(selections)
        self._install_module_hooks(selections)
        if self.prefetch and self.use_checkpoint_recompute_prefetch:
            self._install_checkpoint_contexts(model)
        self.stats.num_matched_module_calls = len(selections)

    @staticmethod
    def _validate_prefetch_selection(selections: Tuple[ResolvedModuleSelection, ...]) -> None:
        paths = [path for selection in selections for path in (selection.module_path, *selection.alias_paths)]
        for index, path in enumerate(paths):
            for other_path in paths[index + 1 :]:
                path_is_ancestor = (not path and bool(other_path)) or other_path.startswith(path + ".")
                other_is_ancestor = (not other_path and bool(path)) or path.startswith(other_path + ".")
                if path_is_ancestor or other_is_ancestor:
                    raise ValueError(
                        "Selective activation prefetch does not support nested module selections: "
                        f"{path!r} and {other_path!r}. Disable prefetch or select non-overlapping modules."
                    )

    # ------------------------------------------------------------------
    # Module hooks
    # ------------------------------------------------------------------
    def _install_module_hooks(self, selections: Tuple[ResolvedModuleSelection, ...]) -> None:
        """Attach forward pre/post hooks to every selected module instance."""
        for selection in selections:
            module = selection.module
            pre_handle = module.register_forward_pre_hook(self._make_forward_pre_hook(module))
            post_handle = module.register_forward_hook(self._make_forward_post_hook())
            self._module_hooks.append((module, pre_handle, post_handle))

    def _install_checkpoint_contexts(self, model: nn.Module) -> None:
        checkpoint_plan_owner = model
        if not hasattr(checkpoint_plan_owner, "_veomni_selective_checkpoint_wrappers"):
            wrapped_model = getattr(model, "module", None)
            if isinstance(wrapped_model, nn.Module):
                checkpoint_plan_owner = wrapped_model
        wrappers = tuple(getattr(checkpoint_plan_owner, "_veomni_selective_checkpoint_wrappers", ()))
        if not wrappers:
            self.use_checkpoint_recompute_prefetch = False
            return
        for wrapper in wrappers:
            wrapper.set_checkpoint_context_fn(self.checkpoint_contexts)
        self._checkpoint_wrappers.extend(wrappers)

    def _make_forward_pre_hook(self, selected_module: nn.Module):
        def hook(module, inputs):
            self._call_counter += 1
            call_id = self._call_counter
            self._call_stack.append(call_id)
            self._forward_order.append(call_id)
            if self.exclude_parameter_views:
                self._parameter_storage_keys_by_call_id[call_id] = {
                    storage_key
                    for parameter in selected_module.parameters()
                    if (storage_key := self._storage_key(parameter)) is not None
                }

        return hook

    def _make_forward_post_hook(self):
        def hook(module, inputs, outputs):
            if not self._call_stack:
                return
            call_id = self._call_stack.pop()
            self._parameter_storage_keys_by_call_id.pop(call_id, None)

        return hook

    # ------------------------------------------------------------------
    # Saved-tensor hooks
    # ------------------------------------------------------------------
    def pack_hook(
        self, tensor: torch.Tensor
    ) -> Union[ActivationOffloadHandle, PackedThresholdActivation, _PackedResidentActivation]:
        """Pack a saved tensor for activation offload.

        Eligible tensors share one accelerator-residency budget. Selected
        tensors over budget use asynchronous offload; non-selected tensors over
        budget use the legacy synchronous fallback.
        """
        call_id = self._current_module_call_id()
        if call_id is not None:
            if self.exclude_parameter_views and self._is_parameter_view(tensor, call_id):
                tensor_num_bytes = tensor.numel() * tensor.element_size()
                self.stats.num_parameter_views_skipped += 1
                self.stats.parameter_view_bytes_skipped += tensor_num_bytes
                return (OffloadPolicy.IGNORE, tensor.device, tensor)
            policy = self.threshold_policy.decide(tensor, min_offload_size=0)
            if policy == OffloadPolicy.OFFLOAD:
                return self._offload_selected(tensor, call_id)
            if policy == OffloadPolicy.KEEP_ON_GPU:
                self.stats.num_threshold_keep_on_gpu += 1
                return self._pack_resident(tensor, policy)
            self.stats.num_ignored_tensors += 1
            return (policy, tensor.device, tensor)

        policy = self.threshold_policy.decide(tensor)
        if policy == OffloadPolicy.OFFLOAD:
            self.stats.num_threshold_fallback_offloads += 1
            return (policy, tensor.device, tensor.cpu())
        if policy == OffloadPolicy.KEEP_ON_GPU:
            self.stats.num_threshold_keep_on_gpu += 1
            return self._pack_resident(tensor, policy)
        self.stats.num_ignored_tensors += 1
        return (policy, tensor.device, tensor)

    def unpack_hook(
        self, packed: Union[ActivationOffloadHandle, PackedThresholdActivation, _PackedResidentActivation]
    ) -> torch.Tensor:
        """Restore a tensor that was packed by :meth:`pack_hook`."""
        if isinstance(packed, ActivationOffloadHandle):
            return self._restore_selected(packed)
        if isinstance(packed, _PackedResidentActivation):
            return packed.tensor

        policy, device, tensor = packed
        self.threshold_policy.release(policy, tensor)
        if policy in (OffloadPolicy.IGNORE, OffloadPolicy.KEEP_ON_GPU):
            return tensor
        return tensor.to(device, non_blocking=False)

    # ------------------------------------------------------------------
    # Core offload / restore (async via dedicated streams)
    # ------------------------------------------------------------------
    def _current_module_call_id(self) -> Optional[int]:
        if self._call_stack:
            return self._call_stack[-1]
        return None

    @staticmethod
    def _storage_key(tensor: torch.Tensor) -> Optional[tuple[str, Optional[int], int]]:
        """Identify a tensor's underlying allocation without retaining it."""
        local_tensor = tensor.to_local() if hasattr(tensor, "to_local") else tensor
        try:
            data_ptr = local_tensor.untyped_storage().data_ptr()
        except RuntimeError:
            return None
        return (local_tensor.device.type, local_tensor.device.index, data_ptr)

    def _is_parameter_view(self, tensor: torch.Tensor, call_id: int) -> bool:
        storage_key = self._storage_key(tensor)
        return storage_key is not None and storage_key in self._parameter_storage_keys_by_call_id.get(call_id, ())

    def _pack_resident(self, tensor: torch.Tensor, policy: OffloadPolicy) -> _PackedResidentActivation:
        if policy != OffloadPolicy.KEEP_ON_GPU:
            raise ValueError(f"Cannot track a non-resident activation with policy {policy}.")
        packed = _PackedResidentActivation(tensor)
        tensor_num_bytes = tensor.numel() * tensor.element_size()
        packed.budget_finalizer = weakref.finalize(
            packed,
            self.threshold_policy.release_bytes,
            tensor_num_bytes,
        )
        return packed

    def _offload_selected(self, tensor: torch.Tensor, call_id: int) -> ActivationOffloadHandle:
        handle = ActivationOffloadHandle(
            tensor,
            call_id,
            offload_stream=self._stream_cache.get_offload_stream(tensor.device),
            prefetch_stream=self._stream_cache.get_prefetch_stream(tensor.device),
        )
        handle.offload(tensor)
        if self.prefetch:
            self._handles_by_call_id.setdefault(call_id, []).append(handle)
            self._live_handles.add(handle)
        self.stats.num_offloaded_tensors += 1
        self.stats.offloaded_bytes += tensor.numel() * tensor.element_size()
        if handle.cpu_tensor is not None:
            pinned_bytes = handle.cpu_tensor.numel() * handle.cpu_tensor.element_size()
            self._current_pinned_bytes += pinned_bytes
            self.stats.peak_pinned_bytes = max(self.stats.peak_pinned_bytes, self._current_pinned_bytes)
            weakref.finalize(handle, self._release_pinned_bytes, pinned_bytes)
        return handle

    def _release_pinned_bytes(self, released_bytes: int) -> None:
        self._current_pinned_bytes = max(0, self._current_pinned_bytes - released_bytes)

    def _restore_selected(self, handle: ActivationOffloadHandle) -> torch.Tensor:
        self.stats.num_ondemand_restores += 1
        tensor = handle.ensure_device_resident(block=True)
        if self.prefetch:
            self._begin_backward_call(handle.call_id)
        self.stats.restored_bytes += tensor.numel() * tensor.element_size()
        handle.release_restored_tensor(tensor)
        return tensor

    def start_backward(self) -> None:
        """Prepare prefetch state before autograd starts consuming activations."""
        if self.prefetch and self.use_checkpoint_recompute_prefetch:
            # Calls after the final checkpoint boundary have no recomputation
            # window in which to hide H2D. Leave them for on-demand restore
            # instead of racing the first FSDP backward all-gather.
            self._checkpoint_group_cursor = len(self._forward_order)
        elif self.prefetch and self._forward_order:
            self._prefetch_call(self._forward_order[-1])

    def _begin_backward_call(self, call_id: int) -> None:
        """Prefetch the preceding forward call at the first current unpack."""
        if call_id in self._backward_started_call_ids:
            return
        self._backward_started_call_ids.add(call_id)
        self._handles_by_call_id.pop(call_id, None)
        if self.use_checkpoint_recompute_prefetch:
            if call_id in self._active_checkpoint_prefetch_call_ids:
                self._active_checkpoint_prefetch_call_ids.discard(call_id)
                if not self._active_checkpoint_prefetch_call_ids:
                    self._active_checkpoint_prefetch_boundary = None
            return
        try:
            idx = self._forward_order.index(call_id)
        except ValueError:
            return
        if idx > 0:
            self._prefetch_call(self._forward_order[idx - 1])

    def _prefetch_call(self, call_id: int) -> None:
        """Prefetch all activations belonging to ``call_id`` to the device.

        The underlying :meth:`ActivationOffloadHandle.ensure_device_resident` is
        idempotent and called with ``block=False`` so that the prefetch copy is
        overlapped with the ongoing backward computation.
        """
        for handle in self._handles_by_call_id.get(call_id, ()):
            handle.ensure_device_resident(block=False)
            self.stats.num_prefetch_hits += 1

    def checkpoint_contexts(self) -> tuple[AbstractContextManager, AbstractContextManager]:
        """Seal the preceding offload group and prefetch it during recomputation."""
        checkpoint_boundary = self._checkpoint_boundary_counter
        self._checkpoint_boundary_counter += 1
        self._seal_checkpoint_prefetch_group(checkpoint_boundary)
        return nullcontext(), _CheckpointRecomputePrefetchContext(self, checkpoint_boundary)

    def _seal_checkpoint_prefetch_group(self, checkpoint_boundary: int) -> bool:
        call_ids = tuple(
            call_id
            for call_id in self._forward_order[self._checkpoint_group_cursor :]
            if call_id in self._handles_by_call_id
        )
        self._checkpoint_group_cursor = len(self._forward_order)
        if call_ids:
            self._checkpoint_prefetch_groups.append((checkpoint_boundary, call_ids))
            return True
        return False

    def _prefetch_for_checkpoint_boundary(self, checkpoint_boundary: int) -> None:
        """Start one pending group after the current FSDP unshard dependency."""
        active_boundary = self._active_checkpoint_prefetch_boundary
        if (
            self._active_checkpoint_prefetch_call_ids
            and active_boundary is not None
            and checkpoint_boundary < active_boundary
        ):
            self._retire_active_checkpoint_prefetch_group()

        if self._active_checkpoint_prefetch_call_ids:
            return

        while self._checkpoint_prefetch_groups:
            target_boundary, call_ids = self._checkpoint_prefetch_groups.pop()
            if target_boundary > checkpoint_boundary:
                continue
            live_call_ids = tuple(call_id for call_id in call_ids if call_id in self._handles_by_call_id)
            if not live_call_ids:
                continue

            self._active_checkpoint_prefetch_call_ids.update(live_call_ids)
            self._active_checkpoint_prefetch_boundary = target_boundary
            self._order_prefetch_after_current_stream(live_call_ids)
            for call_id in live_call_ids:
                self._prefetch_call(call_id)
            self.stats.num_checkpoint_prefetch_groups += 1
            return

    def _order_prefetch_after_current_stream(self, call_ids: Tuple[int, ...]) -> None:
        """Make each device's single prefetch stream wait for the current stream."""
        devices = {handle.device for call_id in call_ids for handle in self._handles_by_call_id.get(call_id, ())}
        for device in devices:
            ready_event = self._stream_cache.order_prefetch_after_current_stream(device)
            if ready_event is not None:
                self._prefetch_barrier_events.append(ready_event)

    def _retire_active_checkpoint_prefetch_group(self) -> None:
        """Release prefetched calls left unused when the next GC boundary starts."""
        for call_id in self._active_checkpoint_prefetch_call_ids:
            for handle in self._handles_by_call_id.pop(call_id, ()):
                handle.release_restored_tensor()
        self._active_checkpoint_prefetch_call_ids.clear()
        self._active_checkpoint_prefetch_boundary = None

    def finish_backward(self) -> None:
        """Release per-step device copies and indexes."""
        for handle in tuple(self._live_handles):
            handle.release_restored_tensor()
        self._call_stack.clear()
        self._forward_order.clear()
        self._backward_started_call_ids.clear()
        self._handles_by_call_id.clear()
        self._checkpoint_group_cursor = 0
        self._checkpoint_boundary_counter = 0
        self._checkpoint_prefetch_groups.clear()
        self._active_checkpoint_prefetch_call_ids.clear()
        self._active_checkpoint_prefetch_boundary = None
        self._prefetch_barrier_events.clear()
        self._parameter_storage_keys_by_call_id.clear()
        self._live_handles.clear()

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
        for wrapper in self._checkpoint_wrappers:
            wrapper.set_checkpoint_context_fn(noop_context_fn)
        self._checkpoint_wrappers.clear()
        for _module, pre_handle, post_handle in self._module_hooks:
            pre_handle.remove()
            post_handle.remove()
        self._module_hooks.clear()
        self.finish_backward()
        self._stream_cache.clear()


class _CheckpointRecomputePrefetchContext:
    """Trigger one pending activation group when checkpoint recomputation starts."""

    def __init__(self, runtime: SelectiveAsyncActivationOffloadRuntime, checkpoint_boundary: int) -> None:
        self.runtime = runtime
        self.checkpoint_boundary = checkpoint_boundary

    def __enter__(self) -> "_CheckpointRecomputePrefetchContext":
        self.runtime._prefetch_for_checkpoint_boundary(self.checkpoint_boundary)
        return self

    def __exit__(self, *exc_info) -> None:
        return None
