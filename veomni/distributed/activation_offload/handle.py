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

"""Activation offload handle with a state machine and async support."""

import threading
import weakref
from enum import Enum
from typing import Optional

import torch

from .utils import _current_stream, _new_event


class HandleState(Enum):
    CREATED = 0
    OFFLOAD_QUEUED = 1
    HOST_READY = 2
    PREFETCH_QUEUED = 3
    DEVICE_READY = 4
    RELEASED = 5


class ActivationOffloadHandle:
    """Lifecycle wrapper for one offloaded saved tensor.

    On accelerator devices (CUDA/NPU) copies are issued asynchronously on
    dedicated offload/prefetch streams and synchronized via events. On CPU
    the implementation falls back to synchronous copies.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        call_id: int,
        offload_stream: Optional[torch.Stream] = None,
        prefetch_stream: Optional[torch.Stream] = None,
    ) -> None:
        self.state = HandleState.CREATED
        self.call_id = call_id
        self.device = tensor.device
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.stride = tensor.stride()
        self.layout = tensor.layout

        self.cpu_tensor: Optional[torch.Tensor] = None
        self._restored_tensor: Optional[torch.Tensor] = None
        self._restored_tensor_ref: Optional[weakref.ReferenceType[torch.Tensor]] = None

        self._offload_stream = offload_stream
        self._prefetch_stream = prefetch_stream
        self._producer_event: Optional[torch.Event] = None
        self._d2h_event: Optional[torch.Event] = None
        self._h2d_event: Optional[torch.Event] = None

        self._lock = threading.Lock()

    @property
    def restored_tensor(self) -> Optional[torch.Tensor]:
        """Return the live device copy without extending its lifetime."""
        if self._restored_tensor is not None:
            return self._restored_tensor
        if self._restored_tensor_ref is not None:
            return self._restored_tensor_ref()
        return None

    def offload(self, tensor: torch.Tensor) -> None:
        """Copy ``tensor`` to CPU and mark the handle host-ready."""
        with self._lock:
            if self.state != HandleState.CREATED:
                return

            if self.device.type == "cpu" or self._offload_stream is None:
                # Synchronous path for CPU or when streams are unavailable.
                self.cpu_tensor = tensor.detach().cpu()
                self.state = HandleState.HOST_READY
                return

            # Asynchronous D2H on the offload stream.
            self.cpu_tensor = torch.empty(
                self.shape,
                dtype=self.dtype,
                layout=self.layout,
                pin_memory=True,
                device="cpu",
            )

            current_stream = _current_stream(self.device)
            if current_stream is not None:
                self._producer_event = _new_event(self.device)
                self._producer_event.record(current_stream)
                self._offload_stream.wait_event(self._producer_event)

            with self._offload_stream:
                self.cpu_tensor.copy_(tensor, non_blocking=True)
                self._d2h_event = _new_event(self.device)
                self._d2h_event.record(self._offload_stream)

            tensor.record_stream(self._offload_stream)
            self.state = HandleState.OFFLOAD_QUEUED

    def ensure_device_resident(self, block: bool = True) -> torch.Tensor:
        """Return a device-resident tensor, launching H2D only if needed.

        This method is idempotent: repeated calls return the same restored
        tensor. When ``block=True`` the caller waits until H2D completes;
        ``block=False`` only submits H2D and returns immediately (used by the
        prefetch scheduler).
        """
        with self._lock:
            if self.state == HandleState.DEVICE_READY:
                restored_tensor = self.restored_tensor
                if restored_tensor is not None:
                    return restored_tensor
                # Autograd released the previous unpack result. Retained
                # graphs may request the tensor again, so rebuild it from the
                # durable CPU copy instead of keeping every restored device
                # allocation alive until Python cyclic GC runs.
                self.state = HandleState.HOST_READY
                self._h2d_event = None

            if self.state == HandleState.PREFETCH_QUEUED:
                if block:
                    self._wait_h2d_and_mark_ready()
                assert self._restored_tensor is not None
                return self._restored_tensor

            if self.state == HandleState.OFFLOAD_QUEUED:
                self._wait_d2h()
                self.state = HandleState.HOST_READY

            if self.state == HandleState.HOST_READY:
                self._restored_tensor_ref = None
                if self._prefetch_stream is None or self.device.type == "cpu":
                    # Synchronous H2D.
                    self._restored_tensor = self.cpu_tensor.to(self.device, non_blocking=False)
                    self.state = HandleState.DEVICE_READY
                else:
                    # Allocate and copy on the same stream. In particular,
                    # torch-npu cannot safely write from the prefetch stream
                    # into storage allocated by the compute stream.
                    if self._d2h_event is not None:
                        self._prefetch_stream.wait_event(self._d2h_event)
                    with self._prefetch_stream:
                        self._restored_tensor = self.cpu_tensor.to(self.device, non_blocking=True)
                        self._h2d_event = _new_event(self.device)
                        self._h2d_event.record(self._prefetch_stream)
                    self.state = HandleState.PREFETCH_QUEUED

            if self.state == HandleState.PREFETCH_QUEUED and block:
                self._wait_h2d_and_mark_ready()

            assert self._restored_tensor is not None
            return self._restored_tensor

    def release_restored_tensor(self, tensor: Optional[torch.Tensor] = None) -> None:
        """Let autograd own the consumed device copy while retaining reuse.

        The strong reference is needed between asynchronous prefetch and the
        first unpack. After unpack, a weak reference preserves idempotent reuse
        while the consumer is alive without extending the allocation lifetime.
        The CPU copy remains available for a later retained-graph backward.
        """
        if self.device.type == "cpu":
            return
        with self._lock:
            if self.state == HandleState.PREFETCH_QUEUED:
                self._wait_h2d_and_mark_ready()
            if self._restored_tensor is None or (tensor is not None and self._restored_tensor is not tensor):
                return
            self._restored_tensor_ref = weakref.ref(self._restored_tensor)
            self._restored_tensor = None

    def _wait_d2h(self) -> None:
        """Block until the asynchronous D2H copy has finished."""
        if self._d2h_event is not None:
            self._d2h_event.synchronize()

    def _wait_h2d_and_mark_ready(self) -> None:
        """Make the current compute stream wait for H2D, then mark ready."""
        if self._h2d_event is not None:
            current_stream = _current_stream(self.device)
            if self.device.type == "npu":
                # torch-npu 2.10 does not reliably order the consumer through
                # Stream.wait_event(), so synchronize the copy event at the
                # point of use. Prefetched copies still run asynchronously
                # until their activation is requested by autograd.
                self._h2d_event.synchronize()
            elif current_stream is not None:
                current_stream.wait_event(self._h2d_event)
        if self._restored_tensor is not None:
            # record_stream tells the allocator that the tensor may still be
            # used by work on the current stream.
            current_stream = _current_stream(self.device)
            if current_stream is not None:
                self._restored_tensor.record_stream(current_stream)
        self.state = HandleState.DEVICE_READY
