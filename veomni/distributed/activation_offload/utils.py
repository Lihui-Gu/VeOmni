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

"""Device-agnostic stream/event helpers for selective activation offload."""

from typing import Dict, Optional

import torch

from ...utils.device import get_torch_device


class _StreamCache:
    """Lazily creates and caches one offload stream and one prefetch stream
    per accelerator device. CPU uses no dedicated streams (synchronous path).
    """

    def __init__(self) -> None:
        self._offload_streams: Dict[torch.device, torch.Stream] = {}
        self._prefetch_streams: Dict[torch.device, torch.Stream] = {}

    def get_offload_stream(self, device: torch.device) -> Optional[torch.Stream]:
        if device.type == "cpu":
            return None
        if device not in self._offload_streams:
            self._offload_streams[device] = _new_stream(device)
        return self._offload_streams[device]

    def get_prefetch_stream(self, device: torch.device) -> Optional[torch.Stream]:
        if device.type == "cpu":
            return None
        if device not in self._prefetch_streams:
            self._prefetch_streams[device] = _new_stream(device)
        return self._prefetch_streams[device]

    def order_prefetch_after_current_stream(self, device: torch.device) -> Optional[torch.Event]:
        """Order the cached prefetch stream after work already queued on the current stream."""
        prefetch_stream = self.get_prefetch_stream(device)
        current_stream = _current_stream(device)
        if prefetch_stream is None or current_stream is None:
            return None

        ready_event = _new_event(device)
        assert ready_event is not None
        ready_event.record(current_stream)
        prefetch_stream.wait_event(ready_event)
        return ready_event

    def clear(self) -> None:
        self._offload_streams.clear()
        self._prefetch_streams.clear()


def _current_stream(device: torch.device) -> Optional[torch.Stream]:
    """Return the current compute stream for the device, or None for CPU."""
    return None if device.type == "cpu" else get_torch_device().current_stream(device)


def _new_stream(device: torch.device) -> torch.Stream:
    """Create a stream using the accelerator's native implementation."""
    return get_torch_device().Stream(device=device)


def _new_event(device: torch.device) -> Optional[torch.Event]:
    """Create a new event for the device. CPU returns a no-op event."""
    if device.type == "cpu":
        return None
    return get_torch_device().Event(enable_timing=False)
