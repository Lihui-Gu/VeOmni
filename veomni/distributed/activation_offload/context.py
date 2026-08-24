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

"""Context managers used to wrap forward/backward passes."""

from typing import Optional

from torch.autograd.graph import saved_tensors_hooks

from .runtime import SelectiveAsyncActivationOffloadRuntime


class _SelectiveForwardContext:
    """Wraps a model forward pass with selective activation offload hooks."""

    def __init__(self, runtime: SelectiveAsyncActivationOffloadRuntime) -> None:
        self.runtime = runtime
        self._hooks: Optional[saved_tensors_hooks] = None

    def __enter__(self) -> "_SelectiveForwardContext":
        self._hooks = saved_tensors_hooks(self.runtime.pack_hook, self.runtime.unpack_hook)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._hooks is not None:
            self._hooks.__exit__(*exc_info)
            self._hooks = None


class _SelectiveBackwardContext:
    """Wraps a loss backward pass with selective activation offload hooks."""

    def __init__(self, runtime: SelectiveAsyncActivationOffloadRuntime) -> None:
        self.runtime = runtime
        self._hooks: Optional[saved_tensors_hooks] = None

    def __enter__(self) -> "_SelectiveBackwardContext":
        self._hooks = saved_tensors_hooks(self.runtime.pack_hook, self.runtime.unpack_hook)
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._hooks is not None:
            self._hooks.__exit__(*exc_info)
            self._hooks = None
