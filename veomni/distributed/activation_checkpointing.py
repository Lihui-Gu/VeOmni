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

"""Transparent module-call wrappers for selective gradient checkpointing."""

import inspect
from collections.abc import Iterator, Sequence
from typing import Any, Optional

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, CheckpointWrapper
from torch.utils.checkpoint import noop_context_fn

from ..utils import logging
from .module_selection import ResolvedModuleSelection


logger = logging.get_logger(__name__)


class TransparentCheckpointWrapper(CheckpointWrapper):
    """Checkpoint a module call while preserving its logical traversal names.

    PyTorch's ``CheckpointWrapper`` already strips its internal prefix from
    state-dict keys. This subclass also hides the wrapper from ``named_modules``
    traversal so callers of ``named_parameters`` and ``named_buffers`` retain
    the same fully-qualified names as the unwrapped model.
    """

    _veomni_selective_checkpoint_wrapper = True

    def __init__(self, module: nn.Module, **checkpoint_fn_kwargs: Any) -> None:
        super().__init__(module, **checkpoint_fn_kwargs)
        try:
            self._veomni_forward_signature: Optional[inspect.Signature] = inspect.signature(module.forward)
        except (TypeError, ValueError):
            self._veomni_forward_signature = None

    def named_modules(
        self,
        memo: Optional[set[nn.Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Module]]:
        yield from self._checkpoint_wrapped_module.named_modules(
            memo=memo,
            prefix=prefix,
            remove_duplicate=remove_duplicate,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        call_arguments = dict(kwargs)
        if self._veomni_forward_signature is not None:
            bound = self._veomni_forward_signature.bind_partial(*args, **kwargs)
            call_arguments.update(bound.arguments)
            for name, parameter in self._veomni_forward_signature.parameters.items():
                if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    call_arguments.update(bound.arguments.get(name, {}))

        if call_arguments.get("use_cache") or any(
            call_arguments.get(key) is not None
            for key in ("past_key_value", "past_key_values", "cache_params", "layer_past")
        ):
            raise RuntimeError("Selective gradient checkpointing is only supported for training without KV cache.")
        return super().forward(*args, **kwargs)


def unwrap_selective_checkpoint_module(module: nn.Module) -> nn.Module:
    """Return the implementation module stored by a transparent wrapper."""
    while getattr(module, "_veomni_selective_checkpoint_wrapper", False):
        module = module._checkpoint_wrapped_module
    return module


def install_selective_checkpoint_wrappers(
    model: nn.Module,
    targets: Sequence[ResolvedModuleSelection],
    *,
    early_stop: bool,
) -> tuple[TransparentCheckpointWrapper, ...]:
    """Replace resolved targets with transparent non-reentrant wrappers."""
    installed: list[TransparentCheckpointWrapper] = []
    for target in targets:
        parent_path, separator, child_name = target.module_path.rpartition(".")
        if not separator:
            parent_path = ""
            child_name = target.module_path
        if not child_name:
            raise ValueError("Selective gradient checkpointing cannot wrap the root module.")

        parent = model.get_submodule(parent_path)
        current = parent._modules.get(child_name)
        if current is not target.module:
            raise RuntimeError(
                f"Module identity for selective checkpoint target {target.module_path!r} changed during "
                "parallelization. Resolve selectors after the transformation or reject this parallel mode."
            )

        wrapper = TransparentCheckpointWrapper(
            target.module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            context_fn=noop_context_fn,
            early_stop=early_stop,
        )
        wrapper._veomni_logical_module_path = target.module_path
        parent._modules[child_name] = wrapper
        installed.append(wrapper)
        logger.info_rank0(f"Enable selective gradient checkpointing for module: {target.module_path}")

    return tuple(installed)


__all__ = [
    "TransparentCheckpointWrapper",
    "install_selective_checkpoint_wrappers",
    "unwrap_selective_checkpoint_module",
]
