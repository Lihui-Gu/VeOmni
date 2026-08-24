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

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch.distributed.fsdp import FSDPModule


@dataclass(frozen=True)
class ResolvedModuleSelection:
    """One module instance selected by one or more configured class names."""

    module_path: str
    module: torch.nn.Module = field(repr=False, compare=False)
    matched_class_names: tuple[str, ...]


def _implementation_class_names(module: torch.nn.Module) -> tuple[str, ...]:
    """Return user-visible implementation classes, excluding framework wrappers."""
    classes = type(module).__mro__
    if len(classes) >= 3 and classes[1] is FSDPModule:
        classes = classes[2:]

    return tuple(cls.__name__ for cls in classes if cls is not torch.nn.Module and issubclass(cls, torch.nn.Module))


def resolve_module_class_selection(
    model: torch.nn.Module,
    module_class_names: Iterable[str],
) -> tuple[ResolvedModuleSelection, ...]:
    """Resolve exact implementation-class names against a model hierarchy.

    FSDP2 composes modules in place with a dynamic ``FSDPModule`` subclass.
    Matching against the implementation MRO keeps configuration stable before
    and after that transformation. Each module instance is returned once even
    when several configured base classes match it.
    """
    requested_names = tuple(dict.fromkeys(module_class_names))
    if not requested_names:
        raise ValueError("At least one activation-offload module class must be configured.")
    if any(not name or name != name.strip() for name in requested_names):
        raise ValueError("Activation-offload module classes must be non-empty, trimmed class names.")

    resolved = []
    matched_names = set()
    for module_path, module in model.named_modules():
        implementation_names = set(_implementation_class_names(module))
        module_matches = tuple(name for name in requested_names if name in implementation_names)
        if not module_matches:
            continue
        matched_names.update(module_matches)
        resolved.append(
            ResolvedModuleSelection(
                module_path=module_path,
                module=module,
                matched_class_names=module_matches,
            )
        )

    missing_names = [name for name in requested_names if name not in matched_names]
    if missing_names:
        missing = ", ".join(missing_names)
        raise ValueError(f"Activation-offload module classes matched no modules: {missing}.")

    return tuple(resolved)
