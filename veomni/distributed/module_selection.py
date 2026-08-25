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

"""Resolve stable module identities from shared class/path selectors."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from functools import lru_cache
from typing import Any

from torch import nn
from torch.distributed.fsdp import FSDPModule


@dataclass(frozen=True)
class ResolvedModuleSelection:
    """A selected module and the logical paths that resolve to its identity."""

    module_path: str
    module: nn.Module = field(repr=False, compare=False)
    matched_class_names: tuple[str, ...] = ()
    matched_path_patterns: tuple[str, ...] = ()
    alias_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActivationMemoryPlan:
    """Resolved selective-checkpoint and selective-offload targets."""

    gradient_checkpoint_targets: tuple[ResolvedModuleSelection, ...] = ()
    activation_offload_targets: tuple[ResolvedModuleSelection, ...] = ()

    @property
    def selective_gradient_checkpointing(self) -> bool:
        return bool(self.gradient_checkpoint_targets)

    @property
    def selective_activation_offload(self) -> bool:
        return bool(self.activation_offload_targets)


def _implementation_class_names(module: nn.Module) -> tuple[str, ...]:
    """Return user-visible implementation classes, excluding framework wrappers."""
    wrapped_module = getattr(module, "_checkpoint_wrapped_module", None)
    if isinstance(wrapped_module, nn.Module):
        module = wrapped_module

    classes = type(module).__mro__
    if len(classes) >= 3 and classes[1] is FSDPModule:
        classes = classes[2:]

    return tuple(cls.__name__ for cls in classes if cls is not nn.Module and issubclass(cls, nn.Module))


def _validate_selector_values(values: Iterable[str], description: str) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(values))
    if any(not value or value != value.strip() for value in requested):
        raise ValueError(f"Module selection {description} must be non-empty and trimmed.")
    return requested


def _path_matches(pattern: str, module_path: str) -> bool:
    """Match a dotted module path, with ``*`` per segment and recursive ``**``."""
    pattern_parts = tuple(pattern.split("."))
    path_parts = tuple(module_path.split(".")) if module_path else ()

    @lru_cache(maxsize=None)
    def matches(pattern_index: int, path_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)

        part = pattern_parts[pattern_index]
        if part == "**":
            return matches(pattern_index + 1, path_index) or (
                path_index < len(path_parts) and matches(pattern_index, path_index + 1)
            )
        return (
            path_index < len(path_parts)
            and fnmatchcase(path_parts[path_index], part)
            and matches(pattern_index + 1, path_index + 1)
        )

    return matches(0, 0)


def resolve_module_selection(model: nn.Module, selection: Any) -> tuple[ResolvedModuleSelection, ...]:
    """Resolve a class/path selector before wrappers or FSDP change the visible tree."""
    requested_classes = _validate_selector_values(getattr(selection, "module_classes", ()), "class names")
    requested_paths = _validate_selector_values(getattr(selection, "module_paths", ()), "path patterns")
    if not requested_classes and not requested_paths:
        raise ValueError("Module selection requires at least one module class name or path pattern.")
    if any("" in pattern.split(".") for pattern in requested_paths):
        raise ValueError("Module selection path patterns must not contain empty path segments.")

    entries_by_id: dict[int, dict[str, Any]] = {}
    for module_path, module in model.named_modules(remove_duplicate=False):
        entry = entries_by_id.setdefault(
            id(module),
            {
                "module": module,
                "paths": [],
                "class_names": set(_implementation_class_names(module)),
            },
        )
        entry["paths"].append(module_path)

    resolved: list[ResolvedModuleSelection] = []
    matched_classes: set[str] = set()
    matched_patterns: set[str] = set()
    for entry in entries_by_id.values():
        class_matches = tuple(name for name in requested_classes if name in entry["class_names"])
        path_matches = tuple(
            pattern
            for pattern in requested_paths
            if any(_path_matches(pattern, module_path) for module_path in entry["paths"])
        )
        if requested_classes and not class_matches:
            continue
        if requested_paths and not path_matches:
            continue

        matched_classes.update(class_matches)
        matched_patterns.update(path_matches)
        module_paths = tuple(entry["paths"])
        resolved.append(
            ResolvedModuleSelection(
                module_path=module_paths[0],
                module=entry["module"],
                matched_class_names=class_matches,
                matched_path_patterns=path_matches,
                alias_paths=module_paths[1:],
            )
        )

    missing_classes = [name for name in requested_classes if name not in matched_classes]
    missing_patterns = [pattern for pattern in requested_paths if pattern not in matched_patterns]
    if missing_classes or missing_patterns:
        details = []
        if missing_classes:
            details.append(f"classes: {', '.join(missing_classes)}")
        if missing_patterns:
            details.append(f"paths: {', '.join(missing_patterns)}")
        raise ValueError(f"Module selectors matched no final targets ({'; '.join(details)}).")
    if not resolved:
        raise ValueError("Module selection matched no modules after combining class and path constraints.")

    return tuple(resolved)


def resolve_module_class_selection(
    model: nn.Module,
    module_class_names: Iterable[str],
) -> tuple[ResolvedModuleSelection, ...]:
    """Backward-compatible class-only selection entry point."""

    @dataclass(frozen=True)
    class _ClassSelection:
        module_classes: tuple[str, ...]
        module_paths: tuple[str, ...] = ()

    return resolve_module_selection(model, _ClassSelection(tuple(module_class_names)))


def _is_ancestor_path(ancestor: str, descendant: str) -> bool:
    if not ancestor:
        return bool(descendant)
    return descendant.startswith(ancestor + ".")


def _all_paths(selection: ResolvedModuleSelection) -> tuple[str, ...]:
    return (selection.module_path, *selection.alias_paths)


def validate_activation_memory_regions(
    gradient_checkpoint_targets: Sequence[ResolvedModuleSelection],
    activation_offload_targets: Sequence[ResolvedModuleSelection],
) -> None:
    """Reject ambiguous or overlapping checkpoint/offload regions."""
    for target in gradient_checkpoint_targets:
        if not target.module_path:
            raise ValueError("Selective gradient checkpointing cannot target the root module.")
        if target.alias_paths:
            raise ValueError(
                f"Selective gradient checkpointing does not support shared module {target.module_path!r} "
                f"with aliases {target.alias_paths!r}."
            )
        if isinstance(target.module, (nn.ModuleDict, nn.ModuleList, nn.Sequential)):
            raise TypeError(
                f"Selective gradient checkpointing target {target.module_path!r} is a container module; "
                "select explicit computation modules instead."
            )
        if getattr(target.module, "_supports_selective_gradient_checkpointing", True) is False:
            raise TypeError(f"Module {target.module_path!r} explicitly opts out of selective gradient checkpointing.")

    for index, target in enumerate(gradient_checkpoint_targets):
        for other in gradient_checkpoint_targets[index + 1 :]:
            if any(
                _is_ancestor_path(path, other_path) or _is_ancestor_path(other_path, path)
                for path in _all_paths(target)
                for other_path in _all_paths(other)
            ):
                raise ValueError(
                    "Selective gradient checkpointing targets must not be nested: "
                    f"{target.module_path!r} and {other.module_path!r}."
                )

    for checkpoint_target in gradient_checkpoint_targets:
        for offload_target in activation_offload_targets:
            if checkpoint_target.module is offload_target.module or any(
                checkpoint_path == offload_path
                or _is_ancestor_path(checkpoint_path, offload_path)
                or _is_ancestor_path(offload_path, checkpoint_path)
                for checkpoint_path in _all_paths(checkpoint_target)
                for offload_path in _all_paths(offload_target)
            ):
                raise ValueError(
                    "Selective gradient-checkpoint and activation-offload targets must not overlap or be nested: "
                    f"{checkpoint_target.module_path!r} and {offload_target.module_path!r}."
                )


def resolve_activation_memory_plan(
    model: nn.Module,
    gradient_checkpointing_config: Any,
    offload_config: Any,
) -> ActivationMemoryPlan:
    """Resolve active selectors once against the logical pre-parallel model tree."""
    gradient_checkpoint_targets: tuple[ResolvedModuleSelection, ...] = ()
    activation_offload_targets: tuple[ResolvedModuleSelection, ...] = ()

    checkpoint_selection = getattr(gradient_checkpointing_config, "selection", None)
    if getattr(gradient_checkpointing_config, "enable", False) and checkpoint_selection is not None:
        gradient_checkpoint_targets = resolve_module_selection(model, checkpoint_selection)

    offload_selection = getattr(offload_config, "selection", None)
    offload_enabled = getattr(offload_config, "enable_activation", False)
    model_wide_checkpointing = getattr(gradient_checkpointing_config, "enable", False) and checkpoint_selection is None
    if offload_enabled and offload_selection is not None and not model_wide_checkpointing:
        activation_offload_targets = resolve_module_selection(model, offload_selection)

    validate_activation_memory_regions(gradient_checkpoint_targets, activation_offload_targets)
    return ActivationMemoryPlan(
        gradient_checkpoint_targets=gradient_checkpoint_targets,
        activation_offload_targets=activation_offload_targets,
    )
