import types

import pytest
import torch
import torch.nn as nn
from torch.autograd.graph import saved_tensors_hooks
from torch.utils.checkpoint import noop_context_fn

from veomni.arguments import GradientCheckpointingConfig, MixedPrecisionConfig, ModuleSelectionConfig
from veomni.distributed.activation_checkpointing import install_selective_checkpoint_wrappers
from veomni.distributed.module_selection import resolve_module_selection
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.module_utils import _find_submodule
from veomni.optim.optimizer import get_parameter_names


class _CheckpointingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs


@pytest.mark.parametrize("early_stop", [True, False])
@pytest.mark.parametrize("use_reentrant", [True, False])
def test_build_parallelize_model_forwards_checkpoint_early_stop(monkeypatch, early_stop, use_reentrant):
    import veomni.distributed.torch_parallelize as torch_parallelize

    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: types.SimpleNamespace(fsdp_enabled=True, tp_enabled=False, dp_mode="fsdp2"),
    )
    monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", lambda model, **kwargs: model)
    model = _CheckpointingModel()

    result = build_parallelize_model(
        model,
        mixed_precision=MixedPrecisionConfig(enable=False),
        early_stop=early_stop,
        enable_reentrant=use_reentrant,
    )

    assert result is model
    expected = {
        "use_reentrant": use_reentrant,
        "context_fn": noop_context_fn,
    }
    if not use_reentrant:
        expected["early_stop"] = early_stop
    assert model.gradient_checkpointing_kwargs == expected


def test_gradient_checkpointing_config_enables_early_stop_by_default():
    assert GradientCheckpointingConfig().early_stop is True


class _CheckpointedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.forward_calls = 0

    def forward(self, hidden_states):
        self.forward_calls += 1
        return torch.sin(self.proj(hidden_states))


class _SelectiveCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _CheckpointedBlock()
        self.output = nn.Linear(4, 1)

    def forward(self, hidden_states):
        return self.output(self.block(hidden_states))


def test_selective_checkpoint_wrapper_recomputes_and_preserves_names():
    model = _SelectiveCheckpointModel()
    original_state_keys = tuple(model.state_dict())
    original_parameter_names = tuple(name for name, _ in model.named_parameters())
    target = resolve_module_selection(model, ModuleSelectionConfig(module_paths=["block"]))

    install_selective_checkpoint_wrappers(model, target, early_stop=True)
    hidden_states = torch.randn(2, 4, requires_grad=True)
    output = model(hidden_states)
    output.sum().backward()

    assert model.block.forward_calls == 2
    assert tuple(model.state_dict()) == original_state_keys
    assert tuple(name for name, _ in model.named_parameters()) == original_parameter_names
    assert hidden_states.grad is not None


def test_selective_checkpoint_internal_saved_tensors_do_not_reach_outer_hook():
    model = _SelectiveCheckpointModel()
    target = resolve_module_selection(model, ModuleSelectionConfig(module_paths=["block"]))
    install_selective_checkpoint_wrappers(model, target, early_stop=True)
    packed_shapes = []

    def pack(tensor):
        packed_shapes.append(tuple(tensor.shape))
        return tensor

    hidden_states = torch.randn(2, 4, requires_grad=True)
    with saved_tensors_hooks(pack, lambda tensor: tensor):
        output = model(hidden_states)
    output.sum().backward()

    assert packed_shapes.count((4, 4)) == 0
    assert model.block.forward_calls == 2


def test_selective_checkpoint_rejects_positional_kv_cache():
    class CacheBlock(nn.Module):
        def forward(self, hidden_states, past_key_values=None):
            return hidden_states

    model = nn.Sequential(CacheBlock())
    targets = resolve_module_selection(model, ModuleSelectionConfig(module_paths=["0"]))
    install_selective_checkpoint_wrappers(model, targets, early_stop=True)

    with pytest.raises(RuntimeError, match="without KV cache"):
        model[0](torch.ones(1, requires_grad=True), object())


def test_selective_checkpoint_wrapper_keeps_direct_parameter_lookup_transparent():
    class DirectParameterBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))

        def forward(self, hidden_states):
            return hidden_states * self.weight

    model = nn.Sequential(DirectParameterBlock())
    original = model[0]
    target = resolve_module_selection(model, ModuleSelectionConfig(module_paths=["0"]))
    install_selective_checkpoint_wrappers(model, target, early_stop=True)

    owner, local_name = _find_submodule(model, "0.weight")

    assert owner is original
    assert local_name == "weight"
    assert [name for name, _ in model.named_parameters()] == ["0.weight"]


def test_selective_checkpoint_wrapper_keeps_optimizer_parameter_groups_transparent():
    model = _SelectiveCheckpointModel()
    targets = resolve_module_selection(model, ModuleSelectionConfig(module_paths=["block"]))
    install_selective_checkpoint_wrappers(model, targets, early_stop=True)

    parameter_names = get_parameter_names(model, forbidden_layer_types=[], forbidden_param_names=[])
    decay_names = get_parameter_names(
        model,
        forbidden_layer_types=["_CheckpointedBlock"],
        forbidden_param_names=[],
    )

    assert set(parameter_names) == {"block.proj.weight", "block.proj.bias", "output.weight", "output.bias"}
    assert set(decay_names) == {"output.weight", "output.bias"}


def test_build_parallelize_model_skips_model_wide_gc_for_selective_targets(monkeypatch):
    import veomni.distributed.torch_parallelize as torch_parallelize

    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: types.SimpleNamespace(fsdp_enabled=True, tp_enabled=False, dp_mode="fsdp2"),
    )
    captured = {}

    def fake_parallelize(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", fake_parallelize)
    model = _CheckpointingModel()
    model.block = _CheckpointedBlock()
    targets = resolve_module_selection(model, ModuleSelectionConfig(module_paths=["block"]))

    result = build_parallelize_model(
        model,
        mixed_precision=MixedPrecisionConfig(enable=False),
        selective_checkpoint_targets=targets,
        enable_gradient_checkpointing=True,
    )

    assert result is model
    assert model.gradient_checkpointing_kwargs is None
    assert captured["selective_checkpoint_targets"] == targets
