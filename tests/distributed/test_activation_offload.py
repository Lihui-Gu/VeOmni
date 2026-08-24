from contextlib import nullcontext

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule

from veomni.arguments import (
    AcceleratorConfig,
    ActivationOffloadSelectionConfig,
    GradientCheckpointingConfig,
    OffloadConfig,
    TorchCompileConfig,
    TrainingArguments,
    VeOmniArguments,
    parse_args,
)
from veomni.distributed.activation_offload import (
    NullActivationOffloadRuntime,
    SelectiveAsyncActivationOffloadRuntime,
    ThresholdActivationOffloadRuntime,
    build_activation_offload_runtime,
    resolve_module_class_selection,
)
from veomni.distributed.offloading import (
    OffloadPolicy,
    _ActivationOffloadThresholdPolicy,
    build_activation_offloading_context,
    custom_save_on_cpu,
)


class SelectedBlock(nn.Module):
    pass


class FrameworkMixin:
    pass


class WrappedSelectedBlock(FrameworkMixin, SelectedBlock):
    """Model the implementation-class MRO retained by FSDP2 composition."""


class OtherBlock(nn.Module):
    pass


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = SelectedBlock()
        self.nested = nn.Sequential(WrappedSelectedBlock(), OtherBlock())


def test_selective_offload_config_parses_from_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  config_path: dummy
data:
  train_path: dummy.jsonl
train:
  gradient_checkpointing:
    enable: false
  accelerator:
    offload_config:
      enable_activation: true
      activation_gpu_limit: 2.0
      selection:
        module_classes:
          - SelectedBlock
      prefetch: true
"""
    )
    monkeypatch.setattr("sys.argv", ["test", str(config_path)])

    args = parse_args(VeOmniArguments)

    offload = args.train.accelerator.offload_config
    assert offload.enable_activation is True
    assert offload.activation_gpu_limit == 2.0
    assert offload.selection == ActivationOffloadSelectionConfig(module_classes=["SelectedBlock"])
    assert offload.prefetch is True


def test_selective_offload_config_parses_from_cli(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "test",
            "--model.config_path",
            "dummy",
            "--data.train_path",
            "dummy.jsonl",
            "--train.gradient_checkpointing.enable",
            "false",
            "--train.accelerator.offload_config.enable_activation",
            "true",
            "--train.accelerator.offload_config.selection.module_classes",
            "SelectedBlock",
            "OtherBlock",
            "--train.accelerator.offload_config.prefetch",
            "true",
        ],
    )

    args = parse_args(VeOmniArguments)

    offload = args.train.accelerator.offload_config
    assert offload.selection == ActivationOffloadSelectionConfig(module_classes=["SelectedBlock", "OtherBlock"])
    assert offload.prefetch is True


def test_legacy_offload_config_parses_without_selection(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  config_path: dummy
data:
  train_path: dummy.jsonl
train:
  accelerator:
    offload_config:
      enable_activation: true
      activation_gpu_limit: 2.0
"""
    )
    monkeypatch.setattr("sys.argv", ["test", str(config_path)])

    args = parse_args(VeOmniArguments)

    offload = args.train.accelerator.offload_config
    assert offload.enable_activation is True
    assert offload.activation_gpu_limit == 2.0
    assert offload.selection is None
    assert offload.prefetch is False


def test_legacy_offload_config_defaults_are_unchanged():
    offload = OffloadConfig(enable_activation=True, activation_gpu_limit=2.0)

    assert offload.selection is None
    assert offload.prefetch is False

    fwd_context, bwd_context = build_activation_offloading_context(
        enable_activation=offload.enable_activation,
        enable_gradient_checkpointing=False,
        activation_gpu_limit=offload.activation_gpu_limit,
    )

    assert isinstance(fwd_context, custom_save_on_cpu)
    assert isinstance(bwd_context, nullcontext)
    assert fwd_context.policy.gpu_limit_in_mb == 2.0 * 1024


def test_gradient_checkpointing_warns_and_keeps_legacy_threshold_contexts(monkeypatch):
    warning_messages = []
    monkeypatch.setattr(
        "veomni.arguments.arguments_types.logger.warning_rank0",
        lambda message: warning_messages.append(message),
    )
    offload = OffloadConfig(
        enable_activation=True,
        activation_gpu_limit=2.0,
        selection=ActivationOffloadSelectionConfig(module_classes=["SelectedBlock"]),
        prefetch=True,
    )
    TrainingArguments(
        accelerator=AcceleratorConfig(offload_config=offload),
        gradient_checkpointing=GradientCheckpointingConfig(enable=True),
    )

    fwd_context, bwd_context = build_activation_offloading_context(
        enable_activation=offload.enable_activation,
        enable_gradient_checkpointing=True,
        activation_gpu_limit=offload.activation_gpu_limit,
    )

    assert len(warning_messages) == 1
    assert "falling back to the legacy threshold" in warning_messages[0]
    assert isinstance(fwd_context, custom_save_on_cpu)
    assert isinstance(bwd_context, custom_save_on_cpu)
    assert fwd_context.policy.gpu_limit_in_mb == 0.0
    assert bwd_context.policy.gpu_limit_in_mb == 2.0 * 1024


def test_selective_offload_rejects_torch_compile():
    offload = OffloadConfig(
        enable_activation=True,
        selection=ActivationOffloadSelectionConfig(module_classes=["SelectedBlock"]),
    )

    with pytest.raises(ValueError, match="Selective activation offload is not supported"):
        TrainingArguments(
            accelerator=AcceleratorConfig(offload_config=offload),
            gradient_checkpointing=GradientCheckpointingConfig(enable=False),
            torch_compile=TorchCompileConfig(enable=True),
        )


def test_module_class_selection_matches_all_instances_and_mro():
    model = ToyModel()

    selected = resolve_module_class_selection(model, ["SelectedBlock"])

    assert [item.module_path for item in selected] == ["first", "nested.0"]
    assert [item.module for item in selected] == [model.first, model.nested[0]]
    assert all(item.matched_class_names == ("SelectedBlock",) for item in selected)


def test_module_class_selection_matches_original_class_after_fsdp_composition():
    model = ToyModel()
    fsdp_class = type("FSDPSelectedBlock", (FSDPModule, SelectedBlock), {})
    model.first.__class__ = fsdp_class

    selected = resolve_module_class_selection(model, ["SelectedBlock"])

    assert [item.module_path for item in selected] == ["first", "nested.0"]
    with pytest.raises(ValueError, match="FSDPSelectedBlock"):
        resolve_module_class_selection(model, ["FSDPSelectedBlock"])


def test_module_class_selection_returns_each_module_once():
    model = ToyModel()

    selected = resolve_module_class_selection(model, ["WrappedSelectedBlock", "SelectedBlock"])

    assert [item.module_path for item in selected] == ["first", "nested.0"]
    assert selected[1].matched_class_names == ("WrappedSelectedBlock", "SelectedBlock")


def test_module_class_selection_rejects_unmatched_class():
    with pytest.raises(ValueError, match="MissingBlock"):
        resolve_module_class_selection(ToyModel(), ["SelectedBlock", "MissingBlock"])


def test_module_class_selection_ignores_non_module_mixins():
    with pytest.raises(ValueError, match="FrameworkMixin"):
        resolve_module_class_selection(ToyModel(), ["FrameworkMixin"])


def test_threshold_policy_preserves_legacy_budget_behavior():
    tensor = torch.ones(4, dtype=torch.float32)
    tensor_size_gb = tensor.numel() * tensor.element_size() / 1024**3
    policy = _ActivationOffloadThresholdPolicy(
        gpu_limit_in_gb=tensor_size_gb / 2,
        min_offload_size=0,
    )

    assert policy.decide(tensor, is_param=True) is OffloadPolicy.IGNORE
    first = policy.decide(tensor)
    second = policy.decide(tensor)

    assert first is OffloadPolicy.KEEP_ON_GPU
    assert second is OffloadPolicy.OFFLOAD
    policy.release(first, tensor)
    assert policy.cur_gpu_ram_in_mb == pytest.approx(0.0)


def test_custom_save_on_cpu_delegates_to_threshold_policy():
    context = custom_save_on_cpu(gpu_limit_in_gb=1.5, pin_memory=False, min_offload_size=0)

    assert isinstance(context.policy, _ActivationOffloadThresholdPolicy)
    assert context.policy.gpu_limit_in_mb == 1.5 * 1024
    assert context.cur_gpu_ram_in_mb == 0.0

    tensor = torch.ones(4)
    packed = context.pack_hook(tensor)
    assert packed[0] is OffloadPolicy.KEEP_ON_GPU
    assert context.cur_gpu_ram_in_mb > 0
    assert context.unpack_hook(packed) is tensor
    assert context.cur_gpu_ram_in_mb == pytest.approx(0.0)


class _SelectedLinear(nn.Linear):
    pass


class _OtherLinear(nn.Linear):
    pass


class _RuntimeToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.selected = _SelectedLinear(4, 4)
        self.other = _OtherLinear(4, 4)

    def forward(self, x):
        x = self.selected(x)
        x = self.other(x)
        return x


def _make_offload_config(
    enable_activation: bool = True,
    activation_gpu_limit: float = 0.0,
    selection_module_classes=None,
    prefetch: bool = False,
):
    selection = None
    if selection_module_classes:
        selection = ActivationOffloadSelectionConfig(module_classes=list(selection_module_classes))
    return OffloadConfig(
        enable_activation=enable_activation,
        activation_gpu_limit=activation_gpu_limit,
        selection=selection,
        prefetch=prefetch,
    )


def test_build_runtime_returns_null_when_disabled():
    config = _make_offload_config(enable_activation=False)
    runtime = build_activation_offload_runtime(None, config)

    assert isinstance(runtime, NullActivationOffloadRuntime)
    assert isinstance(runtime.forward_context, nullcontext)
    assert isinstance(runtime.backward_context, nullcontext)


def test_build_runtime_returns_threshold_when_no_selection():
    config = _make_offload_config(activation_gpu_limit=2.0)
    runtime = build_activation_offload_runtime(None, config)

    assert isinstance(runtime, ThresholdActivationOffloadRuntime)
    assert isinstance(runtime.forward_context, custom_save_on_cpu)


def test_build_runtime_returns_selective_with_selection():
    config = _make_offload_config(selection_module_classes=["_SelectedLinear"])
    runtime = build_activation_offload_runtime(_RuntimeToyModel(), config)

    assert isinstance(runtime, SelectiveAsyncActivationOffloadRuntime)


def test_build_runtime_fallback_to_threshold_when_gradient_checkpointing():
    config = _make_offload_config(
        activation_gpu_limit=2.0,
        selection_module_classes=["_SelectedLinear"],
    )
    runtime = build_activation_offload_runtime(
        _RuntimeToyModel(),
        config,
        enable_gradient_checkpointing=True,
    )

    assert isinstance(runtime, ThresholdActivationOffloadRuntime)


def test_build_runtime_rejects_selection_with_torch_compile():
    config = _make_offload_config(selection_module_classes=["_SelectedLinear"])
    with pytest.raises(ValueError, match="Selective activation offload is not supported"):
        build_activation_offload_runtime(_RuntimeToyModel(), config, enable_compile=True)


def test_selective_runtime_offloads_and_restores_selected_tensors():
    model = _RuntimeToyModel()
    config = _make_offload_config(selection_module_classes=["_SelectedLinear"])
    runtime = build_activation_offload_runtime(model, config)

    x = torch.randn(2, 4, requires_grad=True)
    with runtime.forward_context:
        y = model(x)
    assert y.shape == (2, 4)

    loss = y.sum()
    with runtime.backward_context:
        loss.backward()
    assert x.grad is not None

    runtime.log_summary()
    runtime.close()
