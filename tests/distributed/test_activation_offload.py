import importlib
from contextlib import nullcontext
from types import SimpleNamespace

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
from veomni.utils.device import get_device_type, get_torch_device, synchronize


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


def test_selective_runtime_round_trips_threshold_fallback_metadata():
    model = _RuntimeToyModel()
    config = _make_offload_config(
        activation_gpu_limit=0.0,
        selection_module_classes=["_SelectedLinear"],
    )
    runtime = build_activation_offload_runtime(model, config)
    tensor = torch.ones(1024)

    packed = runtime.pack_hook(tensor)

    assert isinstance(packed, tuple)
    assert packed[0] is OffloadPolicy.OFFLOAD
    assert packed[1] == tensor.device
    torch.testing.assert_close(runtime.unpack_hook(packed), tensor)
    runtime.close()


@pytest.mark.skipif(get_device_type() == "cpu", reason="Requires a CUDA or NPU accelerator")
def test_selective_runtime_restores_nonselected_threshold_fallback_to_accelerator():
    model = _RuntimeToyModel().to(get_device_type())
    model_input = torch.randn(128, 4, device=get_device_type(), requires_grad=True)
    runtime = build_activation_offload_runtime(
        model,
        _make_offload_config(
            activation_gpu_limit=0.0,
            selection_module_classes=["_SelectedLinear"],
        ),
    )

    try:
        with runtime.forward_context:
            output = model(model_input)
        output.sum().backward()
        synchronize()
        assert model_input.grad is not None
        assert runtime.stats.num_threshold_fallback_offloads > 0
    finally:
        runtime.close()


@pytest.mark.skipif(get_device_type() == "cpu", reason="Requires a CUDA or NPU accelerator")
def test_qwen3_5_gated_deltanet_selective_offload_forward_backward_equivalence(monkeypatch):
    """Selected Qwen3.5 GDN saved tensors preserve forward and backward numerics."""
    from veomni.ops.dispatch import OpSlot

    device_type = get_device_type()
    if device_type == "npu":
        pytest.importorskip("triton", reason="Qwen3.5 NPU GatedDeltaNet requires triton-ascend")
        module_suffix = "npu"
        kernel_impl = "npu"
    else:
        pytest.importorskip("fla", reason="Qwen3.5 GPU GatedDeltaNet requires flash-linear-attention")
        module_suffix = "gpu"
        kernel_impl = "fla"
    modeling = importlib.import_module(
        f"veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_{module_suffix}"
    )
    for slot_name, op_name in (
        ("veomni_rms_norm_gated", "rms_norm_gated"),
        ("veomni_causal_conv1d", "causal_conv1d"),
        ("veomni_chunk_gated_delta_rule", "chunk_gated_delta_rule"),
    ):
        slot = OpSlot(op_name, "standard")
        slot.bind(kernel_impl)
        monkeypatch.setattr(modeling, slot_name, slot)

    config = SimpleNamespace(
        hidden_size=256,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        dtype=torch.bfloat16,
    )
    torch.manual_seed(20260824)
    get_torch_device().manual_seed_all(20260824)
    layer = modeling.Qwen3_5GatedDeltaNet(config, layer_idx=0).to(
        device=device_type,
        dtype=torch.bfloat16,
    )
    layer.train()
    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(ulysses_enabled=False))

    sequence_length = 64
    model_input = torch.randn(
        1,
        sequence_length,
        config.hidden_size,
        device=device_type,
        dtype=torch.bfloat16,
    )
    cu_seq_lens = torch.tensor([0, sequence_length], dtype=torch.int32)

    def run(runtime):
        layer.zero_grad(set_to_none=True)
        hidden_states = model_input.detach().clone().requires_grad_(True)
        with runtime.forward_context:
            output = layer(
                hidden_states,
                cu_seq_lens_q=cu_seq_lens,
                cu_seqlens_list=[0, sequence_length],
            )
        loss = output.float().square().mean()
        with runtime.backward_context:
            loss.backward()
        synchronize()
        parameter_grads = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in layer.named_parameters()
            if parameter.grad is not None
        }
        return output.detach().cpu(), hidden_states.grad.detach().cpu(), parameter_grads

    baseline_runtime = build_activation_offload_runtime(layer, _make_offload_config(enable_activation=False))
    baseline_output, baseline_input_grad, baseline_parameter_grads = run(baseline_runtime)

    offload_runtime = build_activation_offload_runtime(
        layer,
        _make_offload_config(selection_module_classes=["Qwen3_5GatedDeltaNet"]),
    )
    try:
        offload_output, offload_input_grad, offload_parameter_grads = run(offload_runtime)

        torch.testing.assert_close(offload_output, baseline_output, rtol=0, atol=0)
        torch.testing.assert_close(offload_input_grad, baseline_input_grad, rtol=5e-3, atol=2e-8)
        assert offload_parameter_grads.keys() == baseline_parameter_grads.keys()
        for name, baseline_grad in baseline_parameter_grads.items():
            torch.testing.assert_close(
                offload_parameter_grads[name],
                baseline_grad,
                rtol=5e-3,
                atol=2e-8,
                msg=lambda msg, parameter_name=name: f"{msg}\nGradient mismatch for {parameter_name}",
            )

        assert offload_runtime.stats.num_offloaded_tensors > 0
        assert offload_runtime.stats.num_ondemand_restores > 0
        assert offload_runtime.stats.offloaded_bytes == offload_runtime.stats.restored_bytes
    finally:
        offload_runtime.close()


def test_selective_runtime_prefetch_is_idempotent():
    model = _RuntimeToyModel()
    config = _make_offload_config(
        selection_module_classes=["_SelectedLinear", "_OtherLinear"],
        prefetch=True,
    )
    runtime = build_activation_offload_runtime(model, config)

    x = torch.randn(2, 4, requires_grad=True)
    with runtime.forward_context:
        y = model(x)

    loss = y.sum()
    with runtime.backward_context:
        loss.backward()

    # Prefetch should have been triggered when the backward reaches the
    # second selected module and prefetches the first one.
    assert runtime.stats.num_prefetch_hits > 0

    # All handles should be in DEVICE_READY state after backward.
    for handle in runtime._handles:
        assert handle.state.name == "DEVICE_READY"

    # Calling ensure_device_resident again is a no-op (idempotent).
    for handle in runtime._handles:
        restored = handle.ensure_device_resident()
        assert restored is handle.restored_tensor

    runtime.close()


class _NestedInnerLinear(nn.Linear):
    pass


class _NestedOuterModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = _NestedInnerLinear(4, 4)

    def forward(self, x):
        return self.inner(x)


class _NestedSelectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.outer = _NestedOuterModule()

    def forward(self, x):
        return self.outer(x)


def test_selective_runtime_nested_selection_attaches_to_innermost_module():
    model = _NestedSelectionModel()
    config = _make_offload_config(
        selection_module_classes=["_NestedOuterModule", "_NestedInnerLinear"],
    )
    runtime = build_activation_offload_runtime(model, config)

    x = torch.randn(2, 4, requires_grad=True)
    with runtime.forward_context:
        y = model(x)

    loss = y.sum()
    with runtime.backward_context:
        loss.backward()

    assert x.grad is not None
    runtime.close()
