import gc
import importlib
import weakref
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
    ModuleSelectionConfig,
    OffloadConfig,
    TorchCompileConfig,
    TrainingArguments,
    VeOmniArguments,
    parse_args,
)
from veomni.distributed.activation_checkpointing import install_selective_checkpoint_wrappers
from veomni.distributed.activation_offload import (
    ActivationOffloadHandle,
    NullActivationOffloadRuntime,
    SelectiveAsyncActivationOffloadRuntime,
    ThresholdActivationOffloadRuntime,
    build_activation_offload_runtime,
    resolve_module_class_selection,
    resolve_module_selection,
)
from veomni.distributed.module_selection import resolve_activation_memory_plan
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


def test_hybrid_selection_config_parses_from_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  config_path: dummy
data:
  train_path: dummy.jsonl
train:
  gradient_checkpointing:
    enable: true
    enable_reentrant: false
    selection:
      module_paths:
        - "**.layers.*.mlp"
  accelerator:
    offload_config:
      enable_activation: true
      selection:
        module_classes:
          - SelectedBlock
        module_paths:
          - "**.input_layernorm"
"""
    )
    monkeypatch.setattr("sys.argv", ["test", str(config_path)])

    args = parse_args(VeOmniArguments)

    assert args.train.gradient_checkpointing.selection == ModuleSelectionConfig(module_paths=["**.layers.*.mlp"])
    assert args.train.accelerator.offload_config.selection == ModuleSelectionConfig(
        module_classes=["SelectedBlock"],
        module_paths=["**.input_layernorm"],
    )


def test_selective_gradient_checkpointing_requires_enable():
    with pytest.raises(ValueError, match="requires train.gradient_checkpointing.enable=True"):
        TrainingArguments(
            gradient_checkpointing=GradientCheckpointingConfig(
                enable=False,
                selection=ModuleSelectionConfig(module_paths=["block"]),
            )
        )


def test_selective_gradient_checkpointing_rejects_reentrant_mode():
    with pytest.raises(ValueError, match="requires enable_reentrant=False"):
        TrainingArguments(
            gradient_checkpointing=GradientCheckpointingConfig(
                enable=True,
                enable_reentrant=True,
                selection=ModuleSelectionConfig(module_paths=["block"]),
            )
        )


def test_selective_gradient_checkpointing_rejects_torch_compile():
    with pytest.raises(ValueError, match="not supported with train.torch_compile.enable"):
        TrainingArguments(
            gradient_checkpointing=GradientCheckpointingConfig(
                enable=True,
                selection=ModuleSelectionConfig(module_paths=["block"]),
            ),
            torch_compile=TorchCompileConfig(enable=True),
        )


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


def test_module_path_selection_uses_segment_aware_globs():
    model = ToyModel()

    selected = resolve_module_selection(
        model,
        ModuleSelectionConfig(module_paths=["**.nested.*"]),
    )

    assert [item.module_path for item in selected] == ["nested.0", "nested.1"]
    assert all(item.matched_path_patterns == ("**.nested.*",) for item in selected)


def test_module_selection_combines_class_and_path_constraints():
    model = ToyModel()

    selected = resolve_module_selection(
        model,
        ModuleSelectionConfig(
            module_classes=["SelectedBlock"],
            module_paths=["nested.*"],
        ),
    )

    assert [item.module_path for item in selected] == ["nested.0"]


def test_module_selection_rejects_selector_without_final_target():
    with pytest.raises(ValueError, match="classes: OtherBlock"):
        resolve_module_selection(
            ToyModel(),
            ModuleSelectionConfig(
                module_classes=["SelectedBlock", "OtherBlock"],
                module_paths=["first"],
            ),
        )


class _RegionParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = _SelectedLinear(4, 4)

    def forward(self, hidden_states):
        return self.child(hidden_states)


class _RegionSelectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _RegionParent()


def test_activation_memory_plan_rejects_nested_gc_targets():
    model = _RegionSelectionModel()
    checkpointing = GradientCheckpointingConfig(
        enable=True,
        selection=ModuleSelectionConfig(module_paths=["block", "block.child"]),
    )

    with pytest.raises(ValueError, match="must not be nested"):
        resolve_activation_memory_plan(model, checkpointing, OffloadConfig())


def test_activation_memory_plan_rejects_gc_offload_overlap():
    model = _RegionSelectionModel()
    checkpointing = GradientCheckpointingConfig(
        enable=True,
        selection=ModuleSelectionConfig(module_paths=["block.child"]),
    )
    offload = OffloadConfig(
        enable_activation=True,
        selection=ModuleSelectionConfig(module_paths=["block"]),
    )

    with pytest.raises(ValueError, match="must not overlap or be nested"):
        resolve_activation_memory_plan(model, checkpointing, offload)


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


class _CountingLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.forward_calls = 0

    def forward(self, hidden_states):
        self.forward_calls += 1
        return super().forward(hidden_states)


class _HybridToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.checkpointed = _CountingLinear(4, 4)
        self.offloaded = _CountingLinear(4, 4)

    def forward(self, hidden_states):
        return self.offloaded(torch.sin(self.checkpointed(hidden_states)))


class _MultiOutputSelectedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 4)

    def forward(self, x):
        return self.first(x), {"nested": self.second(x)}


def _make_offload_config(
    enable_activation: bool = True,
    activation_gpu_limit: float = 0.0,
    selection_module_classes=None,
    selection_module_paths=None,
    prefetch: bool = False,
):
    selection = None
    if selection_module_classes or selection_module_paths:
        selection = ActivationOffloadSelectionConfig(
            module_classes=list(selection_module_classes or ()),
            module_paths=list(selection_module_paths or ()),
        )
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


def test_activation_offload_handle_does_not_retain_source_tensor():
    source = torch.randn(4, 8).transpose(0, 1)
    source_ref = weakref.ref(source)
    expected_stride = source.stride()

    handle = ActivationOffloadHandle(source, call_id=0)
    handle.offload(source)
    del source
    gc.collect()

    assert source_ref() is None
    assert handle.stride == expected_stride


@pytest.mark.skipif(get_device_type() == "cpu", reason="Requires a CUDA or NPU accelerator")
def test_activation_offload_handle_restores_after_device_copy_is_released():
    source = torch.randn(128, 128, device=get_device_type())
    expected = source.cpu()
    handle = ActivationOffloadHandle(source, call_id=0)
    handle.offload(source)

    restored = handle.ensure_device_resident()
    handle.release_restored_tensor(restored)
    synchronize()
    restored_ref = weakref.ref(restored)
    del restored
    gc.collect()

    assert restored_ref() is None
    restored_again = handle.ensure_device_resident()
    torch.testing.assert_close(restored_again.cpu(), expected)


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


def test_build_runtime_enables_hybrid_with_resolved_selection():
    model = _RuntimeToyModel()
    config = _make_offload_config(selection_module_paths=["selected"])
    resolved = resolve_module_selection(model, config.selection)

    runtime = build_activation_offload_runtime(
        model,
        config,
        enable_gradient_checkpointing=True,
        enable_selective_gradient_checkpointing=True,
        resolved_selection=resolved,
    )

    assert isinstance(runtime, SelectiveAsyncActivationOffloadRuntime)
    runtime.close()


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


def test_hybrid_runtime_recomputes_only_gc_target_and_offloads_sibling():
    model = _HybridToyModel()
    checkpointing = GradientCheckpointingConfig(
        enable=True,
        selection=ModuleSelectionConfig(module_paths=["checkpointed"]),
    )
    config = _make_offload_config(
        activation_gpu_limit=1024.0,
        selection_module_paths=["offloaded"],
    )
    plan = resolve_activation_memory_plan(model, checkpointing, config)
    install_selective_checkpoint_wrappers(model, plan.gradient_checkpoint_targets, early_stop=True)
    runtime = build_activation_offload_runtime(
        model,
        config,
        enable_gradient_checkpointing=True,
        enable_selective_gradient_checkpointing=True,
        resolved_selection=plan.activation_offload_targets,
    )

    hidden_states = torch.randn(2, 4, requires_grad=True)
    with runtime.forward_context:
        output = model(hidden_states)
    with runtime.backward_context:
        output.sum().backward()

    assert model.checkpointed.forward_calls == 2
    assert model.offloaded.forward_calls == 1
    assert runtime.stats.num_offloaded_tensors > 0
    assert hidden_states.grad is not None
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


class _AsyncToyBlock(nn.Module):
    def forward(self, hidden_states):
        for _ in range(4):
            hidden_states = torch.sin(hidden_states)
        return hidden_states


@pytest.mark.skipif(get_device_type() == "cpu", reason="Requires a CUDA or NPU accelerator")
@pytest.mark.parametrize("prefetch", [False, True])
def test_selective_async_runtime_preserves_multi_block_gradients(prefetch):
    model = nn.Sequential(*(_AsyncToyBlock() for _ in range(4))).to(get_device_type())
    torch.manual_seed(20260824)
    get_torch_device().manual_seed_all(20260824)
    model_input = torch.randn(1024, 1024, device=get_device_type(), dtype=torch.bfloat16)

    def run(runtime):
        hidden_states = model_input.detach().clone().requires_grad_(True)
        with runtime.forward_context:
            output = model(hidden_states)
        with runtime.backward_context:
            output.float().square().mean().backward()
        synchronize()
        return output.detach().cpu(), hidden_states.grad.detach().cpu()

    baseline_runtime = build_activation_offload_runtime(model, _make_offload_config(enable_activation=False))
    baseline_output, baseline_grad = run(baseline_runtime)
    offload_runtime = build_activation_offload_runtime(
        model,
        _make_offload_config(
            selection_module_classes=["_AsyncToyBlock"],
            prefetch=prefetch,
        ),
    )
    try:
        offload_output, offload_grad = run(offload_runtime)
        torch.testing.assert_close(offload_output, baseline_output, rtol=0, atol=0)
        torch.testing.assert_close(offload_grad, baseline_grad, rtol=0, atol=0)
        assert offload_runtime.stats.num_offloaded_tensors > 0
        if prefetch:
            assert offload_runtime.stats.num_prefetch_hits > 0
    finally:
        offload_runtime.close()


@pytest.mark.skipif(get_device_type() == "cpu", reason="Requires a CUDA or NPU accelerator")
def test_selective_runtime_does_not_retain_consumed_device_copies():
    model = nn.Sequential(*(_AsyncToyBlock() for _ in range(2))).to(get_device_type())
    runtime = build_activation_offload_runtime(
        model,
        _make_offload_config(
            selection_module_classes=["_AsyncToyBlock"],
            prefetch=True,
        ),
    )

    model_input = torch.randn(128, 128, device=get_device_type(), requires_grad=True)
    with runtime.forward_context:
        output = model(model_input)
    handles = [handle for call_handles in runtime._handles_by_call_id.values() for handle in call_handles]

    with runtime.backward_context:
        output.sum().backward()
    synchronize()

    assert handles
    assert all(handle._restored_tensor is None for handle in handles)
    assert len(runtime._live_handles) == 0
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
    handles = [handle for call_handles in runtime._handles_by_call_id.values() for handle in call_handles]

    loss = y.sum()
    with runtime.backward_context:
        loss.backward()

    # Entering backward prefetches the final selected call; the first unpack
    # from that call then prefetches the preceding call.
    assert runtime.stats.num_prefetch_hits > 0

    # All handles should be device-ready and released from the runtime's
    # prefetch index after backward.
    assert runtime._handles_by_call_id == {}
    for handle in handles:
        assert handle.state.name == "DEVICE_READY"

    # Calling ensure_device_resident again is a no-op (idempotent).
    for handle in handles:
        restored = handle.ensure_device_resident()
        assert restored is handle.restored_tensor

    runtime.close()


def test_selective_runtime_releases_per_step_handles():
    model = _RuntimeToyModel()
    runtime = build_activation_offload_runtime(
        model,
        _make_offload_config(
            selection_module_classes=["_SelectedLinear", "_OtherLinear"],
            prefetch=True,
        ),
    )

    for _ in range(2):
        model_input = torch.randn(2, 4, requires_grad=True)
        with runtime.forward_context:
            output = model(model_input)
        with runtime.backward_context:
            output.sum().backward()
        assert runtime._handles_by_call_id == {}
        assert len(runtime._live_handles) == 0
        assert runtime._forward_order == []
        assert runtime._current_pinned_bytes == 0

    runtime.close()


def test_selective_runtime_without_prefetch_does_not_retain_handles():
    model = _RuntimeToyModel()
    runtime = build_activation_offload_runtime(
        model,
        _make_offload_config(
            selection_module_classes=["_SelectedLinear", "_OtherLinear"],
            prefetch=False,
        ),
    )

    model_input = torch.randn(2, 4, requires_grad=True)
    with runtime.forward_context:
        output = model(model_input)

    # Autograd owns the packed handles. Without prefetch there is no reason
    # for the runtime to keep an additional strong reference to them.
    assert runtime._handles_by_call_id == {}

    with runtime.backward_context:
        output.sum().backward()
    assert model_input.grad is not None
    runtime.close()


def test_selective_runtime_prefetch_releases_call_index_at_backward_boundary():
    model = _RuntimeToyModel()
    runtime = build_activation_offload_runtime(
        model,
        _make_offload_config(
            selection_module_classes=["_SelectedLinear", "_OtherLinear"],
            prefetch=True,
        ),
    )

    model_input = torch.randn(2, 4, requires_grad=True)
    with runtime.forward_context:
        output = model(model_input)

    first_call_id, second_call_id = runtime._forward_order
    assert set(runtime._handles_by_call_id) == {first_call_id, second_call_id}

    # The first unpack for each module releases its lookup entry without
    # relying on finish_backward's final cleanup.
    output.sum().backward()
    assert runtime._handles_by_call_id == {}
    assert runtime.stats.num_prefetch_hits > 0

    runtime.close()


def test_selective_runtime_prefetch_supports_nested_output_backward():
    model = _MultiOutputSelectedBlock()
    runtime = build_activation_offload_runtime(
        model,
        _make_offload_config(
            selection_module_classes=["_MultiOutputSelectedBlock"],
            prefetch=True,
        ),
    )

    model_input = torch.randn(2, 4, requires_grad=True)
    with runtime.forward_context:
        first_output, nested_outputs = model(model_input)

    assert runtime._handles_by_call_id

    # The first output is an independent, unused branch. Backward through a
    # nested non-first output must still release the module's prefetch index.
    nested_outputs["nested"].sum().backward()
    assert first_output.grad_fn is not None
    assert runtime._handles_by_call_id == {}

    runtime.close()


def test_selective_runtime_retained_graph_does_not_collide_with_next_step():
    model = _RuntimeToyModel()
    runtime = build_activation_offload_runtime(
        model,
        _make_offload_config(
            selection_module_classes=["_SelectedLinear", "_OtherLinear"],
            prefetch=True,
        ),
    )

    first_input = torch.randn(2, 4, requires_grad=True)
    with runtime.forward_context:
        first_output = model(first_input)
    first_loss = first_output.sum()
    first_call_ids = set(runtime._handles_by_call_id)
    with runtime.backward_context:
        first_loss.backward(retain_graph=True)

    second_input = torch.randn(2, 4, requires_grad=True)
    with runtime.forward_context:
        second_output = model(second_input)
    second_call_ids = set(runtime._handles_by_call_id)
    assert first_call_ids.isdisjoint(second_call_ids)

    with runtime.backward_context:
        second_output.sum().backward()
    assert second_input.grad is not None

    # A later backward through the retained graph must not reuse call IDs or
    # prefetch handles belonging to a newer forward generation.
    prefetch_hits = runtime.stats.num_prefetch_hits
    with runtime.backward_context:
        first_loss.backward()
    assert runtime.stats.num_prefetch_hits == prefetch_hits
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


def test_selective_runtime_rejects_nested_selection_with_prefetch():
    model = _NestedSelectionModel()
    config = _make_offload_config(
        selection_module_classes=["_NestedOuterModule", "_NestedInnerLinear"],
        prefetch=True,
    )

    with pytest.raises(ValueError, match="prefetch does not support nested module selections"):
        build_activation_offload_runtime(model, config)


def test_selective_runtime_rejects_root_and_child_selection_with_prefetch():
    model = _NestedSelectionModel()
    config = _make_offload_config(
        selection_module_classes=["_NestedSelectionModel", "_NestedInnerLinear"],
        prefetch=True,
    )

    with pytest.raises(ValueError, match="prefetch does not support nested module selections"):
        build_activation_offload_runtime(model, config)
