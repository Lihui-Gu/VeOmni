import os

import pytest
import torch
import torch.distributed as dist
from torch import nn

from veomni.arguments import MixedPrecisionConfig, ModuleSelectionConfig
from veomni.distributed.module_selection import resolve_module_selection
from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


class _FSDPCheckpointBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.forward_calls = 0

    def forward(self, hidden_states):
        self.forward_calls += 1
        return torch.sin(self.proj(hidden_states))


class _FSDPCheckpointModel(nn.Module):
    _no_split_modules = ["_FSDPCheckpointBlock"]

    def __init__(self):
        super().__init__()
        self.block = _FSDPCheckpointBlock()
        self.output = nn.Linear(8, 1)

    def forward(self, hidden_states):
        return self.output(self.block(hidden_states))

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


@pytest.mark.skipif(int(os.getenv("WORLD_SIZE", "1")) < 2, reason="Run with torchrun and at least two ranks")
def test_selective_checkpoint_reenters_fsdp_module_call_and_preserves_names():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    get_torch_device().set_device(local_rank)
    dist.init_process_group(backend=get_dist_comm_backend())
    init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        device_type=get_device_type(),
        name="selective-checkpoint-fsdp-test",
    )

    try:
        with torch.device("meta"):
            model = _FSDPCheckpointModel()
        targets = resolve_module_selection(model, ModuleSelectionConfig(module_paths=["block"]))
        model = build_parallelize_model(
            model,
            init_device="meta",
            weights_path=None,
            mixed_precision=MixedPrecisionConfig(enable=False),
            enable_gradient_checkpointing=True,
            selective_checkpoint_targets=targets,
        )

        hidden_states = torch.randn(2, 8, device=get_device_type(), requires_grad=True)
        model(hidden_states).sum().backward()

        assert model.block.forward_calls == 2
        assert all("_checkpoint_wrapped_module" not in key for key in model.state_dict())
        assert all("_checkpoint_wrapped_module" not in name for name, _ in model.named_parameters())
        assert hidden_states.grad is not None
    finally:
        dist.destroy_process_group()
        clear_parallel_state()
