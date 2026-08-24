from types import SimpleNamespace

import torch

import veomni.ops.kernels.cross_entropy.chunk_loss as chunk_loss_module


def test_chunk_loss_reuses_valid_token_denominator(monkeypatch):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))

    original_sum = torch.Tensor.sum
    denominator_sum_calls = 0

    def counting_sum(self, *args, **kwargs):
        nonlocal denominator_sum_calls
        if self.dtype == torch.bool and self.shape == (1, 5):
            denominator_sum_calls += 1
        return original_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", counting_sum)

    hidden_states = torch.randn(1, 6, 4, requires_grad=True)
    weights = torch.randn(8, 4, requires_grad=True)
    labels = torch.tensor([[1, 2, -100, 3, 4, 5]])

    loss, _ = chunk_loss_module.chunk_loss_function(
        hidden_states,
        weights,
        labels,
        chunk_size=2,
    )
    loss.backward()

    assert denominator_sum_calls == 1
    assert hidden_states.grad is not None
    assert weights.grad is not None

def test_chunk_loss_supports_outer_saved_tensor_hooks(monkeypatch):
    monkeypatch.setattr(chunk_loss_module, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))

    hidden_states = torch.randn(1, 6, 4, requires_grad=True)
    weights = torch.randn(8, 4, requires_grad=True)
    labels = torch.tensor([[1, 2, -100, 3, 4, 5]])
    reference_hidden_states = hidden_states.detach().clone().requires_grad_(True)
    reference_weights = weights.detach().clone().requires_grad_(True)
    shifted_labels = labels[..., 1:].reshape(-1)
    reference_logits = torch.nn.functional.linear(
        reference_hidden_states[..., :-1, :].reshape(-1, reference_hidden_states.size(-1)),
        reference_weights,
    ).float()
    reference_loss, _ = chunk_loss_module.eager_cross_entropy(
        reference_logits,
        shifted_labels,
        weights.size(0),
        (shifted_labels != -100).sum(),
    )
    reference_loss.backward()

    packed_shapes = []

    def pack_hook(tensor):
        packed_shapes.append(tensor.shape)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        loss, _ = chunk_loss_module.chunk_loss_function(
            hidden_states,
            weights,
            labels,
            chunk_size=2,
        )

    loss.backward()

    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(hidden_states.grad, reference_hidden_states.grad)
    torch.testing.assert_close(weights.grad, reference_weights.grad)
    assert hidden_states.grad is not None
    assert weights.grad is not None
    assert torch.Size((1, 5, 4)) in packed_shapes
    assert weights.shape in packed_shapes