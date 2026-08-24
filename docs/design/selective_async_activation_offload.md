# RFC: Selective Asynchronous Activation Offload in VeOmni

> **Branch:** `gulihui/async_offload`  
> **Related PR:** #1004  
> **Related Issue:** #1037

## 1. Status and Scope

**Status:** Implemented / Under review.

This RFC defines the **first-stage implementation** of **Selective Asynchronous Activation Offload** in VeOmni. The core objective is to support, without modifying model `forward` signatures or generated modeling files:

- Selecting saved tensors for offload by module class name;
- Asynchronously copying selected saved tensors to pinned CPU memory on dedicated streams;
- Backward prefetching the next selected module's activations in reverse forward order (prefetch depth = 1);
- Preserving numerical equivalence and compatibility with FSDP2 and Ulysses sequence parallelism;
- Keeping the existing `enable_activation` behavior unchanged when no module selection is configured.

**Out of scope for this stage:** operator-level selection, custom Python selectors, configurable prefetch depth, sparse/quantized/DTensor offloading, integration with `torch.compile`.

---

## 2. Motivation

VeOmni currently implements activation offloading in `veomni/distributed/offloading.py` via process-level `saved_tensors_hooks`. The selection policy is limited to:

- A device memory threshold controlled by `activation_gpu_limit`;
- A heuristic to skip linear weight transposes.

It lacks:

- Module-level selection;
- Dedicated D2H/H2D streams;
- Event-based producer/consumer synchronization;
- Backward prefetch;
- Explicit handling of repeated `unpack_hook` calls.

This proposal adds these capabilities without breaking existing configurations, and targets Qwen3.5 Dense 9B on Ascend NPU as the first validation workload.

---

## 3. Configuration Semantics and Behavior Matrix

```yaml
train:
  accelerator:
    offload_config:
      enable_activation: true
      activation_gpu_limit: 2.0
      selection:
        module_classes:
          - Qwen3_5GatedDeltaNet
      prefetch: true
  gradient_checkpointing:
    enable: false
```

| Configuration | Actual Behavior |
|---|---|
| `enable_activation: false` | No activation offloading is installed. |
| `enable_activation: true`, no `selection` | Use the existing synchronous `custom_save_on_cpu` threshold policy. |
| `enable_activation: true`, `selection` present, `gradient_checkpointing.enable: true` | Log a warning, ignore `selection` and `prefetch`, fall back to the legacy threshold path. |
| `enable_activation: true`, `selection` present, `gradient_checkpointing: false`, `torch.compile: true` | Raise an error — selective offload is not supported with `torch.compile`. |
| `enable_activation: true`, `selection` present, `gradient_checkpointing: false`, `torch.compile: false` | Enable the new `SelectiveAsyncActivationOffloadRuntime`. |

`prefetch` is intentionally placed **outside** `selection` because it controls transfer scheduling, not target selection.

- `prefetch: false`: asynchronous D2H during forward; H2D is started on demand in `unpack_hook` when backward actually needs the tensor (correct, but may introduce waits).
- `prefetch: true`: asynchronous D2H during forward, plus depth-1 forward-looking H2D prefetch during backward, overlapping H2D with backward compute.

---

## 4. Module-Level Selector

### 4.1 Selection Semantics

The selector scans the already-parallelized model by module class name and registers forward pre/post hooks on every matching instance.

- It matches against the user-visible implementation class names in the module's Python MRO; base classes like `nn.Module` and generic mixins are ignored.
- FSDP2-aware: when FSDP2 composes a module into `FSDPModule + OriginalClass`, the selector skips the dynamic FSDP wrapper and matches the original implementation class name.
- Every configured class name must match at least one module; empty, untrimmed, or unmatched names raise `ValueError`.
- If the same module instance matches multiple configured class names, it is registered only once.
- When a module instance is called multiple times in one forward, each invocation is assigned a monotonically increasing `call_id`.

Implementation: `veomni/distributed/activation_offload/config.py::resolve_module_class_selection`.

### 4.2 Nested Selection

If both a parent module and a child module are selected, a saved tensor created inside the child belongs to the **innermost active selected call**. This is implemented by maintaining a `_call_stack` of active selected call IDs and using the top of the stack in `pack_hook`.

---

## 5. Threshold Policy Reuse and Budget Semantics

### 5.1 Refactoring `_ActivationOffloadThresholdPolicy`

The decision logic inside the legacy `custom_save_on_cpu` is factored out into a reusable policy class:

```python
class _ActivationOffloadThresholdPolicy:
    def decide(self, tensor: torch.Tensor) -> OffloadPolicy:
        # returns IGNORE / KEEP_ON_GPU / OFFLOAD
```

`custom_save_on_cpu` now only handles pack/unpack execution; all decisions are delegated to the policy.

### 5.2 Hybrid Semantics in Selective Runtime

Inside `SelectiveAsyncActivationOffloadRuntime`:

- Saved tensors belonging to `selection.module_classes`: **always** use selective async offload and **do not count** against `activation_gpu_limit`.
- Saved tensors outside selected modules: fall back to `_ActivationOffloadThresholdPolicy.decide(tensor)`:
  - Within budget → `KEEP_ON_GPU`;
  - Over budget → legacy synchronous `tensor.cpu()` offload;
  - Parameters / small tensors / weight transposes → `IGNORE`.

The GPU budget is only tracked for non-selected tensors.

---

## 6. Handle State Machine and Asynchronous Protocol

### 6.1 Handle States

Each offloaded saved tensor is wrapped by an `ActivationOffloadHandle` with the following state machine:

```text
CREATED
  → OFFLOAD_QUEUED
  → HOST_READY
  → PREFETCH_QUEUED
  → DEVICE_READY
  → RELEASED
```

State transitions are driven by stream events, not by CPU polling.

### 6.2 Asynchronous D2H Protocol

```text
compute stream:  [ produce T ] [ producer_event ]
                                      │
offload stream:                       └──wait──► [ D2H T ] [ d2h_event ]
```

```python
current_stream = _current_stream(device)
producer_event.record(current_stream)
offload_stream.wait_event(producer_event)

with offload_stream:
    cpu_buffer.copy_(tensor, non_blocking=True)
    d2h_event.record(offload_stream)

tensor.record_stream(offload_stream)
```

| Mechanism | Purpose |
|---|---|
| `producer_event` | Prevents the offload stream from reading `tensor` before its producer kernel completes. |
| `tensor.record_stream(offload_stream)` | Prevents the allocator from reusing the source storage before D2H completes. |
| `d2h_event` | Prevents H2D from reading the CPU buffer before D2H finishes. |

### 6.3 Asynchronous H2D Protocol

Prefetch stage:

```text
prefetch stream:  [ wait d2h_event ][ H2D T' ][ h2d_event ]
```

Consumption stage:

```python
backward_stream.wait_event(h2d_event)
restored_tensor.record_stream(backward_stream)
```

### 6.4 Idempotency

```python
def ensure_device_resident(self, block: bool = True) -> torch.Tensor:
    if self.state == HandleState.DEVICE_READY:
        return self.restored_tensor
    if self.state == HandleState.PREFETCH_QUEUED and not block:
        return self.restored_tensor
    ...
```

- `block=False`: used by the prefetch scheduler; only submits H2D and returns immediately.
- `block=True`: used by `unpack_hook`; waits until H2D completes before returning.

### 6.5 CPU Fallback

When the device type is `cpu` or dedicated streams cannot be created, the handle transparently falls back to synchronous `.cpu()` / `.to(device)` copies, enabling CPU/Mac unit tests.

---

## 7. Backward Prefetch Scheduler

### 7.1 Trigger Timing

The runtime records the forward order of selected calls:

```text
forward:  M0 → M1 → M2 → M3
backward: M3 → M2 → M1 → M0
```

- At the start of backward, the output grad hook of the last selected call triggers prefetch for `M3`.
- When `M3`'s backward begins, its output grad hook triggers prefetch for `M2`.
- And so on.

Implementation: in `runtime.py`, the forward post-hook of each selected module registers a `register_hook` on the output tensor; during backward, the hook looks up the previous call ID in `_forward_order` and calls `_prefetch_call(prev_call_id)`.

### 7.2 Duplicate-Trigger Safety

If a parent and child module are both selected and share the same output tensor, multiple grad hooks may be registered on that tensor, potentially triggering prefetch more than once. The idempotent `ensure_device_resident` guarantees that no duplicate H2D copy is issued.

---

## 8. Trainer Integration

### 8.1 Construction Point

`BaseTrainer._build_training_context()` is changed to:

```python
def _build_training_context(self):
    self.activation_offload_runtime = build_activation_offload_runtime(
        model=self.model,
        offload_config=self.args.train.accelerator.offload_config,
        enable_gradient_checkpointing=self.args.train.gradient_checkpointing.enable,
        enable_compile=self.args.train.torch_compile.enable,
    )
    self.model_fwd_context = self.activation_offload_runtime.forward_context
    self.model_bwd_context = self.activation_offload_runtime.backward_context
```

**Important:** `_build_training_context()` must be called **after** `parallelize_model_fsdp2(model)`, otherwise FSDP2 composition may change class names and prevent matching the original implementation classes.

### 8.2 Lifecycle Cleanup

`BaseTrainer.on_train_end()` is extended to call:

```python
if getattr(self, "activation_offload_runtime", None) is not None:
    self.activation_offload_runtime.log_summary()
    self.activation_offload_runtime.close()
```

`close()` removes all module forward hooks, clears handle indexes / call stacks / forward-order lists, releases cached streams, and frees internal buffers.

`log_summary()` reports offloaded bytes, prefetch hits, on-demand restores, threshold fallback counts, and peak pinned memory.

### 8.3 `model_fwd_context` and `model_bwd_context`

- `forward_context` is a context manager that installs `saved_tensors_hooks` during model forward;
- When autograd needs to save an intermediate tensor for backward, it invokes `pack_hook`, which decides whether to keep the tensor on the accelerator or offload it to CPU;
- During backward, autograd calls the recorded `unpack_hook` to restore the tensor to the original device for gradient computation;
- When gradient checkpointing is enabled, `model_bwd_context` is still needed because checkpointing re-executes forward during backward, producing new saved tensors that must also be intercepted.

---

## 9. `build_activation_offload_runtime` Factory

This factory selects the correct runtime implementation based on configuration:

```text
enable_activation=false
   └── NullActivationOffloadRuntime

enable_activation=true
   ├── no selection
   │     └── ActivationOffloadThresholdRuntime
   ├── selection + gradient_checkpointing=true
   │     └── warning + ActivationOffloadThresholdRuntime
   ├── selection + torch.compile=true
   │     └── error
   └── selection + GC off + compile off
         └── SelectiveAsyncActivationOffloadRuntime
```

`ActivationOffloadThresholdRuntime` internally reuses the existing `build_activation_offloading_context()` to preserve legacy behavior.

---

## 10. Chunk Loss Compatibility Fix

### 10.1 Problem

`veomni/ops/kernels/cross_entropy/chunk_loss.py` originally used `torch.func.grad_and_value`. This API is a functorch functional transform and does **not** compose safely with outer `saved_tensors_hooks`. When activation offloading installs `saved_tensors_hooks(pack_hook, unpack_hook)`, calling `torch.func.grad_and_value(...)` inside `ChunkLoss.forward` either silently bypasses the hooks or raises an error, preventing offloading of chunk-CE activations.

### 10.2 Fix

Replace `torch.func.grad_and_value` with `torch.autograd.grad`, and isolate the inner short-lived graph with identity hooks so that only the final `grad_inputs` and `grad_weight` buffers are visible to the outer activation-offload hooks:

```python
with torch.enable_grad(), saved_tensors_hooks(lambda x: x, lambda x: x):
    chunk_input = hidden_states_chunk.detach().requires_grad_(True)
    chunk_weight = head_weight.detach().requires_grad_(True)
    chunk_loss, _ = loss_forward(chunk_input, chunk_weight, None, **loss_kwargs_chunks[i])
    chunk_grad_input, chunk_grad_weight = torch.autograd.grad(
        chunk_loss,
        (chunk_input, chunk_weight),
    )
```

Fix commit: `c4b61690d6208309adddd5f05dd2ac143707fdb6`.

---

## 11. Test Plan

### 11.1 Unit Tests

- Config parsing and validation;
- Module selection (MRO, FSDP2 composition, duplicate class names, unmatched-name errors);
- Threshold policy budget behavior;
- `custom_save_on_cpu` delegation to the policy;
- Factory branches (disabled / threshold / selective / GC fallback / compile rejection);
- Handle state machine and idempotent restore;
- Prefetch triggering and nested-selection attribution;
- Chunk-loss numerical equivalence after the fix.

### 11.2 Accelerator Tests

- D2H waits for the producer stream;
- Source storage is not reused before D2H completes;
- H2D waits for D2H completion;
- Backward only waits for relevant prefetch events;
- Repeated prefetch triggers issue only one H2D copy;
- No device-wide synchronizations on the steady path.

### 11.3 Qwen3.5 Integration Tests

- Single-device forward/backward equivalence on a toy Dense model;
- FSDP2 multi-device equivalence;
- Ulysses sequence parallelism with sizes 1, 2, and 4;
- Multiple gradient-accumulation micro-batches;
- Qwen3.5 Dense 9B: GDN-only, FA-only, and GDN+FA selection configurations.

---

## 12. Risks and Notes

| Risk | Notes |
|---|---|
| NPU stream/event semantics | Verify that `torch.npu.Stream/Event` behave equivalently to CUDA. |
| FSDP2 parameter prefetch contention | FSDP2's own H2D traffic may compete with activation H2D for bandwidth. |
| Pin-memory allocation failure | Large activations may OOM CPU; in this stage we fail loudly rather than silently falling back. |
| `torch.compile` | Explicitly rejected in the first stage. |
| GC + selection | Automatically falls back to the legacy path, preserving correctness. |
| Duplicate grad hooks | Safe due to the idempotent `ensure_device_resident` design. |

---

## 13. Summary

This RFC proposes a backward-compatible enhancement to activation offloading in VeOmni:

- Existing behavior is unchanged when no module selection is configured;
- Selective async offload activates only when `selection` is configured, gradient checkpointing is disabled, and `torch.compile` is disabled;
- The core offload logic is encapsulated in `ActivationOffloadHandle` with a state machine supporting asynchronous streams/events and idempotent prefetch;
- Trainers integrate through the unified `build_activation_offload_runtime` interface, and lifecycle cleanup is handled by `close()` and `log_summary()`.

The implementation has been pushed to `gulihui/async_offload` and is ready for community review.
