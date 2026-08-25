# RFC: Selective Asynchronous Activation Offload in VeOmni

> **Branch:** `gulihui/async_offload`  
> **Related PR:** #1004  
> **Related Issue:** #1037

## 1. Status and Scope

**Status:** Stage 1 and Stage 2 selective-GC integration implemented and validated on Ascend NPU.

This RFC defines the **first-stage implementation** of **Selective Asynchronous Activation Offload** in VeOmni. The core objective is to support, without modifying model `forward` signatures or generated modeling files:

- Selecting saved tensors for offload by module class name;
- Asynchronously copying selected saved tensors to pinned CPU memory on dedicated streams;
- Backward prefetching the next selected module's activations in reverse forward order (prefetch depth = 1);
- Preserving numerical equivalence and compatibility with FSDP2 and Ulysses sequence parallelism;
- Keeping the existing `enable_activation` behavior unchanged when no module selection is configured.

The proposed **second-stage hybrid mode** extends that implementation with:

- Explicit, non-overlapping module selections for non-reentrant gradient checkpointing and asynchronous activation offload;
- A shared selector schema supporting both implementation class names and logical module paths;
- Recomputation of explicitly selected GC modules while saved tensors from explicitly selected offload modules are transferred asynchronously;
- Backward-compatible behavior when `gradient_checkpointing.selection` is absent.

**Out of scope for this stage:** operator-level selection, automatic inference of the complement of an offload selection, custom Python selectors, configurable prefetch depth, sparse/quantized/DTensor offloading, reentrant GC in hybrid mode, and integration with `torch.compile`.

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

The hybrid extension targets a stricter comparison against the GC baseline. A
selective-offload-only run may retain substantially more accelerator memory than
a GC run, so a throughput comparison between those two modes is not necessarily
memory-equivalent. Hybrid mode keeps recomputation for explicitly selected
compute regions and uses offload only where transferring the required saved
tensors is expected to cost less than recomputation.

---

## 3. Configuration Semantics and Behavior Matrix

### 3.1 Stage 1: selective offload

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

### 3.2 Stage 2: selective GC plus selective asynchronous offload

The following is the hybrid configuration. The
`gradient_checkpointing.selection` and `selection.module_paths` fields were
added by Stage 2.

```yaml
train:
  gradient_checkpointing:
    enable: true
    enable_reentrant: false
    early_stop: true
    selection:
      module_paths:
        - "**.layers.*.self_attn"
        - "**.layers.*.linear_attn"
        - "**.layers.*.mlp"

  accelerator:
    offload_config:
      enable_activation: true
      activation_gpu_limit: 80
      selection:
        module_paths:
          - "**.layers.*.input_layernorm"
          - "**.layers.*.post_attention_layernorm"
      prefetch: false
```

The two selections have different meanings and are intentionally not inferred
from each other:

- `gradient_checkpointing.selection` identifies modules to recompute;
- `offload_config.selection` identifies modules whose saved tensors use
  selective asynchronous offload.

Defining GC targets as every module outside the offload selection is unsafe:
the complement contains parents, children, containers, and overlapping
checkpoint regions. Hybrid mode therefore requires explicit GC targets.

| Configuration | Target behavior |
|---|---|
| GC disabled, offload selection present | Preserve the Stage 1 selective-offload behavior. |
| GC enabled, no GC selection, no offload selection | Preserve the existing model-wide GC behavior. |
| GC enabled, no GC selection, offload selection present | Preserve the Stage 1 warning and legacy threshold fallback for backward compatibility. |
| GC enabled, GC selection present, offload disabled | Apply non-reentrant checkpointing only to the explicit GC targets. |
| GC enabled, GC selection present, threshold offload enabled without an offload selection | Apply selective GC plus the legacy threshold offload contexts. |
| GC enabled, GC selection and offload selection present | Enable hybrid selective GC plus selective asynchronous offload. |
| Hybrid mode with `enable_reentrant: true` | Raise a configuration error. |
| Selective GC, selective offload, or hybrid mode with `torch.compile: true` | Raise a configuration error. |
| Selective GC with ChunkMBS | Raise a configuration error. |

---

## 4. Module-Level Selectors

### 4.1 Stage 1 Class Selection

The selector scans the already-parallelized model by module class name and registers forward pre/post hooks on every matching instance.

- It matches against the user-visible implementation class names in the module's Python MRO; base classes like `nn.Module` and generic mixins are ignored.
- FSDP2-aware: when FSDP2 composes a module into `FSDPModule + OriginalClass`, the selector skips the dynamic FSDP wrapper and matches the original implementation class name.
- Every configured class name must match at least one module; empty, untrimmed, or unmatched names raise `ValueError`.
- If the same module instance matches multiple configured class names, it is registered only once.
- When a module instance is called multiple times in one forward, each invocation is assigned a monotonically increasing `call_id`.

Implementation: `veomni/distributed/activation_offload/config.py::resolve_module_class_selection`.

### 4.2 Shared Hybrid Selector Schema

Hybrid mode should use the same selector schema for GC and offload:

```yaml
selection:
  module_classes: []  # optional exact implementation-class names
  module_paths: []    # optional glob patterns over logical named-module paths
```

- Values within one field are ORed.
- When both fields are non-empty, a module must satisfy both the class and path
  constraints. This lets a path narrow a broad class selection.
- Every configured class name and path pattern must match at least one module;
  unmatched selectors fail before training.
- Logical paths are resolved to module identities before checkpoint wrappers or
  FSDP2 composition can alter the visible module tree. Later stages consume the
  resolved identities rather than matching paths again.

Class selection is concise when a class uniquely identifies a computation
boundary. Path selection is necessary when instances of the same class have
different roles. For example, `Qwen3_5RMSNorm` includes decoder input/post-
attention norms and the `q_norm`/`k_norm` modules nested inside attention.
Selecting the class while checkpointing the containing attention module would
silently place selected offload modules inside a GC region. The recommended
Qwen3.5 configuration therefore selects the outer decoder norms by path.

### 4.3 Nested Offload Selection

If both a parent module and a child module are selected, a saved tensor created inside the child belongs to the **innermost active selected call**. This is implemented by maintaining a `_call_stack` of active selected call IDs and using the top of the stack in `pack_hook`.

Nested offload selection is supported with on-demand restore
(`prefetch: false`). It is rejected with `prefetch: true` because overlapping
module calls do not define an unambiguous module-level backward prefetch order.

### 4.4 Hybrid Selection Validation

The first hybrid implementation uses a non-overlapping region model:

- GC targets must not be ancestors or descendants of other GC targets;
- GC and offload targets must not be ancestors or descendants of one another;
- Direct overlap between GC and offload target sets is rejected;
- A GC target shared under multiple logical paths is rejected because replacing
  only one parent edge would give the shared instance inconsistent semantics;
- Container modules and arbitrary members of `model.modules()` are not inferred
  as checkpoint targets;
- Root/container modules and modules that explicitly opt out through
  `_supports_selective_gradient_checkpointing = False` cannot be selected for
  GC. Calls carrying a live KV cache fail at the wrapper boundary.

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

### 5.2 Selective-Plus-Threshold Semantics

Inside `SelectiveAsyncActivationOffloadRuntime`:

- Saved tensors belonging to the resolved offload selection: **always** use selective async offload and **do not count** against `activation_gpu_limit`.
- Saved tensors outside selected modules: fall back to `_ActivationOffloadThresholdPolicy.decide(tensor)`:
  - Within budget → `KEEP_ON_GPU`;
  - Over budget → legacy synchronous `tensor.cpu()` offload;
  - Parameters / small tensors / weight transposes → `IGNORE`.

The GPU budget is only tracked for non-selected tensors.

### 5.3 Selective GC Interaction

In hybrid mode:

- Explicit GC targets are wrapped with non-reentrant `torch.utils.checkpoint`;
- Explicit offload targets remain outside checkpoint regions and use
  `SelectiveAsyncActivationOffloadRuntime`;
- The checkpoint's own `_checkpoint_hook` and `_recomputation_hook` are the
  innermost saved-tensor hooks for a checkpointed region, so internal tensors do
  not reach the outer selective-offload hook;
- Checkpoint boundary tensors that remain outside the internal hook use the
  non-selected threshold policy.

Hybrid checkpointing must use the default/no-op checkpoint context:

```python
checkpoint(
    target_module,
    *args,
    use_reentrant=False,
    context_fn=noop_context_fn,
    early_stop=early_stop,
    **kwargs,
)
```

It must **not** install identity `saved_tensors_hooks` through `context_fn`.
PyTorch permits only the innermost saved-tensor hook pair to run, and enters the
forward context inside `_checkpoint_hook`. An identity hook would therefore
replace the checkpoint hook, retain real forward activations, and defeat
recomputation.

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
- After an unpack returns, the handle keeps only a weak reference to the device
  copy. This permits repeated unpack while the consumer is live without keeping
  all restored activations alive until Python cyclic GC. A retained graph can
  restore the tensor again from the durable CPU copy.

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

- Entering the backward context triggers prefetch for `M3`.
- When `M3`'s first saved tensor is unpacked, the runtime triggers prefetch for `M2`.
- And so on.

Implementation: `start_backward()` submits the last selected call, then the
first `unpack_hook` for each call looks up the previous call ID in
`_forward_order` and invokes `_prefetch_call(prev_call_id)`. Scheduling from
unpack avoids output-tensor hooks retaining completed autograd graphs across
gradient-accumulation micro-batches.

### 7.2 Duplicate-Trigger Safety

Multiple tensors from the same selected call may be unpacked. The runtime marks
the call on its first unpack, and the idempotent `ensure_device_resident`
guarantees that no duplicate H2D copy is issued.

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

### 8.2 Hybrid Checkpoint Installation

When both selections are present, `build_parallelize_model()` must skip the
model-wide `model.gradient_checkpointing_enable()` call and install checkpoint
wrappers only on the resolved GC targets.

The wrapper must checkpoint the target **module call**, not a captured
`module.forward` method. Calling a captured `forward` directly during
recomputation can bypass FSDP2, TP, and ordinary module hooks. The wrapper must
also preserve state-dict keys and expose the underlying implementation class to
parallel-plan and selector logic.

The implementation subclasses PyTorch's non-reentrant `CheckpointWrapper`. In
addition to its state-dict prefix hooks, the VeOmni wrapper preserves logical
`named_modules()` / `named_parameters()` paths so model loading, LoRA matching,
optimizer grouping, and checkpoint keys do not acquire a
`._checkpoint_wrapped_module.` component.

The intended build sequence is:

1. Resolve GC and offload selectors to logical paths and module identities;
2. Apply transformations required by the module's parallel plan;
3. Install transparent non-reentrant checkpoint wrappers on explicit GC targets
   without changing parameter state-dict names;
4. Apply FSDP2 while retaining the resolved module identities;
5. Build the selective-offload runtime on the parallelized model using the
   resolved offload targets.

The exact placement relative to TP and ExtraParallel must be validated by
distributed tests. In every case, recomputation must re-enter the module-call
path that owns the relevant communication hooks.

### 8.3 Lifecycle Cleanup

`BaseTrainer.on_train_end()` is extended to call:

```python
if getattr(self, "activation_offload_runtime", None) is not None:
    self.activation_offload_runtime.log_summary()
    self.activation_offload_runtime.close()
```

`close()` removes all module forward hooks, clears handle indexes / call stacks / forward-order lists, releases cached streams, and frees internal buffers.

`log_summary()` reports offloaded bytes, prefetch hits, on-demand restores, threshold fallback counts, and peak pinned memory.

### 8.4 `model_fwd_context` and `model_bwd_context`

- `forward_context` is a context manager that installs `saved_tensors_hooks` during model forward;
- When autograd needs to save an intermediate tensor for backward, it invokes `pack_hook`, which decides whether to keep the tensor on the accelerator or offload it to CPU;
- During backward, autograd calls the recorded `unpack_hook` to restore the tensor to the original device for gradient computation;
- In the legacy threshold path with model-wide GC, `model_bwd_context` handles
  tensors saved during recomputation;
- In hybrid mode, tensors internal to an explicit GC target are owned by
  PyTorch's checkpoint/recomputation hooks and must not be selectively
  offloaded. The selective runtime continues to restore the handles created by
  offload targets outside those regions.

---

## 9. `build_activation_offload_runtime` Factory

This factory selects the correct runtime implementation based on configuration:

```text
enable_activation=false
   └── NullActivationOffloadRuntime

enable_activation=true
   ├── no selection
   │     └── ActivationOffloadThresholdRuntime
   ├── selection + GC on + no GC selection
   │     └── warning + ActivationOffloadThresholdRuntime (legacy behavior)
   ├── selection + torch.compile=true
   │     └── error
   ├── selection + GC off + compile off
   │     └── SelectiveAsyncActivationOffloadRuntime
   └── selection + explicit non-reentrant GC selection + compile off
         └── SelectiveAsyncActivationOffloadRuntime
               + selective checkpoint plan
```

`ActivationOffloadThresholdRuntime` internally reuses the existing `build_activation_offloading_context()` to preserve legacy behavior.

The activation-offload factory does not infer GC targets. The selective
checkpoint plan is built from `gradient_checkpointing.selection` and supplied
alongside the selective runtime by the training setup.

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

This identity-hook usage is local to the chunk-loss inner autograd graph. It
must not be reused as a non-reentrant checkpoint `context_fn`, where it would
override the checkpoint's own saved-tensor hooks.

---

## 11. Test Plan

### 11.1 Unit Tests

- Config parsing and validation;
- Shared class/path module selection, including FSDP2 composition, duplicate
  selectors, and unmatched-selector errors;
- Rejection of nested/overlapping GC and offload regions;
- Rejection of reentrant GC, ChunkMBS, and `torch.compile` in selective-GC mode;
- Threshold policy budget behavior;
- `custom_save_on_cpu` delegation to the policy;
- Factory branches (disabled / threshold / selective / GC fallback / hybrid /
  compile rejection);
- Handle state machine and idempotent restore;
- Prefetch triggering and nested-selection attribution;
- Hybrid execution counters: GC targets execute once in forward and once in
  recomputation, while offload targets outside those regions execute only once;
- GC-internal saved tensors do not reach the selective pack hook, while selected
  offload tensors do;
- State-dict keys are unchanged after installing selective checkpoint wrappers;
- Chunk-loss numerical equivalence after the fix.

### 11.2 Accelerator Tests

- D2H waits for the producer stream;
- Source storage is not reused before D2H completes;
- H2D waits for D2H completion;
- Backward only waits for relevant prefetch events;
- Repeated prefetch triggers issue only one H2D copy;
- No device-wide synchronizations on the steady path.
- FSDP2/TP/ExtraParallel communication hooks are re-entered correctly during
  recomputation of wrapped targets.

### 11.3 Qwen3.5 Integration Tests

- Single-device forward/backward equivalence on a toy Dense model;
- FSDP2 multi-device equivalence;
- Ulysses sequence parallelism with sizes 1, 2, and 4;
- Multiple gradient-accumulation micro-batches;
- Qwen3.5 Dense 9B: GDN-only, FA-only, and GDN+FA selection configurations.
- Qwen3.5 hybrid configuration: attention/GDN/MLP recomputation plus outer
  decoder-RMSNorm asynchronous offload.
- Compare hybrid mode with model-wide GC at the same batch/sequence shape using
  steady-state step time, peak accelerator memory, and D2H/H2D bytes.

### 11.4 Measured Hybrid Results

On two Ascend 910B2 devices with Qwen3-0.6B, FSDP2 DP2, sequence length
16,384, global/micro batch size 16/1, and six steps, a hybrid plan that
checkpoints every MLP and selectively offloads every decoder input/post-attention
RMSNorm was measured twice:

| Mode | Mean step time | Peak NPU memory |
|---|---:|---:|
| Model-wide GC | 9.96 s, 10.20 s | 8.64 GiB |
| MLP-only GC, no offload control | 8.80 s | 22.27 GiB |
| MLP-only GC + selective RMSNorm offload | 8.54 s, 8.56 s | 20.52 GiB |

The two-run means are 10.08 s for model-wide GC and 8.55 s for hybrid mode:
hybrid reduces step latency by 15.2% (17.9% higher throughput). Relative to the
same selective-GC plan without offload, selective offload reduces peak memory
by 1.75 GiB (7.9%); its roughly 0.25 s timing difference is too small to claim
as an independent speedup. The benefit is therefore a tunable memory/compute
tradeoff, not a free reduction in both time and memory: hybrid uses 11.88 GiB
more memory than model-wide GC because attention is no longer recomputed.

All modes produced losses `2.14, 2.31, 2.20, 1.97, 1.88, 1.97`. The hybrid
runtime moved 90,308,755,488 bytes in each direction per rank across six steps,
with no threshold fallback. On this NPU, `prefetch: true` did not improve this
workload, so the demonstrated configuration uses `prefetch: false`.

The RFC's original Qwen3.5-9B plan (attention/GDN/MLP GC plus outer-norm
offload) did not show a benefit at sequence length 4,096: model-wide GC took
15.93 s/step at 48.05 GiB, while hybrid took 18.19 s/step at 49.08 GiB. This is
recorded to make clear that selector quality is model- and workload-dependent;
the Qwen3-0.6B result must not be generalized to that Qwen3.5 plan.

---

## 12. Risks and Notes

| Risk | Notes |
|---|---|
| NPU stream/event semantics | Verify that `torch.npu.Stream/Event` behave equivalently to CUDA. |
| FSDP2 parameter prefetch contention | FSDP2's own H2D traffic may compete with activation H2D for bandwidth. |
| Pin-memory allocation failure | Large activations may OOM CPU; in this stage we fail loudly rather than silently falling back. |
| `torch.compile` | Explicitly rejected for selective offload, selective GC, and hybrid mode. |
| GC + offload selection without explicit GC selection | Falls back to the legacy path for backward compatibility. |
| Hybrid region overlap | Reject ancestor/descendant relationships in either direction, direct overlap, and nested GC targets. |
| Wrapper transparency | Checkpoint wrappers must preserve module-call hooks, state-dict keys, and parallel-plan matching. |
| Stateful modules | Recompute may duplicate mutation or cache updates; unsupported targets must fail validation. |
| Repeated unpack | Safe due to per-call first-unpack scheduling and the idempotent `ensure_device_resident` design. |

---

## 13. Summary

This RFC proposes a backward-compatible enhancement to activation offloading in VeOmni:

- Existing behavior is unchanged when no module selection is configured;
- Stage 1 selective async offload activates when an offload selection is
  configured, gradient checkpointing is disabled, and `torch.compile` is
  disabled;
- Stage 2 hybrid mode is explicitly enabled by providing both a non-reentrant
  `gradient_checkpointing.selection` and an `offload_config.selection`; no
  complement of either selection is inferred;
- GC and offload use the same class/path selector schema, with path selection
  used to disambiguate instances such as outer decoder norms from attention
  `q_norm`/`k_norm` modules;
- Non-reentrant checkpoint's internal saved-tensor hooks isolate recomputed
  regions from the outer selective-offload runtime; no identity checkpoint
  context is installed;
- The core offload logic is encapsulated in `ActivationOffloadHandle` with a state machine supporting asynchronous streams/events and idempotent prefetch;
- Trainers integrate through the unified `build_activation_offload_runtime` interface, and lifecycle cleanup is handled by `close()` and `log_summary()`.

The Stage 1 implementation and Stage 2 selective-GC integration live on
`gulihui/async_offload`. Accelerator-scale correctness and performance were
validated on the configurations in Section 11.4; broader model/selector tuning
remains workload-specific.
