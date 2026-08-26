# RFC: Selective Asynchronous Activation Offload in VeOmni

> **Branch:** `gulihui/async_offload`  
> **Related PR:** #1004  
> **Related Issue:** #1037

## 1. Status and Scope

**Status:** Stage 1 selective asynchronous offload and the Stage 2 checkpoint
replacement planner are implemented. Accelerator-scale validation of the new
Stage 2 policy remains pending.

Stage 1 selects module activations for asynchronous CPU offload when gradient
checkpointing (GC) is disabled. Stage 2 defines a simpler hybrid policy:

- GC enabled without an offload selection means ordinary model-wide GC;
- GC enabled with an offload selection means that selected module regions use
  activation offload instead of recomputation;
- all remaining checkpointable regions continue to use GC;
- the standard hybrid configuration requires only the offload selection;
  explicit `gradient_checkpointing.selection` remains available for backward
  compatibility and advanced layouts, but is not required for mode B;
- no complement is inferred over the raw `named_modules()` tree.

The implementation must preserve numerical equivalence and compatibility with
FSDP2 and Ulysses sequence parallelism without modifying model `forward`
signatures or patchgen-generated files.

Out of scope: arbitrary module-tree complement inference, operator-level
selection, custom Python selectors, configurable prefetch depth,
sparse/quantized/DTensor offloading, reentrant GC in hybrid mode, and
`torch.compile` integration.

---

## 2. User-Facing Memory Strategies

The design exposes three primary operating points:

| Mode | GC | Selective offload | Intended behavior |
|---|---|---|---|
| A | Disabled | Disabled | Speed upper bound. Activations remain on the accelerator and the workload may OOM. |
| B | Enabled | Enabled with `selection` | Selected regions offload saved activations instead of recomputing; remaining regions use GC. |
| C | Enabled | Disabled | Model-wide recomputation with the lowest expected activation-memory requirement. |

Mode B is a tunable memory/compute/transfer tradeoff. With a zero accelerator
residency budget it should target a peak close to mode C. A positive budget
keeps part of the selected activations on the accelerator and intentionally
moves the operating point toward mode A.

“Close to mode C” is an acceptance target, not a hard invariant. Prefetched
activations, FSDP all-gathers, allocator fragmentation, and operator workspaces
can make the measured peak differ from the pure-GC run.

---

## 3. Configuration Semantics

### 3.1 Hybrid configuration

```yaml
train:
  accelerator:
    offload_config:
      enable_activation: true
      activation_gpu_limit: 2.0
      selection:
        module_paths:
          - "**.layers.*.self_attn"
          - "**.layers.*.linear_attn"
          - "**.layers.*.mlp"
      prefetch: true
      exclude_parameter_views: true

  gradient_checkpointing:
    enable: true
    enable_reentrant: false
    early_stop: true
```

This configuration is mode B. The offload selection is an exception to the
default GC policy: matched attention, linear-attention, and MLP regions retain
their computation results through activation offload, while the activation
memory planner keeps GC on the remaining checkpoint regions.

No extra mode flag is required. The combination of
`gradient_checkpointing.enable: true` and a non-empty
`offload_config.selection` uniquely selects the replacement behavior.

### 3.2 Behavior matrix

| Configuration | Target behavior |
|---|---|
| GC disabled, activation offload disabled | Mode A. No activation-memory transformation. |
| GC enabled, activation offload disabled | Mode C. Preserve model-wide GC. |
| GC disabled, offload selection present | Stage 1 selective asynchronous offload. |
| GC enabled, offload selection present | Mode B. Replace recomputation with offload for selected regions and retain GC elsewhere. |
| Activation offload enabled without a selection | Preserve the legacy threshold offload path. |
| Hybrid mode with `enable_reentrant: true` | Raise a configuration error. |
| Selective offload or hybrid mode with `torch.compile: true` | Raise a configuration error. |

The previous behavior that warned, ignored the selection, and silently fell
back to threshold offload when model-wide GC was enabled is removed. A hybrid
configuration must either build a valid replacement plan or fail before
training.

### 3.3 Option meanings

- `selection`: modules whose saved activations replace recomputation. Class and
  path values follow the selector rules in Section 4.
- `activation_gpu_limit`: soft GiB budget for eligible saved activations from
  selected modules that may remain on the accelerator. `0` requests full
  offload of eligible selected activations; a positive value provides a memory
  compromise. It is not a total device-memory limit.
- `prefetch: false`: restore a selected tensor when autograd first requests it.
- `prefetch: true`: prefetch the next bounded activation group during a GC
  recomputation window when one is available. In Stage 1, use reverse module
  order because there are no GC windows.
- `exclude_parameter_views: true`: keep parameters and storage-sharing
  parameter views on the accelerator. Activation offload should not create
  parameter-transfer traffic.

Prefetch scheduling mode is derived from whether hybrid GC is active; users do
not need a separate `prefetch_mode` setting.

---

## 4. Selection and Checkpoint Replacement

### 4.1 Selector schema

```yaml
selection:
  module_classes: []  # optional exact implementation-class names
  module_paths: []    # optional glob patterns over logical named-module paths
```

- Values within one field are ORed.
- When both fields are non-empty, a module must satisfy both the class and path
  constraints.
- Every configured class name and path pattern must match at least one module;
  unmatched selectors fail before training.
- Logical paths are resolved to module identities before TP, ExtraParallel,
  checkpoint wrappers, or FSDP2 alter the visible module tree.
- If the same module instance matches multiple selectors, it is registered
  once.

Path selection is preferred when one implementation class has different roles.
For example, a norm implementation class may be used for outer decoder norms
and attention-internal q/k norms.

### 4.2 Why a raw module-tree complement is unsafe

The complement of an offload selection contains parents, children, containers,
shared instances, and functional operations that do not appear as standalone
modules. Checkpointing that raw complement would create nested regions or omit
parts of the original computation.

Instead, the runtime derives a validated activation-memory partition from
Transformers-style `GradientCheckpointingLayer` boundaries. For each original
GC region, the planner must:

1. keep the original region checkpointed when it contains no selected target;
2. replace the original parent checkpoint when it contains a selected target;
3. place selected targets outside checkpoint hooks so their saved tensors reach
   the selective-offload runtime;
4. checkpoint the remaining non-overlapping compute regions defined by the
   model partition;
5. preserve the original module call path, communication hooks, state-dict
   names, and parameter names.

If a selected path cannot be represented by the model's partition, plan
construction fails with the unmatched or unsupported paths. It must not degrade
to model-wide GC or threshold offload silently.

The initial implementation supports selecting a complete checkpoint boundary or
one of its direct computation children. For Qwen3.5, the sample selects the
decoder-layer mixer (`self_attn`/`linear_attn`) and MLP children. The unselected
input and post-attention norms receive transparent checkpoint wrappers, while
unaffected vision or text blocks remain whole-module checkpoint regions.

### 4.3 Region validation

- Selected offload targets must not be the root or pure container modules.
- Direct duplicate identities are deduplicated.
- Parent/child offload targets are rejected when prefetch is enabled because a
  flat reverse call order cannot represent nested backward consumption.
- Shared module instances with ambiguous logical ownership are rejected.
- Modules that opt out of activation-memory transformation are rejected.
- Calls carrying a live KV cache fail at the replacement boundary.
- Hybrid mode requires non-reentrant GC.

Nested offload-only selection remains valid with on-demand restore when the
innermost active selected call owns each saved tensor.

### 4.4 Saved-tensor hook ownership

Non-reentrant checkpoint uses internal saved-tensor hooks. Consequently, a
selected module nested inside an unchanged parent checkpoint cannot be
offloaded by an outer runtime: the checkpoint hook consumes its saved tensors
first.

The replacement plan is therefore load-bearing. It moves selected calls outside
checkpoint-owned regions, while internal tensors of the remaining checkpointed
regions continue to be owned by PyTorch's checkpoint and recomputation hooks.
Checkpoint boundary tensors retain ordinary autograd semantics.

Checkpoint contexts must not install identity `saved_tensors_hooks`. PyTorch
runs only the innermost hook pair; an identity hook inside the checkpoint would
replace its bookkeeping, retain forward activations, and defeat recomputation.

---

## 5. Accelerator Residency Budget

The Stage 2 budget applies to eligible tensors saved by selected modules:

```python
class ActivationResidencyPolicy:
    def decide(self, tensor: torch.Tensor) -> OffloadPolicy:
        # returns IGNORE / KEEP_ON_ACCELERATOR / OFFLOAD
```

- With `exclude_parameter_views: true`, parameters and storage-sharing views
  return `IGNORE` and do not consume the activation budget.
- Eligible selected tensors remain on the accelerator while the live resident
  byte count fits `activation_gpu_limit`.
- Further selected tensors use asynchronous D2H offload.
- The resident byte count is released when autograd releases the packed saved
  tensor; retained graphs remain charged to the budget.
- Unselected checkpoint-region tensors are managed by checkpoint hooks rather
  than this budget.

The limit is deliberately a soft residency budget. A tensor that crosses the
boundary, restored tensors needed by the current backward operator, and one
bounded prefetch group may temporarily exceed it. Peak process memory also
includes parameters, gradients, optimizer state, communication buffers, and
operator workspaces.

When no selection is configured, the legacy threshold implementation and its
historical semantics remain unchanged for backward compatibility.

---

## 6. Asynchronous Offload Protocol

Each offloaded saved tensor is represented by an
`ActivationOffloadHandle`:

```text
CREATED
  -> OFFLOAD_QUEUED
  -> HOST_READY
  -> PREFETCH_QUEUED
  -> DEVICE_READY
  -> RELEASED
```

### 6.1 D2H

```text
compute stream:  [ produce T ] [ producer_event ]
                                      |
offload stream:                       +--wait--> [ D2H T ] [ d2h_event ]
```

- The producer event prevents D2H from reading an incomplete tensor.
- `tensor.record_stream(offload_stream)` prevents allocator reuse before D2H
  completes.
- The D2H event prevents H2D from reading an incomplete host buffer.

### 6.2 H2D and consumption

```text
prefetch stream: [ wait d2h_event ] [ H2D T' ] [ h2d_event ]
backward stream:                                  [ wait ] [ consume T' ]
```

The restored tensor records the consumer stream. NPU uses the platform-specific
event synchronization required by torch-npu; CUDA keeps the non-blocking stream
wait path.

### 6.3 Idempotency and lifetime

- Prefetch submission is idempotent.
- Repeated unpack is supported.
- After consumption, the handle keeps only the durable CPU copy and a weak
  reference to any restored device tensor.
- A retained graph may restore the tensor again from the CPU copy.
- With prefetch disabled, the runtime does not retain packed handles after their
  owning backward call starts.

CPU execution or unavailable dedicated streams use a synchronous fallback for
unit testing.

---

## 7. Backward Prefetch

Hybrid prefetch uses recomputation as the overlap window:

```text
backward: [ recompute GC region N ] [ backward offload region N-1 ]
                    +-- prefetch activation group for N-1 --+
```

The planner records the forward order of selected calls and checkpoint
boundaries. At backward time, entry into a checkpoint recomputation context may
submit at most the next bounded group. The first unpack of a selected call
retires that group and advances scheduling.

Stage 1 has no checkpoint boundary, so it starts with the final selected call
and advances in reverse forward order on the first unpack for each call.

Prefetch reduces host-visible H2D waits only when the recomputation window is
long enough and transfer does not interfere with FSDP communication. It is a
workload-tuned optimization, not a guaranteed throughput improvement.

---

## 8. Trainer and FSDP2 Integration

The build order is:

1. Resolve offload selectors against logical paths and module identities.
2. When GC and selection are both enabled, build the model-specific checkpoint
   replacement plan. Do not call model-wide
   `model.gradient_checkpointing_enable()` for the same regions.
3. Apply tensor parallelism, then transformations required by the
   ExtraParallel plan.
4. Install transparent non-reentrant checkpoint wrappers for the plan's GC
   regions without changing state-dict or parameter names.
5. Apply FSDP2 while retaining resolved identities.
6. Install selective-offload hooks directly on the resolved offload targets.
7. Build forward/backward runtime contexts and lifecycle cleanup.

The checkpoint wrapper must checkpoint the target module call, not a captured
`module.forward` method. Directly invoking a captured method during
recomputation can bypass FSDP2, TP, and normal module hooks.

`BaseTrainer`, `TextDPOTrainer`, `DiTTrainer`, and any composed trainer that
overrides forward/backward lifecycle handling must all start and finish the
activation-offload runtime. Unsupported trainer families must reject hybrid
mode explicitly.

---

## 9. Runtime Selection

```text
enable_activation=false
  -> NullActivationOffloadRuntime

enable_activation=true, no selection
  -> ThresholdActivationOffloadRuntime

selection, GC disabled, compile disabled
  -> SelectiveAsyncActivationOffloadRuntime (Stage 1)

selection, GC enabled, non-reentrant, compile disabled
  -> checkpoint replacement plan
     + SelectiveAsyncActivationOffloadRuntime (Stage 2 / mode B)

selection with unsupported GC partition, reentrant GC, ChunkMBS, or compile
  -> configuration error before training
```

The factory never ignores a configured selection. In hybrid mode it receives
the already-resolved offload targets and checkpoint replacement plan from the
training setup.

---

## 10. Compatibility and Migration

This Stage 2 definition intentionally changes the meaning of one previously
accepted combination:

- Previously, GC enabled plus an offload selection but no explicit GC selection
  logged a warning and ignored the selection.
- After Stage 2 is implemented, the same configuration activates checkpoint
  replacement hybrid mode.

`gradient_checkpointing.selection` remains supported for compatibility with
existing explicit non-overlapping plans, but it is no longer required for the
standard hybrid configuration. Existing configurations without any selection
retain their legacy behavior.

---

## 11. Chunk Loss Compatibility

Chunk loss must use `torch.autograd.grad`, not `torch.func.grad_and_value`,
because functorch functional transforms do not compose safely with outer
saved-tensor hooks. Its inner short-lived graph may use identity hooks so only
the final gradient buffers are visible to the outer runtime.

This identity-hook use is local to chunk loss. It must not be reused as a
non-reentrant checkpoint context, where it would replace checkpoint bookkeeping.

---

## 12. Test Plan

### 12.1 Configuration and planning

- Parse the single-selector hybrid configuration.
- Verify GC plus selection activates replacement mode instead of legacy fallback.
- Preserve explicit `gradient_checkpointing.selection` plans for compatibility.
- Resolve class/path selectors before wrappers and FSDP composition.
- Reject unmatched, shared, nested-prefetch, and unsupported-partition targets.
- Reject reentrant GC, ChunkMBS, and `torch.compile` in hybrid mode.
- Preserve ordinary model-wide GC when no selection is present.

### 12.2 Correctness

- Selected targets execute once and restore saved activations during backward.
- Remaining GC targets execute once in forward and once in recomputation.
- Selected tensors reach the offload pack hook; checkpoint-internal tensors do not.
- State-dict, named-parameter, optimizer-group, and LoRA matching keys are unchanged.
- Losses and gradients match mode C within the established numerical tolerance.
- FSDP2, TP, ExtraParallel, and Ulysses communication hooks are re-entered correctly.
- Multiple gradient-accumulation micro-batches do not retain completed graphs or handles.

### 12.3 Memory and performance

- Measure peak accelerator memory for A, B with limit 0, B with a positive
  limit, and C at the same batch and sequence shape.
- Report selected resident bytes, D2H/H2D bytes, prefetch hits, on-demand
  restores, peak pinned memory, and threshold fallback counts.
- Verify no device-wide synchronization on the steady path.
- Compare prefetch enabled/disabled; do not claim a win below repeat-to-repeat noise.
- Treat B's memory proximity to C as a measured acceptance threshold, not an API guarantee.

---

## 13. Existing Evidence and Risks

The existing Qwen3.5-9B experiments evaluated the previous explicit selective-GC
design, not the replacement semantics in this RFC. On four Ascend 910B2 devices
at sequence length 4,096, model-wide GC measured 16.44 s/step and 48.05 GB peak,
while the previous hybrid-prefetch plan measured 20.82 s/step and 49.08 GB.
Broad Attention/GDN or MLP offload also generated tens of GB of transfers per
step and was slower than GC.

These results motivate the simpler interface but do not validate the sample
selector as a performance win. The new mode B must be benchmarked independently.
Selectors should maximize recomputation FLOPs avoided per transferred byte, and
prefetch should remain workload-tuned.

Key risks:

| Risk | Required handling |
|---|---|
| Unsupported model partition | Fail before training; never infer a raw module-tree complement. |
| Peak memory above mode C | Bound residency and prefetch groups; validate with measured peak memory. |
| FSDP communication contention | Trace H2D and collectives together; disable prefetch when it increases idle time. |
| Parameter-view traffic | Exclude parameters and storage-sharing views from activation offload. |
| Pinned-host-memory exhaustion | Track peak pinned bytes and fail loudly on allocation failure. |
| Stateful modules or live KV cache | Reject unsupported targets during planning or at the wrapper boundary. |
| `torch.compile` | Reject selective and hybrid modes until hook/wrapper composition is supported. |

---

## 14. Summary

The Stage 2 public contract is deliberately small:

- GC enabled without an offload selection is mode C;
- GC enabled with an offload selection is mode B;
- selected regions use activation offload instead of recomputation;
- remaining supported regions continue to use GC;
- `activation_gpu_limit` controls the selected activation residency tradeoff;
- no public GC selector or replacement mode flag is required;
- the runtime must construct a validated checkpoint replacement plan or fail
  before training.
