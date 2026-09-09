# Qwen4-Exp integration status

This directory is an internal-validation integration for the experimental
`qwen4_exp` model from Transformers commit
`93d1bcfbd2af5798e2f66bf7955e31a537902b64`.

Supported in this correctness stage:

- GPU and NPU VLM supervised fine-tuning with Ulysses sequence parallelism and
  `cp_size=1` for short-sequence validation.
- GatedDeltaNet sequence-to-head all-to-all using the existing Qwen3.5
  causal-conv and gated-delta kernels.
- Global QSA selection plus main-attention Q/K/V Ulysses exchange. The sole
  `qsa_attention_implementation=eager` path uses local block pooling, compact
  global token indices, and a memory-bounded sparse PyTorch implementation.
  Explicit cu-seqlens isolate packed samples consistently for SP=1 and SP>1.
- PLE n-gram token halos and differentiable dilated-convolution halos across
  Ulysses shard boundaries.
- VeOmni fused cross-entropy and fused MoE dispatch, with global router-logit
  statistics for SP-consistent auxiliary load-balancing loss.
- Concurrent PLE and MoE expert parallelism: PLE tables use the persistent
  two-dimensional `ple_fsdp × ple` layout while expert tensors use the
  independent `ep_fsdp × ep` layout.
- Scalable loading and training of the released ~95 GiB PLE table:
  - preserve its 128 checkpoint-native embedding shards instead of
    concatenating them;
  - persistently shard every table by rows over `ple` and columns over the
    complementary `ple_fsdp` dimension;
  - stream only the local row-by-column rectangle and zero-pad final rows;
  - keep PLE weights outside FSDP2, eliminating their forward/backward
    parameter all-gathers;
  - route lookup requests and differentiable column slices over the flattened
    PLE mesh, so every rank may process a different sample.
- Explicitly discarding `mtp.*` checkpoint tensors; MTP loss is not supported.

The internal example uses `ple_size=8`, `ep_size=4`,
`broadcast_model_weights_from_rank0=false`, and
`ep_sharded_stream_load=true`. The world size and every padded PLE shard row
count must be divisible by `ple_size`; the number of experts must be divisible
by `ep_size`. PLE and EP may be enabled together because their parameter sets
and communication meshes are independent.

The implementation details and constraints of this Qwen4-Exp-only local
parallel path are documented in
[Qwen4-Exp PLE Two-Dimensional Parallelism](../../../../docs/design/qwen4_exp_ple_2d_parallelism.md).

Known limitations:

- Context parallelism, cache prefill/decode, HSDP replicas of persistent PLE,
  and hybrid CP × Ulysses remain unsupported.
- A fused production QSA kernel is not integrated. The compact PyTorch backend
  removes the quadratic mask and attention-score allocation. Its custom
  autograd path saves only the original Q/K/V tensors and compact indices,
  then recomputes and scatters one gathered K/V chunk at a time in backward.
  It still has not passed the 16K performance gate, so the 16K example keeps
  `ulysses_size=1` pending profiling.
- Ulysses GatedDeltaNet requires non-eager, varlen-capable causal-conv and
  chunk gated-delta-rule kernels. Head counts must be divisible by the
  Ulysses size; QSA KV heads may instead divide the Ulysses size (MQA/GQA
  replication case).
- PLE's dilated convolution preserves the upstream SP=1 behavior and does not
  reset at packed-sample boundaries. Changing that semantic requires a
  separate SP=1 model change and checkpoint-level validation.
- Distributed PLE training expects pretrained or DCP weights. Initializing from
  scratch after PLE parameters become DTensors is not supported by the upstream
  Hugging Face initializer.
- The real checkpoint schema and shard reads plus two-process lookup/backward
  semantics are validated without materializing the 95 GiB table. A real
  multi-node load-and-train smoke run remains the final deployment gate.

## Two-device pipeline regression

Run the daily toy regression with:

```bash
pytest -s tests/e2e/test_qwen4_exp_pipeline.py
```

On a host with at least two CUDA or NPU devices, the test uses VLM dummy data,
`ple_size=2`, and `ep_size=2` to cover FSDP2 streaming load, fused MoE
forward/backward, PLE and expert AdamW updates, DCP save/resume, finite loss,
and non-zero PLE/expert gradients. It does not load the released 95 GiB PLE
table.
