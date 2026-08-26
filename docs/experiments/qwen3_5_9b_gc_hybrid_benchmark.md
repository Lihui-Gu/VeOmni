# Qwen3.5-9B GC and Selective-Offload Benchmark

This document was created at 2026-08-25 20:50 CST, before the confirmatory
benchmark started at 20:51 CST. It records the pre-registered comparison,
commands, raw-output locations, analysis method, and conclusion. Preliminary
runs are reported separately and are not mixed into the final aggregate.

## Record status

| Field | Value |
|---|---|
| Pre-registration | completed before the confirmatory runs |
| Confirmatory run | completed |
| Source commit | `ee7587920238bf6a98a6b8f784ce90e61ee8f241` |
| Benchmark script SHA-256 | `2a09d364ae0566164deb6cde451889cfe1d505d2296fa52006a5e5b93de5ed77` |
| Run date | 2026-08-25 CST |

The commit and script hash above apply only to the original confirmatory run.
The exploratory follow-up below used an evolving, uncommitted worktree while
selectors were narrowed; its resolved arguments are preserved in each run's
`veomni_cli.yaml`, but it is not a frozen confirmatory benchmark.

## Objective

Measure which activation-memory strategy gives the best usable step latency on
four Ascend 910B2 NPUs, while checking peak device memory and numerical
consistency. A secondary objective is to determine whether backward activation
prefetch is functioning correctly and whether its transfer overlap improves
end-to-end latency.

## Environment and fixed workload

| Field | Value |
|---|---|
| Code under test | `ee7587920238bf6a98a6b8f784ce90e61ee8f241` plus the benchmark script in this worktree |
| Accelerator | 4 x Ascend 910B2C |
| Scheduler-visible physical devices | `3,5,6,7` |
| Process-local logical devices | `0,1,2,3` |
| Model | `Qwen/Qwen3.5-9B` |
| Dataset | `tulu-3-sft-mixture/data` |
| Sequence length | 4,096 |
| Global / micro batch size | 16 / 1 |
| FSDP data-shard size | 4 |
| GDN implementation | `npu_ascendc` |
| Seed | 42 |
| Full determinism | enabled |
| Per-step device synchronization | enabled |
| Warmup batch-size ratio | 0 |
| Confirmatory repetitions | 3 per runnable mode |
| Steps per repetition | 20 |

`ASCEND_VISIBLE_DEVICES=3,5,6,7` is supplied by the scheduler. The launcher
must receive `NPU_DEVICES=0,1,2,3`, because those are the logical IDs visible
inside the allocation. Passing the physical IDs causes `Invalid device ID` on
ranks 1-3.

## Modes

| Mode | Gradient checkpointing | Selective activation offload | Prefetch |
|---|---|---|---|
| `upper` | disabled | disabled | disabled |
| `gc` | model-wide | disabled | disabled |
| `selective-gc` | attention, GDN, and MLP modules | disabled | disabled |
| `hybrid` | attention, GDN, and MLP modules | outer decoder RMSNorm modules | disabled |
| `hybrid-prefetch` | attention, GDN, and MLP modules | outer decoder RMSNorm modules | enabled |

`compare` runs `gc`, `selective-gc`, and `hybrid`. The base order for `all`
adds `upper` at the beginning and `hybrid-prefetch` at the end; repetitions
rotate that order. These names are suites, not separate memory strategies.

## Pre-registered procedure

1. Keep the model, data order, parallel layout, batch sizes, kernels, seed, and
   determinism setting identical across modes.
2. Run 20 optimizer steps for each mode and repeat each runnable mode three
   times. Rotate the order of modes across repetitions where practical.
3. Treat `upper` as a feasibility bound. If it OOMs, record the failure and do
   not include its partial timing in the ranking.
4. Use the final tqdm elapsed time divided by 20 as the primary end-to-end
   latency. Also derive a steady-state latency with the first step excluded.
5. Report mean, sample standard deviation, minimum, and maximum across the
   three repetitions. Report nominal throughput as `65536 / latency` tokens/s.
6. Compare every mode with model-wide `gc` and compare `hybrid-prefetch`
   directly with `hybrid`.
7. Require identical per-step loss sequences across modes. A mismatch blocks a
   performance recommendation.
8. Record rank-0 lifecycle `max_memory_allocated` as the peak-memory metric.
   It is not an aggregate across ranks.

The fastest usable mode is the one with the lowest three-run mean latency. A
difference below 2% or one that is not repeatable across runs is treated as a
tie rather than a meaningful win.

## Reproduction command

```bash
source /app/.venv/bin/activate

MODEL_PATH=/opt/tiger/gulihui/VeOmni/Qwen/Qwen3.5-9B \
DATA_PATH=/opt/tiger/gulihui/VeOmni/tulu-3-sft-mixture/data \
NPU_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
MAX_STEPS=20 REPEATS=3 SEED=42 FULL_DETERMINISM=true \
RUN_TAG=qwen35_compare_20x3_deterministic_20260825 \
bash scripts/profile/run_qwen3_5_gc_vs_hybrid.sh compare

MODEL_PATH=/opt/tiger/gulihui/VeOmni/Qwen/Qwen3.5-9B \
DATA_PATH=/opt/tiger/gulihui/VeOmni/tulu-3-sft-mixture/data \
NPU_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
MAX_STEPS=20 REPEATS=3 SEED=42 FULL_DETERMINISM=true \
RUN_TAG=qwen35_prefetch_20x3_deterministic_20260825 \
bash scripts/profile/run_qwen3_5_gc_vs_hybrid.sh hybrid-prefetch
```

Raw logs and summaries are written below
`output/qwen3_5_gc_vs_hybrid/repro_<RUN_TAG>/`.

The two confirmatory output roots are:

- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_compare_20x3_deterministic_20260825/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_prefetch_20x3_deterministic_20260825/`

Each root contains a `manifest.txt`, one complete log and resolved
`veomni_cli.yaml` per run, and a concise `summary.txt`. The manifests record
`seed=42` and `full_determinism=true`.

## Preliminary observations (not final aggregate)

A six-step pilot used the same workload except that full determinism was not
explicitly enabled:

| Mode | End-to-end latency | Peak device memory | Status |
|---|---:|---:|---|
| `upper` | not reportable | more than 58.40 GiB allocated before failure | OOM entering step 2 |
| `gc` | 16.28 s/step | 48.05 GB | completed |
| `selective-gc` | 20.47 s/step | 51.08 GB | completed |
| `hybrid` | 19.70 s/step | 49.08 GB | completed |
| `hybrid-prefetch` | 20.49 s/step | 49.08 GB | completed |

The one-run prefetch regression is not sufficient evidence of an implementation
problem. A controlled two-step NPU trace showed equal transfer work in both
hybrid modes (about 606 ms of H2D per two steps), while prefetch reduced host
`aclrtSynchronizeEvent` time from 686.6 ms to 4.5 ms. Device compute time was
also effectively unchanged. The trace therefore confirms that prefetch is
submitted early and hides the H2D wait; larger FSDP/HCCL variation can dominate
short end-to-end runs.

Trace directories:

- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_profile_hybrid_20260825/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_profile_hybrid_prefetch_20260825/`

## Confirmatory results

Status: completed successfully. No measured mode OOMed, and no cross-step memory
growth was observed. The `upper` bound was not repeated because the pilot had
already established that it OOMs while entering step 2 after allocating more
than 58.40 GB.

The primary latency is the closing tqdm `s/it` value. `SD` is the sample
standard deviation across three independent process launches. Steady-state
latency excludes the first step and is reconstructed from the displayed final
and first-step rates. Nominal throughput is `65,536 / mean latency`; it is a
fixed-shape comparison metric rather than a count of unmasked loss tokens.

| Mode | Run 1 | Run 2 | Run 3 | Mean +/- SD | Min-max | Steady-state mean | Nominal tokens/s | Peak memory | Latency vs `gc` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gc` | 16.40 | 16.48 | 16.44 | 16.44 +/- 0.04 | 16.40-16.48 | 16.60 | 3,986 | 48.05 GB | baseline |
| `selective-gc` | 21.62 | 21.64 | 21.64 | 21.63 +/- 0.01 | 21.62-21.64 | 22.10 | 3,029 | 51.08 GB | +31.59% |
| `hybrid` | 21.03 | 20.53 | 21.07 | 20.88 +/- 0.30 | 20.53-21.07 | 21.25 | 3,139 | 49.08 GB | +26.99% |
| `hybrid-prefetch` | 21.08 | 20.51 | 20.86 | 20.82 +/- 0.29 | 20.51-21.08 | 21.19 | 3,148 | 49.08 GB | +26.62% |

All latency values are seconds per optimizer step. Relative to `gc`, nominal
throughput is 24.01% lower for `selective-gc`, 21.25% lower for `hybrid`, and
21.02% lower for `hybrid-prefetch`.

For scale only, two-sided 95% Student-t confidence intervals for the mean are
16.34-16.54 (`gc`), 21.60-21.66 (`selective-gc`), 20.13-21.62 (`hybrid`), and
20.10-21.53 s/step (`hybrid-prefetch`). With only three repeats, these intervals
are descriptive rather than strong inferential evidence. They clearly separate
`gc` from the selective strategies, while the two hybrid intervals almost
completely overlap. The prefetch runs were performed as the immediately
following suite rather than interleaved with hybrid, so sub-percent differences
must also be treated as susceptible to temporal system drift.

### Total elapsed time

The training-loop totals below are the displayed tqdm elapsed times. The
process-log spans run from the first to last timestamped log record and include
model/data setup and teardown; they are approximate because log timestamps
have one-second resolution and do not cover launcher time before the first
record.

| Mode | 20-step loop, run 1 / 2 / 3 | Sum of training loops | Process-log spans, run 1 / 2 / 3 | Sum of log spans |
|---|---|---:|---|---:|
| `gc` | 05:27 / 05:29 / 05:28 | 16:24 | 06:07 / 06:18 / 06:08 | 18:33 |
| `selective-gc` | 07:12 / 07:12 / 07:12 | 21:36 | 07:51 / 07:53 / 07:59 | 23:43 |
| `hybrid` | 07:00 / 06:50 / 07:01 | 20:51 | 07:46 / 07:28 / 07:43 | 22:57 |
| `hybrid-prefetch` | 07:01 / 06:50 / 06:57 | 20:48 | 07:45 / 07:31 / 07:37 | 22:53 |
| **All 12 runs** | - | **1:19:39** | - | **1:28:06** |

### Numerical consistency

The rank-0 loss sequence is identical across all 12 runs at the precision
written by the training log:

```text
0.88, 0.93, 1.16, 0.96, 0.93, 0.79, 1.04, 0.96, 0.91, 0.77,
0.93, 0.97, 0.69, 0.97, 0.75, 0.79, 0.66, 0.58, 0.79, 0.67
```

The logged gradient-norm sequence is also identical. This checks deterministic
equivalence to the log's two-decimal precision; the logs do not serialize
enough digits to claim bitwise identity.

### Prefetch diagnosis

The confirmatory mean for `hybrid-prefetch` is 0.06 s/step (0.29%) lower than
for `hybrid`. The per-repetition differences are +0.05, -0.02, and -0.21
s/step, while the across-run standard deviations are about 0.30 s/step. The
effect is therefore smaller than run-to-run noise and far below the
pre-registered 2% threshold: prefetch and non-prefetch are tied for this
workload.

This does not indicate a broken prefetch implementation. In the controlled
trace, both variants made 3,072 event synchronization calls and 3,242 async
memcpy calls. Enabling prefetch reduced total `aclrtSynchronizeEvent` host time
from 686.6 ms to 4.5 ms while leaving the amount of H2D work effectively
unchanged. Runtime counters also report all 15,360 restored tensors as
prefetch hits in every confirmatory prefetch run. The implementation is moving
the wait earlier as intended, but that saved wait is too small relative to
compute and FSDP/HCCL variability to produce a material end-to-end win on four
910B2 devices.

### Conclusion

Use model-wide `gc` for Qwen3.5-9B under this exact 4 x 910B2, sequence-length
4,096 workload. It is 4.38 s/step faster than `hybrid-prefetch` and also uses
1.03 GB less peak rank-0 device memory. The current attention/GDN/MLP selector
is not suitable for this model: its reduced checkpoint coverage does not
translate into lower end-to-end latency and it materially raises peak memory.

Keep activation prefetch disabled for this configuration. It is functioning
correctly and is safe to test for another selector or workload, but the
present 0.29% difference is not a meaningful performance benefit.

These conclusions are scoped to the exact model, selector, sequence length,
batch sizes, four-device topology, and software commit recorded above. They do
not establish that selective GC or prefetch is generally slower on other
models or configurations.

## Follow-up: prioritize expensive recomputation

An exploratory follow-up tested the proposed inverse split: retain or offload
MLP activations so that the dense MLP GEMMs do not run again, while continuing
to checkpoint the attention/GDN path. Broad module selection initially copied
saved FSDP parameter views as well as activations. The new opt-in
`exclude_parameter_views` policy recognizes storage shared with parameters and
keeps those tensors on the accelerator.

The filter worked on FSDP2/NPU: for the two-step Attention/GDN-offload probe it
skipped 1,600 parameter views (33.40 GB total), reduced D2H traffic from 190.18
GB to 156.78 GB, and reduced latency from 29.50 to 26.21 s/step. Correctness was
unchanged (`0.88, 0.93` loss), but this remained much slower than `gc`.

The search then narrowed offload to the MLP tensors with the highest expected
recompute-to-transfer value. These are exploratory two-step probes unless the
row says otherwise; their short-run times are useful for rejecting designs,
not for claiming small wins.

| Strategy | D2H bytes / step | Latency | Peak memory | Result |
|---|---:|---:|---:|---|
| Whole Attention/GDN offload + MLP GC, parameter views filtered | 78.39 GB | 26.21 s/step | 50.39 GB | too much transfer |
| Whole MLP offload + Attention/GDN GC, parameter views filtered | 59.96 GB | 26.47 s/step | 50.47 GB | too much transfer |
| MLP projection/activation inputs offload + Attention/GDN GC | 34.26 GB | 21.41 s/step | 56.12 GB | runnable, still slower |
| Layers 0-23 full GC; layers 24-31 mixer GC + MLP input offload | 8.57 GB | 17.54 s/step | 50.12 GB | closest two-step candidate |
| `gc`, contemporaneous six-step control | 0 | 15.74 s/step | 48.05 GB | fastest |
| Layers 0-15 full GC; layers 16-31 mixer GC; last four `down_proj` inputs prefetched (six steps) | 1.61 GB | 19.05 s/step | 55.25 GB | slower than `gc` |
| Same selective checkpoint split without offload (six steps) | 0 | 19.11 s/step | 55.58 GB | tied with prefetch within noise |

The final narrow plan demonstrates that prefetch can hide a small transfer:
offloading 1.61 GB/step reduced peak memory by 0.33 GB and changed latency by
only -0.06 s/step relative to its no-offload control. It nevertheless remained
3.31 s/step (21.0%) slower than the contemporaneous model-wide `gc` run. Thus,
under this workload there is still no Qwen3.5-9B `hybrid-prefetch` example that
beats `gc`, even after prioritizing expensive MLP recomputation and filtering
parameter views. The practical recommendation remains model-wide `gc`.

Follow-up raw outputs:

- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_mixer_prefetch_probe_20260825/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_mixer_prefetch_paramfilter_probe_20260825/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_mlp_prefetch_paramfilter_probe_20260825/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_mlp_compute_prefetch_probe_20260825/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_layer_hybrid_quarter_mixer_gc_prefetch_probe_20260825/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_layer_hybrid_half_down4_prefetch_probe_20260825/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_layer_hybrid_down4_compare6_20260825/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_layer_partial_gc6_20260825/`

### Why the 12.89-second first step was misleading

The two-step last-four-layer probe showed 12.89 seconds for its first displayed
step, 17.53 s/step for the complete run, 1.61 GB of D2H traffic per step, and a
55.25 GB peak. The first number is not evidence that this plan can run at
12.89 s/step in steady state.

First, this is a timing-boundary effect. With `sync_each_train_step=true`, the
trainer synchronizes at the beginning of a step, after the step timer has
started, but does not synchronize immediately before the tqdm update at the
end. Consequently, asynchronous tail work from step N can be charged to step
N+1. Step 1 has no preceding tail to absorb, so it is systematically shorter.
The second interval implied by the two displayed values is approximately
`2 * 17.53 - 12.89 = 22.17` seconds. The first batch was not a reduced warmup
batch: `bsz_warmup_steps=0`, and the logged per-rank packed lengths were
4,087-4,095 tokens against a 4,096-token limit.

Second, the same pattern appears in all contemporaneous six-step controls. The
first/final-average pairs were 13.02/15.74 seconds for model-wide `gc`,
13.08/19.05 seconds for the same last-four-layer prefetch plan, and 12.48/19.11
seconds for its no-offload selective-GC control. This shows that the unusually
fast first step is a general measurement artifact, with possible additional
data-dependent variation from packed sequence boundaries, rather than a
hybrid-prefetch speedup. Comparisons should use a multi-step cumulative time
after discarding the first step, or profiler step boundaries with explicit
end-of-step synchronization.

## Follow-up: interleave full GC and retained MLP computation

The next experiment created a regular backward pipeline. Three of every four
decoder layers used full-layer checkpointing. In the remaining layers, only
the existing `linear_attn` module was checkpointed, while the saved compute
inputs of `gate_proj`, `act_fn`, `up_proj`, and `down_proj` were offloaded. This
retains the expensive MLP computation and places three full-layer recompute
windows before the next offloaded MLP activation is consumed in backward.

All rows below used the same deterministic six-step workload. D2H volume is
8.55 GB/step for every interleaved row. All logged loss and gradient-norm
sequences are identical to model-wide `gc`.

| Strategy | Latency | Peak memory | Result |
|---|---:|---:|---|
| Model-wide `gc` | 15.79 s/step | 48.05 GB | fastest baseline |
| Interleaved, on-demand H2D | 18.42 s/step | 50.08 GB | +16.7% versus `gc` |
| Interleaved, module-order prefetch | 22.79 s/step | 50.12 GB | slower than on-demand |
| Prefetch a group at the GC recompute boundary | 23.23 s/step | 50.08 GB | slower than on-demand |
| Prefetch a group after FSDP unshard | 22.61 s/step | 50.08 GB | slower than on-demand |
| Stagger group prefetch after FSDP unshard | 22.40 s/step | 50.12 GB | slower than on-demand |

The GC-boundary mechanism did what it was designed to do: H2D was submitted
during checkpoint recomputation, and host `aclrtSynchronizeEvent` wait almost
disappeared. However, a controlled steady-state trace shows why that did not
improve the step:

| Rank-0 trace metric | On-demand H2D | GC-boundary prefetch |
|---|---:|---:|
| Step duration | 19.15 s | 22.14 s |
| Device compute | approximately 7.63 s | approximately 7.63 s |
| Memcpy work | approximately 599 ms | approximately 599 ms |
| Host `aclrtSynchronizeEvent` | 328 ms | 0.4 ms |
| Unoverlapped communication | 2.56 s | 5.93 s |
| Total communication | 6.29 s | 9.87 s |

Three FSDP all-gathers that took 11.8-59.5 ms in the on-demand trace expanded
to 0.74-1.68 seconds with GC-boundary prefetch. This is not evidence that the
HCCS link itself became slower. The aggregate all-gather HCCS payload was
identical at 105,382.65 MB, and active HCCS transit time was effectively
unchanged at 6,005 versus 5,987 ms. Instead, the profiler classified the
all-gather elapsed-time increase, from 2,379 to 5,560 ms, as idle time.

The early H2D copies were therefore accompanied by collective readiness or
device-scheduling delay rather than slower HCCS wire throughput. At this stage,
the rank-0-only trace could not distinguish cross-rank arrival skew from device
queueing or endpoint resource contention. Operationally, the reduction in
`aclrtSynchronizeEvent` time was accompanied by increased collective idle time,
and the step became roughly three seconds longer.

### Four-rank trace follow-up

Four-rank traces were then collected over the same profiler window for the
on-demand and GC-boundary-prefetch variants:

- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_interleaved_noprefetch_allranks_20260826/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_interleaved_gc_prefetch_allranks_20260826/`

The profiler step took 20.859 seconds without prefetch and 23.699 seconds with
GC-boundary prefetch. Every rank transferred the same 105,382.652 MB of
all-gather HCCS payload in both runs. Rank-0 summed HCCS task time was also
effectively unchanged at 5,980.639 versus 5,958.675 ms. The additional time is
therefore not a reduction in HCCS link throughput.

The root FSDP all-gathers with operation sequences 2, 235, 468, and 701 mark
the four gradient-accumulation microbatches. Sequence 468 isolates the new
prefetch delay:

| Sequence 468 metric | On-demand | GC-boundary prefetch |
|---|---:|---:|
| Host enqueue skew | 18.020 ms | 1,596.058 ms |
| Rank elapsed `[0, 1, 2, 3]` | `[77.517, 66.992, 59.528, 69.383]` ms | `[1654.748, 59.033, 1648.178, 95.772]` ms |
| Enqueue-to-device-start `[0, 1, 2, 3]` | `[1.741, 1.656, 1.707, 1.747]` ms | `[1.718, 1.375, 1.841, 1.627]` ms |
| Pre-HCCS idle `[0, 1, 2, 3]` | `[18.309, 7.779, 0.323, 10.175]` ms | `[1596.041, 0.325, 1589.470, 37.064]` ms |
| Summed HCCS task time `[0, 1, 2, 3]` | `[167.258, 173.608, 162.129, 161.798]` ms | `[165.834, 172.095, 161.412, 161.804]` ms |

The enqueue-to-device-start interval remains below 2 ms on every rank. Ranks 0
and 2 enter the collective approximately 1.6 seconds before ranks 1 and 3 and
spend that difference in pre-HCCS notify waits. At sequence 701 the waiting
pair reverses: ranks 1 and 3 wait approximately 1.665 seconds for ranks 0 and
2. The on-demand baseline already has a 1.677-second skew at sequence 701, but
not the additional sequence-468 skew. GC-boundary prefetch therefore adds a
second large synchronization bubble rather than slowing HCCS transfers.

The interval preceding sequence 468 shows rank-dependent allocator and device
backlog:

| Rank | Interval, on-demand | Interval, prefetch | `aclrtFreePhysical`, prefetch | `aclrtSynchronizeStreamWithTimeout`, prefetch |
|---:|---:|---:|---:|---:|
| 0 | 4,536 ms | 5,309 ms | 1,015 calls / 1,109 ms | 12 calls / 142 ms |
| 1 | 4,549 ms | 6,905 ms | 1,824 calls / 2,004 ms | 12 calls / 1,558 ms |
| 2 | 4,552 ms | 5,313 ms | 655 calls / 724 ms | 12 calls / 1,553 ms |
| 3 | 4,545 ms | 6,865 ms | 1,823 calls / 1,994 ms | 12 calls / 1,539 ms |

These host API durations can overlap and must not be added as independent wall
times. They do show that the same prefetch workload leaves different ranks in
different allocator/device-backlog states before the next root collective.
The alternating rank pairs do not match the static topology: all four selected
NPUs are pairwise HCCS-connected, while the reported CPU-affinity split is one
NPU on cores 0-89 and three NPUs on cores 90-179.

Two single-variable checks narrowed the cause further:

1. Disabling expandable segments removed `aclrtFreePhysical`, but increased the
   profiled step from 23.699 to 65.685 seconds as `aclrtSynchronizeStream`
   expanded to several seconds per microbatch. Disabling the allocator mode is
   not a mitigation.
2. Deferring per-microbatch `loss.item()` calls removed the long
   `aclrtSynchronizeStreamWithTimeout` calls, but the 1.59/1.64-second root
   all-gather skews remained. The `aten::item` calls expose queued work but do
   not create the collective delay; this experimental change was reverted.

The raw outputs for these checks are:

- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_interleaved_gc_prefetch_noexpand_allranks_20260826/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_interleaved_gc_prefetch_deferred_item_allranks_20260826/`

The strongest remaining hypothesis is rank-dependent device allocation and
release pressure caused by the lifetime of burst-prefetched restore buffers.
The next experiment should reduce the number or lifetime of those allocations,
for example by coalescing each checkpoint-prefetch group into a reusable arena,
and then validate it with the same four-rank window. Merely changing the
prefetch trigger, removing host scalar reads, or changing the HCCS
configuration does not address the measured bottleneck.

The cross-rank query is reproducible with:

```bash
python scripts/profile/analyze_npu_collective_skew.py \
  output/qwen3_5_gc_vs_hybrid/repro_qwen35_interleaved_noprefetch_allranks_20260826 \
  output/qwen3_5_gc_vs_hybrid/repro_qwen35_interleaved_gc_prefetch_allranks_20260826 \
  --top 10 --interval-op-ids 2,235,468,701
```

Three scheduling variants failed in the same direction. For this Qwen3.5-9B
workload there is still no measured hybrid-offload configuration faster than
model-wide `gc`. The four-rank analysis indicates that a future attempt needs
allocation reuse or throttling in the H2D restore path, not another layer-order
heuristic alone.

Interleaved raw outputs:

- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_interleaved_quarter_noprefetch6_20260826/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_interleaved_quarter_prefetch6_20260826/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_interleaved_quarter_gc_prefetch6_20260826/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_interleaved_gc_prefetch_post_unshard6_20260826/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_interleaved_gc_staggered_prefetch6_20260826/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_interleaved_noprefetch_steady_profile_20260826/`
- `output/qwen3_5_gc_vs_hybrid/repro_qwen35_interleaved_gc_prefetch_steady_profile_20260826/`

## Review procedure

1. Inspect each run's `manifest.txt` and `veomni_cli.yaml` to confirm the fixed
   workload and the one intended mode change.
2. Read each `*.log` after converting tqdm carriage returns to newlines, for
   example `tr '\r' '\n' < gc_r1.log`.
3. Take the last `20/20` line as the primary per-run latency and the final
   `VRAM usage after epoch 1` line as peak rank-0 memory.
4. Compare the 20 `total_loss` and `grad_norm` fields step by step across all
   logs.
5. Inspect `ASCEND_PROFILER_OUTPUT/api_statistic.csv` under the two trace roots
   listed in the preliminary section to verify event-wait and memcpy counts.
