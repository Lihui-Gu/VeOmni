# GC versus selective activation offload

## Confirmed scenario

The following Ascend setup demonstrates that gradient-checkpointing
recomputation costs more than selective activation offload:

- Commit: `afb0670`
- Hardware: 2 x Ascend 910B2 (`NPU_DEVICES=0,2`)
- Model: `Qwen/Qwen3-0.6B` at revision
  `c1899de289a04d12100db370d81485cdf75e47ca`
- Dataset: `allenai/tulu-3-sft-mixture` at revision
  `b14afda60f1bbebe55d5d2fa1e4df5042f97f8be`
- Sequence length: 16,384
- Micro/global batch size: 1/16 (8 micro-batches per rank)
- FSDP2 data-parallel size: 2
- Steps: 4; seed: 42
- Selective target: `Qwen3RMSNorm`
- Selective GPU budget: 40 GiB
- Selective prefetch: disabled

| Mode | Mean step time | Peak NPU memory | Overhead versus upper bound |
|---|---:|---:|---:|
| GC off, offload off (upper bound) | 7.54 s | 32.78 GiB | - |
| GC off, selective offload | 7.58 s | 28.30 GiB | 0.04 s (0.5%) |
| GC on, offload off | 8.76 s | 8.64 GiB | 1.22 s (16.2%) |

Selective offload is 13.5% faster than GC over all four steps. Excluding the
first step, their mean times are approximately 7.05 s and 8.29 s,
respectively, so the result is not caused by first-step warmup.

The four loss values match between runs: `1.31, 1.38, 1.32, 1.14`. Selective
offload moved 152,842,526,728 bytes in each direction across four steps and
reported zero threshold-fallback offloads. It therefore measures the selected
RMSNorm path rather than the legacy synchronous fallback.

## Reproduction

Download the model and dataset once:

```bash
/app/.venv/bin/hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir ./Qwen3-0.6B
/app/.venv/bin/hf download allenai/tulu-3-sft-mixture \
  --repo-type dataset \
  --revision b14afda60f1bbebe55d5d2fa1e4df5042f97f8be \
  --local-dir ./tulu-3-sft-mixture
```

Run all three cases on the Ascend worker:

```bash
MODEL_PATH=./Qwen3-0.6B NPU_DEVICES=0,2 \
  bash scripts/profile/run_gc_vs_selective_offload.sh all
```

The script accepts `upper`, `gc`, or `selective` to run one case. Paths and
launch settings can be overridden with `MODEL_PATH`, `DATA_PATH`,
`OUTPUT_ROOT`, `TORCHRUN_BIN`, `NPU_DEVICES`, `NPROC_PER_NODE`,
`MASTER_PORT_BASE`, `MAX_STEPS`, and `RUN_TAG`.

Results are written below `output/gc_vs_selective_offload/repro_<timestamp>/`:

- Logs: `repro_<timestamp>/<case>.log`
- Resolved configs: `repro_<timestamp>/<case>/veomni_cli.yaml`

## Pending Qwen3.5-9B validation

The confirmed result above does not establish the same performance result on
Qwen3.5-9B. Run the corresponding three-case text-only experiment directly
with `train.sh` and the existing Qwen3.5 text config. The commands below avoid
the expensive whole-Attention/GDN selection and differ only in the settings
under test.

Upper bound (GC off, activation offload off):

```bash
ASCEND_RT_VISIBLE_DEVICES=0,2,4,6 NPROC_PER_NODE=4 \
bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
  --model.model_path /path/to/Qwen3.5-9B \
  --model.ops_implementation.rms_norm_gated_implementation npu \
  --model.ops_implementation.causal_conv1d_implementation npu \
  --model.ops_implementation.chunk_gated_delta_rule_implementation npu_ascendc \
  --data.train_path ./tulu-3-sft-mixture/data \
  --data.max_seq_len 4096 \
  --train.accelerator.dp_shard_size 4 \
  --train.micro_batch_size 1 \
  --train.global_batch_size 16 \
  --train.max_steps 4 \
  --train.seed 42 \
  --train.wandb.enable false \
  --train.checkpoint.output_dir output/gc_vs_selective_offload/qwen3_5_9b_upper \
  --train.gradient_checkpointing.enable false \
  --train.accelerator.offload_config.enable_activation false
```

GC baseline (GC on, activation offload off): use the same command, change the
output directory to `qwen3_5_9b_gc`, and replace the final two options with:

```bash
  --train.gradient_checkpointing.enable true \
  --train.accelerator.offload_config.enable_activation false
```

Selective activation offload (GC off): use the same command, change the output
directory to `qwen3_5_9b_selective`, and replace the final two options with:

```bash
  --train.gradient_checkpointing.enable false \
  --train.accelerator.offload_config.enable_activation true \
  --train.accelerator.offload_config.activation_gpu_limit 80 \
  --train.accelerator.offload_config.selection.module_classes Qwen3_5RMSNorm \
  --train.accelerator.offload_config.prefetch false
```

All three cases use 4 Ascend NPUs, DP4, sequence length 4,096, and four steps.
If the upper bound does not fit, lower `--data.max_seq_len` consistently for
all three cases. If the selective summary reports threshold fallback, raise
`--train.accelerator.offload_config.activation_gpu_limit` before using the run
as an isolated selective-offload comparison. `train.sh` writes the console log
to `log.txt`, so preserve or rename it before starting the next case.

Qwen3.5's GatedDeltaNet kernels must be available on the NPU worker. The
commands use `npu_ascendc`; change
`--model.ops_implementation.chunk_gated_delta_rule_implementation` to `npu`
when the external `fla_npu` package is unavailable. This follow-up has a
reproduction procedure but no confirmed performance result yet.

## Historical confirmation logs

The measurements above come from these local reference logs. Their names are
historical and differ from the `upper`, `gc`, and `selective` names generated
by the reproduction script:

- `Qwen3.5-9B-vl-sft-selective-offload/gc_vs_selective/qwen3_06b_text_no_gc_no_offload_16k_dp2_confirm4.log`
- `Qwen3.5-9B-vl-sft-selective-offload/gc_vs_selective/qwen3_06b_text_selective_rmsnorm_16k_dp2_confirm4.log`
- `Qwen3.5-9B-vl-sft-selective-offload/gc_vs_selective/qwen3_06b_text_gc_on_16k_dp2_confirm4.log`

## Why this setup works

Selecting whole Attention, GDN, MLP, or Decoder modules also captures large
weight-related saved tensors. On Qwen3.5-9B this caused roughly 94-200 GB of
selected traffic per step, and the selective runs took 51-57 seconds versus
18 seconds with GC. Selecting the parameter-light RMSNorm boundary avoids that
failure mode.

The 40 GiB threshold budget is intentionally above the measured no-offload
peak. The confirmed selective run reports zero threshold fallback, so only
the explicitly selected module tensors are offloaded. Prefetch is disabled:
on torch-npu 2.10, enabling it expanded the restored-tensor live range enough
to cause OOM in the tested configurations.
