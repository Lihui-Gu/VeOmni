#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/text/qwen3.yaml}"
MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_mlsys_ssd_lfrz_search/ecom/models/Qwen3-0.6B}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/tulu-3-sft-mixture/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/gc_vs_selective_offload}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/app/.venv/bin/torchrun}"
NPU_DEVICES="${NPU_DEVICES:-0,2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29650}"
MAX_STEPS="${MAX_STEPS:-4}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MODE="${1:-all}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing training config: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Missing model config: ${MODEL_PATH}/config.json" >&2
  exit 1
fi

shopt -s nullglob
dataset_shards=("${DATA_PATH}"/*.parquet)
if (( ${#dataset_shards[@]} == 0 )); then
  echo "No parquet shards found under ${DATA_PATH}" >&2
  exit 1
fi

RUN_ROOT="${OUTPUT_ROOT}/repro_${RUN_TAG}"
mkdir -p "${RUN_ROOT}"

common_args=(
  tasks/train_text.py
  "${CONFIG_PATH}"
  --model.model_path "${MODEL_PATH}"
  --data.train_path "${DATA_PATH}"
  --data.max_seq_len 16384
  --data.train_size 10000000
  --data.dataloader.num_workers 2
  --train.micro_batch_size 1
  --train.global_batch_size 16
  --train.bsz_warmup_ratio 0
  --train.max_steps "${MAX_STEPS}"
  --train.num_train_epochs 1
  --train.seed 42
  --train.wandb.enable false
  --train.profile.enable false
  --train.checkpoint.save_steps 0
  --train.checkpoint.save_epochs 0
  --train.checkpoint.save_hf_weights false
)

run_case() {
  local name="$1"
  local port_offset="$2"
  shift 2

  local case_dir="${RUN_ROOT}/${name}"
  local log_file="${RUN_ROOT}/${name}.log"
  local master_port=$((MASTER_PORT_BASE + port_offset))

  echo "Running ${name}; log=${log_file}"
  PYTHONUNBUFFERED=1 ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}" \
    "${TORCHRUN_BIN}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${master_port}" \
    "${common_args[@]}" \
    --train.checkpoint.output_dir "${case_dir}" \
    "$@" 2>&1 | tee "${log_file}"

  grep -E "Epoch 1/1: 100%|VRAM usage after epoch|Selective activation offload summary" "${log_file}" | tail -n 3 || true
}

case "${MODE}" in
  upper)
    run_case upper 0 \
      --train.gradient_checkpointing.enable false \
      --train.accelerator.offload_config.enable_activation false
    ;;
  gc)
    run_case gc 1 \
      --train.gradient_checkpointing.enable true \
      --train.accelerator.offload_config.enable_activation false
    ;;
  selective)
    run_case selective 2 \
      --train.gradient_checkpointing.enable false \
      --train.accelerator.offload_config.enable_activation true \
      --train.accelerator.offload_config.activation_gpu_limit 40 \
      --train.accelerator.offload_config.selection.module_classes Qwen3RMSNorm \
      --train.accelerator.offload_config.prefetch false
    ;;
  selective-gc)
    run_case selective_gc 3 \
      --train.gradient_checkpointing.enable true \
      --train.gradient_checkpointing.enable_reentrant false \
      --train.gradient_checkpointing.selection.module_paths '**.layers.*.mlp' \
      --train.accelerator.offload_config.enable_activation false
    ;;
  hybrid)
    run_case hybrid 4 \
      --train.gradient_checkpointing.enable true \
      --train.gradient_checkpointing.enable_reentrant false \
      --train.gradient_checkpointing.selection.module_paths '**.layers.*.mlp' \
      --train.accelerator.offload_config.enable_activation true \
      --train.accelerator.offload_config.activation_gpu_limit 40 \
      --train.accelerator.offload_config.selection.module_paths \
        '**.layers.*.input_layernorm' \
        '**.layers.*.post_attention_layernorm' \
      --train.accelerator.offload_config.prefetch false
    ;;
  hybrid-all)
    run_case gc 1 \
      --train.gradient_checkpointing.enable true \
      --train.accelerator.offload_config.enable_activation false
    run_case selective_gc 3 \
      --train.gradient_checkpointing.enable true \
      --train.gradient_checkpointing.enable_reentrant false \
      --train.gradient_checkpointing.selection.module_paths '**.layers.*.mlp' \
      --train.accelerator.offload_config.enable_activation false
    run_case hybrid 4 \
      --train.gradient_checkpointing.enable true \
      --train.gradient_checkpointing.enable_reentrant false \
      --train.gradient_checkpointing.selection.module_paths '**.layers.*.mlp' \
      --train.accelerator.offload_config.enable_activation true \
      --train.accelerator.offload_config.activation_gpu_limit 40 \
      --train.accelerator.offload_config.selection.module_paths \
        '**.layers.*.input_layernorm' \
        '**.layers.*.post_attention_layernorm' \
      --train.accelerator.offload_config.prefetch false
    ;;
  all)
    run_case upper 0 \
      --train.gradient_checkpointing.enable false \
      --train.accelerator.offload_config.enable_activation false
    run_case gc 1 \
      --train.gradient_checkpointing.enable true \
      --train.accelerator.offload_config.enable_activation false
    run_case selective 2 \
      --train.gradient_checkpointing.enable false \
      --train.accelerator.offload_config.enable_activation true \
      --train.accelerator.offload_config.activation_gpu_limit 40 \
      --train.accelerator.offload_config.selection.module_classes Qwen3RMSNorm \
      --train.accelerator.offload_config.prefetch false
    ;;
  *)
    echo "Usage: $0 [all|upper|gc|selective|selective-gc|hybrid|hybrid-all]" >&2
    exit 2
    ;;
esac

echo "Results: ${RUN_ROOT}"
