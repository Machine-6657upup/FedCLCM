#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
DATASET_ROOT="${SRC_ROOT}/dataset"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_overnight_thesis_rescue_1000}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
CURVES_DIR="${RUN_ROOT}/curves"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
MANIFEST="${RUN_ROOT}/manifest.tsv"
DATASET_LOG_DIR="${RUN_ROOT}/dataset_logs"

PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_PARALLEL="${BATCH_PARALLEL:-4}"
ROUNDS="${ROUNDS:-1000}"
EVAL_GAP="${EVAL_GAP:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ONLY_TAG="${ONLY_TAG:-}"
PROFILE="${PROFILE:-all}"

ensure_dir "${RUN_ROOT}"
ensure_dir "${LOG_DIR}"
ensure_dir "${CURVES_DIR}"
ensure_dir "${DATASET_LOG_DIR}"

dataset_name() {
  local seed="$1"
  if [[ "${seed}" == "42" ]]; then
    echo "Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10"
  else
    echo "Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10_seed${seed}"
  fi
}

ensure_badnet_dataset() {
  local seed="$1"
  local dataset
  dataset="$(dataset_name "${seed}")"
  local ds_dir="${DATASET_ROOT}/${dataset}"
  if [[ -f "${ds_dir}/config.json" ]]; then
    return
  fi
  echo "[DATASET] generate ${dataset}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/generate_dataset_via_existing.py" \
    --generator "${DATASET_ROOT}/utils/generate_Cifar10_badnet.py" \
    --dir-path "${ds_dir}" \
    --rawdata-path "${DATASET_ROOT}/rawdata" \
    --num-clients 100 \
    --backdoor-rate 0.2 \
    --adversary-num 10 \
    --target-y 0 \
    --alpha 0.5 \
    --train-ratio 0.8 \
    --batch-size 10 \
    --partition dir \
    --niid \
    --balance \
    --generator-seed "${seed}" \
    > "${DATASET_LOG_DIR}/${dataset}.log" 2>&1
}

declare -a CONFIGS=()

add_config() {
  local tag="$1"
  local seed="$2"
  local attack_mode="$3"
  local lr="$4"
  local lambda_cl="$5"
  local mask_tau="$6"
  local mask_alpha="$7"
  local join_ratio="$8"
  local local_epochs="$9"
  local plocal_epochs="${10}"
  local purpose="${11}"
  local dataset
  dataset="$(dataset_name "${seed}")"
  CONFIGS+=("${tag}|${seed}|${dataset}|${attack_mode}|${lr}|${lambda_cl}|${mask_tau}|${mask_alpha}|${join_ratio}|${local_epochs}|${plocal_epochs}|${purpose}")
}

add_static_configs() {
  # Main result / multiseed.
  add_config "BN_LOWCL_T5A03_S42" 42 "static" 0.08 0.05 5.0 0.3 0.10 1 1 "main_badnet_lowcl_seed42"
  add_config "BN_LOWCL_T5A03_S43" 43 "static" 0.08 0.05 5.0 0.3 0.10 1 1 "main_badnet_lowcl_seed43"
  add_config "BN_LOWCL_T5A03_S44" 44 "static" 0.08 0.05 5.0 0.3 0.10 1 1 "main_badnet_lowcl_seed44"

  # Contrastive-loss ablation around the new good setting.
  add_config "BN_CL0_T5A03" 42 "static" 0.08 0.00 5.0 0.3 0.10 1 1 "cl_ablation_zero"
  add_config "BN_CL002_T5A03" 42 "static" 0.08 0.02 5.0 0.3 0.10 1 1 "cl_ablation_002"
  add_config "BN_CL01_T5A03" 42 "static" 0.08 0.10 5.0 0.3 0.10 1 1 "cl_ablation_01"
  add_config "BN_CL02_T5A03" 42 "static" 0.08 0.20 5.0 0.3 0.10 1 1 "cl_ablation_02"

  # Mask-neighborhood sweep for a cleaner Pareto frontier.
  add_config "BN_M_T4A02_LC005" 42 "static" 0.08 0.05 4.0 0.2 0.10 1 1 "mask_tau4_alpha02"
  add_config "BN_M_T4A03_LC005" 42 "static" 0.08 0.05 4.0 0.3 0.10 1 1 "mask_tau4_alpha03"
  add_config "BN_M_T4A04_LC005" 42 "static" 0.08 0.05 4.0 0.4 0.10 1 1 "mask_tau4_alpha04"
  add_config "BN_M_T5A02_LC005" 42 "static" 0.08 0.05 5.0 0.2 0.10 1 1 "mask_tau5_alpha02"
  add_config "BN_M_T5A04_LC005" 42 "static" 0.08 0.05 5.0 0.4 0.10 1 1 "mask_tau5_alpha04"
  add_config "BN_M_T6A02_LC005" 42 "static" 0.08 0.05 6.0 0.2 0.10 1 1 "mask_tau6_alpha02"
  add_config "BN_M_T6A03_LC005" 42 "static" 0.08 0.05 6.0 0.3 0.10 1 1 "mask_tau6_alpha03"
  add_config "BN_M_T6A04_LC005" 42 "static" 0.08 0.05 6.0 0.4 0.10 1 1 "mask_tau6_alpha04"

  # Join-ratio analysis under the good recipe.
  add_config "BN_JR005_LOWCL_T5A03" 42 "static" 0.08 0.05 5.0 0.3 0.05 1 1 "join_ratio_005_lowcl"
  add_config "BN_JR02_LOWCL_T5A03" 42 "static" 0.08 0.05 5.0 0.3 0.20 1 1 "join_ratio_02_lowcl"

  # Local-epoch analysis under the good recipe.
  add_config "BN_LE2_LOWCL_T5A03" 42 "static" 0.08 0.05 5.0 0.3 0.10 2 1 "local_epoch_2_lowcl"
  add_config "BN_LE5_LOWCL_T5A03" 42 "static" 0.08 0.05 5.0 0.3 0.10 5 1 "local_epoch_5_lowcl"
}

add_pfedba_configs() {
  local lr
  local pair
  for lr in 0.03 0.05 0.08; do
    for pair in "5.0 0.3" "8.0 0.5" "10.0 0.7" "12.0 0.7"; do
      read -r tau alpha <<< "${pair}"
      local tag_lr="${lr/./}"
      local tag_tau="${tau/./}"
      local tag_alpha="${alpha/./}"
      add_config "PF_LR${tag_lr}_T${tag_tau}A${tag_alpha}_LC005" 42 "pfedba" "${lr}" 0.05 "${tau}" "${alpha}" 0.10 1 1 "pfedba_lr${lr}_tau${tau}_alpha${alpha}_lowcl"
    done
  done
}

case "${PROFILE}" in
  static)
    add_static_configs
    ;;
  pfedba)
    add_pfedba_configs
    ;;
  all)
    add_static_configs
    add_pfedba_configs
    ;;
  *)
    echo "[FATAL] unknown PROFILE=${PROFILE}; use static, pfedba, all" >&2
    exit 1
    ;;
esac

if [[ -n "${ONLY_TAG}" ]]; then
  FILTERED=()
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r tag _ <<< "${cfg}"
    if [[ "${tag}" == "${ONLY_TAG}" ]]; then
      FILTERED+=("${cfg}")
    fi
  done
  CONFIGS=("${FILTERED[@]}")
fi

if (( ${#CONFIGS[@]} == 0 )); then
  echo "[FATAL] no configs selected" >&2
  exit 1
fi

for seed in 42 43 44; do
  if printf '%s\n' "${CONFIGS[@]}" | grep -q "|${seed}|"; then
    ensure_badnet_dataset "${seed}"
  fi
done

echo -e "tag\tseed\tdataset\tattack_mode\tlr\tlambda_cl\tmask_tau\tmask_alpha\tjoin_ratio\tlocal_epochs\tplocal_epochs\trounds\tpurpose" > "${MANIFEST}"
for cfg in "${CONFIGS[@]}"; do
  printf '%s|%s\n' "${cfg}" "${ROUNDS}" | awk -F'|' 'BEGIN{OFS="\t"} {print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$13,$12}' >> "${MANIFEST}"
done

launch_one() {
  local cfg="$1"
  local gpu="$2"
  IFS='|' read -r tag seed dataset attack_mode lr lambda_cl mask_tau mask_alpha join_ratio local_epochs plocal_epochs purpose <<< "${cfg}"
  local log_file="${LOG_DIR}/${tag}.log"
  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda -did "${gpu}"
    -data "${dataset}"
    -m ResNet18 -algo FedCLCM
    -ncl 10 -nc 100 -jr "${join_ratio}" -lbs "${BATCH_SIZE}"
    -lr "${lr}" -lr_head "${lr}" -ls "${local_epochs}" -pls "${plocal_epochs}"
    -gr "${ROUNDS}" -eg "${EVAL_GAP}"
    -go "${tag}"
    --num_adv_clients 10
    --rt_beta 0.20
    --lambda_cl "${lambda_cl}"
    --aug_strength 0.10
    --mask_tau "${mask_tau}"
    --mask_alpha "${mask_alpha}"
    --enable_channel_mask true
    --adv_eps 0.00
    --adv_num_iter 0
  )

  if [[ "${attack_mode}" == "pfedba" ]]; then
    cmd+=(
      --attack pfedba
      --pfedba_target_label 0
      --pfedba_poison_rate 0.5
      --pfedba_attack_start 30
      --pfedba_trigger_opt_steps 15
      --pfedba_trigger_lr 0.1
      --pfedba_patch_size 8
      --pfedba_patch_position center
      --pfedba_align_grad_weight 1.0
      --pfedba_align_loss_weight 1.0
      --pfedba_attack_loss_weight 1.0
    )
  fi

  echo "[RUN] ${tag} gpu=${gpu} mode=${attack_mode} rounds=${ROUNDS} purpose=${purpose}"
  (
    cd "${SRC_ROOT}"
    "${cmd[@]}" > "${log_file}" 2>&1
  )
}

run_config_batches CONFIGS launch_one "${BATCH_PARALLEL}"
collect_metrics "${LOG_DIR}" "*.log" "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${CURVES_DIR}"

echo "[READY] run_root=${RUN_ROOT}"
echo "[READY] manifest=${MANIFEST}"
echo "[READY] summary=${SUMMARY_CSV}"
