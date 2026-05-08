#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
DATASET_ROOT="${SRC_ROOT}/dataset"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_next_static_badnet_rescue}"
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
  local lr="$3"
  local lambda_cl="$4"
  local mask_tau="$5"
  local mask_alpha="$6"
  local join_ratio="$7"
  local local_epochs="$8"
  local plocal_epochs="$9"
  local adv_eps="${10}"
  local adv_num_iter="${11}"
  local schedule="${12}"
  local purpose="${13}"
  local dataset
  dataset="$(dataset_name "${seed}")"
  CONFIGS+=("${tag}|${seed}|${dataset}|${lr}|${lambda_cl}|${mask_tau}|${mask_alpha}|${join_ratio}|${local_epochs}|${plocal_epochs}|${adv_eps}|${adv_num_iter}|${schedule}|${purpose}")
}

add_grid_configs() {
  local seed
  for seed in 42 43 44; do
    add_config "G_LC002_T5A02_S${seed}" "${seed}" 0.08 0.02 5.0 0.2 0.10 1 1 0.00 0 "" "fine_grid_lowcl002_tau5_alpha02"
    add_config "G_LC002_T5A03_S${seed}" "${seed}" 0.08 0.02 5.0 0.3 0.10 1 1 0.00 0 "" "fine_grid_lowcl002_tau5_alpha03"
    add_config "G_LC005_T5A02_S${seed}" "${seed}" 0.08 0.05 5.0 0.2 0.10 1 1 0.00 0 "" "fine_grid_lowcl005_tau5_alpha02"
    add_config "G_LC005_T5A03_S${seed}" "${seed}" 0.08 0.05 5.0 0.3 0.10 1 1 0.00 0 "" "fine_grid_lowcl005_tau5_alpha03"
    add_config "G_LC008_T5A02_S${seed}" "${seed}" 0.08 0.08 5.0 0.2 0.10 1 1 0.00 0 "" "fine_grid_lowcl008_tau5_alpha02"
    add_config "G_LC008_T5A03_S${seed}" "${seed}" 0.08 0.08 5.0 0.3 0.10 1 1 0.00 0 "" "fine_grid_lowcl008_tau5_alpha03"
  done
}

add_schedule_configs() {
  add_config "SCH_STRONG_TO_MID_A_S42" 42 0.03 0.02 3.0 0.3 0.10 1 1 0.00 0 "round=0,lr=0.03,lr_head=0.03,lambda_cl=0.02,mask_tau=3,mask_alpha=0.3;round=200,lr=0.05,lr_head=0.05,mask_tau=4,mask_alpha=0.25;round=500,lr=0.08,lr_head=0.08,mask_tau=5,mask_alpha=0.2" "schedule_strong_to_mid_a"
  add_config "SCH_STRONG_TO_MID_B_S42" 42 0.03 0.05 3.0 0.3 0.10 1 1 0.00 0 "round=0,lr=0.03,lr_head=0.03,lambda_cl=0.05,mask_tau=3,mask_alpha=0.3;round=200,lr=0.05,lr_head=0.05,mask_tau=4,mask_alpha=0.25;round=500,lr=0.08,lr_head=0.08,mask_tau=5,mask_alpha=0.3" "schedule_strong_to_mid_b"
  add_config "SCH_MASK_RELAX_A_S42" 42 0.08 0.05 4.0 0.2 0.10 1 1 0.00 0 "round=0,mask_tau=4,mask_alpha=0.2;round=300,mask_tau=5,mask_alpha=0.2;round=700,mask_tau=5,mask_alpha=0.3" "schedule_mask_relax_a"
  add_config "SCH_CL_DECAY_A_S42" 42 0.08 0.10 5.0 0.2 0.10 1 1 0.00 0 "round=0,lambda_cl=0.10,mask_tau=5,mask_alpha=0.2;round=300,lambda_cl=0.05;round=600,lambda_cl=0.02" "schedule_cl_decay_a"
  add_config "SCH_CL_DECAY_B_S42" 42 0.08 0.08 5.0 0.3 0.10 1 1 0.00 0 "round=0,lambda_cl=0.08,mask_tau=5,mask_alpha=0.3;round=300,lambda_cl=0.05;round=600,lambda_cl=0.02" "schedule_cl_decay_b"
  add_config "SCH_MASK_RELAX_A_S43" 43 0.08 0.05 4.0 0.2 0.10 1 1 0.00 0 "round=0,mask_tau=4,mask_alpha=0.2;round=300,mask_tau=5,mask_alpha=0.2;round=700,mask_tau=5,mask_alpha=0.3" "schedule_mask_relax_a_confirm"
  add_config "SCH_MASK_RELAX_A_S44" 44 0.08 0.05 4.0 0.2 0.10 1 1 0.00 0 "round=0,mask_tau=4,mask_alpha=0.2;round=300,mask_tau=5,mask_alpha=0.2;round=700,mask_tau=5,mask_alpha=0.3" "schedule_mask_relax_a_confirm"
  add_config "SCH_STRONG_TO_MID_B_S43" 43 0.03 0.05 3.0 0.3 0.10 1 1 0.00 0 "round=0,lr=0.03,lr_head=0.03,lambda_cl=0.05,mask_tau=3,mask_alpha=0.3;round=200,lr=0.05,lr_head=0.05,mask_tau=4,mask_alpha=0.25;round=500,lr=0.08,lr_head=0.08,mask_tau=5,mask_alpha=0.3" "schedule_strong_to_mid_b_confirm"
  add_config "SCH_MASK_RELAX_A_JR005_S42" 42 0.08 0.05 4.0 0.2 0.05 1 1 0.00 0 "round=0,mask_tau=4,mask_alpha=0.2;round=300,mask_tau=5,mask_alpha=0.2;round=700,mask_tau=5,mask_alpha=0.3" "schedule_mask_relax_join005"
  add_config "SCH_MASK_RELAX_A_JR015_S42" 42 0.08 0.05 4.0 0.2 0.15 1 1 0.00 0 "round=0,mask_tau=4,mask_alpha=0.2;round=300,mask_tau=5,mask_alpha=0.2;round=700,mask_tau=5,mask_alpha=0.3" "schedule_mask_relax_join015"
}

add_pgd_configs() {
  add_config "PGD_E002_I3_T5A02_S42" 42 0.05 0.05 5.0 0.2 0.10 1 1 0.02 3 "" "pgd_light_eps002_tau5_alpha02"
  add_config "PGD_E002_I3_T5A03_S42" 42 0.05 0.05 5.0 0.3 0.10 1 1 0.02 3 "" "pgd_light_eps002_tau5_alpha03"
  add_config "PGD_E005_I3_T5A02_S42" 42 0.03 0.05 5.0 0.2 0.10 1 1 0.05 3 "" "pgd_light_eps005_tau5_alpha02"
  add_config "PGD_E005_I3_T5A03_S42" 42 0.03 0.05 5.0 0.3 0.10 1 1 0.05 3 "" "pgd_light_eps005_tau5_alpha03"
  add_config "PGD_E002_I3_T5A02_S43" 43 0.05 0.05 5.0 0.2 0.10 1 1 0.02 3 "" "pgd_light_eps002_tau5_alpha02_confirm"
  add_config "PGD_E002_I3_T5A03_S43" 43 0.05 0.05 5.0 0.3 0.10 1 1 0.02 3 "" "pgd_light_eps002_tau5_alpha03_confirm"
}

case "${PROFILE}" in
  grid)
    add_grid_configs
    ;;
  schedule)
    add_schedule_configs
    ;;
  pgd)
    add_pgd_configs
    ;;
  all)
    add_grid_configs
    add_schedule_configs
    add_pgd_configs
    ;;
  *)
    echo "[FATAL] unknown PROFILE=${PROFILE}; use grid, schedule, pgd, all" >&2
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

echo -e "tag\tseed\tdataset\tlr\tlambda_cl\tmask_tau\tmask_alpha\tjoin_ratio\tlocal_epochs\tplocal_epochs\tadv_eps\tadv_num_iter\tschedule\trounds\tpurpose" > "${MANIFEST}"
for cfg in "${CONFIGS[@]}"; do
  printf '%s|%s\n' "${cfg}" "${ROUNDS}" | awk -F'|' 'BEGIN{OFS="\t"} {print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$15,$14}' >> "${MANIFEST}"
done

launch_one() {
  local cfg="$1"
  local gpu="$2"
  IFS='|' read -r tag seed dataset lr lambda_cl mask_tau mask_alpha join_ratio local_epochs plocal_epochs adv_eps adv_num_iter schedule purpose <<< "${cfg}"
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
    --adv_eps "${adv_eps}"
    --adv_num_iter "${adv_num_iter}"
  )
  if [[ -n "${schedule}" ]]; then
    cmd+=(--clcm_schedule "${schedule}")
  fi

  echo "[RUN] ${tag} gpu=${gpu} rounds=${ROUNDS} purpose=${purpose}"
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
