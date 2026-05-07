#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_clcm_badnet_resnet18_rescue}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
CURVES_DIR="${RUN_ROOT}/curves"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
MANIFEST="${RUN_ROOT}/manifest.tsv"

PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_PARALLEL="${BATCH_PARALLEL:-4}"
GPUS="${GPUS:-0 1 2 3}"
ROUNDS="${ROUNDS:-600}"
EVAL_GAP="${EVAL_GAP:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ONLY_TAG="${ONLY_TAG:-}"
PROFILE="${PROFILE:-full}"

ensure_dir "${RUN_ROOT}"
ensure_dir "${LOG_DIR}"
ensure_dir "${CURVES_DIR}"

declare -a CONFIGS=()

dataset_name() {
  local seed="$1"
  if [[ "${seed}" == "42" ]]; then
    echo "Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10"
  else
    echo "Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10_seed${seed}"
  fi
}

add_config() {
  local tag="$1"
  local seed="$2"
  local algorithm="$3"
  local rounds="$4"
  local jr="$5"
  local lr="$6"
  local lr_head="$7"
  local rt_beta="$8"
  local lambda_cl="$9"
  local aug_strength="${10}"
  local mask_tau="${11}"
  local mask_alpha="${12}"
  local adv_eps="${13}"
  local adv_num_iter="${14}"
  local purify_beta="${15}"
  local purify_feature_beta="${16}"
  local purify_logit_beta="${17}"
  local purify_start="${18}"
  local purify_layers="${19}"
  local schedule="${20}"
  local purpose="${21}"
  local dataset
  dataset="$(dataset_name "${seed}")"
  CONFIGS+=("${tag}|${seed}|${dataset}|${algorithm}|${rounds}|${jr}|${lr}|${lr_head}|${rt_beta}|${lambda_cl}|${aug_strength}|${mask_tau}|${mask_alpha}|${adv_eps}|${adv_num_iter}|${purify_beta}|${purify_feature_beta}|${purify_logit_beta}|${purify_start}|${purify_layers}|${schedule}|${purpose}")
}

add_matrix_seed() {
  local seed="$1"

  add_config "R18_S${seed}_BASE_T6A03" "${seed}" "FedCLCM" "${ROUNDS}" 0.10 0.10 0.10 0.20 0.20 0.10 6.0 0.30 0.00 0 0.0 0.0 0.0 1 "layer4" "" "current_thesis_style_resnet18_control"
  add_config "R18_S${seed}_NOCL_T6A03" "${seed}" "FedCLCM" "${ROUNDS}" 0.10 0.10 0.10 0.20 0.00 0.10 6.0 0.30 0.00 0 0.0 0.0 0.0 1 "layer4" "" "ablation_inspired_low_asr_no_cl_control"
  add_config "R18_S${seed}_LOWCL_T5A03" "${seed}" "FedCLCM" "${ROUNDS}" 0.10 0.08 0.08 0.20 0.05 0.10 5.0 0.30 0.00 0 0.0 0.0 0.0 1 "layer4" "" "stronger_mask_low_cl_static"
  add_config "R18_S${seed}_LOWCL_T6A02" "${seed}" "FedCLCM" "${ROUNDS}" 0.10 0.08 0.08 0.20 0.05 0.10 6.0 0.20 0.00 0 0.0 0.0 0.0 1 "layer4" "" "mild_mask_low_cl_static"

  add_config "R18_S${seed}_SCH_A_STRONG_TO_MILD" "${seed}" "FedCLCM" "${ROUNDS}" 0.10 0.01 0.01 0.20 0.10 0.10 3.0 0.50 0.10 5 0.0 0.0 0.0 1 "layer4" "round=0,lr=0.01,lr_head=0.01,adv_eps=0.10,adv_num_iter=5,lambda_cl=0.10,mask_tau=3,mask_alpha=0.50;round=120,lr=0.03,lr_head=0.03,adv_eps=0.02,adv_num_iter=2,lambda_cl=0.05,mask_tau=5,mask_alpha=0.30;round=320,lr=0.05,lr_head=0.05,adv_eps=0,adv_num_iter=0,lambda_cl=0.05,mask_tau=6,mask_alpha=0.30" "old_resnet18_safety_window_then_utility_recovery"
  add_config "R18_S${seed}_SCH_B_NOCL_RECOVERY" "${seed}" "FedCLCM" "${ROUNDS}" 0.10 0.01 0.01 0.20 0.00 0.10 3.0 0.50 0.10 5 0.0 0.0 0.0 1 "layer4" "round=0,lr=0.01,lr_head=0.01,adv_eps=0.10,adv_num_iter=5,lambda_cl=0,mask_tau=3,mask_alpha=0.50;round=120,lr=0.03,lr_head=0.03,adv_eps=0.02,adv_num_iter=2,lambda_cl=0,mask_tau=5,mask_alpha=0.30;round=320,lr=0.05,lr_head=0.05,adv_eps=0,adv_num_iter=0,lambda_cl=0,mask_tau=6,mask_alpha=0.30" "no_cl_schedule_based_on_ablation"
  add_config "R18_S${seed}_SCH_C_LONG_SUPPRESS" "${seed}" "FedCLCM" "${ROUNDS}" 0.10 0.01 0.01 0.20 0.05 0.10 3.0 0.50 0.10 5 0.0 0.0 0.0 1 "layer4" "round=0,lr=0.01,lr_head=0.01,adv_eps=0.10,adv_num_iter=5,lambda_cl=0.05,mask_tau=3,mask_alpha=0.50;round=220,lr=0.03,lr_head=0.03,adv_eps=0.02,adv_num_iter=2,lambda_cl=0.05,mask_tau=5,mask_alpha=0.30;round=420,lr=0.05,lr_head=0.05,adv_eps=0,adv_num_iter=0,lambda_cl=0.05,mask_tau=6,mask_alpha=0.30" "longer_suppression_if_asr_rebounds_early"
  add_config "R18_S${seed}_SCH_D_T2_RECOVERY" "${seed}" "FedCLCM" "${ROUNDS}" 0.10 0.01 0.01 0.20 0.05 0.10 2.0 0.30 0.10 5 0.0 0.0 0.0 1 "layer4" "round=0,lr=0.01,lr_head=0.01,adv_eps=0.10,adv_num_iter=5,lambda_cl=0.05,mask_tau=2,mask_alpha=0.30;round=160,lr=0.03,lr_head=0.03,adv_eps=0.02,adv_num_iter=2,lambda_cl=0.05,mask_tau=5,mask_alpha=0.30;round=360,lr=0.05,lr_head=0.05,adv_eps=0,adv_num_iter=0,lambda_cl=0.05,mask_tau=6,mask_alpha=0.30" "aggressive_tau2_warmup_with_recovery"

  add_config "R18_S${seed}_PUR_L4_B1" "${seed}" "FedCLCMPurify" "${ROUNDS}" 0.10 0.08 0.08 0.20 0.05 0.10 6.0 0.30 0.00 0 1.0 0.0 0.0 20 "layer4" "" "light_attention_purify_not_resnetp"
  add_config "R18_S${seed}_PUR_L34_B01F01" "${seed}" "FedCLCMPurify" "${ROUNDS}" 0.10 0.08 0.08 0.20 0.05 0.10 6.0 0.30 0.00 0 0.1 0.1 0.0 20 "layer3,layer4" "" "attention_plus_feature_purify_high_layers"
  add_config "R18_S${seed}_SCH_PUR_A" "${seed}" "FedCLCMPurify" "${ROUNDS}" 0.10 0.01 0.01 0.20 0.05 0.10 3.0 0.50 0.10 5 0.1 0.0 0.0 20 "layer4" "round=0,lr=0.01,lr_head=0.01,adv_eps=0.10,adv_num_iter=5,lambda_cl=0.05,mask_tau=3,mask_alpha=0.50,purify_beta=0;round=120,lr=0.03,lr_head=0.03,adv_eps=0.02,adv_num_iter=2,lambda_cl=0.05,mask_tau=5,mask_alpha=0.30,purify_beta=0.1;round=320,lr=0.05,lr_head=0.05,adv_eps=0,adv_num_iter=0,lambda_cl=0.05,mask_tau=6,mask_alpha=0.30,purify_beta=0.1" "scheduled_suppression_then_light_purify"
  add_config "R18_S${seed}_SCH_GATE_LAYERTRIM" "${seed}" "FedCLCM" "${ROUNDS}" 0.10 0.01 0.01 0.20 0.05 0.10 3.0 0.50 0.10 5 0.0 0.0 0.0 1 "layer4" "round=0,lr=0.01,lr_head=0.01,adv_eps=0.10,adv_num_iter=5,lambda_cl=0.05,mask_tau=3,mask_alpha=0.50;round=160,lr=0.03,lr_head=0.03,adv_eps=0.02,adv_num_iter=2,lambda_cl=0.05,mask_tau=5,mask_alpha=0.30;round=360,lr=0.05,lr_head=0.05,adv_eps=0,adv_num_iter=0,lambda_cl=0.05,mask_tau=6,mask_alpha=0.30" "schedule_plus_existing_cosine_gate_layer_trim"
}

case "${PROFILE}" in
  smoke)
    add_config "SMOKE_R18_S42_SCH_A" 42 "FedCLCM" 3 0.10 0.01 0.01 0.20 0.10 0.10 3.0 0.50 0.10 5 0.0 0.0 0.0 1 "layer4" "round=0,lr=0.01,lr_head=0.01,adv_eps=0.10,adv_num_iter=5,lambda_cl=0.10,mask_tau=3,mask_alpha=0.50;round=2,lr=0.03,lr_head=0.03,adv_eps=0.02,adv_num_iter=2,lambda_cl=0.05,mask_tau=5,mask_alpha=0.30" "schedule_smoke"
    ;;
  core)
    add_matrix_seed 42
    ;;
  multiseed)
    add_matrix_seed 43
    add_matrix_seed 44
    ;;
  full)
    add_matrix_seed 42
    add_matrix_seed 43
    add_matrix_seed 44
    ;;
  *)
    echo "[FATAL] unknown PROFILE=${PROFILE}; use smoke, core, multiseed, full" >&2
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

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r _tag _seed dataset _algorithm _rounds _jr _lr _lr_head _rt_beta _lambda_cl _aug_strength _mask_tau _mask_alpha _adv_eps _adv_num_iter _purify_beta _purify_feature_beta _purify_logit_beta _purify_start _purify_layers _schedule _purpose <<< "${cfg}"
  if [[ ! -f "${SRC_ROOT}/dataset/${dataset}/config.json" ]]; then
    echo "[FATAL] missing dataset ${dataset}. Prepare static datasets first." >&2
    exit 1
  fi
done

echo -e "tag\tseed\tdataset\talgorithm\trounds\tjr\tlr\tlr_head\trt_beta\tlambda_cl\taug_strength\tmask_tau\tmask_alpha\tadv_eps\tadv_num_iter\tpurify_beta\tpurify_feature_beta\tpurify_logit_beta\tpurify_start\tpurify_layers\tschedule\tpurpose" > "${MANIFEST}"
for cfg in "${CONFIGS[@]}"; do
  printf '%s\n' "${cfg}" | tr '|' '\t' >> "${MANIFEST}"
done

launch_one() {
  local cfg="$1"
  local gpu="$2"
  IFS='|' read -r tag seed dataset algorithm rounds jr lr lr_head rt_beta lambda_cl aug_strength mask_tau mask_alpha adv_eps adv_num_iter purify_beta purify_feature_beta purify_logit_beta purify_start purify_layers schedule purpose <<< "${cfg}"
  local log_file="${LOG_DIR}/${tag}.log"
  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda -did "${gpu}"
    -data "${dataset}"
    -m ResNet18 -algo "${algorithm}"
    -ncl 10 -nc 100 -jr "${jr}" -lbs "${BATCH_SIZE}"
    -lr "${lr}" -lr_head "${lr_head}" -ls 1 -pls 1
    -gr "${rounds}" -eg "${EVAL_GAP}"
    -go "${tag}"
    --num_adv_clients 10
    --rt_beta "${rt_beta}"
    --lambda_cl "${lambda_cl}"
    --aug_strength "${aug_strength}"
    --mask_tau "${mask_tau}"
    --mask_alpha "${mask_alpha}"
    --enable_channel_mask true
    --adv_eps "${adv_eps}"
    --adv_num_iter "${adv_num_iter}"
  )

  if [[ "${algorithm}" == "FedCLCMPurify" ]]; then
    cmd+=(
      --purify_beta "${purify_beta}"
      --purify_feature_beta "${purify_feature_beta}"
      --purify_logit_beta "${purify_logit_beta}"
      --purify_start_round "${purify_start}"
      --purify_layers "${purify_layers}"
      --purify_teacher_momentum 0.90
      --purify_teacher_cpu_half true
    )
  fi

  if [[ -n "${schedule}" ]]; then
    cmd+=(--clcm_schedule "${schedule}")
  fi

  if [[ "${tag}" == *"GATE_LAYERTRIM" ]]; then
    cmd+=(--cosine_gate --cosine_gate_threshold 0.2 --cosine_gate_alpha 0.5 --trim_high_layers "layer3,layer4,fc" --trim_beta_low 0.1 --trim_beta_high 0.3)
  fi

  echo "[RUN] ${tag} gpu=${gpu} seed=${seed} algo=${algorithm} rounds=${rounds} purpose=${purpose}"
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
