#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
PROFILE="${PROFILE:-badnet_focus}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_static_fedclcm_purify_${PROFILE}}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
CURVES_DIR="${RUN_ROOT}/curves"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
MANIFEST="${RUN_ROOT}/manifest.tsv"

BATCH_PARALLEL="${BATCH_PARALLEL:-4}"
ONLY_TAG="${ONLY_TAG:-}"
EVAL_GAP="${EVAL_GAP:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FINAL_ROUNDS="${FINAL_ROUNDS:-600}"
SMOKE_ROUNDS="${SMOKE_ROUNDS:-3}"

ensure_dir "${RUN_ROOT}"
ensure_dir "${LOG_DIR}"
ensure_dir "${CURVES_DIR}"

declare -a CONFIGS=()

dataset_name() {
  local clients="$1"
  local attack="$2"
  local adv_clients="$3"
  local bdoor="$4"
  echo "Cifar10_dir0.5_bdoor${bdoor}_nclient_${clients}_${attack}_adv${adv_clients}"
}

add_config() {
  local tag="$1"
  local algorithm="$2"
  local attack="$3"
  local clients="$4"
  local adv_clients="$5"
  local bdoor="$6"
  local join_ratio="$7"
  local model="$8"
  local local_lr="$9"
  local lr_head="${10}"
  local local_epochs="${11}"
  local plocal_epochs="${12}"
  local rounds="${13}"
  local rt_beta="${14}"
  local lambda_cl="${15}"
  local aug_strength="${16}"
  local mask_tau="${17}"
  local mask_alpha="${18}"
  local enable_channel_mask="${19}"
  local adv_eps="${20}"
  local adv_num_iter="${21}"
  local purify_beta="${22}"
  local purify_feature_beta="${23}"
  local purify_logit_beta="${24}"
  local purify_start_round="${25}"
  local purify_layers="${26}"
  local purify_teacher_momentum="${27}"
  local purpose="${28}"
  local data
  data="$(dataset_name "${clients}" "${attack}" "${adv_clients}" "${bdoor}")"

  CONFIGS+=("${tag}|${algorithm}|${attack}|${data}|${clients}|${adv_clients}|${join_ratio}|${model}|${local_lr}|${lr_head}|${local_epochs}|${plocal_epochs}|${rounds}|${rt_beta}|${lambda_cl}|${aug_strength}|${mask_tau}|${mask_alpha}|${enable_channel_mask}|${adv_eps}|${adv_num_iter}|${purify_beta}|${purify_feature_beta}|${purify_logit_beta}|${purify_start_round}|${purify_layers}|${purify_teacher_momentum}|${purpose}")
}

add_smoke_configs() {
  add_config "SMK_B04_PUR_L4_B500" "FedCLCMPurify" "badnet" 100 10 0.2 0.1 "ResNet18" 0.1 0.1 1 1 "${SMOKE_ROUNDS}" 0.20 0.20 0.10 6.0 0.30 true 0.00 0 500.0 0.0 0.0 1 "layer4" 0.90 "smoke_current_best_plus_l4_purify"
  add_config "SMK_OLD_PUR_L4_B500" "FedCLCMPurify" "badnet" 40 5 0.2 0.25 "ResNetP" 0.003 0.005 1 1 "${SMOKE_ROUNDS}" 0.05 0.20 0.10 12.0 0.70 true 0.00 0 500.0 0.0 0.0 1 "layer4" 0.90 "smoke_old_success_recipe_plus_l4_purify"
}

add_badnet_focus_configs() {
  # Paired controls: current best 100-client formal recipe vs the same recipe
  # with lightweight client-side purification.
  add_config "BN100_B04_BASE" "FedCLCM" "badnet" 100 10 0.2 0.1 "ResNet18" 0.1 0.1 1 1 "${FINAL_ROUNDS}" 0.20 0.20 0.10 6.0 0.30 true 0.00 0 0.0 0.0 0.0 1 "layer4" 0.90 "current_best_static_baseline"
  add_config "BN100_B04_PUR_L4_B500" "FedCLCMPurify" "badnet" 100 10 0.2 0.1 "ResNet18" 0.1 0.1 1 1 "${FINAL_ROUNDS}" 0.20 0.20 0.10 6.0 0.30 true 0.00 0 500.0 0.0 0.0 1 "layer4" 0.90 "current_best_plus_paper_scale_attention_purify"
  add_config "BN100_B04_PUR_L4_B1500" "FedCLCMPurify" "badnet" 100 10 0.2 0.1 "ResNet18" 0.1 0.1 1 1 "${FINAL_ROUNDS}" 0.20 0.20 0.10 6.0 0.30 true 0.00 0 1500.0 0.0 0.0 1 "layer4" 0.90 "current_best_plus_pflalp_beta_attention_purify"

  # Old-success transfer: preserve the validated CLCM ingredients instead of
  # stacking more server-side filtering.
  add_config "BN100_OLD_BASE" "FedCLCM" "badnet" 100 10 0.2 0.1 "ResNetP" 0.003 0.005 1 1 "${FINAL_ROUNDS}" 0.05 0.20 0.10 12.0 0.70 true 0.00 0 0.0 0.0 0.0 1 "layer4" 0.90 "old_success_transfer_baseline"
  add_config "BN100_OLD_PUR_L4_B500" "FedCLCMPurify" "badnet" 100 10 0.2 0.1 "ResNetP" 0.003 0.005 1 1 "${FINAL_ROUNDS}" 0.05 0.20 0.10 12.0 0.70 true 0.00 0 500.0 0.0 0.0 1 "layer4" 0.90 "old_success_transfer_plus_paper_scale_purify"
  add_config "BN100_OLD_PUR_L4_B1500" "FedCLCMPurify" "badnet" 100 10 0.2 0.1 "ResNetP" 0.003 0.005 1 1 "${FINAL_ROUNDS}" 0.05 0.20 0.10 12.0 0.70 true 0.00 0 1500.0 0.0 0.0 1 "layer4" 0.90 "old_success_transfer_plus_pflalp_beta_purify"
  add_config "BN100_OLD_PUR_NOMASK_L4_B500" "FedCLCMPurify" "badnet" 100 10 0.2 0.1 "ResNetP" 0.003 0.005 1 1 "${FINAL_ROUNDS}" 0.05 0.20 0.10 1000000000.0 1.00 false 0.00 0 500.0 0.0 0.0 1 "layer4" 0.90 "old_logs_no_mask_check_plus_purify"

  # Positive control from the historical successful regime. This is not the
  # thesis setting, but it tells us whether the cleaned repo still reproduces
  # the old CLCM behavior.
  add_config "BN40_PC_BASE" "FedCLCM" "badnet" 40 5 0.2 0.25 "ResNetP" 0.003 0.005 1 1 "${FINAL_ROUNDS}" 0.05 0.20 0.10 12.0 0.70 true 0.00 0 0.0 0.0 0.0 1 "layer4" 0.90 "positive_control_old_success_recipe"
  add_config "BN40_PC_PUR_L4_B500" "FedCLCMPurify" "badnet" 40 5 0.2 0.25 "ResNetP" 0.003 0.005 1 1 "${FINAL_ROUNDS}" 0.05 0.20 0.10 12.0 0.70 true 0.00 0 500.0 0.0 0.0 1 "layer4" 0.90 "positive_control_old_success_plus_purify"
}

add_static_extension_configs() {
  # Run only after badnet_focus shows a positive ACC/ASR tradeoff.
  add_config "BL100_OLD_PUR_L4_B500" "FedCLCMPurify" "blend" 100 10 0.2 0.1 "ResNetP" 0.003 0.005 1 1 "${FINAL_ROUNDS}" 0.05 0.20 0.10 12.0 0.70 true 0.00 0 500.0 0.0 0.0 1 "layer4" 0.90 "blend_transfer_after_badnet"
  add_config "SG100_OLD_PUR_L4_B500" "FedCLCMPurify" "sig" 100 10 0.2 0.1 "ResNetP" 0.003 0.005 1 1 "${FINAL_ROUNDS}" 0.05 0.20 0.10 12.0 0.70 true 0.00 0 500.0 0.0 0.0 1 "layer4" 0.90 "sig_transfer_after_badnet"
}

case "${PROFILE}" in
  smoke)
    add_smoke_configs
    ;;
  badnet_focus)
    add_badnet_focus_configs
    ;;
  static_extension)
    add_static_extension_configs
    ;;
  all)
    add_badnet_focus_configs
    add_static_extension_configs
    ;;
  *)
    echo "[FATAL] unknown PROFILE=${PROFILE}; use smoke, badnet_focus, static_extension, or all" >&2
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
  IFS='|' read -r _tag _algorithm _attack dataset _clients _adv_clients _jr _model _lr _lrh _le _ple _rounds _rt _lam _aug _tau _alpha _mask _adv_eps _adv_iter _pb _pfb _plb _ps _layers _pm _purpose <<< "${cfg}"
  if [[ ! -f "${SRC_ROOT}/dataset/${dataset}/config.json" ]]; then
    echo "[FATAL] missing dataset ${dataset}. Run scripts/prepare_static_datasets_srcutils.sh first." >&2
    exit 1
  fi
done

echo -e "tag\talgorithm\tattack\tdataset\tnum_clients\tadv_clients\tjoin_ratio\tmodel\tlocal_lr\tlr_head\tlocal_epochs\tplocal_epochs\trounds\trt_beta\tlambda_cl\taug_strength\tmask_tau\tmask_alpha\tenable_channel_mask\tadv_eps\tadv_num_iter\tpurify_beta\tpurify_feature_beta\tpurify_logit_beta\tpurify_start_round\tpurify_layers\tpurify_teacher_momentum\tpurpose" > "${MANIFEST}"
for cfg in "${CONFIGS[@]}"; do
  printf '%s\n' "${cfg}" | tr '|' '\t' >> "${MANIFEST}"
done

launch_one() {
  local cfg="$1"
  local gpu="$2"
  IFS='|' read -r tag algorithm attack dataset clients adv_clients join_ratio model local_lr lr_head local_epochs plocal_epochs rounds rt_beta lambda_cl aug_strength mask_tau mask_alpha enable_channel_mask adv_eps adv_num_iter purify_beta purify_feature_beta purify_logit_beta purify_start_round purify_layers purify_teacher_momentum purpose <<< "${cfg}"

  local log_file="${LOG_DIR}/${tag}.log"
  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda -did "${gpu}"
    -data "${dataset}"
    -m "${model}" -algo "${algorithm}"
    -ncl 10 -nc "${clients}" -jr "${join_ratio}" -lbs "${BATCH_SIZE}"
    -lr "${local_lr}" -lr_head "${lr_head}" -ls "${local_epochs}" -pls "${plocal_epochs}"
    -gr "${rounds}" -eg "${EVAL_GAP}"
    -go "${tag}"
    --num_adv_clients "${adv_clients}"
    --rt_beta "${rt_beta}"
    --lambda_cl "${lambda_cl}"
    --aug_strength "${aug_strength}"
    --mask_tau "${mask_tau}"
    --mask_alpha "${mask_alpha}"
    --enable_channel_mask "${enable_channel_mask}"
    --adv_eps "${adv_eps}"
    --adv_num_iter "${adv_num_iter}"
  )

  if [[ "${algorithm}" == "FedCLCMPurify" ]]; then
    cmd+=(
      --purify_beta "${purify_beta}"
      --purify_feature_beta "${purify_feature_beta}"
      --purify_logit_beta "${purify_logit_beta}"
      --purify_start_round "${purify_start_round}"
      --purify_layers "${purify_layers}"
      --purify_teacher_momentum "${purify_teacher_momentum}"
      --purify_teacher_cpu_half true
    )
  fi

  echo "[RUN] ${tag} algo=${algorithm} attack=${attack} gpu=${gpu} rounds=${rounds} purpose=${purpose}"
  (
    cd "${SRC_ROOT}"
    "${cmd[@]}" > "${log_file}" 2>&1
  )
}

run_config_batches CONFIGS launch_one "${BATCH_PARALLEL}"
collect_metrics "${LOG_DIR}" "*.log" "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${CURVES_DIR}"

echo "[READY] manifest=${MANIFEST}"
echo "[READY] summary=${SUMMARY_CSV}"
