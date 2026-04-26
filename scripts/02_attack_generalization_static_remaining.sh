#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

LOG_DIR="${PROJECT_DIR}/thesis_log/attack_generalization_static"
RUN_TS="$(timestamp)"
SUMMARY_CSV="${LOG_DIR}/summary_remaining_${RUN_TS}.csv"
SUMMARY_JSON="${LOG_DIR}/summary_remaining_${RUN_TS}.json"
CURVES_DIR="${LOG_DIR}/curves_remaining_${RUN_TS}"
FAIL_FILE="${LOG_DIR}/failed_remaining_${RUN_TS}.txt"

ensure_dir "${LOG_DIR}"
: > "${FAIL_FILE}"

# Resume only the unfinished tasks from the interrupted 02 run.
# tag|algo|dataset|extra
CONFIGS=(
  "blend_fedrep|FedRep|Cifar10_dir0.5_bdoor0.2_nclient_40_blend_adv5|"
  "blend_fedbulyan|FedBulyan|Cifar10_dir0.5_bdoor0.2_nclient_40_blend_adv5|"
  "blend_fedflip|FedFLIP|Cifar10_dir0.5_bdoor0.2_nclient_40_blend_adv5|"
  "blend_fedclcm|FedCLCM|Cifar10_dir0.5_bdoor0.2_nclient_40_blend_adv5|--rt_beta 0.0 --lambda_cl 0.10 --aug_strength 0.05 --adv_eps 0.0 --adv_num_iter 0 --mask_tau 10.0 --mask_alpha 0.90"
  "sig_fedavg|FedAvg|Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5|"
  "sig_fedrep|FedRep|Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5|"
  "sig_fedmedian|FedMedian|Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5|"
  "sig_fedtrimmed|FedTrimmed|Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5|"
  "sig_fedbulyan|FedBulyan|Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5|"
  "sig_fedflip|FedFLIP|Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5|"
  "sig_fedclcm|FedCLCM|Cifar10_dir0.5_bdoor0.2_nclient_40_sig_adv5|--rt_beta 0.0 --lambda_cl 0.10 --aug_strength 0.05 --adv_eps 0.0 --adv_num_iter 0 --mask_tau 10.0 --mask_alpha 0.90"
)

run_one() {
  local cfg="$1"
  local gpu="$2"
  IFS='|' read -r tag algo dataset extra <<< "${cfg}"

  local log_file="${LOG_DIR}/${tag}_${RUN_TS}.log"
  local -a extra_flags=()
  if [[ -n "${extra}" ]]; then
    read -r -a extra_flags <<< "${extra}"
  fi

  echo "[START] gpu=${gpu} tag=${tag} log=${log_file}"
  "${PYTHON_BIN}" -u "${PROJECT_DIR}/main.py" \
    -dev cuda -did "${gpu}" \
    -data "${dataset}" \
    -m ResNetP \
    -algo "${algo}" \
    -ncl 10 -nc 40 -jr 1.0 -lbs 64 \
    -lr 0.003 -lr_head 0.01 \
    -ls 1 -pls 1 -gr 800 -eg 1 \
    -go "${tag}" \
    --num_adv_clients 5 \
    "${extra_flags[@]}" \
    > "${log_file}" 2>&1
  echo "[DONE] gpu=${gpu} tag=${tag}"
}

run_config_batches_lenient() {
  local -n configs_ref="$1"

  load_gpu_array

  local batch_size="${#GPU_LIST[@]}"
  local idx=0
  while (( idx < ${#configs_ref[@]} )); do
    local -a pids=()
    local -a labels=()
    local slot=0

    while (( slot < batch_size && idx < ${#configs_ref[@]} )); do
      local cfg="${configs_ref[$idx]}"
      local gpu="${GPU_LIST[$slot]}"
      local label
      label="$(printf '%s' "${cfg}" | awk -F'|' '{print $1}')"
      labels+=("${label}")
      run_one "${cfg}" "${gpu}" &
      pids+=("$!")
      idx=$((idx + 1))
      slot=$((slot + 1))
    done

    echo "Batch running: ${labels[*]}"
    local i
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        :
      else
        echo "${labels[$i]}" | tee -a "${FAIL_FILE}"
        echo "[FAILED] ${labels[$i]}"
      fi
    done
    echo "Batch finished: ${labels[*]}"
  done
}

echo "Run timestamp: ${RUN_TS}"
run_config_batches_lenient CONFIGS
collect_metrics "${LOG_DIR}" "*_${RUN_TS}.log" "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${CURVES_DIR}"
echo "Summary CSV : ${SUMMARY_CSV}"
echo "Summary JSON: ${SUMMARY_JSON}"
echo "Curves dir  : ${CURVES_DIR}"
if [[ -s "${FAIL_FILE}" ]]; then
  echo "Failed tasks:"
  sed 's/^/- /' "${FAIL_FILE}"
else
  rm -f "${FAIL_FILE}"
  echo "All remaining tasks finished."
fi
