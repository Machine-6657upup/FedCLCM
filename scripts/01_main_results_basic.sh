#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

LOG_DIR="${PROJECT_DIR}/thesis_log/main_results_basic"
RUN_TS="$(timestamp)"
SUMMARY_CSV="${LOG_DIR}/summary_${RUN_TS}.csv"
SUMMARY_JSON="${LOG_DIR}/summary_${RUN_TS}.json"
CURVES_DIR="${LOG_DIR}/curves_${RUN_TS}"

ensure_dir "${LOG_DIR}"

# tag|algo|dataset|model|ncl|nc|jr|lbs|lr|lr_head|ls|pls|gr|eg|num_adv|extra
CONFIGS=(
  "mnist_fedavg|FedAvg|MNIST_dir0.5_bdoor0.5_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|"
  "mnist_fedrep|FedRep|MNIST_dir0.5_bdoor0.5_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|"
  "mnist_fedmedian|FedMedian|MNIST_dir0.5_bdoor0.5_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|"
  "mnist_fedtrimmed|FedTrimmed|MNIST_dir0.5_bdoor0.5_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|"
  "mnist_fedbulyan|FedBulyan|MNIST_dir0.5_bdoor0.5_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|"
  "mnist_fedflip|FedFLIP|MNIST_dir0.5_bdoor0.5_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|"
  "mnist_fedclcm|FedCLCM|MNIST_dir0.5_bdoor0.5_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|--rt_beta 0.18 --lambda_cl 0.05 --aug_strength 0.03 --adv_eps 0.0 --adv_num_iter 0 --mask_tau 8.0 --mask_alpha 0.80 --trim_high_layers fc1 --trim_beta_high 0.18 --trim_beta_low 0.10"
  "fmnist_fedavg|FedAvg|FashionMNIST_dir0.5_bdoor0.2_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|"
  "fmnist_fedrep|FedRep|FashionMNIST_dir0.5_bdoor0.2_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|"
  "fmnist_fedmedian|FedMedian|FashionMNIST_dir0.5_bdoor0.2_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|"
  "fmnist_fedtrimmed|FedTrimmed|FashionMNIST_dir0.5_bdoor0.2_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|"
  "fmnist_fedbulyan|FedBulyan|FashionMNIST_dir0.5_bdoor0.2_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|"
  "fmnist_fedflip|FedFLIP|FashionMNIST_dir0.5_bdoor0.2_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|"
  "fmnist_fedclcm|FedCLCM|FashionMNIST_dir0.5_bdoor0.2_nclient_40_badnet_adv5|CNN|10|40|1.0|64|0.03|0.03|20|1|320|1|5|--rt_beta 0.18 --lambda_cl 0.05 --aug_strength 0.03 --adv_eps 0.0 --adv_num_iter 0 --mask_tau 8.0 --mask_alpha 0.80 --trim_high_layers fc1 --trim_beta_high 0.18 --trim_beta_low 0.10"
  "cifar_fedavg|FedAvg|Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5|ResNetP|10|40|1.0|64|0.003|0.01|1|1|800|1|5|"
  "cifar_fedrep|FedRep|Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5|ResNetP|10|40|1.0|64|0.003|0.01|1|1|800|1|5|"
  "cifar_fedmedian|FedMedian|Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5|ResNetP|10|40|1.0|64|0.003|0.01|1|1|800|1|5|"
  "cifar_fedtrimmed|FedTrimmed|Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5|ResNetP|10|40|1.0|64|0.003|0.01|1|1|800|1|5|"
  "cifar_fedbulyan|FedBulyan|Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5|ResNetP|10|40|1.0|64|0.003|0.01|1|1|800|1|5|"
  "cifar_fedflip|FedFLIP|Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5|ResNetP|10|40|1.0|64|0.003|0.01|1|1|800|1|5|"
  "cifar_fedclcm|FedCLCM|Cifar10_dir0.5_bdoor0.2_nclient_40_badnet_adv5|ResNetP|10|40|1.0|64|0.003|0.01|1|1|800|1|5|--rt_beta 0.0 --lambda_cl 0.10 --aug_strength 0.05 --adv_eps 0.0 --adv_num_iter 0 --mask_tau 10.0 --mask_alpha 0.90"
)

run_one() {
  local cfg="$1"
  local gpu="$2"
  IFS='|' read -r tag algo dataset model ncl nc jr lbs lr lr_head ls pls gr eg num_adv extra <<< "${cfg}"

  local log_file="${LOG_DIR}/${tag}_${RUN_TS}.log"
  local -a extra_flags=()
  if [[ -n "${extra}" ]]; then
    read -r -a extra_flags <<< "${extra}"
  fi

  echo "[START] gpu=${gpu} tag=${tag} log=${log_file}"
  "${PYTHON_BIN}" -u "${PROJECT_DIR}/main.py" \
    -dev cuda -did "${gpu}" \
    -data "${dataset}" \
    -m "${model}" \
    -algo "${algo}" \
    -ncl "${ncl}" -nc "${nc}" -jr "${jr}" -lbs "${lbs}" \
    -lr "${lr}" -lr_head "${lr_head}" \
    -ls "${ls}" -pls "${pls}" -gr "${gr}" -eg "${eg}" \
    -go "${tag}" \
    --num_adv_clients "${num_adv}" \
    "${extra_flags[@]}" \
    > "${log_file}" 2>&1
  echo "[DONE] gpu=${gpu} tag=${tag}"
}

echo "Run timestamp: ${RUN_TS}"
run_config_batches CONFIGS run_one 4
collect_metrics "${LOG_DIR}" "*_${RUN_TS}.log" "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${CURVES_DIR}"
echo "Summary CSV : ${SUMMARY_CSV}"
echo "Summary JSON: ${SUMMARY_JSON}"
echo "Curves dir  : ${CURVES_DIR}"
