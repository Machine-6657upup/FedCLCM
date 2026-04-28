#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
GPU_ID="${GPU_ID:-1}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_TS}_static_attack_sanity_fedavg_fedrep"
LOG_DIR="${RUN_ROOT}/train_logs"
STATUS_DIR="${RUN_ROOT}/status"
MANIFEST="${RUN_ROOT}/manifest.tsv"

mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

cat > "${MANIFEST}" <<EOF
tag	gpu	algorithm	attack	dataset	model	num_clients	adv_clients	join_ratio	lr	lr_head	local_epochs	rounds	purpose
SA01_FEDAVG_BADNET	${GPU_ID}	FedAvg	badnet	Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10	ResNet18	100	10	0.1	0.1	-	1	800	attack sanity: no-defense FedAvg under BadNet
SA02_FEDREP_BADNET	${GPU_ID}	FedRep	badnet	Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10	ResNet18	100	10	0.1	0.1	0.1	1	800	attack sanity: FedRep under BadNet
SA03_FEDAVG_BLEND	${GPU_ID}	FedAvg	blend	Cifar10_dir0.5_bdoor0.2_nclient_100_blend_adv10	ResNet18	100	10	0.1	0.1	-	1	800	attack sanity: no-defense FedAvg under Blend
SA04_FEDREP_BLEND	${GPU_ID}	FedRep	blend	Cifar10_dir0.5_bdoor0.2_nclient_100_blend_adv10	ResNet18	100	10	0.1	0.1	0.1	1	800	attack sanity: FedRep under Blend
SA05_FEDAVG_SIG	${GPU_ID}	FedAvg	sig	Cifar10_dir0.5_bdoor0.2_nclient_100_sig_adv10	ResNet18	100	10	0.1	0.1	-	1	800	attack sanity: no-defense FedAvg under SIG
SA06_FEDREP_SIG	${GPU_ID}	FedRep	sig	Cifar10_dir0.5_bdoor0.2_nclient_100_sig_adv10	ResNet18	100	10	0.1	0.1	0.1	1	800	attack sanity: FedRep under SIG
EOF

run_one() {
  local tag="$1"
  local algorithm="$2"
  local dataset="$3"
  local attack="$4"
  local log_file="${LOG_DIR}/${tag}.log"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"

  echo "[RUN] ${tag} algo=${algorithm} attack=${attack} dataset=${dataset} gpu=${GPU_ID}"
  rm -f "${ok_file}" "${fail_file}"

  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda -did "${GPU_ID}"
    -data "${dataset}"
    -m ResNet18
    -algo "${algorithm}"
    -ncl 10
    -nc 100
    -jr 0.1
    -lbs 64
    -lr 0.1
    -ls 1
    -gr 800
    -eg 10
    -go "${tag}"
    --num_adv_clients 10
  )

  if [[ "${algorithm}" == "FedRep" ]]; then
    cmd+=( -lr_head 0.1 -pls 1 )
  fi

  set +e
  (
    cd "${SRC_ROOT}"
    "${cmd[@]}" > "${log_file}" 2>&1
  )
  local rc=$?
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    echo "ok rc=0 $(date '+%F %T')" > "${ok_file}"
    echo "[OK] ${tag}"
  else
    echo "fail rc=${rc} $(date '+%F %T')" > "${fail_file}"
    echo "[FAIL] ${tag} rc=${rc}"
    return "${rc}"
  fi
}

for dataset in \
  Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10 \
  Cifar10_dir0.5_bdoor0.2_nclient_100_blend_adv10 \
  Cifar10_dir0.5_bdoor0.2_nclient_100_sig_adv10; do
  if [[ ! -f "${SRC_ROOT}/dataset/${dataset}/config.json" ]]; then
    echo "[FATAL] missing dataset ${dataset}" >&2
    exit 1
  fi
done

run_one SA01_FEDAVG_BADNET FedAvg Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10 badnet
run_one SA02_FEDREP_BADNET FedRep Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10 badnet
run_one SA03_FEDAVG_BLEND FedAvg Cifar10_dir0.5_bdoor0.2_nclient_100_blend_adv10 blend
run_one SA04_FEDREP_BLEND FedRep Cifar10_dir0.5_bdoor0.2_nclient_100_blend_adv10 blend
run_one SA05_FEDAVG_SIG FedAvg Cifar10_dir0.5_bdoor0.2_nclient_100_sig_adv10 sig
run_one SA06_FEDREP_SIG FedRep Cifar10_dir0.5_bdoor0.2_nclient_100_sig_adv10 sig

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/summarize_thesis_logs.py" "${LOG_DIR}" > "${RUN_ROOT}/summary.tsv" || true
echo "[READY] ${RUN_ROOT}"
