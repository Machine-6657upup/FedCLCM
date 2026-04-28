#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
DATASET_ROOT="${SRC_ROOT}/dataset"
GEN_DIR="${DATASET_ROOT}/utils"
RAWDATA_ROOT="${DATASET_ROOT}/rawdata"
WRAPPER="${PROJECT_DIR}/scripts/generate_dataset_via_existing.py"
PYTHON_BIN="${PYTHON_BIN:-/home/huangtu/miniconda3/envs/torch/bin/python}"
if [[ "${PYTHON_BIN}" == "python" ]]; then
  PYTHON_BIN="/home/huangtu/miniconda3/envs/torch/bin/python"
fi
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_thesis_multiseed_static_3090}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
STATUS_DIR="${RUN_ROOT}/status"
DATASET_LOG_DIR="${RUN_ROOT}/dataset_logs"
MANIFEST="${RUN_ROOT}/manifest.csv"
DATASET_MANIFEST="${RUN_ROOT}/dataset_manifest.csv"
ROUNDS="${ROUNDS:-800}"
EVAL_GAP="${EVAL_GAP:-10}"
GPUS="${GPUS:-0 1 2 3}"

mkdir -p "${LOG_DIR}" "${STATUS_DIR}" "${DATASET_LOG_DIR}"

dataset_name() {
  local attack="$1"
  local seed="$2"
  echo "Cifar10_dir0.5_bdoor0.2_nclient_100_${attack}_adv10_seed${seed}"
}

generator_for_attack() {
  case "$1" in
    badnet) echo "generate_Cifar10_badnet.py" ;;
    blend) echo "generate_Cifar10_blend.py" ;;
    sig) echo "generate_Cifar10_sig.py" ;;
    *) echo "[FATAL] unknown attack $1" >&2; exit 1 ;;
  esac
}

ensure_dataset() {
  local attack="$1"
  local seed="$2"
  local dataset
  dataset="$(dataset_name "${attack}" "${seed}")"
  local generator
  generator="$(generator_for_attack "${attack}")"
  local log_file="${DATASET_LOG_DIR}/${dataset}.log"

  echo "[DATASET] ${dataset}"
  local -a cmd=(
    "${PYTHON_BIN}" "${WRAPPER}"
    --generator "${GEN_DIR}/${generator}"
    --dir-path "${DATASET_ROOT}/${dataset}"
    --rawdata-path "${RAWDATA_ROOT}"
    --num-clients 100
    --backdoor-rate 0.2
    --adversary-num 10
    --target-y 0
    --alpha 0.5
    --train-ratio 0.8
    --batch-size 10
    --partition dir
    --niid
    --balance
    --generator-seed "${seed}"
  )
  if [[ "${attack}" == "blend" ]]; then
    cmd+=(--blend-alpha 0.2)
  elif [[ "${attack}" == "sig" ]]; then
    cmd+=(--sig-delta 0.11764705882352941 --sig-f 6 --sig-label-mode dirty)
  fi
  "${cmd[@]}" > "${log_file}" 2>&1
  echo "${dataset},${attack},${seed},${generator},${log_file}" >> "${DATASET_MANIFEST}"
}

echo "dataset,attack,seed,generator,log_file" > "${DATASET_MANIFEST}"
for seed in 43 44; do
  for attack in badnet blend sig; do
    ensure_dataset "${attack}" "${seed}"
  done
done

cat > "${MANIFEST}" <<CSV
tag,gpu,algorithm,attack,dataset,seed,rounds,jr,lr,lr_head,local_epochs,purpose
CSV

declare -a Q0=()
declare -a Q1=()
declare -a Q2=()
declare -a Q3=()

add_task() {
  local queue="$1"
  local tag="$2"
  local algorithm="$3"
  local attack="$4"
  local seed="$5"
  local dataset
  dataset="$(dataset_name "${attack}" "${seed}")"
  local row="${tag}|${algorithm}|${attack}|${dataset}|${seed}"
  case "${queue}" in
    0) Q0+=("${row}") ;;
    1) Q1+=("${row}") ;;
    2) Q2+=("${row}") ;;
    3) Q3+=("${row}") ;;
    *) echo "[FATAL] bad queue ${queue}" >&2; exit 1 ;;
  esac
}

add_raw_task() {
  local queue="$1"
  local tag="$2"
  local algorithm="$3"
  local attack="$4"
  local dataset="$5"
  local seed="$6"
  local extra="$7"
  local row="${tag}|${algorithm}|${attack}|${dataset}|${seed}|${extra}"
  case "${queue}" in
    0) Q0+=("${row}") ;;
    1) Q1+=("${row}") ;;
    2) Q2+=("${row}") ;;
    3) Q3+=("${row}") ;;
    *) echo "[FATAL] bad queue ${queue}" >&2; exit 1 ;;
  esac
}

# Balance by expected cost and thesis value.
add_task 0 MS01_CLCM_BADNET_S43 FedCLCM badnet 43
add_task 0 MS02_CLCM_BLEND_S43 FedCLCM blend 43
add_task 0 MS03_CLCM_SIG_S43 FedCLCM sig 43
add_task 0 MS04_FEDAVG_BADNET_S43 FedAvg badnet 43
add_task 0 MS05_FEDAVG_BLEND_S43 FedAvg blend 43

add_task 1 MS06_FEDREP_BADNET_S43 FedRep badnet 43
add_task 1 MS07_FEDREP_BLEND_S43 FedRep blend 43
add_task 1 MS08_FEDREP_SIG_S43 FedRep sig 43
add_task 1 MS09_FEDAVG_SIG_S43 FedAvg sig 43

add_task 2 MS10_CLCM_BADNET_S44 FedCLCM badnet 44
add_task 2 MS11_CLCM_BLEND_S44 FedCLCM blend 44
add_task 2 MS12_CLCM_SIG_S44 FedCLCM sig 44
add_task 2 MS13_FEDAVG_BADNET_S44 FedAvg badnet 44
add_task 2 MS14_FEDAVG_BLEND_S44 FedAvg blend 44

add_task 3 MS15_FEDREP_BADNET_S44 FedRep badnet 44
add_task 3 MS16_FEDREP_BLEND_S44 FedRep blend 44
add_task 3 MS17_FEDREP_SIG_S44 FedRep sig 44
add_task 3 MS18_FEDAVG_SIG_S44 FedAvg sig 44

# FedCLCM static best-parameter exploration on the canonical seed42 datasets.
# These runs are intentionally extra: they are for finding a better ACC/ASR
# tradeoff and for enriching sensitivity/robustness sections.
BD="Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10"
BL="Cifar10_dir0.5_bdoor0.2_nclient_100_blend_adv10"
SG="Cifar10_dir0.5_bdoor0.2_nclient_100_sig_adv10"

add_raw_task 0 HP01_BN_T8A05 FedCLCM badnet "${BD}" 42 "mask_tau=8.0 mask_alpha=0.5"
add_raw_task 0 HP02_BN_T10A07 FedCLCM badnet "${BD}" 42 "mask_tau=10.0 mask_alpha=0.7"
add_raw_task 0 HP03_BN_T10A09 FedCLCM badnet "${BD}" 42 "mask_tau=10.0 mask_alpha=0.9"
add_raw_task 0 HP04_BN_LR005 FedCLCM badnet "${BD}" 42 "lr=0.05 lr_head=0.05"
add_raw_task 0 HP05_BN_LR02 FedCLCM badnet "${BD}" 42 "lr=0.2 lr_head=0.2"

add_raw_task 1 HP06_BL_T8A05 FedCLCM blend "${BL}" 42 "mask_tau=8.0 mask_alpha=0.5"
add_raw_task 1 HP07_BL_T10A07 FedCLCM blend "${BL}" 42 "mask_tau=10.0 mask_alpha=0.7"
add_raw_task 1 HP08_BL_LAM01 FedCLCM blend "${BL}" 42 "lambda_cl=0.1"
add_raw_task 1 HP09_BL_LAM05 FedCLCM blend "${BL}" 42 "lambda_cl=0.5"
add_raw_task 1 HP10_BL_LR005 FedCLCM blend "${BL}" 42 "lr=0.05 lr_head=0.05"

add_raw_task 2 HP11_SIG_T6A02 FedCLCM sig "${SG}" 42 "mask_tau=6.0 mask_alpha=0.2"
add_raw_task 2 HP12_SIG_T8A05 FedCLCM sig "${SG}" 42 "mask_tau=8.0 mask_alpha=0.5"
add_raw_task 2 HP13_SIG_T10A07 FedCLCM sig "${SG}" 42 "mask_tau=10.0 mask_alpha=0.7"
add_raw_task 2 HP14_SIG_LAM01 FedCLCM sig "${SG}" 42 "lambda_cl=0.1"
add_raw_task 2 HP15_SIG_LAM05 FedCLCM sig "${SG}" 42 "lambda_cl=0.5"

add_raw_task 3 HP16_BN_LE2_PLE2 FedCLCM badnet "${BD}" 42 "local_epochs=2 plocal_epochs=2"
add_raw_task 3 HP17_BL_LE2_PLE2 FedCLCM blend "${BL}" 42 "local_epochs=2 plocal_epochs=2"
add_raw_task 3 HP18_SIG_LE2_PLE2 FedCLCM sig "${SG}" 42 "local_epochs=2 plocal_epochs=2"
add_raw_task 3 HP19_BN_PLE5 FedCLCM badnet "${BD}" 42 "plocal_epochs=5"
add_raw_task 3 HP20_BL_PLE5 FedCLCM blend "${BL}" 42 "plocal_epochs=5"
add_raw_task 0 HP21_BN_T4A01 FedCLCM badnet "${BD}" 42 "mask_tau=4.0 mask_alpha=0.1"
add_raw_task 0 HP22_BN_T6A02 FedCLCM badnet "${BD}" 42 "mask_tau=6.0 mask_alpha=0.2"
add_raw_task 1 HP23_BN_LAM01 FedCLCM badnet "${BD}" 42 "lambda_cl=0.1"
add_raw_task 1 HP24_BN_LAM05 FedCLCM badnet "${BD}" 42 "lambda_cl=0.5"
add_raw_task 2 HP25_BN_LE5 FedCLCM badnet "${BD}" 42 "local_epochs=5"
add_raw_task 2 HP26_BN_PLE10 FedCLCM badnet "${BD}" 42 "plocal_epochs=10"
add_raw_task 3 HP27_BL_LE5 FedCLCM blend "${BL}" 42 "local_epochs=5"
add_raw_task 3 HP28_BL_PLE10 FedCLCM blend "${BL}" 42 "plocal_epochs=10"
add_raw_task 0 HP29_SIG_LE5 FedCLCM sig "${SG}" 42 "local_epochs=5"
add_raw_task 1 HP30_SIG_PLE5 FedCLCM sig "${SG}" 42 "plocal_epochs=5"
add_raw_task 2 HP31_SIG_PLE10 FedCLCM sig "${SG}" 42 "plocal_epochs=10"

write_manifest_row() {
  local gpu="$1"
  local item="$2"
  IFS='|' read -r tag algorithm attack dataset seed extra <<< "${item}"
  echo "${tag},${gpu},${algorithm},${attack},${dataset},${seed},${ROUNDS},0.1,0.1,0.1,1,${extra:-multiseed main table}" >> "${MANIFEST}"
}
for item in "${Q0[@]}"; do write_manifest_row 0 "${item}"; done
for item in "${Q1[@]}"; do write_manifest_row 1 "${item}"; done
for item in "${Q2[@]}"; do write_manifest_row 2 "${item}"; done
for item in "${Q3[@]}"; do write_manifest_row 3 "${item}"; done

run_one() {
  local gpu="$1"
  local item="$2"
  IFS='|' read -r tag algorithm attack dataset seed extra <<< "${item}"
  local log_file="${LOG_DIR}/${tag}.log"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"
  local lr=0.1
  local lr_head=0.1
  local local_epochs=1
  local plocal_epochs=1
  local rt_beta=0.2
  local lambda_cl=0.2
  local aug_strength=0.1
  local mask_tau=6.0
  local mask_alpha=0.3

  echo "[RUN] ${tag} gpu=${gpu} algo=${algorithm} attack=${attack} seed=${seed}"
  rm -f "${ok_file}" "${fail_file}"

  if [[ -n "${extra:-}" ]]; then
    for kv in ${extra}; do
      case "${kv}" in
        lr=*) lr="${kv#lr=}" ;;
        lr_head=*) lr_head="${kv#lr_head=}" ;;
        local_epochs=*) local_epochs="${kv#local_epochs=}" ;;
        plocal_epochs=*) plocal_epochs="${kv#plocal_epochs=}" ;;
        rt_beta=*) rt_beta="${kv#rt_beta=}" ;;
        lambda_cl=*) lambda_cl="${kv#lambda_cl=}" ;;
        aug_strength=*) aug_strength="${kv#aug_strength=}" ;;
        mask_tau=*) mask_tau="${kv#mask_tau=}" ;;
        mask_alpha=*) mask_alpha="${kv#mask_alpha=}" ;;
        *) echo "[WARN] ${tag}: ignored extra token ${kv}" ;;
      esac
    done
  fi

  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda -did "${gpu}"
    -data "${dataset}"
    -m ResNet18
    -algo "${algorithm}"
    -ncl 10 -nc 100 -jr 0.1 -lbs 64
    -lr "${lr}" -lr_head "${lr_head}" -ls "${local_epochs}" -pls "${plocal_epochs}"
    -gr "${ROUNDS}" -eg "${EVAL_GAP}"
    -go "${tag}"
    --num_adv_clients 10
  )

  if [[ "${algorithm}" == "FedCLCM" ]]; then
    cmd+=(--rt_beta "${rt_beta}" --lambda_cl "${lambda_cl}" --aug_strength "${aug_strength}" --mask_tau "${mask_tau}" --mask_alpha "${mask_alpha}" --enable_channel_mask true --adv_eps 0.0 --adv_num_iter 0)
  fi

  set +e
  (cd "${SRC_ROOT}" && "${cmd[@]}" > "${log_file}" 2>&1)
  local rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    echo "ok rc=0 $(date '+%F %T')" > "${ok_file}"
    echo "[OK] ${tag}"
  else
    echo "fail rc=${rc} $(date '+%F %T')" > "${fail_file}"
    echo "[FAIL] ${tag} rc=${rc}" >&2
  fi
}

run_queue() {
  local gpu="$1"
  shift
  local -a queue=("$@")
  for item in "${queue[@]}"; do
    run_one "${gpu}" "${item}"
  done
}

run_queue 0 "${Q0[@]}" &
pid0=$!
run_queue 1 "${Q1[@]}" &
pid1=$!
run_queue 2 "${Q2[@]}" &
pid2=$!
run_queue 3 "${Q3[@]}" &
pid3=$!

wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"
echo "[READY] ${RUN_ROOT}"
