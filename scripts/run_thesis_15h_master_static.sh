#!/usr/bin/env bash
set -euo pipefail

THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${THIS_SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
SERVER_ROLE="${SERVER_ROLE:-3090}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_thesis_15h_master_${SERVER_ROLE}}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
STATUS_DIR="${RUN_ROOT}/status"
CURVES_DIR="${RUN_ROOT}/curves"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
MANIFEST="${RUN_ROOT}/manifest.tsv"
DATASET_MANIFEST="${RUN_ROOT}/dataset_manifest.tsv"
REFERENCE_DIR="${RUN_ROOT}/reference_only"

ROUNDS="${ROUNDS:-800}"
EVAL_GAP="${EVAL_GAP:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PREPARE_DATASETS="${PREPARE_DATASETS:-1}"

ensure_dir "${RUN_ROOT}"
ensure_dir "${LOG_DIR}"
ensure_dir "${STATUS_DIR}"
ensure_dir "${CURVES_DIR}"
ensure_dir "${REFERENCE_DIR}"

load_gpu_array

ds() {
  local alpha="$1"
  local bdoor="$2"
  local attack="$3"
  local adv="$4"
  echo "Cifar10_dir${alpha}_bdoor${bdoor}_nclient_100_${attack}_adv${adv}"
}

declare -a DATASET_SPECS=()
add_dataset_spec() {
  DATASET_SPECS+=("$1|$2|$3")
}

declare -a CONFIGS=()
add_task() {
  local queue="$1"
  local tag="$2"
  local dataset="$3"
  local attack="$4"
  local algorithm="$5"
  local adv_clients="$6"
  local join_ratio="$7"
  local local_epochs="$8"
  local lambda_cl="$9"
  local mask_tau="${10}"
  local mask_alpha="${11}"
  local channel_mask="${12}"
  local purpose="${13}"
  CONFIGS+=("${queue}|${tag}|${dataset}|${attack}|${algorithm}|${adv_clients}|${join_ratio}|${local_epochs}|${lambda_cl}|${mask_tau}|${mask_alpha}|${channel_mask}|${purpose}")
}

add_common_dataset_specs() {
  add_dataset_spec "0.5" "0.2" "10"
  add_dataset_spec "0.1" "0.2" "10"
  add_dataset_spec "1.0" "0.2" "10"
  add_dataset_spec "0.5" "0.1" "10"
  add_dataset_spec "0.5" "0.3" "10"
  add_dataset_spec "0.5" "0.2" "5"
  add_dataset_spec "0.5" "0.2" "20"
}

build_3090_configs() {
  local bd_default blend_default sig_default
  bd_default="$(ds 0.5 0.2 badnet 10)"
  blend_default="$(ds 0.5 0.2 blend 10)"
  sig_default="$(ds 0.5 0.2 sig 10)"

  # GPU0: FedCLCM main effectiveness, module ablation, and mask/lambda sensitivity.
  add_task 0 "F00_CLCM_BADNET_T6A03_800" "${bd_default}" "badnet" "FedCLCM" 10 0.1 1 0.2 6.0 0.3 true "main effect: FedCLCM vs BadNet, strict 800-round formal run"
  add_task 0 "F01_CLCM_BLEND_T6A03_800" "${blend_default}" "blend" "FedCLCM" 10 0.1 1 0.2 6.0 0.3 true "main effect: FedCLCM vs Blend, strict 800-round formal run"
  add_task 0 "F02_CLCM_SIG_T6A03_800" "${sig_default}" "sig" "FedCLCM" 10 0.1 1 0.2 6.0 0.3 true "main effect: FedCLCM vs SIG, strict 800-round formal run"
  add_task 0 "A01_BADNET_NO_CL" "${bd_default}" "badnet" "FedCLCM" 10 0.1 1 0.0 6.0 0.3 true "ablation: remove contrastive loss"
  add_task 0 "A02_BADNET_NO_MASK" "${bd_default}" "badnet" "FedCLCM" 10 0.1 1 0.2 6.0 1.0 false "ablation: remove channel mask"
  add_task 0 "A03_BADNET_NO_BOTH" "${bd_default}" "badnet" "FedCLCM" 10 0.1 1 0.0 6.0 1.0 false "ablation: remove contrastive loss and channel mask"
  add_task 0 "A04_BLEND_NO_CL" "${blend_default}" "blend" "FedCLCM" 10 0.1 1 0.0 6.0 0.3 true "ablation: remove contrastive loss on Blend"
  add_task 0 "A05_BLEND_NO_MASK" "${blend_default}" "blend" "FedCLCM" 10 0.1 1 0.2 6.0 1.0 false "ablation: remove channel mask on Blend"
  add_task 0 "A06_SIG_NO_CL" "${sig_default}" "sig" "FedCLCM" 10 0.1 1 0.0 6.0 0.3 true "ablation: remove contrastive loss on SIG"
  add_task 0 "A07_SIG_NO_MASK" "${sig_default}" "sig" "FedCLCM" 10 0.1 1 0.2 6.0 1.0 false "ablation: remove channel mask on SIG"
  add_task 0 "S01_BADNET_T5A02" "${bd_default}" "badnet" "FedCLCM" 10 0.1 1 0.2 5.0 0.2 true "sensitivity: stronger channel mask"
  add_task 0 "S02_BADNET_T12A07" "${bd_default}" "badnet" "FedCLCM" 10 0.1 1 0.2 12.0 0.7 true "sensitivity: old good mask setting under current architecture"

  # GPU1: defense baselines beyond FedAvg/FedRep.
  add_task 1 "M01_MEDIAN_BADNET" "${bd_default}" "badnet" "FedMedian" 10 0.1 1 0.2 6.0 0.3 true "baseline: coordinate median"
  add_task 1 "M02_MEDIAN_BLEND" "${blend_default}" "blend" "FedMedian" 10 0.1 1 0.2 6.0 0.3 true "baseline: coordinate median"
  add_task 1 "M03_MEDIAN_SIG" "${sig_default}" "sig" "FedMedian" 10 0.1 1 0.2 6.0 0.3 true "baseline: coordinate median"
  add_task 1 "T01_TRIM_BADNET" "${bd_default}" "badnet" "FedTrimmed" 10 0.1 1 0.2 6.0 0.3 true "baseline: selected-adv-count trimmed mean"
  add_task 1 "T02_TRIM_BLEND" "${blend_default}" "blend" "FedTrimmed" 10 0.1 1 0.2 6.0 0.3 true "baseline: selected-adv-count trimmed mean"
  add_task 1 "T03_TRIM_SIG" "${sig_default}" "sig" "FedTrimmed" 10 0.1 1 0.2 6.0 0.3 true "baseline: selected-adv-count trimmed mean"
  add_task 1 "K01_MK_BADNET" "${bd_default}" "badnet" "FedMK" 10 0.1 1 0.2 6.0 0.3 true "baseline: Multi-Krum-style robust aggregation"
  add_task 1 "K02_MK_BLEND" "${blend_default}" "blend" "FedMK" 10 0.1 1 0.2 6.0 0.3 true "baseline: Multi-Krum-style robust aggregation"
  add_task 1 "K03_MK_SIG" "${sig_default}" "sig" "FedMK" 10 0.1 1 0.2 6.0 0.3 true "baseline: Multi-Krum-style robust aggregation"
  add_task 1 "B01_BULYAN_BADNET" "${bd_default}" "badnet" "FedBulyan" 10 0.1 1 0.2 6.0 0.3 true "baseline: Bulyan-style robust aggregation"
  add_task 1 "B02_BULYAN_BLEND" "${blend_default}" "blend" "FedBulyan" 10 0.1 1 0.2 6.0 0.3 true "baseline: Bulyan-style robust aggregation"
  add_task 1 "B03_BULYAN_SIG" "${sig_default}" "sig" "FedBulyan" 10 0.1 1 0.2 6.0 0.3 true "baseline: Bulyan-style robust aggregation"

  # GPU2: default-dataset FedCLCM sensitivity. New dataset splits are left to 4090.
  add_task 2 "S16_BADNET_T5A03" "${bd_default}" "badnet" "FedCLCM" 10 0.1 1 0.2 5.0 0.3 true "mask sensitivity: lower tau only"
  add_task 2 "S17_BADNET_T6A02" "${bd_default}" "badnet" "FedCLCM" 10 0.1 1 0.2 6.0 0.2 true "mask sensitivity: stronger alpha only"
  add_task 2 "S18_BADNET_T7A03" "${bd_default}" "badnet" "FedCLCM" 10 0.1 1 0.2 7.0 0.3 true "mask sensitivity: higher tau only"
  add_task 2 "S19_BADNET_T6A04" "${bd_default}" "badnet" "FedCLCM" 10 0.1 1 0.2 6.0 0.4 true "mask sensitivity: milder alpha only"
  add_task 2 "S20_BLEND_T6A02" "${blend_default}" "blend" "FedCLCM" 10 0.1 1 0.2 6.0 0.2 true "mask sensitivity: Blend stronger alpha"
  add_task 2 "S21_SIG_T6A02" "${sig_default}" "sig" "FedCLCM" 10 0.1 1 0.2 6.0 0.2 true "mask sensitivity: SIG stronger alpha"
  add_task 2 "S22_BLEND_T5A03" "${blend_default}" "blend" "FedCLCM" 10 0.1 1 0.2 5.0 0.3 true "mask sensitivity: Blend lower tau"
  add_task 2 "S23_SIG_T5A03" "${sig_default}" "sig" "FedCLCM" 10 0.1 1 0.2 5.0 0.3 true "mask sensitivity: SIG lower tau"
  add_task 2 "J01_BADNET_JR005" "${bd_default}" "badnet" "FedCLCM" 10 0.05 1 0.2 6.0 0.3 true "system sensitivity: join_ratio=0.05"
  add_task 2 "J02_BADNET_JR02" "${bd_default}" "badnet" "FedCLCM" 10 0.2 1 0.2 6.0 0.3 true "system sensitivity: join_ratio=0.2"
  add_task 2 "J03_BLEND_JR005" "${blend_default}" "blend" "FedCLCM" 10 0.05 1 0.2 6.0 0.3 true "system sensitivity: Blend join_ratio=0.05"
  add_task 2 "J04_SIG_JR02" "${sig_default}" "sig" "FedCLCM" 10 0.2 1 0.2 6.0 0.3 true "system sensitivity: SIG join_ratio=0.2"

  # GPU3: PFL baselines and local training sensitivity.
  add_task 3 "PFL01_FEDPROTO_BADNET" "${bd_default}" "badnet" "FedProto" 10 0.1 1 0.2 6.0 0.3 true "PFL baseline: FedProto"
  add_task 3 "PFL02_FEDPROTO_BLEND" "${blend_default}" "blend" "FedProto" 10 0.1 1 0.2 6.0 0.3 true "PFL baseline: FedProto"
  add_task 3 "PFL03_FEDPROTO_SIG" "${sig_default}" "sig" "FedProto" 10 0.1 1 0.2 6.0 0.3 true "PFL baseline: FedProto"
  add_task 3 "PFL04_FEDPD_BADNET" "${bd_default}" "badnet" "FedPD" 10 0.1 1 0.2 6.0 0.3 true "PFL baseline: local FedPD/prototype-defense variant"
  add_task 3 "PFL05_FEDPD_BLEND" "${blend_default}" "blend" "FedPD" 10 0.1 1 0.2 6.0 0.3 true "PFL baseline: local FedPD/prototype-defense variant"
  add_task 3 "PFL06_FEDPD_SIG" "${sig_default}" "sig" "FedPD" 10 0.1 1 0.2 6.0 0.3 true "PFL baseline: local FedPD/prototype-defense variant"
  add_task 3 "L01_BADNET_LE2" "${bd_default}" "badnet" "FedCLCM" 10 0.1 2 0.2 6.0 0.3 true "training sensitivity: local_epochs=2"
  add_task 3 "L02_BADNET_LE5" "${bd_default}" "badnet" "FedCLCM" 10 0.1 5 0.2 6.0 0.3 true "training sensitivity: local_epochs=5"
  add_task 3 "L03_BADNET_LE10" "${bd_default}" "badnet" "FedCLCM" 10 0.1 10 0.2 6.0 0.3 true "training sensitivity: local_epochs=10"
  add_task 3 "L04_BLEND_LE2" "${blend_default}" "blend" "FedCLCM" 10 0.1 2 0.2 6.0 0.3 true "training sensitivity: Blend local_epochs=2"
  add_task 3 "L05_SIG_LE2" "${sig_default}" "sig" "FedCLCM" 10 0.1 2 0.2 6.0 0.3 true "training sensitivity: SIG local_epochs=2"
  add_task 3 "L06_BLEND_LE5" "${blend_default}" "blend" "FedCLCM" 10 0.1 5 0.2 6.0 0.3 true "training sensitivity: Blend local_epochs=5"
}

build_4090_configs() {
  local bd_default blend_default sig_default
  bd_default="$(ds 0.5 0.2 badnet 10)"
  blend_default="$(ds 0.5 0.2 blend 10)"
  sig_default="$(ds 0.5 0.2 sig 10)"

  add_task 0 "H01_BADNET_DIR01" "$(ds 0.1 0.2 badnet 10)" "badnet" "FedCLCM" 10 0.1 1 0.2 6.0 0.3 true "data heterogeneity: Dirichlet alpha=0.1"
  add_task 0 "H02_BADNET_DIR10" "$(ds 1.0 0.2 badnet 10)" "badnet" "FedCLCM" 10 0.1 1 0.2 6.0 0.3 true "data heterogeneity: Dirichlet alpha=1.0"
  add_task 0 "H03_BLEND_DIR01" "$(ds 0.1 0.2 blend 10)" "blend" "FedCLCM" 10 0.1 1 0.2 6.0 0.3 true "data heterogeneity: Blend alpha=0.1"
  add_task 0 "H04_SIG_DIR01" "$(ds 0.1 0.2 sig 10)" "sig" "FedCLCM" 10 0.1 1 0.2 6.0 0.3 true "data heterogeneity: SIG alpha=0.1"
  add_task 0 "V01_BADNET_ADV5" "$(ds 0.5 0.2 badnet 5)" "badnet" "FedCLCM" 5 0.1 1 0.2 6.0 0.3 true "malicious ratio: 5 adversarial clients"
  add_task 0 "V02_BADNET_ADV20" "$(ds 0.5 0.2 badnet 20)" "badnet" "FedCLCM" 20 0.1 1 0.2 6.0 0.3 true "malicious ratio: 20 adversarial clients"
  add_task 0 "V03_BLEND_ADV20" "$(ds 0.5 0.2 blend 20)" "blend" "FedCLCM" 20 0.1 1 0.2 6.0 0.3 true "malicious ratio: Blend with 20 adversarial clients"
  add_task 0 "V04_SIG_ADV20" "$(ds 0.5 0.2 sig 20)" "sig" "FedCLCM" 20 0.1 1 0.2 6.0 0.3 true "malicious ratio: SIG with 20 adversarial clients"
  add_task 0 "P01_BADNET_BDOOR01" "$(ds 0.5 0.1 badnet 10)" "badnet" "FedCLCM" 10 0.1 1 0.2 6.0 0.3 true "attack strength: backdoor rate=0.1"
  add_task 0 "P02_BADNET_BDOOR03" "$(ds 0.5 0.3 badnet 10)" "badnet" "FedCLCM" 10 0.1 1 0.2 6.0 0.3 true "attack strength: backdoor rate=0.3"
  add_task 0 "P03_BLEND_BDOOR03" "$(ds 0.5 0.3 blend 10)" "blend" "FedCLCM" 10 0.1 1 0.2 6.0 0.3 true "attack strength: Blend backdoor rate=0.3"
  add_task 0 "P04_SIG_BDOOR03" "$(ds 0.5 0.3 sig 10)" "sig" "FedCLCM" 10 0.1 1 0.2 6.0 0.3 true "attack strength: SIG backdoor rate=0.3"
  add_task 0 "Z01_BADNET_LE20" "${bd_default}" "badnet" "FedCLCM" 10 0.1 20 0.2 6.0 0.3 true "training sensitivity: BadNet local_epochs=20"
  add_task 0 "Z02_BLEND_LE10" "${blend_default}" "blend" "FedCLCM" 10 0.1 10 0.2 6.0 0.3 true "training sensitivity: Blend local_epochs=10"
  add_task 0 "Z03_SIG_LE10" "${sig_default}" "sig" "FedCLCM" 10 0.1 10 0.2 6.0 0.3 true "training sensitivity: SIG local_epochs=10"
  add_task 0 "Z04_BADNET_LAM05" "${bd_default}" "badnet" "FedCLCM" 10 0.1 1 0.5 6.0 0.3 true "contrastive sensitivity: lambda_cl=0.5"
}

copy_reference_logs() {
  local pfedba_dir="/home/fch/FedCLCM_purify/runs/20260427_pfedba_clcmpur_priority_4090/train_logs"
  if [[ -d "${pfedba_dir}" ]]; then
    cp -f "${pfedba_dir}"/PF01_CLCM_FORMAL_T12A07.log "${REFERENCE_DIR}/" 2>/dev/null || true
    cp -f "${pfedba_dir}"/PF03_CLCM_T6A03_BASE.log "${REFERENCE_DIR}/" 2>/dev/null || true
  fi
}

prepare_datasets() {
  echo -e "alpha\tbdoor_rate\tadv_clients\tstatus" > "${DATASET_MANIFEST}"
  for spec in "${DATASET_SPECS[@]}"; do
    IFS='|' read -r alpha bdoor adv <<< "${spec}"
    echo "[DATASET] alpha=${alpha} bdoor=${bdoor} adv=${adv}"
    (
      cd "${PROJECT_DIR}"
      PYTHON_BIN="${PYTHON_BIN}" NUM_CLIENTS=100 ADV_CLIENTS="${adv}" \
        BACKDOOR_RATE="${bdoor}" TARGET_LABEL=0 ALPHA="${alpha}" FORCE_REBUILD=0 \
        bash "${THIS_SCRIPT_DIR}/prepare_static_datasets_srcutils.sh"
    )
    echo -e "${alpha}\t${bdoor}\t${adv}\tensured" >> "${DATASET_MANIFEST}"
  done
}

if [[ "${SERVER_ROLE}" == "3090" ]]; then
  add_common_dataset_specs
  build_3090_configs
elif [[ "${SERVER_ROLE}" == "4090" ]]; then
  add_common_dataset_specs
  build_4090_configs
else
  echo "[FATAL] SERVER_ROLE must be 3090 or 4090, got ${SERVER_ROLE}" >&2
  exit 1
fi

copy_reference_logs

if [[ "${PREPARE_DATASETS}" == "1" ]]; then
  prepare_datasets
fi

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r _ _ dataset _ _ _ _ _ _ _ _ _ _ <<< "${cfg}"
  if [[ ! -f "${SRC_ROOT}/dataset/${dataset}/config.json" ]]; then
    echo "[FATAL] missing dataset ${dataset}" >&2
    exit 1
  fi
done

echo -e "queue\ttag\tdataset\tattack\talgorithm\tmodel\tadv_clients\tjoin_ratio\tlocal_lr\tlr_head\tlocal_epochs\tplocal_epochs\trounds\teval_gap\trt_beta\tlambda_cl\taug_strength\tmask_tau\tmask_alpha\tenable_channel_mask\tpurpose" > "${MANIFEST}"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r queue tag dataset attack algorithm adv_clients join_ratio local_epochs lambda_cl mask_tau mask_alpha channel_mask purpose <<< "${cfg}"
  echo -e "${queue}\t${tag}\t${dataset}\t${attack}\t${algorithm}\tResNet18\t${adv_clients}\t${join_ratio}\t0.1\t0.1\t${local_epochs}\t1\t${ROUNDS}\t${EVAL_GAP}\t0.2\t${lambda_cl}\t0.1\t${mask_tau}\t${mask_alpha}\t${channel_mask}\t${purpose}" >> "${MANIFEST}"
done

run_one() {
  local cfg="$1"
  local gpu="$2"
  IFS='|' read -r queue tag dataset attack algorithm adv_clients join_ratio local_epochs lambda_cl mask_tau mask_alpha channel_mask purpose <<< "${cfg}"
  local log_file="${LOG_DIR}/${tag}.log"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"

  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda -did "${gpu}"
    -data "${dataset}"
    -m ResNet18
    -algo "${algorithm}"
    -ncl 10 -nc 100 -jr "${join_ratio}" -lbs "${BATCH_SIZE}"
    -lr 0.1 -lr_head 0.1 -ls "${local_epochs}" -pls 1
    -gr "${ROUNDS}" -eg "${EVAL_GAP}"
    -go "${tag}"
    --num_adv_clients "${adv_clients}"
    --rt_beta 0.2
    --lambda_cl "${lambda_cl}"
    --aug_strength 0.1
    --mask_tau "${mask_tau}"
    --mask_alpha "${mask_alpha}"
    --enable_channel_mask "${channel_mask}"
    --adv_eps 0.0
    --adv_num_iter 0
  )

  echo "[RUN] tag=${tag} gpu=${gpu} algo=${algorithm} dataset=${dataset} jr=${join_ratio} le=${local_epochs} purpose=${purpose}"
  rm -f "${ok_file}" "${fail_file}"
  set +e
  (
    cd "${SRC_ROOT}"
    "${cmd[@]}" > "${log_file}" 2>&1
  )
  local status=$?
  set -e

  if [[ "${status}" -eq 0 ]]; then
    echo "ok $(date '+%F %T')" > "${ok_file}"
    echo "[OK] ${tag}"
  else
    echo "fail status=${status} $(date '+%F %T')" > "${fail_file}"
    echo "[FAIL] ${tag} status=${status}"
  fi
}

run_queue() {
  local queue_idx="$1"
  local gpu="$2"
  echo "[QUEUE] queue=${queue_idx} gpu=${gpu} start"
  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r queue _ <<< "${cfg}"
    if [[ "${queue}" == "${queue_idx}" ]]; then
      run_one "${cfg}" "${gpu}"
    fi
  done
  echo "[QUEUE] queue=${queue_idx} gpu=${gpu} done"
}

declare -a pids=()
for idx in "${!GPU_LIST[@]}"; do
  run_queue "${idx}" "${GPU_LIST[$idx]}" &
  pids+=("$!")
done

wait "${pids[@]}"

collect_metrics "${LOG_DIR}" "*.log" "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${CURVES_DIR}" || true

echo "[READY] run_root=${RUN_ROOT}"
echo "[READY] manifest=${MANIFEST}"
echo "[READY] summary=${SUMMARY_CSV}"
