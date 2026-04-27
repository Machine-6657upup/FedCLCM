#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

SRC_ROOT="${PROJECT_DIR}/src"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
SERVER_ROLE="${SERVER_ROLE:-3090}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_thesis_15h_static_${SERVER_ROLE}}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
STATUS_DIR="${RUN_ROOT}/status"
CURVES_DIR="${RUN_ROOT}/curves"
REUSE_DIR="${RUN_ROOT}/reuse"
REFERENCE_DIR="${RUN_ROOT}/reference_only"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
MANIFEST="${RUN_ROOT}/manifest.tsv"
REUSE_MANIFEST="${RUN_ROOT}/reuse_manifest.tsv"

ROUNDS="${ROUNDS:-800}"
EVAL_GAP="${EVAL_GAP:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
ADV_CLIENTS="${ADV_CLIENTS:-10}"
JOIN_RATIO="${JOIN_RATIO:-0.1}"
LOCAL_LR="${LOCAL_LR:-0.1}"
LR_HEAD="${LR_HEAD:-0.1}"
LOCAL_EPOCHS_DEFAULT="${LOCAL_EPOCHS_DEFAULT:-1}"
PLOCAL_EPOCHS="${PLOCAL_EPOCHS:-1}"
RT_BETA="${RT_BETA:-0.2}"
LAMBDA_CL_DEFAULT="${LAMBDA_CL_DEFAULT:-0.2}"
AUG_STRENGTH="${AUG_STRENGTH:-0.1}"
MASK_TAU_DEFAULT="${MASK_TAU_DEFAULT:-6.0}"
MASK_ALPHA_DEFAULT="${MASK_ALPHA_DEFAULT:-0.3}"
PREPARE_DATASETS="${PREPARE_DATASETS:-1}"

ensure_dir "${RUN_ROOT}"
ensure_dir "${LOG_DIR}"
ensure_dir "${STATUS_DIR}"
ensure_dir "${CURVES_DIR}"
ensure_dir "${REUSE_DIR}"
ensure_dir "${REFERENCE_DIR}"

load_gpu_array

dataset_for_attack() {
  local attack="$1"
  echo "Cifar10_dir0.5_bdoor0.2_nclient_${NUM_CLIENTS}_${attack}_adv${ADV_CLIENTS}"
}

copy_if_exists() {
  local src="$1"
  local dst_dir="$2"
  local role="$3"
  local note="$4"

  if [[ -f "${src}" ]]; then
    cp -f "${src}" "${dst_dir}/"
    echo -e "$(basename "${src}")\t${role}\t${src}\t${note}" >> "${REUSE_MANIFEST}"
  fi
}

copy_reuse_logs() {
  echo -e "file\trole\tsource\tnote" > "${REUSE_MANIFEST}"

  local static_old="/home/huangtu/FedCLCM_purify/thesis_formal_logs/final_static_main_20260424/static"
  if [[ -d "${static_old}" ]]; then
    for f in "${static_old}"/S*.log; do
      [[ -f "${f}" ]] || continue
      copy_if_exists "${f}" "${REUSE_DIR}" "static_20260424" "FedAvg/FedRep 600-round converged or attack-calibration reusable log"
    done
  fi

  copy_if_exists \
    "/home/huangtu/FedCLCM_purify/runs/20260427_hit_setting_t6a03_baseline_3090/train_logs/HIT_CLCM_T6A03_BASE.log" \
    "${REUSE_DIR}" \
    "fedclcm_badnet_current" \
    "FedCLCM BadNet current setting, 600 rounds, tau=6 alpha=0.3, stable"

  copy_if_exists \
    "/home/fch/FedCLCM_purify/runs/20260427_12h_4090_static_final_confirm/train_logs/B04_600.log" \
    "${REUSE_DIR}" \
    "fedclcm_badnet_current_4090" \
    "FedCLCM BadNet current setting from 4090, 600 rounds, reference duplicate"

  local pfedba_old="/home/fch/FedCLCM_purify/runs/20260427_pfedba_clcmpur_priority_4090/train_logs"
  if [[ -d "${pfedba_old}" ]]; then
    for f in "${pfedba_old}"/PF01_CLCM_FORMAL_T12A07.log "${pfedba_old}"/PF03_CLCM_T6A03_BASE.log; do
      copy_if_exists "${f}" "${REFERENCE_DIR}" "pfedba_20260427" "PFedBA completed 1000-round FedCLCM reference log"
    done
  fi
}

declare -a CONFIGS=()

add_config() {
  local queue="$1"
  local tag="$2"
  local attack="$3"
  local algorithm="$4"
  local mask_tau="$5"
  local mask_alpha="$6"
  local lambda_cl="$7"
  local channel_mask="$8"
  local local_epochs="$9"
  local purpose="${10}"
  CONFIGS+=("${queue}|${tag}|${attack}|${algorithm}|${mask_tau}|${mask_alpha}|${lambda_cl}|${channel_mask}|${local_epochs}|${purpose}")
}

build_3090_configs() {
  # GPU0: final FedCLCM static table plus key ablations on BadNet.
  add_config 0 "F01_CLCM_BLEND_T6A03" "blend" "FedCLCM" 6.0 0.3 0.2 true 1 "main table missing: FedCLCM vs Blend, current good hyperparams"
  add_config 0 "F02_CLCM_SIG_T6A03" "sig" "FedCLCM" 6.0 0.3 0.2 true 1 "main table missing: FedCLCM vs SIG, current good hyperparams"
  add_config 0 "A01_CLCM_BADNET_NO_CL" "badnet" "FedCLCM" 6.0 0.3 0.0 true 1 "ablation: remove contrastive loss"
  add_config 0 "A02_CLCM_BADNET_NO_MASK" "badnet" "FedCLCM" 6.0 1.0 0.2 false 1 "ablation: remove channel mask"
  add_config 0 "A07_CLCM_BADNET_NO_CL_NOMASK" "badnet" "FedCLCM" 6.0 1.0 0.0 false 1 "ablation: remove contrastive loss and channel mask"
  add_config 0 "A08_CLCM_BLEND_NO_CL_NOMASK" "blend" "FedCLCM" 6.0 1.0 0.0 false 1 "ablation: remove both modules on Blend"
  add_config 0 "A09_CLCM_SIG_NO_CL_NOMASK" "sig" "FedCLCM" 6.0 1.0 0.0 false 1 "ablation: remove both modules on SIG"
  add_config 0 "S01_CLCM_BADNET_T5A02" "badnet" "FedCLCM" 5.0 0.2 0.2 true 1 "sensitivity: stronger mask than tau=6 alpha=0.3"
  add_config 0 "S02_CLCM_BADNET_T12A07" "badnet" "FedCLCM" 12.0 0.7 0.2 true 1 "reference: old good ResNetP-style mask on current ResNet18 setting"
  add_config 0 "S05_CLCM_BADNET_T4A02" "badnet" "FedCLCM" 4.0 0.2 0.2 true 1 "sensitivity: aggressive threshold on BadNet"
  add_config 0 "S06_CLCM_BADNET_T8A04" "badnet" "FedCLCM" 8.0 0.4 0.2 true 1 "sensitivity: milder threshold/downweight on BadNet"
  add_config 0 "S07_CLCM_BADNET_LAM01" "badnet" "FedCLCM" 6.0 0.3 0.1 true 1 "sensitivity: weaker contrastive loss on BadNet"

  # GPU1: coordinate robust baselines plus strict 800-round FedAvg/FedRep baselines.
  add_config 1 "M01_MEDIAN_BADNET" "badnet" "FedMedian" 6.0 0.3 0.2 true 1 "baseline: coordinate-wise median"
  add_config 1 "M02_MEDIAN_BLEND" "blend" "FedMedian" 6.0 0.3 0.2 true 1 "baseline: coordinate-wise median"
  add_config 1 "M03_MEDIAN_SIG" "sig" "FedMedian" 6.0 0.3 0.2 true 1 "baseline: coordinate-wise median"
  add_config 1 "T01_TRIM_BADNET" "badnet" "FedTrimmed" 6.0 0.3 0.2 true 1 "baseline: oracle-f trimmed mean matched to selected adv clients"
  add_config 1 "T02_TRIM_BLEND" "blend" "FedTrimmed" 6.0 0.3 0.2 true 1 "baseline: oracle-f trimmed mean matched to selected adv clients"
  add_config 1 "T03_TRIM_SIG" "sig" "FedTrimmed" 6.0 0.3 0.2 true 1 "baseline: oracle-f trimmed mean matched to selected adv clients"
  add_config 1 "G01_FEDAVG_BADNET_800" "badnet" "FedAvg" 6.0 0.3 0.2 true 1 "strict 800-round attack baseline: FedAvg BadNet"
  add_config 1 "G02_FEDAVG_BLEND_800" "blend" "FedAvg" 6.0 0.3 0.2 true 1 "strict 800-round attack baseline: FedAvg Blend"
  add_config 1 "G03_FEDAVG_SIG_800" "sig" "FedAvg" 6.0 0.3 0.2 true 1 "strict 800-round attack baseline: FedAvg SIG"
  add_config 1 "R01_FEDREP_BADNET_800" "badnet" "FedRep" 6.0 0.3 0.2 true 1 "strict 800-round PFL baseline: FedRep BadNet"
  add_config 1 "R02_FEDREP_BLEND_800" "blend" "FedRep" 6.0 0.3 0.2 true 1 "strict 800-round PFL baseline: FedRep Blend"
  add_config 1 "R03_FEDREP_SIG_800" "sig" "FedRep" 6.0 0.3 0.2 true 1 "strict 800-round PFL baseline: FedRep SIG"

  # GPU2: Krum-family robust baselines plus CLCM hyperparameter sensitivity.
  add_config 2 "K01_MK_BADNET" "badnet" "FedMK" 6.0 0.3 0.2 true 1 "baseline: Multi-Krum-style aggregation"
  add_config 2 "K02_MK_BLEND" "blend" "FedMK" 6.0 0.3 0.2 true 1 "baseline: Multi-Krum-style aggregation"
  add_config 2 "K03_MK_SIG" "sig" "FedMK" 6.0 0.3 0.2 true 1 "baseline: Multi-Krum-style aggregation"
  add_config 2 "B01_BULYAN_BADNET" "badnet" "FedBulyan" 6.0 0.3 0.2 true 1 "baseline: Bulyan-style, trimmed fallback if n < 4f+3"
  add_config 2 "B02_BULYAN_BLEND" "blend" "FedBulyan" 6.0 0.3 0.2 true 1 "baseline: Bulyan-style, trimmed fallback if n < 4f+3"
  add_config 2 "B03_BULYAN_SIG" "sig" "FedBulyan" 6.0 0.3 0.2 true 1 "baseline: Bulyan-style, trimmed fallback if n < 4f+3"
  add_config 2 "S16_CLCM_BADNET_T5A03" "badnet" "FedCLCM" 5.0 0.3 0.2 true 1 "sensitivity: lower tau only on BadNet"
  add_config 2 "S17_CLCM_BADNET_T6A02" "badnet" "FedCLCM" 6.0 0.2 0.2 true 1 "sensitivity: stronger alpha only on BadNet"
  add_config 2 "S18_CLCM_BADNET_T7A03" "badnet" "FedCLCM" 7.0 0.3 0.2 true 1 "sensitivity: higher tau only on BadNet"
  add_config 2 "S19_CLCM_BADNET_T6A04" "badnet" "FedCLCM" 6.0 0.4 0.2 true 1 "sensitivity: milder alpha only on BadNet"
  add_config 2 "S20_CLCM_BLEND_T6A02" "blend" "FedCLCM" 6.0 0.2 0.2 true 1 "sensitivity: stronger alpha only on Blend"
  add_config 2 "S21_CLCM_SIG_T6A02" "sig" "FedCLCM" 6.0 0.2 0.2 true 1 "sensitivity: stronger alpha only on SIG"

  # GPU3: PFL baselines from existing local implementations plus FedCLCM cross-attack sensitivity.
  add_config 3 "P01_FEDPROTO_BADNET" "badnet" "FedProto" 6.0 0.3 0.2 true 1 "PFL baseline: FedProto, prototype aggregation"
  add_config 3 "P02_FEDPROTO_BLEND" "blend" "FedProto" 6.0 0.3 0.2 true 1 "PFL baseline: FedProto, prototype aggregation"
  add_config 3 "P03_FEDPROTO_SIG" "sig" "FedProto" 6.0 0.3 0.2 true 1 "PFL baseline: FedProto, prototype aggregation"
  add_config 3 "D01_FEDPD_BADNET" "badnet" "FedPD" 6.0 0.3 0.2 true 1 "PFL baseline: local FedPD/prototype-defense variant"
  add_config 3 "D02_FEDPD_BLEND" "blend" "FedPD" 6.0 0.3 0.2 true 1 "PFL baseline: local FedPD/prototype-defense variant"
  add_config 3 "D03_FEDPD_SIG" "sig" "FedPD" 6.0 0.3 0.2 true 1 "PFL baseline: local FedPD/prototype-defense variant"
  add_config 3 "S08_CLCM_BLEND_T5A02" "blend" "FedCLCM" 5.0 0.2 0.2 true 1 "sensitivity: stronger mask on Blend"
  add_config 3 "S09_CLCM_SIG_T5A02" "sig" "FedCLCM" 5.0 0.2 0.2 true 1 "sensitivity: stronger mask on SIG"
  add_config 3 "S10_CLCM_BLEND_T12A07" "blend" "FedCLCM" 12.0 0.7 0.2 true 1 "sensitivity: old mask setting on Blend"
  add_config 3 "S11_CLCM_SIG_T12A07" "sig" "FedCLCM" 12.0 0.7 0.2 true 1 "sensitivity: old mask setting on SIG"
  add_config 3 "S12_CLCM_BLEND_LAM01" "blend" "FedCLCM" 6.0 0.3 0.1 true 1 "sensitivity: weaker contrastive loss on Blend"
  add_config 3 "S13_CLCM_SIG_LAM01" "sig" "FedCLCM" 6.0 0.3 0.1 true 1 "sensitivity: weaker contrastive loss on SIG"
}

build_4090_configs() {
  # 4090 complements 3090: FedCLCM ablations/generalization and local epoch sensitivity.
  add_config 0 "A03_CLCM_BLEND_NO_CL" "blend" "FedCLCM" 6.0 0.3 0.0 true 1 "ablation: remove contrastive loss on Blend"
  add_config 0 "A04_CLCM_BLEND_NO_MASK" "blend" "FedCLCM" 6.0 1.0 0.2 false 1 "ablation: remove channel mask on Blend"
  add_config 0 "A05_CLCM_SIG_NO_CL" "sig" "FedCLCM" 6.0 0.3 0.0 true 1 "ablation: remove contrastive loss on SIG"
  add_config 0 "A06_CLCM_SIG_NO_MASK" "sig" "FedCLCM" 6.0 1.0 0.2 false 1 "ablation: remove channel mask on SIG"
  add_config 0 "L02_CLCM_BADNET_LE2" "badnet" "FedCLCM" 6.0 0.3 0.2 true 2 "sensitivity: local_epochs=2, BadNet"
  add_config 0 "L05_CLCM_BADNET_LE5" "badnet" "FedCLCM" 6.0 0.3 0.2 true 5 "sensitivity: local_epochs=5, BadNet"
  add_config 0 "S03_CLCM_BLEND_T5A02" "blend" "FedCLCM" 5.0 0.2 0.2 true 1 "sensitivity: stronger mask on Blend"
  add_config 0 "S04_CLCM_SIG_T5A02" "sig" "FedCLCM" 5.0 0.2 0.2 true 1 "sensitivity: stronger mask on SIG"
  add_config 0 "L10_CLCM_BADNET_LE10" "badnet" "FedCLCM" 6.0 0.3 0.2 true 10 "sensitivity: local_epochs=10, BadNet"
  add_config 0 "L20_CLCM_BADNET_LE20" "badnet" "FedCLCM" 6.0 0.3 0.2 true 20 "sensitivity: local_epochs=20, BadNet; long filler and thesis sensitivity"
  add_config 0 "L12_CLCM_BLEND_LE2" "blend" "FedCLCM" 6.0 0.3 0.2 true 2 "sensitivity: local_epochs=2, Blend"
  add_config 0 "L15_CLCM_BLEND_LE5" "blend" "FedCLCM" 6.0 0.3 0.2 true 5 "sensitivity: local_epochs=5, Blend"
  add_config 0 "L22_CLCM_SIG_LE2" "sig" "FedCLCM" 6.0 0.3 0.2 true 2 "sensitivity: local_epochs=2, SIG"
  add_config 0 "L25_CLCM_SIG_LE5" "sig" "FedCLCM" 6.0 0.3 0.2 true 5 "sensitivity: local_epochs=5, SIG"
  add_config 0 "S14_CLCM_BADNET_T4A02" "badnet" "FedCLCM" 4.0 0.2 0.2 true 1 "sensitivity: aggressive mask on BadNet"
  add_config 0 "S15_CLCM_BADNET_T8A04" "badnet" "FedCLCM" 8.0 0.4 0.2 true 1 "sensitivity: milder mask on BadNet"
}

if [[ "${SERVER_ROLE}" == "3090" ]]; then
  build_3090_configs
elif [[ "${SERVER_ROLE}" == "4090" ]]; then
  build_4090_configs
else
  echo "[FATAL] SERVER_ROLE must be 3090 or 4090, got ${SERVER_ROLE}" >&2
  exit 1
fi

copy_reuse_logs

if [[ "${PREPARE_DATASETS}" == "1" ]]; then
  echo "[DATASET] ensure static datasets exist"
  (
    cd "${PROJECT_DIR}"
    PYTHON_BIN="${PYTHON_BIN}" NUM_CLIENTS="${NUM_CLIENTS}" ADV_CLIENTS="${ADV_CLIENTS}" \
      BACKDOOR_RATE=0.2 TARGET_LABEL=0 ALPHA=0.5 FORCE_REBUILD=0 \
      bash "${SCRIPT_DIR}/prepare_static_datasets_srcutils.sh"
  )
fi

for attack in badnet blend sig; do
  dataset="$(dataset_for_attack "${attack}")"
  if [[ ! -f "${SRC_ROOT}/dataset/${dataset}/config.json" ]]; then
    echo "[FATAL] missing dataset ${dataset}" >&2
    exit 1
  fi
done

echo -e "queue\ttag\tattack\tdataset\talgorithm\tmodel\tnum_clients\tadv_clients\tjoin_ratio\tlocal_lr\tlr_head\tlocal_epochs\tplocal_epochs\trounds\teval_gap\trt_beta\tlambda_cl\taug_strength\tmask_tau\tmask_alpha\tenable_channel_mask\tpurpose" > "${MANIFEST}"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r queue tag attack algorithm mask_tau mask_alpha lambda_cl channel_mask local_epochs purpose <<< "${cfg}"
  dataset="$(dataset_for_attack "${attack}")"
  echo -e "${queue}\t${tag}\t${attack}\t${dataset}\t${algorithm}\tResNet18\t${NUM_CLIENTS}\t${ADV_CLIENTS}\t${JOIN_RATIO}\t${LOCAL_LR}\t${LR_HEAD}\t${local_epochs}\t${PLOCAL_EPOCHS}\t${ROUNDS}\t${EVAL_GAP}\t${RT_BETA}\t${lambda_cl}\t${AUG_STRENGTH}\t${mask_tau}\t${mask_alpha}\t${channel_mask}\t${purpose}" >> "${MANIFEST}"
done

run_one() {
  local cfg="$1"
  local gpu="$2"
  IFS='|' read -r queue tag attack algorithm mask_tau mask_alpha lambda_cl channel_mask local_epochs purpose <<< "${cfg}"
  dataset="$(dataset_for_attack "${attack}")"
  log_file="${LOG_DIR}/${tag}.log"
  ok_file="${STATUS_DIR}/${tag}.ok"
  fail_file="${STATUS_DIR}/${tag}.fail"

  local -a cmd=(
    "${PYTHON_BIN}" -u main.py
    -dev cuda -did "${gpu}"
    -data "${dataset}"
    -m ResNet18
    -algo "${algorithm}"
    -ncl 10 -nc "${NUM_CLIENTS}" -jr "${JOIN_RATIO}" -lbs "${BATCH_SIZE}"
    -lr "${LOCAL_LR}" -lr_head "${LR_HEAD}" -ls "${local_epochs}" -pls "${PLOCAL_EPOCHS}"
    -gr "${ROUNDS}" -eg "${EVAL_GAP}"
    -go "${tag}"
    --num_adv_clients "${ADV_CLIENTS}"
    --rt_beta "${RT_BETA}"
    --lambda_cl "${lambda_cl}"
    --aug_strength "${AUG_STRENGTH}"
    --mask_tau "${mask_tau}"
    --mask_alpha "${mask_alpha}"
    --enable_channel_mask "${channel_mask}"
    --adv_eps 0.0
    --adv_num_iter 0
  )

  echo "[RUN] tag=${tag} gpu=${gpu} algo=${algorithm} attack=${attack} le=${local_epochs} tau=${mask_tau} alpha=${mask_alpha} mask=${channel_mask}"
  rm -f "${ok_file}" "${fail_file}"
  set +e
  (
    cd "${SRC_ROOT}"
    "${cmd[@]}" > "${log_file}" 2>&1
  )
  status=$?
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
echo "[READY] reuse_manifest=${REUSE_MANIFEST}"
echo "[READY] summary=${SUMMARY_CSV}"
