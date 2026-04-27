#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib/common.sh"

PFEDBA_ROOT="${PROJECT_DIR}/pfedba_local"
PFEDBA_MAIN="${PFEDBA_ROOT}/main.py"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-priority}"
RUN_NAME="${RUN_NAME:-${RUN_TS}_pfedba_clcmpur_${MODE}}"
RUN_ROOT="${PROJECT_DIR}/runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/train_logs"
STATUS_DIR="${RUN_ROOT}/status"
MANIFEST="${RUN_ROOT}/manifest.tsv"
SUMMARY="${RUN_ROOT}/summary_hint.txt"

ROUNDS="${ROUNDS:-1000}"
SMOKE_ROUNDS="${SMOKE_ROUNDS:-3}"
EVAL_GAP="${EVAL_GAP:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BATCH_PARALLEL="${BATCH_PARALLEL:-0}"
ONLY_TAG="${ONLY_TAG:-}"
TIMES="${TIMES:-1}"
SEED="${SEED:-1}"

ensure_dir "${RUN_ROOT}"
ensure_dir "${LOG_DIR}"
ensure_dir "${STATUS_DIR}"

declare -a CONFIGS=()

add_config() {
  local tag="$1"
  local algorithm="$2"
  local rounds="$3"
  local rt_beta="$4"
  local lambda_cl="$5"
  local aug_strength="$6"
  local mask_tau="$7"
  local mask_alpha="$8"
  local enable_channel_mask="$9"
  local purify_beta="${10}"
  local purify_feature_beta="${11}"
  local purify_logit_beta="${12}"
  local purify_start_round="${13}"
  local purify_layers="${14}"
  local purify_teacher_momentum="${15}"
  local purpose="${16}"

  CONFIGS+=("${tag}|${algorithm}|${rounds}|${rt_beta}|${lambda_cl}|${aug_strength}|${mask_tau}|${mask_alpha}|${enable_channel_mask}|${purify_beta}|${purify_feature_beta}|${purify_logit_beta}|${purify_start_round}|${purify_layers}|${purify_teacher_momentum}|${purpose}")
}

add_smoke_configs() {
  add_config "SMK_PUR_T6A03_ATT_B0P03_S1" "FedCLCMPurify" "${SMOKE_ROUNDS}" 0.20 0.20 0.10 6.0 0.30 1 0.03 0.0 0.0 1 "layer4" 0.90 "smoke_pfedsba_adapter_results_dir_and_teacher_path"
}

add_priority_configs() {
  # Directly comparable to the preserved PFedBA logs: Cifar10/resnet/no-pretrain,
  # 100 clients, 10 selected per round, lr=0.1/lr_head=0.1, LE=1/PLE=1,
  # attack_start=30, malclient=10, attackall, per_epoch=1.
  add_config "PF01_CLCM_FORMAL_T12A07" "FedCLCM" "${ROUNDS}" 0.20 0.20 0.10 12.0 0.70 1 0.0 0.0 0.0 1 "layer4" 0.90 "same_as_20260424_pfedba_fedclcm_baseline"
  add_config "PF02_PUR_FORMAL_T12A07_ATT_B0P03_S5" "FedCLCMPurify" "${ROUNDS}" 0.20 0.20 0.10 12.0 0.70 1 0.03 0.0 0.0 5 "layer4" 0.90 "same_formal_mask_plus_light_attention_purify"

  # Current static evidence says tau=6/alpha=0.3 is the real promising CLCM
  # setting. These two runs separate mask gain from the new purification gain.
  add_config "PF03_CLCM_T6A03_BASE" "FedCLCM" "${ROUNDS}" 0.20 0.20 0.10 6.0 0.30 1 0.0 0.0 0.0 1 "layer4" 0.90 "current_static_best_mask_baseline_under_pfedba"
  add_config "PF04_PUR_T6A03_ATT_B0P03_S5" "FedCLCMPurify" "${ROUNDS}" 0.20 0.20 0.10 6.0 0.30 1 0.03 0.0 0.0 5 "layer4" 0.90 "highest_priority_light_attention_purify_under_pfedba"
}

add_ablation_configs() {
  # The ablation starts from PF04. It tests whether the useful part is actually
  # attention purification, feature/logit distillation, channel masking,
  # contrastive learning, or trigger-breaking augmentation.
  add_config "AB01_PUR_T6A03_ATT_B0P1_S20" "FedCLCMPurify" "${ROUNDS}" 0.20 0.20 0.10 6.0 0.30 1 0.10 0.0 0.0 20 "layer4" 0.90 "stronger_attention_later_start_acc_asr_tradeoff"
  add_config "AB02_PUR_T6A03_ATT_B1_S20" "FedCLCMPurify" "${ROUNDS}" 0.20 0.20 0.10 6.0 0.30 1 1.00 0.0 0.0 20 "layer4" 0.90 "stress_test_attention_strength_seen_useful_on_3090"
  add_config "AB03_PUR_T6A03_FEAT_B0P03_S5" "FedCLCMPurify" "${ROUNDS}" 0.20 0.20 0.10 6.0 0.30 1 0.0 0.03 0.0 5 "layer4" 0.90 "feature_distillation_only"
  add_config "AB04_PUR_T6A03_LOGIT_B0P03_S5" "FedCLCMPurify" "${ROUNDS}" 0.20 0.20 0.10 6.0 0.30 1 0.0 0.0 0.03 5 "layer4" 0.90 "logit_distillation_only"
  add_config "AB05_PUR_T6A03_ATT_FEAT_B0P03_S5" "FedCLCMPurify" "${ROUNDS}" 0.20 0.20 0.10 6.0 0.30 1 0.03 0.03 0.0 5 "layer4" 0.90 "attention_plus_feature_distillation"
  add_config "AB06_PUR_T6A03_NOMASK_ATT_B0P03_S5" "FedCLCMPurify" "${ROUNDS}" 0.20 0.20 0.10 1000000000.0 1.00 0 0.03 0.0 0.0 5 "layer4" 0.90 "remove_channel_mask_test_purify_only"
  add_config "AB07_PUR_T6A03_NOCL_ATT_B0P03_S5" "FedCLCMPurify" "${ROUNDS}" 0.20 0.00 0.10 6.0 0.30 1 0.03 0.0 0.0 5 "layer4" 0.90 "remove_contrastive_loss_keep_trigger_breaking"
  add_config "AB08_PUR_T6A03_NOAUG_ATT_B0P03_S5" "FedCLCMPurify" "${ROUNDS}" 0.20 0.20 0.00 6.0 0.30 1 0.03 0.0 0.0 5 "layer4" 0.90 "remove_trigger_breaking_and_effective_cl_views"
}

case "${MODE}" in
  smoke)
    add_smoke_configs
    ;;
  priority)
    add_priority_configs
    ;;
  ablation)
    add_ablation_configs
    ;;
  all)
    add_priority_configs
    add_ablation_configs
    ;;
  *)
    echo "[FATAL] unknown MODE=${MODE}; use smoke, priority, ablation, all" >&2
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

if [[ ! -f "${PFEDBA_MAIN}" ]]; then
  echo "[FATAL] missing ${PFEDBA_MAIN}" >&2
  exit 1
fi

echo -e "tag\talgorithm\tdataset\tmodel\tresnet_pretrained\ttotal_users\tselected_users\tlr\tlr_head\tlocal_epochs\tplocal_epochs\trounds\teval_gap\tattack_start\tattack_method\tpoisoning_per_batch\tper_epoch\tmalclient\ttimes\tseed\trt_beta\tlambda_cl\taug_strength\tmask_tau\tmask_alpha\tenable_channel_mask\tpurify_beta\tpurify_feature_beta\tpurify_logit_beta\tpurify_start_round\tpurify_layers\tpurify_teacher_momentum\tpurpose" > "${MANIFEST}"
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tag algorithm rounds rt_beta lambda_cl aug_strength mask_tau mask_alpha enable_channel_mask purify_beta purify_feature_beta purify_logit_beta purify_start_round purify_layers purify_teacher_momentum purpose <<< "${cfg}"
  echo -e "${tag}\t${algorithm}\tCifar10\tresnet\t0\t100\t10\t0.1\t0.1\t1\t1\t${rounds}\t${EVAL_GAP}\t30\tattackall\t1\t1\t10\t${TIMES}\t${SEED}\t${rt_beta}\t${lambda_cl}\t${aug_strength}\t${mask_tau}\t${mask_alpha}\t${enable_channel_mask}\t${purify_beta}\t${purify_feature_beta}\t${purify_logit_beta}\t${purify_start_round}\t${purify_layers}\t${purify_teacher_momentum}\t${purpose}" >> "${MANIFEST}"
done

launch_one() {
  local cfg="$1"
  local gpu="$2"
  IFS='|' read -r tag algorithm rounds rt_beta lambda_cl aug_strength mask_tau mask_alpha enable_channel_mask purify_beta purify_feature_beta purify_logit_beta purify_start_round purify_layers purify_teacher_momentum purpose <<< "${cfg}"

  local log_file="${LOG_DIR}/${tag}.log"
  local meta_file="${LOG_DIR}/${tag}.meta.log"
  local running_file="${STATUS_DIR}/${tag}.running"
  local ok_file="${STATUS_DIR}/${tag}.ok"
  local fail_file="${STATUS_DIR}/${tag}.fail"

  rm -f "${ok_file}" "${fail_file}"
  : > "${running_file}"

  local -a cmd=(
    "${PYTHON_BIN}" -u "${PFEDBA_MAIN}"
    --dataset Cifar10
    --model resnet
    --resnet_pretrained 0
    --algorithm "${algorithm}"
    --batch_size "${BATCH_SIZE}"
    --learning_rate 0.1
    --lr_head 0.1
    --num_global_iters "${rounds}"
    --local_epochs 1
    --plocal_epochs 1
    --numusers 10
    --times "${TIMES}"
    --seed "${SEED}"
    --malclient 10
    --attack_start 30
    --poisoning_per_batch 1
    --attack_method attackall
    --per_epoch 1
    --defense none
    --rt_beta "${rt_beta}"
    --lambda_cl "${lambda_cl}"
    --aug_strength "${aug_strength}"
    --mask_tau "${mask_tau}"
    --mask_alpha "${mask_alpha}"
    --enable_channel_mask "${enable_channel_mask}"
    --adv_eps 0.0
    --adv_num_iter 0
    --purify_beta "${purify_beta}"
    --purify_feature_beta "${purify_feature_beta}"
    --purify_logit_beta "${purify_logit_beta}"
    --purify_temperature 2.0
    --purify_start_round "${purify_start_round}"
    --purify_layers "${purify_layers}"
    --purify_teacher_momentum "${purify_teacher_momentum}"
    --purify_teacher_cpu_half 1
    --eval_gap "${EVAL_GAP}"
    --personalized_eval_gap 0
  )

  {
    echo "=================================================="
    echo "[START] ${tag}"
    echo "TIME=$(date '+%F %T')"
    echo "GPU=${gpu}"
    echo "PURPOSE=${purpose}"
    printf 'CMD=%q ' "${cmd[@]}"
    echo
    echo "=================================================="
  } > "${meta_file}"

  echo "[RUN] ${tag} gpu=${gpu} algo=${algorithm} rounds=${rounds} purpose=${purpose}"
  set +e
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONUNBUFFERED=1
    cd "${PFEDBA_ROOT}"
    "${cmd[@]}" > "${log_file}" 2>&1
  )
  local rc=$?
  set -e

  rm -f "${running_file}"
  if [[ "${rc}" -eq 0 ]]; then
    : > "${ok_file}"
    echo "[OK] ${tag}" >> "${meta_file}"
  else
    : > "${fail_file}"
    echo "[FAIL] ${tag} rc=${rc}" >> "${meta_file}"
    return "${rc}"
  fi
}

run_config_batches CONFIGS launch_one "${BATCH_PARALLEL}"

cat > "${SUMMARY}" <<EOF
Run root: ${RUN_ROOT}
Manifest: ${MANIFEST}
Logs: ${LOG_DIR}

Parsing note:
  PFedBA logs are not the same format as src/static logs.
  Use tail/grep for "Average Global Accurancy", "Average Personalized Accurancy",
  "ASR", "attack_auc", and final h5 files under pfedba_local/results.
EOF

echo "[READY] run_root=${RUN_ROOT}"
echo "[READY] manifest=${MANIFEST}"
