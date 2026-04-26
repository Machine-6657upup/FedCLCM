#!/usr/bin/env bash
set -u

PFEDBA_ROOT="/home/huangtu/PFL_Backdoor_Defense/PFedBA"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${PFEDBA_ROOT}/log/fixedle10_global12_${RUN_TS}"
STATUS_DIR="${LOGDIR}/status"
WORKER_SCRIPT="/home/huangtu/PFL_clean_workspace/root_static/scripts/pfedba_fixedle10_global12_worker.sh"

mkdir -p "${LOGDIR}" "${STATUS_DIR}"

cat > "${LOGDIR}/manifest.tsv" <<'EOF'
tag	gpu	algorithm	local_epochs	plocal_epochs	global_iters	variant
G01	0	FedRep	10	1	250	baseline
G02	1	FedRT	10	1	250	baseline
G03	2	FedRPD	10	1	250	full
G04	3	FedRPD	10	1	250	purify2
G05	0	FedRPD	10	1	250	no_purify
G06	1	FedRPD	10	1	250	no_distill
G07	2	FedRPD	10	1	300	purify2
G08	3	FedRep	10	1	300	baseline
G09	0	FedRT	10	1	300	baseline
G10	1	FedRPD	10	1	300	full
G11	2	FedRPD	10	1	300	no_purify
G12	3	FedRPD	10	1	300	no_distill
EOF

{
  echo "PFEDBA_ROOT=${PFEDBA_ROOT}"
  echo "RUN_TS=${RUN_TS}"
  echo "LOGDIR=${LOGDIR}"
  echo "STATUS_DIR=${STATUS_DIR}"
  echo "WORKER_SCRIPT=${WORKER_SCRIPT}"
  echo "LOCAL_EPOCHS=10"
  echo "PLOCAL_EPOCHS=1"
  echo "GLOBAL_ITERS_SET=250,300"
} | tee "${LOGDIR}/launcher.env"

nohup bash "${WORKER_SCRIPT}" gpu0 "${LOGDIR}" "${STATUS_DIR}" "${RUN_TS}" > "${LOGDIR}/gpu0.worker.log" 2>&1 &
PID0=$!
nohup bash "${WORKER_SCRIPT}" gpu1 "${LOGDIR}" "${STATUS_DIR}" "${RUN_TS}" > "${LOGDIR}/gpu1.worker.log" 2>&1 &
PID1=$!
nohup bash "${WORKER_SCRIPT}" gpu2 "${LOGDIR}" "${STATUS_DIR}" "${RUN_TS}" > "${LOGDIR}/gpu2.worker.log" 2>&1 &
PID2=$!
nohup bash "${WORKER_SCRIPT}" gpu3 "${LOGDIR}" "${STATUS_DIR}" "${RUN_TS}" > "${LOGDIR}/gpu3.worker.log" 2>&1 &
PID3=$!

{
  echo -e "queue\tpid"
  echo -e "gpu0\t${PID0}"
  echo -e "gpu1\t${PID1}"
  echo -e "gpu2\t${PID2}"
  echo -e "gpu3\t${PID3}"
} | tee "${LOGDIR}/worker_pids.tsv"
