#!/usr/bin/env bash
set -u

PFEDBA_ROOT="/home/huangtu/PFL_Backdoor_Defense/PFedBA"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${PFEDBA_ROOT}/log/localepoch22_${RUN_TS}"
STATUS_DIR="${LOGDIR}/status"
WORKER_SCRIPT="/home/huangtu/PFL_clean_workspace/root_static/scripts/pfedba_localepoch22_worker.sh"

mkdir -p "${LOGDIR}" "${STATUS_DIR}"

cat > "${LOGDIR}/manifest.tsv" <<'EOF'
tag	gpu	algorithm	local_epochs	variant
E01	0	FedRep	1	baseline
E02	1	FedRep	5	baseline
E03	2	FedRep	10	baseline
E04	3	FedRT	1	baseline
E05	0	FedRT	5	baseline
E06	1	FedRT	10	baseline
E07	2	FedRPD	1	full
E08	3	FedRPD	5	full
E09	0	FedRPD	10	full
E10	1	FedRPD	20	full
E11	2	FedRPD	1	no_purify
E12	3	FedRPD	5	no_purify
E13	0	FedRPD	10	no_purify
E14	1	FedRPD	20	no_purify
E15	2	FedRPD	1	no_distill
E16	3	FedRPD	5	no_distill
E17	0	FedRPD	10	no_distill
E18	1	FedRPD	20	no_distill
E19	2	FedRPD	1	purify2
E20	3	FedRPD	5	purify2
E21	0	FedRPD	10	purify2
E22	1	FedRPD	20	purify2
EOF

{
  echo "PFEDBA_ROOT=${PFEDBA_ROOT}"
  echo "RUN_TS=${RUN_TS}"
  echo "LOGDIR=${LOGDIR}"
  echo "STATUS_DIR=${STATUS_DIR}"
  echo "WORKER_SCRIPT=${WORKER_SCRIPT}"
  echo "REUSE_BASELINES=FedRep(local_epochs=20),FedRT(local_epochs=20)"
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
