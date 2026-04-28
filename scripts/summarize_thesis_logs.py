#!/usr/bin/env python3
import os
import re
import sys


ROUND_RE = re.compile(r"Round number:\s*(\d+)")
ACC_RE = re.compile(r"Averaged Test Accurancy:\s*([0-9.]+)")
ASR_RE = re.compile(r"^ASR\s*:\s*([0-9.]+)")
BEST_RE = re.compile(r"mean for best accurancy:\s*([0-9.]+)")


def parse_log(path):
    rows = []
    current_round = None
    pending_acc = None
    best_acc = None
    failed = False
    with open(path, "r", errors="replace") as f:
        for line in f:
            if "Traceback" in line or "RuntimeError" in line or "IndexError" in line:
                failed = True
            m = ROUND_RE.search(line)
            if m:
                current_round = int(m.group(1))
                pending_acc = None
                continue
            m = ACC_RE.search(line)
            if m:
                pending_acc = float(m.group(1))
                continue
            m = ASR_RE.search(line.strip())
            if m and current_round is not None and pending_acc is not None:
                rows.append((current_round, pending_acc, float(m.group(1))))
                pending_acc = None
                continue
            m = BEST_RE.search(line)
            if m:
                best_acc = float(m.group(1))
    return rows, best_acc, failed


def summarize_dir(log_dir):
    print("tag\tstatus\tlast_round\tfinal_acc\tfinal_asr\tbest_acc\tmin_asr_acc75\tacc_at_min_asr_acc75")
    for name in sorted(os.listdir(log_dir)):
        if not name.endswith(".log"):
            continue
        tag = name[:-4]
        rows, best_acc, failed = parse_log(os.path.join(log_dir, name))
        status = "fail" if failed else "ok"
        if rows:
            last_round, final_acc, final_asr = rows[-1]
            eligible = [(asr, acc, rnd) for rnd, acc, asr in rows if acc >= 0.75]
            if eligible:
                min_asr, acc_at_min, _ = min(eligible)
                min_asr_s = f"{min_asr:.6f}"
                acc_at_min_s = f"{acc_at_min:.6f}"
            else:
                min_asr_s = ""
                acc_at_min_s = ""
            print(
                f"{tag}\t{status}\t{last_round}\t{final_acc:.6f}\t{final_asr:.6f}\t"
                f"{best_acc if best_acc is not None else ''}\t{min_asr_s}\t{acc_at_min_s}"
            )
        else:
            print(f"{tag}\t{status}\t\t\t\t{best_acc if best_acc is not None else ''}\t\t")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_thesis_logs.py <train_logs_dir>")
    summarize_dir(sys.argv[1])


if __name__ == "__main__":
    main()
