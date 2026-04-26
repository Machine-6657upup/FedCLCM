#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


ACC_RE = re.compile(r"Averaged Test Accurancy:\s*([0-9.]+)")
AUC_RE = re.compile(r"Averaged Test AUC:\s*([0-9.]+)")
LOSS_RE = re.compile(r"Averaged Train Loss:\s*([0-9.]+)")
ROUND_RE = re.compile(r"Round number:\s*(\d+)")
ASR_RE = re.compile(r"ASR\s*:?\s*([0-9.]+)")
ATTACK_AUC_RE = re.compile(r"attack_auc\s*:?\s*([0-9.]+)")
BEST_RE = re.compile(r"Best accuracy\.\s*\n([0-9.]+)")
TIME_RE = re.compile(r"time cost\s*-+\s*([0-9.]+)")
TS_SUFFIX_RE = re.compile(r"_(\d{8}_\d{6})$")


def strip_timestamp(stem: str) -> str:
    return TS_SUFFIX_RE.sub("", stem)


def pad(values, size):
    if len(values) >= size:
        return values[:size]
    return values + [None] * (size - len(values))


def mean_tail(values, tail):
    if not values:
        return None
    chunk = values[-tail:]
    return sum(chunk) / len(chunk)


def parse_log(path: Path):
    text = path.read_text(errors="ignore")

    rounds = [int(v) for v in ROUND_RE.findall(text)]
    accs = [float(v) for v in ACC_RE.findall(text)]
    aucs = [float(v) for v in AUC_RE.findall(text)]
    losses = [float(v) for v in LOSS_RE.findall(text)]
    asrs = [float(v) for v in ASR_RE.findall(text)]
    attack_aucs = [float(v) for v in ATTACK_AUC_RE.findall(text)]
    time_costs = [float(v) for v in TIME_RE.findall(text)]
    best_matches = [float(v) for v in BEST_RE.findall(text)]

    max_len = max(len(rounds), len(accs), len(aucs), len(losses), len(asrs), len(attack_aucs))
    curve_rows = []
    if max_len > 0:
      curve_rounds = pad(rounds, max_len)
      curve_accs = pad(accs, max_len)
      curve_aucs = pad(aucs, max_len)
      curve_losses = pad(losses, max_len)
      curve_asrs = pad(asrs, max_len)
      curve_attack_aucs = pad(attack_aucs, max_len)
      for idx in range(max_len):
          curve_rows.append(
              {
                  "step_idx": idx,
                  "round": curve_rounds[idx],
                  "train_loss": curve_losses[idx],
                  "test_acc": curve_accs[idx],
                  "test_auc": curve_aucs[idx],
                  "asr": curve_asrs[idx],
                  "attack_auc": curve_attack_aucs[idx],
              }
          )

    final_acc = accs[-1] if accs else None
    final_asr = asrs[-1] if asrs else None
    final_auc = aucs[-1] if aucs else None
    final_train_loss = losses[-1] if losses else None
    final_attack_auc = attack_aucs[-1] if attack_aucs else None
    best_acc = best_matches[-1] if best_matches else (max(accs) if accs else None)
    best_round = None
    if best_acc is not None and accs:
        best_idx = max(range(len(accs)), key=lambda i: accs[i])
        if best_idx < len(rounds):
            best_round = rounds[best_idx]

    result = {
        "tag": strip_timestamp(path.stem),
        "file": path.name,
        "rounds_seen": (max(rounds) + 1) if rounds else 0,
        "best_acc": best_acc,
        "best_round": best_round,
        "final_acc": final_acc,
        "final_auc": final_auc,
        "final_train_loss": final_train_loss,
        "final_asr": final_asr,
        "final_attack_auc": final_attack_auc,
        "mean_acc_last_10": mean_tail(accs, 10),
        "mean_asr_last_10": mean_tail(asrs, 10),
        "mean_round_time": (sum(time_costs) / len(time_costs)) if time_costs else None,
    }
    return result, curve_rows


def write_curve_csv(path: Path, rows):
    fieldnames = ["step_idx", "round", "train_loss", "test_acc", "test_auc", "asr", "attack_auc"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--pattern", default="*.log")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--curves-dir", required=True)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    summary_csv = Path(args.summary_csv)
    summary_json = Path(args.summary_json)
    curves_dir = Path(args.curves_dir)
    curves_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for log_path in sorted(log_dir.glob(args.pattern)):
        if log_path.name.startswith("scheduler_"):
            continue
        result, curve_rows = parse_log(log_path)
        rows.append(result)
        write_curve_csv(curves_dir / f"{strip_timestamp(log_path.stem)}.csv", curve_rows)

    with summary_json.open("w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "tag",
        "file",
        "rounds_seen",
        "best_acc",
        "best_round",
        "final_acc",
        "final_auc",
        "final_train_loss",
        "final_asr",
        "final_attack_auc",
        "mean_acc_last_10",
        "mean_asr_last_10",
        "mean_round_time",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
