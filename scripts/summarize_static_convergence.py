#!/usr/bin/env python3
"""Summarize static experiment convergence from FedCLCM-style logs.

The parser is intentionally conservative: it only reports rounds that include
both "Averaged Test Accurancy" and the following aggregate "ASR" line.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROUND_RE = re.compile(r"-+Round number:\s*(\d+)-+")
ACC_RE = re.compile(r"Averaged Test Accurancy:\s*([0-9.]+)")
ASR_RE = re.compile(r"^ASR\s*:\s*([0-9.eE+-]+)")
BEST_RE = re.compile(r"^([0-9.]+)$")


def parse_log(path: Path) -> dict[str, object]:
    current_round = None
    pending_acc = None
    evals: list[tuple[int, float, float]] = []
    best_acc = None
    expect_best_value = False
    done = False

    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        m = ROUND_RE.search(line)
        if m:
            current_round = int(m.group(1))
            pending_acc = None
            continue

        m = ACC_RE.search(line)
        if m and current_round is not None:
            pending_acc = float(m.group(1))
            continue

        m = ASR_RE.search(line)
        if m and current_round is not None and pending_acc is not None:
            evals.append((current_round, pending_acc, float(m.group(1))))
            pending_acc = None
            continue

        if line == "Best accuracy.":
            expect_best_value = True
            continue
        if expect_best_value:
            m = BEST_RE.match(line)
            if m:
                best_acc = float(m.group(1))
                expect_best_value = False
            continue

        if line == "All done!":
            done = True

    by_round = {round_no: (acc, asr) for round_no, acc, asr in evals}
    last_round, last_acc, last_asr = evals[-1] if evals else (None, None, None)
    r500 = by_round.get(500)
    r600 = by_round.get(600)

    if r500 and r600:
        delta_acc_500_600 = r600[0] - r500[0]
        delta_asr_500_600 = r600[1] - r500[1]
        stable_500_600 = abs(delta_acc_500_600) <= 0.02 and abs(delta_asr_500_600) <= 0.05
    else:
        delta_acc_500_600 = None
        delta_asr_500_600 = None
        stable_500_600 = False

    return {
        "file": path.name,
        "status": "done" if done else "running_or_stopped",
        "eval_count": len(evals),
        "last_round": last_round,
        "last_acc": last_acc,
        "last_asr": last_asr,
        "round500_acc": r500[0] if r500 else None,
        "round500_asr": r500[1] if r500 else None,
        "round600_acc": r600[0] if r600 else None,
        "round600_asr": r600[1] if r600 else None,
        "delta_acc_500_600": delta_acc_500_600,
        "delta_asr_500_600": delta_asr_500_600,
        "stable_500_600": stable_500_600,
        "best_acc": best_acc,
        "path": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Log files or directories.")
    parser.add_argument("--csv", dest="csv_path", default=None)
    args = parser.parse_args()

    logs: list[Path] = []
    for item in args.paths:
        path = Path(item)
        if path.is_dir():
            logs.extend(sorted(path.rglob("*.log")))
        elif path.is_file():
            logs.append(path)

    rows = [parse_log(path) for path in logs if path.name.endswith(".log") and not path.name.endswith(".meta.log")]
    rows.sort(key=lambda row: row["file"])

    fields = [
        "file",
        "status",
        "eval_count",
        "last_round",
        "last_acc",
        "last_asr",
        "round500_acc",
        "round500_asr",
        "round600_acc",
        "round600_asr",
        "delta_acc_500_600",
        "delta_asr_500_600",
        "stable_500_600",
        "best_acc",
        "path",
    ]

    if args.csv_path:
        out = Path(args.csv_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
