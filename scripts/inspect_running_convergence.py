#!/usr/bin/env python3
import re
import sys

ROUND_RE = re.compile(r"Round number:\s*(\d+)")
ACC_RE = re.compile(r"Averaged Test Accurancy:\s*([0-9.]+)")
ASR_RE = re.compile(r"^ASR\s*:\s*([0-9.]+)")


def parse(path):
    rows = []
    rnd = None
    acc = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = ROUND_RE.search(line)
            if m:
                rnd = int(m.group(1))
                acc = None
                continue
            m = ACC_RE.search(line)
            if m:
                acc = float(m.group(1))
                continue
            m = ASR_RE.search(line.strip())
            if m and rnd is not None and acc is not None:
                rows.append((rnd, acc, float(m.group(1))))
                acc = None
    return rows


def main():
    for path in sys.argv[1:]:
        rows = parse(path)
        print(f"==== {path} ====")
        if not rows:
            print("no eval rows")
            continue
        print("last_round", rows[-1][0], "last_acc", f"{rows[-1][1]:.4f}", "last_asr", f"{rows[-1][2]:.4f}")
        print("last10")
        for rnd, acc, asr in rows[-10:]:
            print(f"{rnd}\tacc={acc:.4f}\tasr={asr:.4f}")
        if len(rows) >= 6:
            old = rows[-6]
            new = rows[-1]
            print("delta_last50", f"acc={new[1]-old[1]:+.4f}", f"asr={new[2]-old[2]:+.4f}")


if __name__ == "__main__":
    main()
