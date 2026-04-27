#!/usr/bin/env python3
import argparse
import json
import os
import re


ACC_PATTERNS = [
    re.compile(r"Averaged Test Accurancy:\s*([0-9.]+)"),
    re.compile(r"Average (?:Personal|Global) Accurancy(?: \(k local SGD\))?:\s*([0-9.]+)"),
    re.compile(r"Average Global Accurancy:\s*([0-9.]+)"),
    re.compile(r"Average Personal Accurancy \(k local SGD\):\s*([0-9.]+)"),
]

ASR_PATTERNS = [
    re.compile(r"\bASR:\s*([0-9.]+)"),
    re.compile(r"Average (?:Personal|Global) ATTACK ALL ASR(?: \(k local SGD\))?:\s*([0-9.]+)"),
    re.compile(r"Average Global ATTACK ALL ASR:\s*([0-9.]+)"),
    re.compile(r"Average Personal ATTACK ALL ASR \(k local SGD\):\s*([0-9.]+)"),
]

ROUND_PATTERN = re.compile(r"Round number:\s*([0-9]+)")
PARAM_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
PARAM_KEYS = {
    "goal",
    "dataset",
    "model",
    "join_ratio",
    "local_learning_rate",
    "lr_head",
    "global_rounds",
    "local_epochs",
    "plocal_epochs",
    "algorithm",
    "num_clients",
    "num_adv_clients",
    "rt_beta",
    "lambda_cl",
    "aug_strength",
    "mask_tau",
    "mask_alpha",
    "enable_channel_mask",
    "adv_eps",
    "adv_num_iter",
}


def values(patterns, text):
    out = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            try:
                out.append(float(match.group(1)))
            except ValueError:
                pass
    return out


def parse_file(path, root):
    rel = os.path.relpath(path, root)
    try:
        size = os.path.getsize(path)
        if size > 50_000_000:
            return None
        with open(path, "rb") as handle:
            text = handle.read().decode("utf-8", "ignore")
    except OSError:
        return None

    lowered = rel.lower()
    if "clcm" not in lowered and "fedclcm" not in text and "CLCM" not in text:
        return None

    accs = values(ACC_PATTERNS, text)
    asrs = values(ASR_PATTERNS, text)
    rounds = []
    params = {}
    for line in text.splitlines()[:260]:
        match = PARAM_PATTERN.match(line.strip())
        if match and match.group(1) in PARAM_KEYS:
            params[match.group(1)] = match.group(2).strip()
    for match in ROUND_PATTERN.finditer(text):
        try:
            rounds.append(int(match.group(1)))
        except ValueError:
            pass

    if not accs and not asrs:
        return None

    return {
        "rel": rel,
        "size": size,
        "mtime": os.path.getmtime(path),
        "acc_last": accs[-1] if accs else None,
        "acc_best": max(accs) if accs else None,
        "asr_last": asrs[-1] if asrs else None,
        "asr_min": min(asrs) if asrs else None,
        "round_last": rounds[-1] if rounds else None,
        "n_acc": len(accs),
        "n_asr": len(asrs),
        **params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--acc-threshold", type=float, default=0.75)
    parser.add_argument("--asr-threshold", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Case-insensitive substring that must appear in the relative path or parsed parameters. Can be repeated.",
    )
    args = parser.parse_args()

    rows = []
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(args.root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {".git", "__pycache__"}
            and not os.path.join(dirpath, name).endswith(os.path.join("dataset", "rawdata"))
        ]
        for name in filenames:
            if not name.endswith((".log", ".out", ".txt", ".csv")):
                continue
            row = parse_file(os.path.join(dirpath, name), args.root)
            if row is None:
                continue
            haystack = " ".join(str(value) for value in row.values()).lower()
            if any(needle.lower() not in haystack for needle in args.contains):
                continue
            scanned += 1
            rows.append(row)

    strict = []
    loose = []
    for row in rows:
        if (
            row["acc_last"] is not None
            and row["asr_last"] is not None
            and row["acc_last"] >= args.acc_threshold
            and row["asr_last"] <= args.asr_threshold
        ):
            strict.append(row)
        if (
            row["acc_best"] is not None
            and row["asr_min"] is not None
            and row["acc_best"] >= args.acc_threshold
            and row["asr_min"] <= args.asr_threshold
        ):
            loose.append(row)

    print(json.dumps({"scanned": scanned, "with_metrics": len(rows), "strict_candidates": len(strict), "loose_candidates": len(loose)}, ensure_ascii=False))
    print("---STRICT_FINAL_ACC_ASR---")
    for row in sorted(strict, key=lambda r: (r["asr_last"], -r["acc_last"]))[: args.limit]:
        print(json.dumps(row, ensure_ascii=False))
    print("---LOOSE_BESTACC_MINASR---")
    for row in sorted(loose, key=lambda r: (r["asr_min"], -r["acc_best"]))[: args.limit]:
        print(json.dumps(row, ensure_ascii=False))
    print("---LOWEST_FINAL_ASR_WITH_ACC---")
    valid = [row for row in rows if row["acc_last"] is not None and row["asr_last"] is not None]
    for row in sorted(valid, key=lambda r: (r["asr_last"], -r["acc_last"]))[: args.limit]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
