#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


PATTERNS = {
    "round": re.compile(r"Round number:\s*(\d+)"),
    "global_acc": re.compile(r"Average Global Accurancy:\s*([0-9.]+)"),
    "personal_acc": re.compile(r"Average Personal Accurancy.*?:\s*([0-9.]+)"),
    "global_asr": re.compile(r"Average Global ATTACK ALL ASR:\s*([0-9.]+)"),
    "personal_asr": re.compile(r"Average Personal ATTACK ALL ASR.*?:\s*([0-9.]+)"),
    "static_acc": re.compile(r"Averaged Test Accurancy:\s*([0-9.]+)"),
    "static_asr": re.compile(r"^ASR:\s*([0-9.]+)"),
    "best_acc": re.compile(r"mean for best accurancy:\s*([0-9.]+)"),
}


def last_match(lines, key):
    pattern = PATTERNS[key]
    value = None
    for line in lines:
        match = pattern.search(line)
        if match:
            value = match.group(1)
    return value


def summarize(path):
    lines = path.read_text(errors="ignore").splitlines()
    status = "done" if any("All done!" in line for line in lines[-80:]) else "running"
    return {
        "file": path.name,
        "status": status,
        "round": last_match(lines, "round"),
        "global_acc": last_match(lines, "global_acc"),
        "global_asr": last_match(lines, "global_asr"),
        "personal_acc": last_match(lines, "personal_acc"),
        "personal_asr": last_match(lines, "personal_asr"),
        "static_acc": last_match(lines, "static_acc"),
        "static_asr": last_match(lines, "static_asr"),
        "best_acc": last_match(lines, "best_acc"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    files = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.log")))
        elif path.exists():
            files.append(path)

    headers = [
        "file",
        "status",
        "round",
        "global_acc",
        "global_asr",
        "personal_acc",
        "personal_asr",
        "static_acc",
        "static_asr",
        "best_acc",
    ]
    print("\t".join(headers))
    for path in files:
        row = summarize(path)
        print("\t".join(row.get(h) or "" for h in headers))


if __name__ == "__main__":
    main()
