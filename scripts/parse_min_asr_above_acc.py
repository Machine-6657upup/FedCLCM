import argparse
import re
from pathlib import Path


ROUND_RE = re.compile(r"Round number:\s*(\d+)")
ACC_RE = re.compile(r"Averaged Test Accurancy:\s*([0-9.]+)")
ASR_RE = re.compile(r"ASR\s*:?\s*([0-9.]+)")


def parse_log(path: Path):
    cur_round = None
    cur_acc = None
    pairs = []
    for line in path.read_text(errors="ignore").splitlines():
        m = ROUND_RE.search(line)
        if m:
            cur_round = int(m.group(1))
            cur_acc = None
            continue
        m = ACC_RE.search(line)
        if m:
            cur_acc = float(m.group(1))
            continue
        m = ASR_RE.search(line)
        if m and cur_acc is not None:
            pairs.append((cur_round, cur_acc, float(m.group(1))))
            cur_acc = None
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    for raw in args.logs:
        path = Path(raw)
        print(f"\nFILE\t{path}")
        if not path.exists():
            print("status\tmissing")
            continue

        pairs = parse_log(path)
        valid = [row for row in pairs if row[1] > args.threshold]
        print(f"pairs\t{len(pairs)}")
        print(f"valid_acc_gt_{args.threshold}\t{len(valid)}")
        if not pairs:
            continue

        final = pairs[-1]
        max_acc = max(pairs, key=lambda row: row[1])
        print(f"final\tround={final[0]}\tacc={final[1]:.6f}\tasr={final[2]:.6f}")
        print(f"max_acc\tround={max_acc[0]}\tacc={max_acc[1]:.6f}\tasr={max_acc[2]:.6f}")

        if valid:
            best = min(valid, key=lambda row: row[2])
            print(f"min_asr_acc_gt_threshold\tround={best[0]}\tacc={best[1]:.6f}\tasr={best[2]:.6f}")
            print("lowest_5")
            for row in sorted(valid, key=lambda row: row[2])[:5]:
                print(f"round={row[0]}\tacc={row[1]:.6f}\tasr={row[2]:.6f}")


if __name__ == "__main__":
    main()
