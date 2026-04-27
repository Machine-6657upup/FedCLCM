#!/usr/bin/env python3
import argparse
import os


PARAM_PREFIXES = (
    "goal =",
    "device =",
    "join_ratio =",
    "dataset =",
    "model =",
    "batch_size =",
    "local_learning_rate =",
    "global_rounds =",
    "local_epochs =",
    "algorithm =",
    "num_clients =",
    "eval_gap =",
    "plocal_epochs =",
    "lr_head =",
    "rt_beta =",
    "lambda_cl =",
    "aug_strength =",
    "mask_tau =",
    "mask_alpha =",
    "enable_channel_mask =",
    "adv_eps =",
    "adv_num_iter =",
    "num_adv_clients =",
)

METRIC_MARKERS = (
    "Round number",
    "Averaged Test Accurancy",
    "Averaged Test AUC",
    "Averaged Train Loss",
    "ASR:",
    "attack_auc",
    "Best accuracy",
    "Average time cost",
)


def read_lines(path):
    with open(path, "rb") as handle:
        return handle.read().decode("utf-8", "ignore").splitlines()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    for rel in args.files:
        path = os.path.join(args.root, rel)
        print(f"===== {rel} =====")
        if not os.path.exists(path):
            print("MISSING")
            continue
        lines = read_lines(path)
        print("---PARAMS---")
        count = 0
        for line in lines[:250]:
            if line.startswith(PARAM_PREFIXES):
                print(line)
                count += 1
        if count == 0:
            for line in lines[:80]:
                print(line)
        print("---TAIL METRICS---")
        metrics = [line for line in lines if any(marker in line for marker in METRIC_MARKERS)]
        for line in metrics[-60:]:
            print(line)


if __name__ == "__main__":
    main()
