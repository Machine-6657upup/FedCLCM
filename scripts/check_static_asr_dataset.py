#!/usr/bin/env python3
import json
import os
import sys

import numpy as np


def inspect_dataset(root, name, num_clients=100, target_y=0):
    dataset_dir = os.path.join(root, name)
    print(f"==== {name} exists={os.path.isdir(dataset_dir)}")
    if not os.path.isdir(dataset_dir):
        return

    config_path = os.path.join(dataset_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print("fingerprint", config.get("generation_fingerprint", {}))

    clean_sizes = []
    backdoor_sizes = []
    target_only_clean = 0
    empty_backdoor = []

    for client_id in range(num_clients):
        test_path = os.path.join(dataset_dir, "test", f"{client_id}.npz")
        backdoor_path = os.path.join(dataset_dir, "test", f"{client_id}_backdoored.npz")

        if os.path.exists(test_path):
            data = np.load(test_path, allow_pickle=True)["data"].tolist()
            labels = np.asarray(data["y"])
            clean_sizes.append(len(labels))
            target_only_clean += int(len(labels) > 0 and np.all(labels == target_y))

        if os.path.exists(backdoor_path):
            data = np.load(backdoor_path, allow_pickle=True)["data"].tolist()
            n = len(data["y"])
            backdoor_sizes.append(n)
            if n == 0:
                empty_backdoor.append(client_id)

    print("clean", _summarize(clean_sizes), "target_only_clean", target_only_clean)
    print("backdoor", _summarize(backdoor_sizes), "empty_backdoor", empty_backdoor[:20], "n_empty", len(empty_backdoor))


def _summarize(values):
    if not values:
        return None
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "sum": int(sum(values)),
        "unique_first20": [int(v) for v in sorted(set(values))[:20]],
    }


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "src/dataset"
    names = sys.argv[2:] or [
        "Cifar10_dir0.5_bdoor0.2_nclient_100_badnet_adv10",
        "Cifar10_dir0.5_bdoor0.2_nclient_100_blend_adv10",
        "Cifar10_dir0.5_bdoor0.2_nclient_100_sig_adv10",
    ]
    for name in names:
        inspect_dataset(root, name)


if __name__ == "__main__":
    main()
