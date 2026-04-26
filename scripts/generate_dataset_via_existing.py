#!/usr/bin/env python3
import argparse
import importlib
import importlib.util
import inspect
import os
import shutil
import sys
from pathlib import Path


def load_generator_module(generator_path: Path):
    generator_dir = generator_path.parent
    if str(generator_dir) not in sys.path:
        sys.path.insert(0, str(generator_dir))

    dataset_utils = importlib.import_module("dataset_utils")
    spec = importlib.util.spec_from_file_location("legacy_generator", generator_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, dataset_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True)
    parser.add_argument("--dir-path", required=True)
    parser.add_argument("--rawdata-path", default=None)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--backdoor-rate", type=float, default=0.0)
    parser.add_argument("--adversary-num", type=int, default=0)
    parser.add_argument("--target-y", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--partition", default="dir")
    parser.add_argument("--niid", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--blend-alpha", type=float, default=0.2)
    parser.add_argument("--sig-delta", type=float, default=30/255)
    parser.add_argument("--sig-f", type=int, default=6)
    parser.add_argument("--sig-label-mode", choices=["dirty", "clean"], default="dirty")
    args = parser.parse_args()

    generator_path = Path(args.generator).resolve()
    dir_path = Path(args.dir_path).resolve()
    rawdata_path = Path(args.rawdata_path).resolve() if args.rawdata_path else None
    dir_path_str = f"{dir_path}/"

    if args.force_rebuild and dir_path.exists():
        # Keep raw cache reusable across rebuilds.
        raw_link = dir_path / "rawdata"
        if raw_link.exists() and raw_link.is_symlink():
            raw_link.unlink()
        shutil.rmtree(dir_path)

    dir_path.mkdir(parents=True, exist_ok=True)

    module, dataset_utils = load_generator_module(generator_path)
    dataset_utils.alpha = args.alpha
    dataset_utils.train_ratio = args.train_ratio
    dataset_utils.batch_size = args.batch_size
    dataset_utils.generation_fingerprint = {
        "generator": generator_path.name,
        "num_clients": args.num_clients,
        "backdoor_rate": args.backdoor_rate,
        "adversary_num": args.adversary_num,
        "target_y": args.target_y,
        "alpha": args.alpha,
        "train_ratio": args.train_ratio,
        "batch_size": args.batch_size,
        "partition": None if args.partition == "-" else args.partition,
        "niid": args.niid,
        "balance": args.balance,
        "generator_seed": getattr(module, "GENERATOR_SEED", None),
        "blend_alpha": args.blend_alpha,
        "sig_delta": args.sig_delta,
        "sig_f": args.sig_f,
        "sig_label_mode": args.sig_label_mode,
    }

    setattr(module, "adversary_num", args.adversary_num)
    setattr(module, "target_y", args.target_y)
    setattr(module, "aux_path", str(dir_path / "test") + "/")
    setattr(module, "server_clean_path", str(dir_path / "test") + "/")
    setattr(module, "blend_alpha", args.blend_alpha)
    setattr(module, "sig_delta", args.sig_delta)
    setattr(module, "sig_f", args.sig_f)
    setattr(module, "sig_label_mode", args.sig_label_mode)
    if rawdata_path is not None:
        rawdata_path.mkdir(parents=True, exist_ok=True)
        setattr(module, "rawdata_path", str(rawdata_path))
        # Many legacy generators ignore rawdata_path and hardcode root=dir_path+"rawdata".
        # Create a symlink so all generators share one global cache directory.
        local_raw = dir_path / "rawdata"
        if local_raw.exists() and not local_raw.is_symlink():
            shutil.rmtree(local_raw)
        if not local_raw.exists():
            local_raw.symlink_to(rawdata_path, target_is_directory=True)

    generate_fn = getattr(module, "generate_dataset")
    signature = inspect.signature(generate_fn)

    original_cwd = Path.cwd()
    os.chdir(generator_path.parent)
    try:
        if "backdoor_rate" in signature.parameters and "target_y" in signature.parameters:
            generate_fn(
                dir_path_str,
                args.num_clients,
                args.niid,
                args.balance,
                None if args.partition == "-" else args.partition,
                args.backdoor_rate,
                args.target_y,
            )
        else:
            generate_fn(
                dir_path_str,
                args.num_clients,
                args.niid,
                args.balance,
                None if args.partition == "-" else args.partition,
            )
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()

