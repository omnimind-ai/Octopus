from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

from octopus.data import add_dataset_arguments

ROOT = Path(__file__).resolve().parent


def command(script, options):
    cmd = [sys.executable, str(script)]
    for key, value in options.items():
        if value is None or value is False:
            continue
        cmd.append("--" + key)
        if value is not True:
            cmd.extend(map(str, value if isinstance(value, (tuple, list)) else [value]))
    return cmd


def make_parser():
    parser = argparse.ArgumentParser(description="Octopus offline preparation and online inference")
    parser.add_argument("--stage", choices=("all", "offline", "inference"), default="all")
    parser.add_argument("--model-path", type=Path, required=True, help="Local Qwen, Llama, or Gemma text model directory")
    parser.add_argument("--dataset-path", type=Path, help="Local dataset file or datasets.save_to_disk directory")
    add_dataset_arguments(parser)
    parser.add_argument("--work-dir", type=Path, required=True, help="Offline artifacts and inference metrics")
    parser.add_argument("--disk-paths", nargs="+", type=Path, help="New KV files on already-mounted SSDs, in disk order")
    parser.add_argument("--dram-only", action="store_true", help="Skip SSD preparation and run a DRAM-only check")
    parser.add_argument("--device", default="cuda", help="Online device, e.g. cuda:0 (cpu only with --dram-only)")
    parser.add_argument("--offline-device", help="Prefill/profiling device; defaults to --device")
    parser.add_argument("--prefix-length", type=int, default=2048)
    parser.add_argument("--profile-steps", type=int, default=64)
    parser.add_argument("--profile-runs", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--tau", type=float, default=.7, help="Offline clustering distance threshold")
    parser.add_argument("--online-tau", type=float, default=.25)
    parser.add_argument("--dram-ratio", type=float, default=.2)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1, help="Offline modeling workers")
    parser.add_argument("--distance-cache-mb", type=float, default=128,
                        help="Maximum cached distance data per offline worker in MiB")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=.7, help="Online sampling temperature; 0 is greedy")
    parser.add_argument("--disable-prefetch", action="store_true")
    parser.add_argument("--disable-cache-update", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Print generated text during decoding")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without loading models or accessing KV files")
    return parser


def plan(args, parser):
    for key in ("prefix_length", "profile_steps", "profile_runs", "decode_tokens", "local_window", "workers"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not math.isfinite(args.distance_cache_mb) or args.distance_cache_mb < 0:
        parser.error("--distance-cache-mb must be finite and nonnegative")
    if args.profile_runs > args.profile_steps or args.local_window > args.prefix_length:
        parser.error("profile-runs must not exceed profile-steps; local-window must not exceed prefix-length")
    if not 0 <= args.dram_ratio <= 1 or not 0 <= args.tau <= 1 or not 0 <= args.online_tau <= 1 or not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("Ratios/thresholds must be in [0,1], and temperature must be nonnegative")
    if args.sample_index < 0:
        parser.error("--sample-index must be nonnegative")
    if args.stage != "inference" and args.dataset_path is None and args.dataset == "context":
        parser.error("--dataset-path is required for offline preparation")
    if (args.stage != "inference" and args.dataset_path is None
            and args.dataset in ("longbench", "infinitebench") and not args.dataset_config):
        parser.error("Use --dataset-config TASK for a LongBench or InfiniteBench Hub dataset")
    if not args.dram_only and not args.disk_paths:
        parser.error("Supply --disk-paths or choose --dram-only")
    if args.stage != "offline" and not args.dram_only and not args.device.startswith("cuda"):
        parser.error("SSD inference requires a CUDA device")
    model, work = args.model_path.expanduser().resolve(), args.work_dir.expanduser().resolve()
    dataset = args.dataset_path.expanduser().resolve() if args.dataset_path else None
    disks = [str(p.expanduser().resolve()) for p in (args.disk_paths or [])]
    if len(disks) != len(set(disks)) or any(Path(p).is_relative_to('/dev') for p in disks):
        parser.error("Disk paths must be distinct regular files, never raw devices")
    num_disks = len(disks) or 1
    commands = []
    if args.stage in ("all", "offline"):
        commands.append(command(ROOT / "offline/offline.py", {
            "model_path": model, "dataset_path": dataset, "output_dir": work,
            "dataset": args.dataset, "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split, "sample_index": args.sample_index,
            "device": args.offline_device or args.device, "prefix_length": args.prefix_length,
            "total_t": args.profile_steps, "runs": args.profile_runs, "tau": args.tau,
            "budget_ratio": args.dram_ratio, "local_window": args.local_window,
            "co_activation_workers": args.workers, "distance_cache_mb": args.distance_cache_mb,
            "num_disks": num_disks, "disk_paths": disks or None, "skip_physical": args.dram_only,
        }))
    if args.stage in ("all", "inference"):
        commands.append(command(ROOT / "online/inference.py", {
            "model_path": model, "kv_cache": work / "full_kv_cache.pt", "offline_dir": work,
            "device": args.device, "max_new_tokens": args.decode_tokens,
            "local_window": args.local_window, "tau": args.online_tau, "num_disks": num_disks,
            "disk_paths": disks or None, "dram_only": args.dram_only,
            "ban_prefetch": args.disable_prefetch, "ban_cache_update": args.disable_cache_update,
            "debug": args.debug, "stream": args.stream, "seed": args.seed, "temperature": args.temperature,
            "metrics_output": work / "inference_metrics.json",
        }))
    return commands, model, dataset, work, disks


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    commands, model, dataset, work, disks = plan(args, parser)
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if not (model / "config.json").is_file():
        parser.error(f"Model config not found: {model / 'config.json'}")
    if args.stage in ("all", "offline"):
        if dataset is not None and not dataset.exists():
            parser.error(f"Input file not found: {dataset}")
        if work.exists() and (not work.is_dir() or any(work.iterdir())):
            parser.error("Offline preparation requires a new or empty --work-dir")
        if not args.dram_only and any(Path(p).exists() for p in disks):
            parser.error("Offline preparation requires new KV filenames; existing files are never overwritten")
    if not args.dram_only:
        if any(not Path(p).parent.is_dir() for p in disks):
            parser.error("KV parent directories must exist on already-mounted storage")
        if args.stage == "inference" and any(not Path(p).is_file() for p in disks):
            parser.error("Every KV file must exist for inference")
    if args.stage == "inference" and not (work / "full_kv_cache.pt").is_file():
        parser.error("Offline artifacts are missing; run --stage offline first")
    work.mkdir(parents=True, exist_ok=True)
    with (work / "launch.json").open("w") as f:
        json.dump({"commands": commands, "stage": args.stage}, f, indent=2)
    for cmd in commands:
        subprocess.run(cmd, check=True, cwd=Path(cmd[1]).parent)


if __name__ == "__main__":
    main()
