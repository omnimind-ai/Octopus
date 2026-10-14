import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from octopus.models import ModelAdapter
from octopus.data import add_dataset_arguments


def run_step(step_name, command_list):
    """Execute a subprocess command.  Halts the entire pipeline on failure."""
    command_list[1] = str(Path(__file__).resolve().parent / command_list[1])
    cmd_str = " ".join(command_list)
    print(f"\n{'='*80}")
    print(f"[Phase]: {step_name}")
    print(f"[Exec]:  {cmd_str}")
    print(f"{'-'*80}\n")

    try:
        subprocess.run(command_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL ERROR] Phase '{step_name}' failed with return code {e.returncode}.")
        print("Pipeline halted. Please check the error logs above.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n[FATAL ERROR] Could not find the script for Phase '{step_name}'.")
        print("Please ensure you are running this from the Octopus/octopus/offline directory.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Octopus: End-to-End Layer-wise Offline Pipeline")

    # Prefill/decode params.
    parser.add_argument("--dataset_path", type=str)
    add_dataset_arguments(parser, internal=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prefix_length", type=int, default=3000)
    parser.add_argument("--total_t", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--strategy", type=str, default="quantile_mass")
    parser.add_argument("--param", type=float, default=0.90)

    # Clustering params.
    parser.add_argument("--tau", type=float, default=0.05, help="Co-activation distance threshold")
    parser.add_argument("--max_clusters", type=int, default=0, help="0 means run until full coverage")
    parser.add_argument("--co_activation_workers", type=int, default=1)
    parser.add_argument("--distance_cache_mb", type=float, default=128)

    # Dispatch params.
    parser.add_argument("--alpha", type=float, default=10.0, help="SSD seek penalty ratio")
    parser.add_argument("--budget_ratio", type=float, default=0.1)
    parser.add_argument("--local_window", type=int, default=128)
    parser.add_argument("--min_c_size", type=int, default=8)

    # Physical allocation params.
    parser.add_argument("--num_disks", type=int, default=8)
    parser.add_argument("--block_size", type=int, help="Optional check of the inferred aligned KV record size")

    parser.add_argument("--output_dir", type=str, default="results_3k")
    parser.add_argument("--disk_paths", nargs='+')

    # Skip physical disk phases (for environments without multi-disk NVMe).
    parser.add_argument("--skip_physical", action="store_true",
                        help="Verify the logical routing index without writing physical KV files")

    args = parser.parse_args()
    args.model_path = str(Path(args.model_path).expanduser().resolve())
    args.dataset_path = str(Path(args.dataset_path).expanduser().resolve()) if args.dataset_path else None
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    args.disk_paths = [str(Path(p).expanduser().resolve()) for p in (args.disk_paths or [])]
    adapter = ModelAdapter.from_path(args.model_path)
    if args.num_disks < 1 or args.block_size not in (None, adapter.layout.block_bytes):
        parser.error("Disk count must be positive and block_size must match the model KV geometry")
    args.block_size = adapter.layout.block_bytes
    if not args.skip_physical:
        if len(args.disk_paths) != args.num_disks or len(set(args.disk_paths)) != args.num_disks:
            parser.error("Supply one distinct --disk_paths entry per disk")
        if any(Path(p).exists() or Path(p).is_relative_to('/dev') for p in args.disk_paths):
            parser.error("Use new regular KV files; never existing files or raw devices")
    os.makedirs(args.output_dir, exist_ok=True)
    py_exec = sys.executable

    print("\nStarting Octopus Layer-wise Offline Storage Preparation Pipeline...")

    # Phase 1: collect activation matrix and full KV cache.
    cmd_1 = [
        py_exec, "prefill_and_decode.py",
        "--model_path", args.model_path,
        "--device", args.device,
        "--prefix_length", str(args.prefix_length),
        "--total_t", str(args.total_t),
        "--runs", str(args.runs),
        "--temperature", str(args.temperature),
        "--strategy", args.strategy,
        "--param", str(args.param),
        "--output_dir", args.output_dir,
    ]
    cmd_1 += ['--dataset', args.dataset, '--dataset_split', args.dataset_split,
              '--sample_index', str(args.sample_index)]
    if args.dataset_path:
        cmd_1 += ['--dataset_path', args.dataset_path]
    if args.dataset_config:
        cmd_1 += ['--dataset_config', args.dataset_config]
    run_step("1. Prefill & Decode -> Multi-Layer Activation Tensor", cmd_1)
    profile = json.loads((Path(args.output_dir) / 'config.json').read_text())
    args.prefix_length = profile['prefix_length']
    requested_window = args.local_window
    args.local_window = min(args.local_window, args.prefix_length)
    with open(os.path.join(args.output_dir, "storage.json"), "w") as f:
        json.dump({"disk_paths": args.disk_paths, "num_disks": args.num_disks,
                   **adapter.layout.metadata(), "model": adapter.metadata(),
                   "disk_base": 0, "physical": not args.skip_physical,
                   "dram_budget_tokens": int(args.prefix_length * args.budget_ratio),
                   "local_window": args.local_window, "requested_local_window": requested_window}, f, indent=2)

    # Compute distances on demand and cluster each layer in one pass.
    cmd_2 = [
        py_exec, "cluster_builder.py",
        "--activations", os.path.join(args.output_dir, "activation_matrix.npy"),
        "--output_dir", args.output_dir,
        "--tau", str(args.tau),
        "--max_clusters", str(args.max_clusters),
        "--workers", str(args.co_activation_workers),
        "--distance_cache_mb", str(args.distance_cache_mb),
    ]
    run_step("2. Co-Activation Modeling & Clustering", cmd_2)

    # Phase 3: score and dispatch tokens between DRAM/SSD.
    cmd_4 = [
        py_exec, "token_dispatcher.py",
        "--matrix", os.path.join(args.output_dir, "activation_matrix.npy"),
        "--clusters", os.path.join(args.output_dir, "clusters.json"),
        "--output", os.path.join(args.output_dir, "scored_routing_index.json"),
        "--alpha", str(args.alpha),
        "--budget_ratio", str(args.budget_ratio),
        "--local_window", str(args.local_window),
        "--min_c_size", str(args.min_c_size),
    ]
    run_step("3. Token Scoring & Atomic DRAM/SSD Dispatch", cmd_4)

    # Phase 4: assign physical SSD locations.
    cmd_5 = [
        py_exec, "multi_disk_allocator.py",
        "--scored_index", os.path.join(args.output_dir, "scored_routing_index.json"),
        "--output", os.path.join(args.output_dir, "routing_index.json"),
        "--num_disks", str(args.num_disks),
        "--block_size", str(args.block_size),
    ]
    run_step("4. Multi-Disk Global Round-Robin Striping", cmd_5)

    if args.skip_physical:
        # Phase 4b: logical verification (no disk required).
        cmd_5b = [
            py_exec, "verify_routing_index.py",
            "--routing_index", os.path.join(args.output_dir, "routing_index.json"),
            "--kv_cache", os.path.join(args.output_dir, "full_kv_cache.pt"),
            "--clusters", os.path.join(args.output_dir, "clusters.json"),
            "--num_disks", str(args.num_disks),
            "--block_size", str(args.block_size),
        ]
        run_step("4b. Logical Routing Index Verification (no disk required)", cmd_5b)

        print(f"\n{'='*80}")
        print("[SUCCESS] Octopus Offline Pipeline Completed (physical disk phases skipped).")
        print(f"Routing index: {os.path.join(args.output_dir, 'routing_index.json')}")
        print(f"{'='*80}")
    else:
        # Phase 5: dump offline KV to physical disks.
        cmd_6 = [
            py_exec, "dump_physical_kv_offline.py",
            "--routing_index", os.path.join(args.output_dir, "routing_index.json"),
            "--kv_cache", os.path.join(args.output_dir, "full_kv_cache.pt"),
        ]
        cmd_6.extend(["--disk_paths"] + args.disk_paths)
        run_step("5. Physical Storage Allocation & O_DIRECT Dumping", cmd_6)

        # Phase 6: verify physical bytes against tensor ground truth.
        cmd_7 = [
            py_exec, "verify_physical_kv.py",
            "--routing_index", os.path.join(args.output_dir, "routing_index.json"),
            "--kv_cache", os.path.join(args.output_dir, "full_kv_cache.pt"),
        ]
        cmd_7.extend(["--disk_paths"] + args.disk_paths)
        run_step("6. Absolute Byte-to-Byte Physical Verification", cmd_7)

        print(f"\n{'='*80}")
        print("[SUCCESS] Octopus Layer-wise Offline Pipeline Fully Completed!")
        print(f"Routing index: {os.path.join(args.output_dir, 'routing_index.json')}")
        print(f"KV tensors striped across {len(args.disk_paths)} NVMe disks.")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
