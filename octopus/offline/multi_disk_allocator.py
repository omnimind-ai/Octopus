import os
import json
import argparse


def allocate_striping(routing_data, num_disks, block_size):
    """Assign physical disk locations via global round-robin to all STRIPED_SSD clusters.

    Returns (updated routing_data, per-disk token counts).
    """
    global_disk_cursor = 0
    disk_offsets = [0] * num_disks
    disk_token_counts = [0] * num_disks

    for layer_str, layer_info in routing_data.items():
        for cid, cluster_info in layer_info.items():
            if cid == "global_dram_tokens":
                continue

            if cluster_info.get("type") == "STRIPED_SSD":
                physical_members = cluster_info.get("physical_ssd_members", [])
                physical_locations = []

                for token in physical_members:
                    disk_id = global_disk_cursor % num_disks
                    offset = disk_offsets[disk_id]

                    physical_locations.append({
                        "token": token,
                        "disk_idx": disk_id,
                        "offset": offset,
                        "length": block_size,
                    })

                    disk_offsets[disk_id] += block_size
                    disk_token_counts[disk_id] += 1
                    global_disk_cursor += 1

                cluster_info["physical_locations"] = physical_locations

    return routing_data, disk_token_counts


def print_allocation_report(disk_token_counts, block_size):
    """Print a summary of per-disk allocation."""
    print("-" * 75)
    print("Physical Disk Allocation Report:")
    print(f"{'Disk ID':<10} | {'Allocated Blocks (Tokens)':<30} | {'Total Size (MB)':<20}")
    print("-" * 75)

    total = 0
    for i, blocks in enumerate(disk_token_counts):
        size_mb = (blocks * block_size) / (1024 * 1024)
        total += blocks
        print(f"Disk {i:<5} | {blocks:<30} | {size_mb:.2f} MB")

    print("-" * 75)
    print(f"Total Striped SSD Tokens: {total}")


def main():
    parser = argparse.ArgumentParser(description="Octopus Multi-Disk Physical Allocator")
    parser.add_argument("--scored_index", type=str, required=True,
                        help="Path to scored_routing_index.json")
    parser.add_argument("--output", type=str, default="results/routing_index.json",
                        help="Path for final routing_index.json")
    parser.add_argument("--num_disks", type=int, default=6,
                        help="Number of physical NVMe SSDs available")
    parser.add_argument("--block_size", type=int, default=4096,
                        help="Aligned block size per token in bytes")
    args = parser.parse_args()

    print("Octopus Multi-Disk Allocator: Global Round-Robin Striping")
    print("-" * 75)
    print(f"Target Disks: {args.num_disks}")
    print(f"Block Size: {args.block_size} bytes")

    with open(args.scored_index, "r") as f:
        routing_data = json.load(f)

    routing_data, disk_token_counts = allocate_striping(
        routing_data, args.num_disks, args.block_size,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(routing_data, f, indent=2)

    print_allocation_report(disk_token_counts, args.block_size)
    print(f"Final routing index saved to: {args.output}")


if __name__ == "__main__":
    main()
