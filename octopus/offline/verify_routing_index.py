import os
import json
import argparse
from collections import defaultdict

import torch


def verify_structure(routing_data, num_layers):
    """Check that routing_index.json has the expected per-layer structure."""
    errors = []

    for layer_idx in range(num_layers):
        layer_str = str(layer_idx)
        if layer_str not in routing_data:
            errors.append(f"Layer {layer_idx}: missing from routing data")
            continue

        layer_info = routing_data[layer_str]

        if "global_dram_tokens" not in layer_info:
            errors.append(f"Layer {layer_idx}: missing 'global_dram_tokens'")
        else:
            dram_tokens = layer_info["global_dram_tokens"]
            if dram_tokens != sorted(dram_tokens):
                errors.append(f"Layer {layer_idx}: global_dram_tokens not sorted")
            if len(dram_tokens) != len(set(dram_tokens)):
                errors.append(f"Layer {layer_idx}: global_dram_tokens has duplicates")

        for cid, info in layer_info.items():
            if cid == "global_dram_tokens":
                continue
            if not isinstance(info, dict):
                errors.append(f"Layer {layer_idx}, cluster {cid}: expected dict, got {type(info).__name__}")
                continue

            for key in ("center_status", "type", "routes"):
                if key not in info:
                    errors.append(f"Layer {layer_idx}, cluster {cid}: missing '{key}'")

            ctype = info.get("type")
            if ctype == "STRIPED_SSD":
                if "physical_ssd_members" not in info:
                    errors.append(f"Layer {layer_idx}, cluster {cid}: STRIPED_SSD missing 'physical_ssd_members'")
                if "physical_locations" not in info:
                    errors.append(f"Layer {layer_idx}, cluster {cid}: STRIPED_SSD missing 'physical_locations'")

    return errors


def verify_token_coverage(routing_data, clusters_data):
    """Verify all cluster tokens appear in either DRAM or SSD routing."""
    errors = []

    for layer_data in clusters_data.get("layers", []):
        layer_idx = layer_data["layer_idx"]
        layer_str = str(layer_idx)

        if layer_str not in routing_data:
            continue

        layer_info = routing_data[layer_str]
        dram_set = set(layer_info.get("global_dram_tokens", []))

        ssd_set = set()
        for cid, info in layer_info.items():
            if cid == "global_dram_tokens":
                continue
            if isinstance(info, dict):
                for route in info.get("routes", []):
                    ssd_set.add(route["token"])

        routed = dram_set | ssd_set

        for c_info in layer_data.get("clusters", []):
            center = c_info["center"]
            all_tokens = [center] + c_info.get("others", [])
            for tok in all_tokens:
                if tok not in routed:
                    errors.append(f"Layer {layer_idx}: token {tok} from cluster(center={center}) not routed")

    return errors


def verify_striping_consistency(routing_data, num_disks, block_size):
    """Verify physical_locations use valid disk indices, correct block sizes, and no overlaps."""
    errors = []
    disk_offsets = defaultdict(list)

    for layer_str, layer_info in routing_data.items():
        if not isinstance(layer_info, dict):
            continue

        for cid, info in layer_info.items():
            if cid == "global_dram_tokens" or not isinstance(info, dict):
                continue

            if info.get("type") != "STRIPED_SSD":
                continue

            locations = info.get("physical_locations", [])
            members = info.get("physical_ssd_members", [])

            if len(locations) != len(members):
                errors.append(
                    f"Layer {layer_str}, cluster {cid}: "
                    f"physical_locations count ({len(locations)}) != physical_ssd_members count ({len(members)})"
                )

            for loc in locations:
                disk_idx = loc.get("disk_idx")
                offset = loc.get("offset")
                length = loc.get("length")

                if disk_idx is None or not (0 <= disk_idx < num_disks):
                    errors.append(
                        f"Layer {layer_str}, cluster {cid}: invalid disk_idx={disk_idx} (num_disks={num_disks})"
                    )
                if length != block_size:
                    errors.append(
                        f"Layer {layer_str}, cluster {cid}: length={length} != block_size={block_size}"
                    )
                if disk_idx is not None:
                    disk_offsets[disk_idx].append((offset, offset + (length or 0), layer_str, cid))

    for disk_id, intervals in disk_offsets.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            _, end_a, la, ca = intervals[i]
            start_b, _, lb, cb = intervals[i + 1]
            if end_a > start_b:
                errors.append(
                    f"Disk {disk_id}: overlap between "
                    f"L{la}/C{ca}(end={end_a}) and L{lb}/C{cb}(start={start_b})"
                )

    return errors


def verify_kv_cache_compatibility(routing_data, kv_cache_path):
    """Verify all referenced token IDs are within the KV cache sequence length."""
    errors = []

    cache = torch.load(kv_cache_path, map_location="cpu", weights_only=False)
    num_layers = len(cache.layers)
    seq_len = cache.layers[0].keys.shape[2]

    for layer_str, layer_info in routing_data.items():
        if not isinstance(layer_info, dict):
            continue

        layer_idx = int(layer_str)
        if layer_idx >= num_layers:
            errors.append(f"Layer {layer_idx}: exceeds KV cache layer count ({num_layers})")
            continue

        for tok in layer_info.get("global_dram_tokens", []):
            if tok < 0 or tok >= seq_len:
                errors.append(f"Layer {layer_idx}: DRAM token {tok} out of range [0, {seq_len})")

        for cid, info in layer_info.items():
            if cid == "global_dram_tokens" or not isinstance(info, dict):
                continue
            for route in info.get("routes", []):
                tok = route["token"]
                if tok < 0 or tok >= seq_len:
                    errors.append(f"Layer {layer_idx}, cluster {cid}: token {tok} out of range [0, {seq_len})")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Octopus Logical Routing Index Verification")
    parser.add_argument("--routing_index", type=str, required=True)
    parser.add_argument("--kv_cache", type=str, required=True)
    parser.add_argument("--clusters", type=str, required=True)
    parser.add_argument("--num_disks", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=4096)
    args = parser.parse_args()

    print("Octopus Logical Routing Index Verification")
    print("-" * 70)

    with open(args.routing_index, "r") as f:
        routing_data = json.load(f)
    with open(args.clusters, "r") as f:
        clusters_data = json.load(f)

    num_layers = clusters_data.get("num_layers", len(routing_data))
    all_errors = []

    print("[1/4] Structure check...")
    errs = verify_structure(routing_data, num_layers)
    all_errors.extend(errs)
    print(f"      {'PASS' if not errs else f'FAIL ({len(errs)} errors)'}")

    print("[2/4] Token coverage check...")
    errs = verify_token_coverage(routing_data, clusters_data)
    all_errors.extend(errs)
    print(f"      {'PASS' if not errs else f'FAIL ({len(errs)} errors)'}")

    print("[3/4] Striping consistency check...")
    errs = verify_striping_consistency(routing_data, args.num_disks, args.block_size)
    all_errors.extend(errs)
    print(f"      {'PASS' if not errs else f'FAIL ({len(errs)} errors)'}")

    print("[4/4] KV cache compatibility check...")
    errs = verify_kv_cache_compatibility(routing_data, args.kv_cache)
    all_errors.extend(errs)
    print(f"      {'PASS' if not errs else f'FAIL ({len(errs)} errors)'}")

    print("-" * 70)
    if all_errors:
        print(f"VERIFICATION FAILED: {len(all_errors)} error(s) found:")
        for e in all_errors[:20]:
            print(f"  - {e}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")
        exit(1)
    else:
        print("VERIFICATION PASSED: Routing index is logically consistent.")


if __name__ == "__main__":
    main()
