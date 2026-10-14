import os
import json
import torch
import math
import mmap
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from octopus.layout import KVLayout


def aligned_direct_write(fd, offset, data_bytes, aligned_buf):
    """
    Perform an O_DIRECT safe write.
    Uses a pre-allocated mmap buffer to ensure 4KB memory alignment.
    """
    size = len(data_bytes)
    if size % 4096:
        raise ValueError("Record must be 4 KiB aligned")

    aligned_buf[:size] = data_bytes
    os.lseek(fd, offset, os.SEEK_SET)

    written = 0
    while written < size:
        w = os.write(fd, memoryview(aligned_buf)[written:])
        if w <= 0:
            raise RuntimeError(f"[FATAL] Disk write failed at offset {offset}.")
        written += w

def main():
    parser = argparse.ArgumentParser(description="Octopus Offline KV Cache Physical Dumper")
    parser.add_argument("--routing_index", type=str, required=True, help="Path to routing_index.json")
    parser.add_argument("--kv_cache", type=str, required=True, help="Path to the full_kv_cache.pt tensor file")
    parser.add_argument("--disk_paths", nargs='+', required=True, help="Paths to the physical disk files (e.g. /mnt/kv_disk1/octo.bin)")
    args = parser.parse_args()

    print("Octopus Physical KV Dumper Initializing...")
    print("-" * 70)

    print(f"Loading routing index from {args.routing_index}")
    with open(args.routing_index, "r") as f:
        routing_data = json.load(f)

    print(f"Loading full KV cache from {args.kv_cache}")
    raw_cache = torch.load(args.kv_cache, map_location="cpu", weights_only=False)
    
    layout = KVLayout.from_cache(raw_cache)
    full_cache = []
    for layer in raw_cache.layers:
        full_cache.append((layer.keys, layer.values))
    
    num_layers = len(full_cache)
    print(f"Detected {num_layers} layers in KV cache.")


    disk_fds = []
    if len(set(str(Path(p).resolve()) for p in args.disk_paths)) != len(args.disk_paths):
        raise ValueError("KV files must be distinct")
    if any(Path(p).exists() or Path(p).resolve().is_relative_to('/dev') for p in args.disk_paths):
        raise ValueError("Use new regular KV files, never existing files or raw devices")
    for path in args.disk_paths:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_DIRECT | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            disk_fds.append(fd)
            print(f"Successfully opened O_DIRECT stream for {path}")
        except PermissionError:
            print(f"[FATAL] Permission denied for {path}. Check file ownership.")
            exit(1)

    aligned_buf = mmap.mmap(-1, layout.block_bytes)

    total_tokens_written = 0
    disk_write_counts = [0] * len(args.disk_paths)

    print("-" * 70)
    print("Starting physical data striping...")

    for layer_str in sorted(routing_data.keys(), key=lambda x: int(x)):
        layer_idx = int(layer_str)
        layer_info = routing_data[layer_str]
        
        K = full_cache[layer_idx][0]
        V = full_cache[layer_idx][1]
        
        layer_tokens_written = 0
        
        for cid, cluster_info in layer_info.items():
            if cid == "global_dram_tokens":
                continue
                
            if cluster_info.get("type") == "STRIPED_SSD":
                locations = cluster_info.get("physical_locations", [])
                
                for loc in locations:
                    token_id = loc["token"]
                    disk_idx = loc["disk_idx"]
                    offset = loc["offset"]

                    # Payload is K followed by V, padded to an aligned disk record.
                    k_slice = K[0, :, token_id, :]
                    v_slice = V[0, :, token_id, :]
                    data_bytes = layout.pack(k_slice, v_slice)
                    aligned_direct_write(disk_fds[disk_idx], offset, data_bytes, aligned_buf)
                    
                    layer_tokens_written += 1
                    total_tokens_written += 1
                    disk_write_counts[disk_idx] += 1
                    
        print(f"Layer {layer_idx:<2} | Dumped {layer_tokens_written} tokens to NVMe array.")

    aligned_buf.close()
    for fd in disk_fds:
        os.fsync(fd)
        os.close(fd)

    print("-" * 70)
    print("Physical Dump Completed Successfully!")
    print(f"Total Tokens Transferred: {total_tokens_written}")
    total_mb = (total_tokens_written * layout.block_bytes) / (1024 * 1024)
    print(f"Total IO Volume: {total_mb:.2f} MB")
    
    print("\nDisk Write Distribution:")
    for i, count in enumerate(disk_write_counts):
        print(f"  Disk {i}: {count} blocks written")

if __name__ == "__main__":
    main()
