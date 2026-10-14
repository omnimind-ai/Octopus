import os
import json
import mmap
import torch
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from octopus.layout import KVLayout


def aligned_direct_read(fd, offset, size, aligned_buf):
    """
    Perform an O_DIRECT safe read into an aligned buffer.
    Bypasses Python's internal unaligned memory allocation using os.readv.
    """
    os.lseek(fd, offset, os.SEEK_SET)
    bytes_read = 0
    while bytes_read < size:
        buffer_view = memoryview(aligned_buf)[bytes_read:size]
        r_len = os.readv(fd, [buffer_view])
        
        if r_len == 0:
            raise EOFError(f"[FATAL] Reached EOF unexpectedly at offset {offset}")
        
        bytes_read += r_len
        
    return memoryview(aligned_buf)[:size].tobytes()

def main():
    parser = argparse.ArgumentParser(description="Octopus Physical KV Cache Verifier")
    parser.add_argument("--routing_index", type=str, required=True, help="Path to routing_index.json")
    parser.add_argument("--kv_cache", type=str, required=True, help="Path to the full_kv_cache.pt tensor file")
    parser.add_argument("--disk_paths", nargs='+', required=True, help="Paths to the physical disk files")
    args = parser.parse_args()

    print("Octopus Physical KV Verifier Initializing...")
    print("-" * 75)

    with open(args.routing_index, "r") as f:
        routing_data = json.load(f)

    print(f"Loading ground truth KV cache from {args.kv_cache}...")
    raw_cache = torch.load(args.kv_cache, map_location="cpu", weights_only=False)
    layout = KVLayout.from_cache(raw_cache)
    full_cache = []
    for layer in raw_cache.layers:
        full_cache.append((layer.keys, layer.values))

    num_layers = len(full_cache)

    disk_fds = []
    for path in args.disk_paths:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            disk_fds.append(fd)
        except Exception as e:
            print(f"[FATAL] Failed to open {path} for reading: {e}")
            exit(1)

    aligned_buf = mmap.mmap(-1, layout.block_bytes)
    total_verified = 0

    print("Starting byte-to-byte read-back verification...")
    print("-" * 75)

    for layer_str in sorted(routing_data.keys(), key=lambda x: int(x)):
        layer_idx = int(layer_str)
        layer_info = routing_data[layer_str]
        
        K = full_cache[layer_idx][0]
        V = full_cache[layer_idx][1]
        
        layer_verified = 0
        
        for cid, cluster_info in layer_info.items():
            if cid == "global_dram_tokens":
                continue
                
            if cluster_info.get("type") == "STRIPED_SSD":
                locations = cluster_info.get("physical_locations", [])
                
                for loc in locations:
                    token_id = loc["token"]
                    disk_idx = loc["disk_idx"]
                    offset = loc["offset"]

                    k_slice = K[0, :, token_id, :]
                    v_slice = V[0, :, token_id, :]
                    truth_bytes = layout.pack(k_slice, v_slice)

                    disk_bytes = aligned_direct_read(disk_fds[disk_idx], offset, layout.block_bytes, aligned_buf)

                    if truth_bytes != disk_bytes:
                        print(f"\n[FATAL DATA CORRUPTION] Mismatch found!")
                        print(f"  Layer: {layer_idx}, Token ID: {token_id}")
                        print(f"  Disk Index: {disk_idx}, Offset: {offset}")
                        exit(1)
                        
                    layer_verified += 1
                    total_verified += 1
                    
        print(f"Layer {layer_idx:<2} | Verified {layer_verified} physical token blocks. Status: MATCH")

    aligned_buf.close()
    for fd in disk_fds:
        os.close(fd)

    print("-" * 75)
    print("Physical KV verification passed.")
    print(f"Total Block Matches: {total_verified} / {total_verified}")
    print("The routing index perfectly represents the physical NVMe state.")

if __name__ == "__main__":
    main()
