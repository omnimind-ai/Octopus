import os
import json
import numpy as np
import argparse


def compute_frequency_matrix(activation_matrix):
    """Sum activation observations over the T dimension to get per-layer token frequencies.

    Handles both 3D (T, L, N) and 2D (L, N) input.
    """
    if len(activation_matrix.shape) == 3:
        return activation_matrix.sum(axis=0)
    return activation_matrix


def build_mandatory_dram_set(seq_len, local_window):
    """Initialize the DRAM set with local window tokens (recency bias)."""
    local_start = max(0, seq_len - local_window)
    return set(range(local_start, seq_len))


def score_clusters(clusters_raw, freqs, alpha, min_c_size):
    """Score each cluster and prepare routing metadata.

    Score(Ci) = fi * (alpha / |Ci| + 1) where fi is the center's activation
    frequency.  Also determines center_status based on min_c_size.

    Returns (cluster_payloads sorted by score descending, initial layer_routing dict).
    """
    layer_routing = {}
    cluster_payloads = []

    for c_dict in clusters_raw:
        center = c_dict["center"]
        others = c_dict.get("others", [])
        all_tokens = [center] + others
        c_size = len(all_tokens)

        center_status = "PINNED_CENTER" if c_size >= min_c_size else "NONE"
        layer_routing[str(center)] = {"center_status": center_status, "routes": []}

        score = float(freqs[center] * ((alpha / c_size) + 1.0))
        cluster_payloads.append({
            "center_id": center,
            "all_tokens": all_tokens,
            "score": score,
        })

    cluster_payloads.sort(key=lambda x: x["score"], reverse=True)
    return cluster_payloads, layer_routing


def dispatch_clusters(cluster_payloads, layer_routing, dram_set, dram_budget_max):
    """Assign each cluster as PINNED_DRAM or STRIPED_SSD based on score-ordered budget.

    Returns the count of physical SSD tokens for this layer.
    """
    ssd_physical_count = 0

    for payload in cluster_payloads:
        cid = str(payload["center_id"])
        all_tokens = payload["all_tokens"]
        center_token = payload["center_id"]

        layer_routing[cid]["score"] = payload["score"]
        novel_tokens = [t for t in all_tokens if t not in dram_set]

        if (len(dram_set) + len(novel_tokens) <= dram_budget_max
                and layer_routing[cid]["center_status"] == "PINNED_CENTER"):
            dram_set.update(novel_tokens)
            layer_routing[cid]["type"] = "PINNED_DRAM"
            layer_routing[cid]["routes"] = [{"token": t} for t in all_tokens]
        else:
            if layer_routing[cid]["center_status"] == "PINNED_CENTER":
                ssd_members = [t for t in all_tokens if t != center_token]
            else:
                ssd_members = all_tokens

            ssd_physical_count += len(ssd_members)
            layer_routing[cid]["type"] = "STRIPED_SSD"
            layer_routing[cid]["routes"] = [{"token": t} for t in all_tokens]
            layer_routing[cid]["physical_ssd_members"] = ssd_members

    return ssd_physical_count


def dispatch_layer(layer_data, freq_matrix, alpha, budget_ratio, local_window, min_c_size):
    """Full dispatch pipeline for one layer.

    Returns (layer_routing dict, dram_count, ssd_count).
    """
    layer_idx = layer_data["layer_idx"]
    freqs = freq_matrix[layer_idx]
    seq_len = freq_matrix.shape[1]

    dram_set = build_mandatory_dram_set(seq_len, local_window)
    clusters_raw = layer_data.get("clusters", [])

    cluster_payloads, layer_routing = score_clusters(clusters_raw, freqs, alpha, min_c_size)

    # Pin eligible cluster centers into DRAM.
    for payload in cluster_payloads:
        cid = str(payload["center_id"])
        if layer_routing[cid]["center_status"] == "PINNED_CENTER":
            dram_set.add(payload["center_id"])

    dram_budget_max = int(seq_len * budget_ratio)
    ssd_count = dispatch_clusters(cluster_payloads, layer_routing, dram_set, dram_budget_max)

    layer_routing["global_dram_tokens"] = sorted(dram_set)
    return layer_routing, len(dram_set), ssd_count


def main():
    parser = argparse.ArgumentParser(description="Octopus Token Dispatch & Scoring Engine")
    parser.add_argument("--matrix", type=str, required=True, help="Path to activation_matrix.npy")
    parser.add_argument("--clusters", type=str, required=True, help="Path to clusters.json")
    parser.add_argument("--output", type=str, default="results/scored_routing_index.json")
    parser.add_argument("--alpha", type=float, default=10.0, help="SSD seek penalty ratio")
    parser.add_argument("--budget_ratio", type=float, default=0.15, help="DRAM budget ratio")
    parser.add_argument("--local_window", type=int, default=128, help="Local window size")
    parser.add_argument("--min_c_size", type=int, default=2, help="Threshold to pin center")
    args = parser.parse_args()

    activation_matrix = np.load(args.matrix)
    freq_matrix = compute_frequency_matrix(activation_matrix)

    with open(args.clusters, "r") as f:
        raw_clusters = json.load(f)

    seq_len = freq_matrix.shape[1]
    print(f"Loaded activation matrix with shape {activation_matrix.shape}")
    print(f"Sequence Length: {seq_len}, Layers: {freq_matrix.shape[0]}")

    scored_routing = {}
    global_total_dram = 0
    global_total_ssd = 0

    print("Octopus Token Dispatcher: Scoring and Dispatching")
    print("-" * 80)
    print(f"{'Layer':<6} | {'DRAM Physical (Tokens)':<25} | {'SSD Physical (Tokens)':<25} | {'DRAM / Budget'}")
    print("-" * 80)

    for layer_data in raw_clusters.get("layers", []):
        layer_idx = layer_data["layer_idx"]
        layer_routing, dram_count, ssd_count = dispatch_layer(
            layer_data, freq_matrix, args.alpha, args.budget_ratio,
            args.local_window, args.min_c_size,
        )

        scored_routing[str(layer_idx)] = layer_routing
        global_total_dram += dram_count
        global_total_ssd += ssd_count

        dram_budget_max = int(seq_len * args.budget_ratio)
        print(f"L{layer_idx:<5} | {dram_count:<25} | {ssd_count:<25} | {dram_count} / {dram_budget_max}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scored_routing, f, indent=2)

    num_layers = len(raw_clusters.get("layers", []))
    print("-" * 80)
    print("Overall Allocation Summary:")
    print(f"  Total Evaluated Layers: {num_layers}")
    print(f"  Total CPU DRAM Physical Tokens: {global_total_dram}")
    print(f"  Total SSD Physical Tokens (with replicas): {global_total_ssd}")
    print(f"Routing index saved to: {args.output}")


if __name__ == "__main__":
    main()
