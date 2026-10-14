import os
import json
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm

if __package__:
    from .co_activation import DenseDistances, JaccardDistances
else:
    from co_activation import DenseDistances, JaccardDistances


def _build_raw_clusters(dist_matrix, tau, max_clusters, is_full_coverage_mode):
    """Build clusters via greedy density-based expansion.

    For each candidate center (sorted by density descending), start a new
    cluster and greedily add tokens whose average distance to the current
    cluster members is within tau.  All tokens are scanned for each cluster,
    which naturally introduces replicas.

    Returns (clusters, stop_reason, total_covered) where each cluster is a
    list of token indices with the center at position 0.
    """
    N = dist_matrix.size
    densities = dist_matrix.densities(tau)
    sorted_candidates = np.argsort(densities)[::-1]

    clusters = []
    covered = np.zeros(N, dtype=bool)
    stop_reason = "Exhausted all candidates"

    for R_i in sorted_candidates:
        if not is_full_coverage_mode and len(clusters) >= max_clusters:
            stop_reason = f"Reached max_clusters limit ({max_clusters})"
            break

        if covered[R_i]:
            continue

        C_i = [int(R_i)]

        center_row = dist_matrix.row(R_i)
        candidate_js = np.flatnonzero(center_row <= tau)
        candidate_js = candidate_js[np.argsort(center_row[candidate_js])]
        coordinates, groups = dist_matrix.candidate_space(candidate_js)
        dist_sum = dist_matrix.candidate_row(R_i, coordinates).astype(np.float64)
        cluster_count = 1

        for position, j in enumerate(candidate_js):
            j = int(j)
            if j == R_i:
                continue
            if dist_sum[groups[position]] / cluster_count <= tau:
                C_i.append(j)
                dist_sum += dist_matrix.candidate_row(j, coordinates)
                cluster_count += 1

        clusters.append(C_i)
        covered[C_i] = True

        if np.all(covered):
            stop_reason = "Successfully covered ALL tokens"
            break

    return clusters, stop_reason, int(np.sum(covered))


def _prune_contained_clusters(clusters):
    """Remove redundant clusters contained in another cluster."""
    ordered = sorted(enumerate(map(frozenset, clusters)), key=lambda item: len(item[1]), reverse=True)
    kept, postings = {}, {}
    for index, members in ordered:
        anchor = min(members, key=lambda token: len(postings.get(token, ())))
        if any(members <= kept[other] for other in postings.get(anchor, ())):
            continue
        kept[index] = members
        for token in members:
            postings.setdefault(token, []).append(index)
    return [cluster for index, cluster in enumerate(clusters) if index in kept]


def _compute_cluster_metrics(clusters, dist_matrix):
    """Compute intra-cluster distance metrics for each cluster."""
    results = []
    for C_i in clusters:
        center = C_i[0]
        others = C_i[1:]
        size = len(C_i)

        if size > 1:
            avg_intra_dist = dist_matrix.mean_pairwise(C_i)
            avg_dist_to_center = float(np.mean(dist_matrix.row(center, others)))
        else:
            avg_intra_dist = 0.0
            avg_dist_to_center = 0.0

        results.append({
            "center": center,
            "others": others,
            "size": size,
            "avg_intra_cluster_distance": round(avg_intra_dist, 6),
            "avg_dist_to_center": round(avg_dist_to_center, 6),
        })

    results.sort(key=lambda x: x["size"], reverse=True)
    return results


def build_clusters_for_layer(dist_matrix, tau, max_clusters, is_full_coverage_mode):
    """Full clustering pipeline for a single layer's distance matrix.

    Steps:
      1. Build raw clusters via density-based greedy expansion
      2. Prune redundant clusters contained in others
      3. Compute per-cluster distance metrics
    """
    if isinstance(dist_matrix, np.ndarray):
        dist_matrix = DenseDistances(dist_matrix)
    if not np.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("Clustering threshold must be in [0, 1]")
    N = dist_matrix.size
    clusters, stop_reason, total_covered = _build_raw_clusters(
        dist_matrix, tau, max_clusters, is_full_coverage_mode,
    )

    pre_prune_count = len(clusters)
    clusters = _prune_contained_clusters(clusters)
    post_prune_count = len(clusters)

    results_list = _compute_cluster_metrics(clusters, dist_matrix)

    return {
        "stop_reason": stop_reason,
        "final_cluster_count": post_prune_count,
        "pruned_count": pre_prune_count - post_prune_count,
        "coverage_ratio": float(total_covered / N),
        "clusters": results_list,
    }


def _build_layer(task):
    path, layer, tau, max_clusters, cache_bytes = task
    activations = np.load(path, mmap_mode="r")
    start = time.perf_counter()
    distances = JaccardDistances(activations[:, layer, :], cache_bytes=cache_bytes)
    result = build_clusters_for_layer(distances, tau, max_clusters, max_clusters <= 0)
    stats = {"layer_idx": layer, "unique_signatures": distances.unique_count,
             "seconds": time.perf_counter() - start}
    return {"layer_idx": layer, **result}, stats


def main():
    parser = argparse.ArgumentParser(description="Octopus co-activation clustering")
    parser.add_argument("--activations", required=True, help="Binary (steps, layers, tokens) activation tensor")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--max_clusters", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--distance_cache_mb", type=float, default=128,
                        help="Maximum cached distance data per worker in MiB")
    args = parser.parse_args()
    if args.workers < 1 or not np.isfinite(args.distance_cache_mb) or args.distance_cache_mb < 0:
        parser.error("Use positive workers and a finite nonnegative distance cache size")
    if not np.isfinite(args.tau) or not 0 <= args.tau <= 1:
        parser.error("tau must be in [0, 1]")
    activations = np.load(args.activations, mmap_mode="r")
    if activations.ndim != 3 or min(activations.shape) == 0:
        parser.error("Expected a nonempty (steps, layers, tokens) activation tensor")
    _, layers, tokens = activations.shape
    tasks = [(args.activations, layer, args.tau, args.max_clusters, int(args.distance_cache_mb * 1024**2))
             for layer in range(layers)]
    started = time.perf_counter()
    if args.workers == 1:
        results = list(tqdm(map(_build_layer, tasks), total=layers, desc="Clustering layers"))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, layers)) as pool:
            results = list(tqdm(pool.map(_build_layer, tasks), total=layers, desc="Clustering layers"))
    mode = "Full Coverage Mode" if args.max_clusters <= 0 else f"Limited Mode (Max {args.max_clusters})"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "clusters.json"), "w") as f:
        json.dump({"N_tokens": tokens, "num_layers": layers, "tau": args.tau, "mode": mode,
                   "layers": [result for result, _ in results]}, f, indent=2)
    with open(os.path.join(args.output_dir, "modeling_metrics.json"), "w") as f:
        json.dump({"seconds": time.perf_counter() - started, "workers": min(args.workers, layers),
                   "distance_cache_mb_per_worker": args.distance_cache_mb,
                   "layers": [stats for _, stats in results]}, f, indent=2)
    print(f"Clustered {layers} layers of {tokens} tokens; no pairwise distance file required.")


if __name__ == "__main__":
    main()
