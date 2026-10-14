import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from octopus.offline.co_activation import DenseDistances, JaccardDistances
from octopus.offline.cluster_builder import build_clusters_for_layer, _prune_contained_clusters


def reference_distances(observations):
    x = observations.astype(np.float32).T
    intersection = x @ x.T
    sums = x.sum(axis=1)
    union = sums[:, None] + sums[None, :] - intersection
    distance = 1 - np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    np.fill_diagonal(distance, 0)
    return distance


def reference_clusters(distance, tau, limit=0):
    covered = np.zeros(len(distance), dtype=bool)
    clusters = []
    for center in np.argsort((distance <= tau).sum(axis=1))[::-1]:
        if limit > 0 and len(clusters) >= limit:
            break
        if covered[center]:
            continue
        members = [int(center)]
        candidates = np.flatnonzero(distance[center] <= tau)
        candidates = candidates[np.argsort(distance[center, candidates])]
        for candidate in candidates:
            if candidate != center and sum(float(distance[candidate, m]) for m in members) / len(members) <= tau:
                members.append(int(candidate))
        covered[members] = True
        clusters.append(members)
        if covered.all():
            break
    ordered = sorted(enumerate(map(frozenset, clusters)), key=lambda item: len(item[1]), reverse=True)
    removed = set()
    for i, (idx, members) in enumerate(ordered):
        if idx in removed:
            continue
        for other_idx, other in ordered[i + 1:]:
            if other <= members:
                removed.add(other_idx)
    retained = [c for i, c in enumerate(clusters) if i not in removed]
    return sorted(retained, key=len, reverse=True)


class DistanceTests(unittest.TestCase):
    def test_bit_packing_distances_and_weighted_densities(self):
        rng = np.random.default_rng(17)
        for steps in (1, 7, 63, 64, 65, 125, 257):
            original = rng.integers(0, 2, (steps, 19), dtype=np.uint8)
            original[:, :2] = 0
            original[:, 2:4] = 1
            observations = np.tile(original, (1, 3))
            dense = reference_distances(observations)
            for budget in (0, 512, 1024**2):
                with self.subTest(steps=steps, budget=budget):
                    packed = JaccardDistances(observations, cache_bytes=budget)
                    actual = np.stack([packed.row(i) for i in range(len(dense))])
                    np.testing.assert_array_equal(actual, dense)
                    for tau in (0, .2, .7, 1):
                        np.testing.assert_array_equal(packed.densities(tau), (dense <= tau).sum(axis=1))
                    np.testing.assert_array_equal(packed.row(3, [5, 3, 0, 5]), dense[3, [5, 3, 0, 5]])
                    cached = packed._matrix.nbytes if packed._matrix is not None else sum(r.nbytes for r in packed._rows.values())
                    self.assertLessEqual(cached, budget)

    def test_never_activated_tokens_are_distinct(self):
        packed = JaccardDistances(np.zeros((65, 4), dtype=np.uint8))
        np.testing.assert_array_equal(packed.row(2), [1, 1, 0, 1])
        np.testing.assert_array_equal(packed.densities(.99), [1, 1, 1, 1])
        np.testing.assert_array_equal(packed.densities(1), [4, 4, 4, 4])
        self.assertEqual(packed.mean_pairwise([0, 1, 2]), 1)

    def test_bounded_cache_eviction_keeps_values(self):
        observations = np.eye(40, dtype=np.uint8)
        packed = JaccardDistances(observations, cache_bytes=160)
        dense = reference_distances(observations)
        for i in [3, 20, 3, 39, 0]:
            np.testing.assert_array_equal(packed.row(i), dense[i])
            self.assertLessEqual(len(packed._rows), 1)

    def test_input_validation(self):
        for observations in (np.zeros((0, 4)), np.zeros((2, 0)), np.ones(5),
                             np.array([[0, 2]]), np.array([[0., .5]]), np.array([[np.nan]])):
            with self.subTest(shape=observations.shape), self.assertRaises(ValueError):
                JaccardDistances(observations)
        with self.assertRaises(ValueError):
            JaccardDistances(np.ones((2, 2), dtype=np.uint8), cache_bytes=-1)


class ClusteringTests(unittest.TestCase):
    def test_exact_membership_against_paper_algorithm(self):
        rng = np.random.default_rng(43)
        datasets = [np.zeros((5, 15), dtype=np.uint8), np.ones((5, 15), dtype=np.uint8),
                    np.eye(15, dtype=np.uint8)]
        for steps in (5, 63, 65, 125):
            unique = (rng.random((steps, 19)) < .7).astype(np.uint8)
            unique[:, :2] = 0
            datasets.append(np.tile(unique, (1, 2)))
        for observations in datasets:
            dense = reference_distances(observations)
            for tau in (0, .2, .5, .7, 1):
                for limit in (0, 2):
                    expected = reference_clusters(dense, tau, limit)
                    for budget in (0, 128, 1024**2):
                        with self.subTest(shape=observations.shape, tau=tau, limit=limit, budget=budget):
                            packed = JaccardDistances(observations, cache_bytes=budget)
                            result = build_clusters_for_layer(packed, tau, limit, limit <= 0)
                            actual = [[c['center'], *c['others']] for c in result['clusters']]
                            self.assertEqual(actual, expected)
                            for c, members in zip(result['clusters'], actual):
                                if len(members) > 1:
                                    truth = np.mean(dense[np.ix_(members, members)][np.triu_indices(len(members), 1)], dtype=np.float64)
                                    self.assertAlmostEqual(c['avg_intra_cluster_distance'], truth, places=6)
                            dense_result = build_clusters_for_layer(dense, tau, limit, limit <= 0)
                            self.assertEqual(result, dense_result)

    def test_pruning_preserves_order_and_overlapping_replicas(self):
        clusters = [[2, 1], [0, 1, 2], [4, 2, 3], [0, 1, 2], [6], [5, 6]]
        self.assertEqual(_prune_contained_clusters(clusters), [[0, 1, 2], [4, 2, 3], [5, 6]])

    def test_128k_identical_histories_keep_every_token(self):
        observations = np.ones((65, 131072), dtype=np.uint8)
        packed = JaccardDistances(observations, cache_bytes=4)
        result = build_clusters_for_layer(packed, .7, 0, True)
        self.assertEqual(packed._matrix.shape, (1, 1))
        self.assertEqual(result['final_cluster_count'], 1)
        self.assertEqual(result['coverage_ratio'], 1)
        cluster = result['clusters'][0]
        self.assertEqual(cluster['size'], 131072)
        self.assertEqual(set([cluster['center'], *cluster['others']]), set(range(131072)))

    def test_matrix_validation(self):
        for matrix in (np.zeros((0, 0)), np.zeros((2, 3))):
            with self.assertRaises(ValueError):
                DenseDistances(matrix)
        for tau in (-1, 2, float('nan')):
            with self.assertRaises(ValueError):
                build_clusters_for_layer(np.eye(2), tau, 0, True)


class ModelingPipelineTests(unittest.TestCase):
    def test_parallel_and_serial_layer_outputs_match(self):
        script = Path(__file__).resolve().parents[1] / 'octopus/offline/cluster_builder.py'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            observations = np.random.default_rng(21).integers(0, 2, (65, 3, 96), dtype=np.uint8)
            source = root / 'activations.npy'
            np.save(source, observations)
            outputs = []
            for workers in (1, 2):
                output = root / str(workers)
                subprocess.run([sys.executable, str(script), '--activations', str(source),
                                '--output_dir', str(output), '--tau', '.7', '--workers', str(workers),
                                '--distance_cache_mb', '.01'], cwd=root, check=True, capture_output=True)
                self.assertFalse((output / 'co_activation_distance.npy').exists())
                outputs.append(json.loads((output / 'clusters.json').read_text()))
            self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
