import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from octopus.online import cache as cache_module
from octopus.online import cache_manager as manager_module
from octopus.online.cache_state import HitWindow, TokenStore


class FakeIO:
    def __init__(self):
        self.data, self.pending = {}, {}
        self.writes = []
        self.next_handle = 1

    def async_append_kv(self, requests):
        for disk, offset, k, v in requests:
            self.data[1, disk, offset] = (k.reshape(8, 128).clone(), v.reshape(8, 128).clone())
            self.writes.append((disk, offset))
        return True

    def sync_read_clusters(self, layer, plan):
        rows = [r for bucket in plan for r in sorted(bucket, key=lambda r: (r[1], r[0]))]
        if not rows:
            return None, None, []
        pairs = [self.data[r[4], r[3], r[1]] for r in rows]
        k = torch.stack([p[0] for p in pairs]).transpose(0, 1).unsqueeze(0)
        v = torch.stack([p[1] for p in pairs]).transpose(0, 1).unsqueeze(0)
        return k, v, [r[0] for r in rows]

    def async_read_clusters(self, layer, plan):
        handle = self.next_handle
        self.next_handle += 1
        self.pending[handle] = self.sync_read_clusters(layer, plan)
        return handle

    def wait_and_get(self, handle):
        return self.pending.pop(handle)


def fixture(root, fake, layers=1, hot=(0,), random_keys=False):
    n, disks = 12, 3
    groups = {0: [0, 1, 2, 3], 4: [4, 2, 3, 5], 6: [6, 7, 8, 9], 10: [10, 11]}
    routing, tensors = {}, []
    offsets = [0] * disks
    for layer in range(layers):
        keys = (torch.arange(n).view(1, 1, n, 1) + layer * 100).expand(1, 8, n, 128).to(torch.bfloat16).clone()
        if random_keys:
            keys = torch.randn(keys.shape, generator=torch.Generator().manual_seed(12 + layer)).to(torch.bfloat16)
        values = -keys - 1
        tensors.append(SimpleNamespace(keys=keys, values=values))
        global_dram = {0, 4, 6, 10, 11}
        layer_info = {}
        for center, members in groups.items():
            resident = center in hot
            if resident:
                global_dram.update(members)
            info = {'type': 'PINNED_DRAM' if resident else 'STRIPED_SSD',
                    'center_status': 'PINNED_CENTER' if center != 10 else 'NONE',
                    'routes': [{'token': t} for t in members],
                    'score': (100 if center == 4 else 1) * (10 / len(members) + 1)}
            if not resident:
                physical = [t for t in members if t != center or center == 10]
                info['physical_ssd_members'] = physical
                info['physical_locations'] = []
                for token in physical:
                    disk = (token + center) % disks
                    offset = offsets[disk]
                    offsets[disk] += 4096
                    info['physical_locations'].append({'token': token, 'disk_idx': disk, 'offset': offset, 'length': 4096})
                    fake.data[0, disk + 1, offset] = (keys[0, :, token].clone(), values[0, :, token].clone())
            layer_info[str(center)] = info
        layer_info['global_dram_tokens'] = sorted(global_dram)
        routing[str(layer)] = layer_info
    (root / 'routing_index.json').write_text(json.dumps(routing))
    (root / 'storage.json').write_text(json.dumps({'disk_base': 0}))
    return SimpleNamespace(layers=tensors, get_seq_length=lambda: n)


class StateTests(unittest.TestCase):
    def test_packed_store_growth_swap_removal_and_ids(self):
        keys = torch.arange(12).view(1, 1, 12, 1).expand(1, 8, 12, 128).to(torch.bfloat16)
        store = TokenStore(keys, -keys, [0, 2, 4])
        for token in range(12, 40):
            k = torch.full((8, 128), token, dtype=torch.bfloat16)
            store.put(token, k, -k)
        for token in [2, 13, 38]:
            store.remove(token)
        data, ids = store.packed()
        self.assertEqual(len(ids), data.shape[0])
        for slot, token in enumerate(ids):
            self.assertEqual(store.slots[token], slot)
            self.assertTrue(torch.all(data[slot, 0] == token))
            self.assertTrue(torch.all(data[slot, 1] == -token))
        self.assertFalse(store.ordered)

    def test_rolling_counts_match_explicit_windows(self):
        window = HitWindow(7, 3)
        history = []
        for hits in ([1, 2], [2, 4], [], [2, 2, 5], [1, 5, 6], [3]):
            history.append(set(hits))
            window.record(hits)
            np.testing.assert_array_equal(window.total, [sum(i in h for h in history) for i in range(7)])
            np.testing.assert_array_equal(window.recent, [sum(i in h for h in history[-3:]) for i in range(7)])


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.io = FakeIO()
        self.patches = [patch.object(manager_module, 'octopus_io', self.io), patch.object(cache_module, 'octopus_io', self.io)]
        for p in self.patches:
            p.start()
        self.full = fixture(self.root, self.io)
        cache_module.OCTOPUS_GLOBAL_Q.clear()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def manager(self, **kwargs):
        return manager_module.CacheManager(self.root, self.full, len(self.full.layers), 2, 3, **kwargs)

    def add(self, manager, hits, step):
        token = manager.past_len + step
        manager.step_start(token)
        for i in range(manager.num_layers):
            k = torch.full((1, 8, 1, 128), token + i * 100, dtype=torch.bfloat16)
            manager.add_online_kv_for_layer(i, k, -k - 1)
            manager.update_cache(i, hits)

    def test_gqa_scores_preserve_original_topk(self):
        self.full = fixture(self.root, self.io, random_keys=True)
        manager = self.manager(ban_cache_update=True)
        layer = manager.layers[0]
        for heads in (4, 8, 16, 32):
            for seed in range(20):
                q = torch.randn((1, heads, 1, 128), generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)
                k, query = layer.centers, q
                if heads > 8:
                    k = k.repeat_interleave(heads // 8, dim=1)
                elif heads < 8:
                    query = q.repeat_interleave(8 // heads, dim=1)
                scores = torch.matmul(query, k.transpose(-1, -2)).mean(dim=(0, 1, 2))
                expected = [layer.cids[i] for i in torch.topk(scores, 3).indices.tolist()]
                self.assertEqual(manager.predict_hits_for_layer(0, q, 3), expected)

    def test_load_balanced_routes_and_missing_backing(self):
        manager = self.manager(ban_cache_update=True)
        layer = manager.layers[0]
        hits = ['0', '4', '6', '10']
        plan = manager.generate_io_plan(0, hits)
        expected = set().union(*(layer.members[layer.index[c]] for c in hits)) - layer.store.slots.keys()
        requests = [r for bucket in plan for r in bucket]
        self.assertEqual({r[0] for r in requests}, expected)
        self.assertEqual(len(requests), len(expected))
        for r in requests:
            self.assertIn(r, layer.candidates(r[0]))
        lost = next(iter(expected))
        layer.locations[lost] = ()
        with self.assertRaisesRegex(RuntimeError, 'no SSD location'):
            manager.generate_io_plan(0, hits)

    def test_observations_cover_entire_residency_and_fifo_is_bounded(self):
        self.full = fixture(self.root, self.io, hot=())
        manager = self.manager(tau=0, ban_cache_update=True)
        for step in range(8):
            self.add(manager, ['0'] if step % 2 == 0 else ['4'], step)
            layer = manager.layers[0]
            self.assertLessEqual(len(layer.pending), 2)
            self.assertLessEqual(len(layer.store), layer.capacity + len(layer.pending))
        layer = manager.layers[0]
        self.assertIn(12, layer.dynamic[layer.index['0']])
        self.assertNotIn(12, layer.store)
        self.assertTrue(layer.candidates(12))
        np.testing.assert_array_equal(layer.hits.recent, [1, 1, 0, 0])

    def test_hot_replacement_writes_back_and_preserves_shared_tokens(self):
        manager = self.manager()
        self.add(manager, ['4'], 0)
        layer = manager.layers[0]
        plan = manager.generate_io_plan(0, ['4'])
        fetched = self.io.sync_read_clusters(0, plan)
        manager.complete_fetch(0, *fetched)
        self.assertIn(layer.index['4'], layer.hot)
        self.assertNotIn(layer.index['0'], layer.hot)
        for token in (0, 2, 3, 4, 5):
            self.assertIn(token, layer.store)
        self.assertNotIn(1, layer.store)
        location = layer.candidates(1)[0]
        self.assertEqual(location[4], 1)
        self.assertTrue(torch.all(self.io.data[1, location[3], location[1]][0] == 1))
        self.assertLessEqual(len(layer.store), layer.capacity)

    def test_reference_counts_protect_overlapping_hot_clusters(self):
        self.full = fixture(self.root, self.io, hot=(0, 4))
        manager = self.manager()
        layer = manager.layers[0]
        self.assertEqual(layer.refs[2], 2)
        manager._demote(layer, layer.index['0'])
        self.assertEqual(layer.refs[2], 1)
        self.assertIn(2, layer.store)
        self.assertIn(3, layer.store)
        self.assertIn(0, layer.store)

    def test_pending_counter_snapshots_survive_byte_boundaries(self):
        self.full = fixture(self.root, self.io, hot=())
        manager = self.manager(tau=0, ban_cache_update=True)
        for step in range(270):
            self.add(manager, ['0'] if step % 2 == 0 else ['4'], step)
        layer = manager.layers[0]
        self.assertLessEqual(len(layer.pending), 2)
        self.assertLessEqual(len(layer.store), layer.capacity + 2)
        self.assertTrue(all(snapshot.dtype == np.uint16 for _, snapshot in layer.pending.values()))
        np.testing.assert_array_equal(layer.hits.total, [135, 135, 0, 0])

    def test_explicit_disk_base_without_a_zero_disk_entry(self):
        routing = {'0': {'0': {'type': 'STRIPED_SSD', 'center_status': 'NONE',
            'routes': [{'token': 0}, {'token': 1}], 'score': 1,
            'physical_ssd_members': [0, 1],
            'physical_locations': [{'token': t, 'disk_idx': 1, 'offset': t*4096, 'length': 4096} for t in (0, 1)]},
            'global_dram_tokens': list(range(2, 12))}}
        (self.root/'routing_index.json').write_text(json.dumps(routing))
        manager = self.manager()
        requests = [r for bucket in manager.generate_io_plan(0, ['0']) for r in bucket]
        self.assertEqual([r[3] for r in requests], [2, 2])

    def test_frequency_updates_follow_hits_and_cached_misses(self):
        manager = self.manager()
        layer = manager.layers[0]
        before = layer.frequency.copy()
        self.add(manager, ['4'], 0)
        self.assertEqual(layer.frequency[layer.index['0']], max(0, before[layer.index['0']] - 1))
        self.assertEqual(layer.frequency[layer.index['4']], before[layer.index['4']] + 1)
        self.assertEqual(layer.frequency[layer.index['6']], before[layer.index['6']])

    def test_offline_budget_and_window_metadata_are_honored(self):
        (self.root/'storage.json').write_text(json.dumps({'disk_base': 0, 'dram_budget_tokens': 10, 'local_window': 2}))
        manager = self.manager()
        self.assertEqual(manager.layers[0].capacity, 10)
        with self.assertRaisesRegex(ValueError, 'local-window'):
            manager_module.CacheManager(self.root, self.full, 1, 3, 3)

    def test_full_prefix_is_not_retained_and_compatibility_snapshot_is_stable(self):
        manager = self.manager()
        self.assertFalse(hasattr(manager, 'full_cache_ref'))
        before = manager.get_dram_kv_for_layer(0)
        snapshot = before[0].clone()
        self.add(manager, ['4'], 0)
        self.assertTrue(torch.equal(before[0], snapshot))
        self.assertLess(len(manager.layers[0].store), self.full.get_seq_length())

    def test_streaming_cache_retrieves_every_token_and_has_logical_length(self):
        self.full = fixture(self.root, self.io, layers=2)
        for prefetch in (False, True):
            with self.subTest(prefetch=prefetch):
                manager = self.manager(tau=0)
                cache = cache_module.OctopusCache(manager, torch.device('cpu'), ban_prefetch=not prefetch)
                retained = []
                for step in range(10):
                    manager.step_start(12 + step)
                    cache_module.OCTOPUS_GLOBAL_Q.clear()
                    for layer in range(2):
                        cache_module.OCTOPUS_GLOBAL_Q.append(torch.ones((1, 8, 1, 128), dtype=torch.bfloat16))
                        k = torch.full((1, 8, 1, 128), 12 + step + layer*100, dtype=torch.bfloat16)
                        actual_k, actual_v = cache.update(k, -k-1, layer)
                        expected = (torch.arange(13+step).view(1, 1, -1, 1)+layer*100).expand(1, 8, -1, 128).to(torch.bfloat16)
                        self.assertTrue(torch.equal(actual_k, expected), (step, layer, actual_k[0,0,:,0]))
                        self.assertTrue(torch.equal(actual_v, -expected-1))
                        self.assertEqual(cache.get_seq_length(layer), 13+step)
                        retained.append((actual_k, expected.clone()))
                for actual, expected in retained:
                    self.assertTrue(torch.equal(actual, expected))

    def test_dram_only_reuses_multidisk_artifacts_without_io(self):
        manager = manager_module.CacheManager(self.root, self.full, 1, 2, 1, dram_only=True)
        cache = cache_module.OctopusCache(manager, torch.device('cpu'))
        for step in range(5):
            manager.step_start(12+step)
            k = torch.full((1,8,1,128),12+step,dtype=torch.bfloat16)
            output, _ = cache.update(k, -k-1, 0)
            self.assertEqual(output.shape[2],13+step)
        self.assertEqual(self.io.writes, [])
        self.assertEqual(self.io.pending, {})

    def test_invalid_lifecycle_and_duplicate_assembly_fail(self):
        manager = self.manager()
        with self.assertRaises(ValueError):
            manager.step_start(15)
        manager.step_start(12)
        manager.update_cache(0, ['4'])
        with self.assertRaises(RuntimeError):
            manager.update_cache(0, ['4'])
        with self.assertRaisesRegex(RuntimeError, 'incomplete KV'):
            manager.step_start(13)
        cache = cache_module.OctopusCache(manager, torch.device('cpu'))
        k = torch.zeros((1,8,1,128),dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            cache.assemble_and_sort(k,k,[1],k,k,[1])
        with self.assertRaises(ValueError):
            manager.predict_hits_for_layer(0,torch.zeros((1,3,1,128),dtype=torch.bfloat16))


if __name__ == '__main__':
    unittest.main()
