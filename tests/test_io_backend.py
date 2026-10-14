import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
except ImportError:
    torch = None

DISKS = [p for p in os.environ.get('OCTOPUS_TEST_DISK_DIRS', '').split(os.pathsep) if p]
ENABLED = bool(DISKS) and torch is not None and torch.cuda.is_available()


@unittest.skipUnless(ENABLED, 'Set OCTOPUS_TEST_DISK_DIRS to mounted SSD directories on a CUDA host')
class IOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'octopus/online'))
        from octopus.online import octopus_io
        cls.io = octopus_io
        if not hasattr(octopus_io, 'get_stats'):
            raise unittest.SkipTest('Build the current octopus_io extension first')
        cls.directories = [Path(tempfile.mkdtemp(prefix='octopus-io-test-', dir=p)) for p in DISKS]

    @classmethod
    def tearDownClass(cls):
        cls.io.close()
        for directory in cls.directories:
            shutil.rmtree(directory)

    def setUp(self):
        self.io.close()
        self.paths = [str(d / 'kv.bin') for d in self.directories]
        self.appends = [str(d / 'append.bin') for d in self.directories]
        self.values = {}
        for disk, path in enumerate(self.paths):
            rows = []
            for block in range(32):
                k = (torch.arange(1024).reshape(8, 128) % 31 + disk * 4 + block).to(torch.bfloat16)
                v = -k
                self.values[disk, block] = (k, v)
                rows.append(torch.stack((k, v)))
            with open(path, 'wb') as f:
                f.write(torch.stack(rows).view(torch.uint8).numpy().tobytes())
                f.flush()
                os.fsync(f.fileno())
            Path(self.appends[disk]).unlink(missing_ok=True)
        self.io.configure_storage(self.paths, self.appends)

    def tearDown(self):
        self.io.close()

    def plan(self, blocks=(2, 0, 1)):
        return [[(disk * 1000 + block, block * 4096, 4096, disk + 1, 0) for block in blocks]
                for disk in range(len(self.paths))]

    def check(self, result):
        k, v, ids = result
        for i, token in enumerate(ids):
            expected_k, expected_v = self.values[token // 1000, token % 1000]
            self.assertTrue(torch.equal(k[0, :, i].cpu(), expected_k))
            self.assertTrue(torch.equal(v[0, :, i].cpu(), expected_v))

    def test_sync_async_data_layout_and_merge_counts(self):
        for mode in ('sync', 'async'):
            before = self.io.get_stats()
            plan = self.plan()
            result = (self.io.sync_read_clusters(0, plan) if mode == 'sync'
                      else self.io.wait_and_get(self.io.async_read_clusters(0, plan)))
            self.check(result)
            stats = self.io.get_stats()
            self.assertEqual(stats['submitted_reads'] - before['submitted_reads'], len(self.paths))
            self.assertEqual(stats['read_bytes'] - before['read_bytes'], len(self.paths) * 3 * 4096)
            self.assertEqual(result[2], [d * 1000 + b for d in range(len(self.paths)) for b in range(3)])

    def test_model_geometries_padding_and_writeback(self):
        from octopus.layout import KVLayout
        for heads, dim in ((1, 64), (1, 256), (8, 256), (12, 256)):
            with self.subTest(heads=heads, dim=dim):
                self.io.close()
                layout = KVLayout(heads, dim)
                key = torch.arange(heads * dim).reshape(heads, dim).to(torch.bfloat16)
                value = -key
                for path in self.paths:
                    Path(path).write_bytes(layout.pack(key, value) * 3)
                self.io.configure_storage(self.paths, self.appends, heads, dim)
                plan = [[(disk * 10 + t, t * layout.block_bytes, layout.block_bytes, disk + 1, 0)
                         for t in (2, 0, 1)] for disk in range(len(self.paths))]
                before = self.io.get_stats()['read_bytes']
                handle = self.io.async_read_clusters(0, plan)
                for k, v, ids in (self.io.sync_read_clusters(0, plan), self.io.wait_and_get(handle)):
                    self.assertEqual(tuple(k.shape), (1, heads, len(ids), dim))
                    for i in range(len(ids)):
                        self.assertTrue(torch.equal(k[0, :, i].cpu(), key))
                        self.assertTrue(torch.equal(v[0, :, i].cpu(), value))
                self.assertEqual(self.io.get_stats()['read_bytes'] - before,
                                 2 * 3 * len(self.paths) * layout.block_bytes)
                self.io.async_append_kv([(1, 0, key, value)])
                k, v, _ = self.io.sync_read_clusters(0, [[(100, 0, layout.block_bytes, 1, 1)]])
                self.assertTrue(torch.equal(k[0, :, 0].cpu(), key))
                self.assertTrue(torch.equal(v[0, :, 0].cpu(), value))
                raw = Path(self.appends[0]).read_bytes()[:layout.block_bytes]
                self.assertEqual(raw, layout.pack(key, value))

    def test_resource_reuse_and_retained_outputs(self):
        handles = [self.io.async_read_clusters(0, self.plan()) for _ in range(8)]
        retained = [self.io.wait_and_get(h) for h in reversed(handles)]
        before = self.io.get_stats()
        for _ in range(12):
            self.io.wait_and_get(self.io.async_read_clusters(0, self.plan((4, 7))))
        after = self.io.get_stats()
        for key in ('rings_created', 'pinned_allocations', 'file_opens', 'worker_threads_created'):
            self.assertEqual(before[key], after[key], key)
        for result in retained:
            self.check(result)

    def test_more_requests_than_queue_depth(self):
        # Repeated physical offsets cannot merge; every issued 4 KiB read remains observable.
        plan = [[(disk * 1000, 0, 4096, disk + 1, 0)] * 4097 for disk in range(len(self.paths))]
        before = self.io.get_stats()
        k, v, ids = self.io.sync_read_clusters(0, plan)
        self.assertEqual(self.io.get_stats()['submitted_reads'] - before['submitted_reads'], 4097 * len(self.paths))
        for disk in range(len(self.paths)):
            expected = self.values[disk, 0]
            self.assertTrue(torch.equal(k[0, :, disk*4097:(disk+1)*4097].cpu(), expected[0][:, None].expand(8, 4097, 128)))
            self.assertTrue(torch.equal(v[0, :, disk*4097:(disk+1)*4097].cpu(), expected[1][:, None].expand(8, 4097, 128)))
        warmed = self.io.get_stats()
        self.io.sync_read_clusters(0, plan)
        self.assertEqual(self.io.get_stats()['pinned_allocations'], warmed['pinned_allocations'])

    def test_concurrent_sync_and_async_callers(self):
        def read(index):
            plan = self.plan((index % 16,))
            if index % 2:
                return self.io.wait_and_get(self.io.async_read_clusters(0, plan))
            return self.io.sync_read_clusters(0, plan)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(read, range(16)))
        for result in results:
            self.check(result)

    def test_invalid_plans_eof_and_error_recovery(self):
        for request in ((0, 1, 4096, 1, 0), (0, 0, 2048, 1, 0), (0, 0, 4096, 0, 0),
                        (0, 0, 4096, 1, 2), (0, 32*4096, 4096, 1, 0)):
            with self.subTest(request=request), self.assertRaises(RuntimeError):
                self.io.wait_and_get(self.io.async_read_clusters(0, [[request]]))
        self.check(self.io.sync_read_clusters(0, self.plan()))
        os.truncate(self.paths[0], 4096)
        with self.assertRaisesRegex(RuntimeError, 'short SSD read'):
            self.io.sync_read_clusters(0, self.plan())
        self.check(self.io.sync_read_clusters(0, self.plan((0,))))

    def test_append_visibility_and_submission_order(self):
        first = torch.full((8, 128), 3, dtype=torch.bfloat16)
        last = torch.full((8, 128), 7, dtype=torch.bfloat16)
        self.io.async_append_kv([(1, 0, first, -first)])
        self.io.async_append_kv([(1, 0, last, -last), (1, 4096, first, -first)])
        handles = [self.io.async_read_clusters(0, [[(901, 4096, 4096, 1, 1), (900, 0, 4096, 1, 1)]]) for _ in range(6)]
        for h in handles:
            k, v, ids = self.io.wait_and_get(h)
            self.assertEqual(ids, [900, 901])
            self.assertTrue(torch.equal(k[0, :, 0].cpu(), last))
            self.assertTrue(torch.equal(v[0, :, 1].cpu(), -first))

    def test_append_failure_is_reported_and_reconfiguration_recovers(self):
        bad = [str(self.directories[0] / 'missing' / 'append.bin'), *self.appends[1:]]
        self.io.configure_storage(self.paths, bad)
        k = torch.ones((8, 128), dtype=torch.bfloat16)
        self.io.async_append_kv([(1, 0, k, k)])
        with self.assertRaises(RuntimeError):
            self.io.wait_and_get(self.io.async_read_clusters(0, [[(0, 0, 4096, 1, 1)]]))
        with self.assertRaises(RuntimeError):
            self.io.synchronize()
        self.io.configure_storage(self.paths, self.appends)
        self.check(self.io.sync_read_clusters(0, self.plan()))

    def test_cuda_consumer_stream_owns_output_lifetime(self):
        stream = torch.cuda.Stream()
        results = []
        expected = sum(self.values[disk, 0][0].float().sum().item() for disk in range(len(self.paths)))
        for _ in range(16):
            handle = self.io.async_read_clusters(0, self.plan((0,)))
            with torch.cuda.stream(stream):
                k, v, _ = self.io.wait_and_get(handle)
                torch.cuda._sleep(2_000_000)
                results.append(k.float().sum())
            del k, v
            self.io.sync_read_clusters(0, self.plan((5,)))
        stream.synchronize()
        self.assertEqual([x.item() for x in results], [expected] * len(results))

    def test_empty_handles_close_and_process_exit(self):
        self.assertEqual(self.io.sync_read_clusters(0, []), (None, None, []))
        self.assertEqual(self.io.async_read_clusters(0, []), -1)
        self.assertEqual(self.io.wait_and_get(-1), (None, None, []))
        with self.assertRaises(RuntimeError):
            self.io.wait_and_get(2147483646)
        h = self.io.async_read_clusters(0, self.plan())
        self.io.synchronize()
        with self.assertRaises(RuntimeError):
            self.io.wait_and_get(h)
        module_dir = str(Path(self.io.__file__).parent)
        code = ('import sys, torch; sys.path.insert(0, sys.argv[1]); import octopus_io as io; '
                'import json; io.configure_storage(json.loads(sys.argv[2]),json.loads(sys.argv[3])); '
                'io.async_read_clusters(0,[[(0,0,4096,1,0)]])')
        subprocess.run([sys.executable, '-c', code, module_dir, json.dumps(self.paths), json.dumps(self.appends)],
                       check=True, capture_output=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
