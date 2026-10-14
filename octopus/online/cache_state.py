from collections import Counter, deque
import heapq

import numpy as np
import torch


class TokenStore:
    def __init__(self, keys, values, token_ids):
        if keys.shape != values.shape or keys.dtype != values.dtype or keys.ndim != 4 or keys.shape[0] != 1:
            raise ValueError('Expected matching batch-one [1, heads, tokens, dim] KV tensors')
        self.heads, self.dim = keys.shape[1], keys.shape[3]
        self.dtype = keys.dtype
        self.ids = list(sorted(token_ids))
        self.ordered = True
        self.slots = {token: slot for slot, token in enumerate(self.ids)}
        self.data = torch.empty((max(16, len(self.ids)), 2, self.heads, self.dim), dtype=self.dtype)
        if self.ids:
            index = torch.tensor(self.ids, dtype=torch.long, device=keys.device)
            self.data[:len(self.ids), 0].copy_(keys.detach()[0].index_select(1, index).transpose(0, 1))
            self.data[:len(self.ids), 1].copy_(values.detach()[0].index_select(1, index).transpose(0, 1))

    def __contains__(self, token):
        return token in self.slots

    def __len__(self):
        return len(self.ids)

    def put(self, token, key, value):
        if key.numel() != self.heads * self.dim or value.numel() != key.numel():
            raise ValueError('Expected a single token with matching KV geometry')
        slot = self.slots.get(token)
        if slot is None:
            slot = len(self.ids)
            if slot == len(self.data):
                grown = torch.empty((max(slot + 16, (slot * 9 + 7) // 8), 2, self.heads, self.dim), dtype=self.dtype)
                grown[:slot].copy_(self.data)
                self.data = grown
            self.ordered = self.ordered and (not self.ids or token > self.ids[-1])
            self.ids.append(token)
            self.slots[token] = slot
        self.data[slot, 0].copy_(key.detach().reshape(self.heads, self.dim))
        self.data[slot, 1].copy_(value.detach().reshape(self.heads, self.dim))

    def get(self, token):
        pair = self.data[self.slots[token]]
        return pair[0], pair[1]

    def remove(self, token):
        slot = self.slots.pop(token)
        last = self.ids.pop()
        if slot < len(self.ids):
            self.ordered = False
            self.data[slot].copy_(self.data[len(self.ids)])
            self.ids[slot] = last
            self.slots[last] = slot

    def packed(self):
        return self.data[:len(self.ids)], self.ids


class HitWindow:
    def __init__(self, clusters, width):
        self.width = width
        self.total = np.zeros(clusters, dtype=np.int64)
        self.recent = np.zeros(clusters, dtype=np.int64)
        self.history = deque()

    def record(self, indices):
        indices = np.unique(np.asarray(indices, dtype=np.intp))
        if len(self.history) == self.width:
            self.recent[self.history.popleft()] -= 1
        self.total[indices] += 1
        self.recent[indices] += 1
        self.history.append(indices)


class LayerState:
    def __init__(self, routing, keys, values, tokens, window, num_disks, disk_base, alpha, dram_only, budget_tokens=None):
        self.cids = tuple(cid for cid in routing if cid != 'global_dram_tokens')
        self.index = {cid: i for i, cid in enumerate(self.cids)}
        self.members, self.dynamic = [], []
        self.hot = set()
        self.refs = Counter()
        self.locations = [()] * len(tokens)
        self.online_locations = {}
        self.primary = [-1] * len(tokens)
        self.next_disk = [1] * len(self.cids)
        self.frequency = np.zeros(len(self.cids), dtype=np.float64)
        self.sizes = np.zeros(len(self.cids), dtype=np.int64)
        self.alpha = alpha
        self.num_disks = num_disks
        payload_bytes = 2 * keys.shape[1] * keys.shape[3] * keys.element_size()
        self.bytes_per_kv = (payload_bytes + 4095) // 4096 * 4096
        self.heap = []
        self.heap_dirty = True
        self.pending_admission = None
        self.pending = {}
        self.last_recorded_step = -1
        self.hits = HitWindow(len(self.cids), len(window))
        pins = set()
        hot_tokens = set()
        locations = {}
        for i, cid in enumerate(self.cids):
            info = routing[cid]
            ids = [int(item['token']) for item in info.get('routes', [])]
            if not ids or any(t < 0 or t >= len(tokens) for t in ids) or int(cid) not in ids:
                raise ValueError(f'Invalid membership for cluster {cid}')
            members = frozenset(tokens[t] for t in ids)
            self.members.append(members)
            self.dynamic.append(set())
            self.sizes[i] = len(members)
            self.frequency[i] = float(info.get('score', 1.0)) / (alpha / len(members) + 1)
            for token in members:
                if self.primary[token] == -1:
                    self.primary[token] = i
            if info.get('center_status') == 'PINNED_CENTER':
                pins.add(int(cid))
            if info.get('type') == 'PINNED_DRAM':
                self.hot.add(i)
                self.refs.update(members)
                hot_tokens.update(members)
            elif info.get('type') != 'STRIPED_SSD':
                raise ValueError(f'Unknown placement for cluster {cid}')
            for loc in (() if dram_only else info.get('physical_locations', [])):
                token, disk = int(loc['token']), int(loc['disk_idx']) - disk_base + 1
                offset, length = int(loc['offset']), int(loc['length'])
                if token not in members or not 1 <= disk <= num_disks or offset < 0 or offset % self.bytes_per_kv or length != self.bytes_per_kv:
                    raise ValueError(f'Invalid physical location for cluster {cid}')
                request = (token, offset, length, disk, int(bool(loc.get('is_append', False))))
                old = locations.setdefault(token, {}).get(disk)
                if old is None or offset < old[1]:
                    locations[token][disk] = request
                self.next_disk[i] = disk % num_disks + 1
        for token, candidates in locations.items():
            self.locations[token] = tuple(candidates[d] for d in sorted(candidates))
        global_dram = {int(t) for t in routing.get('global_dram_tokens', [])}
        if any(t < 0 or t >= len(tokens) for t in global_dram):
            raise ValueError('Global DRAM token is outside the prefix')
        pins.update(global_dram - set(window) - hot_tokens)
        pins.update(t for t in global_dram if self.primary[t] < 0)
        self.pinned = frozenset(pins)
        resident = set(tokens) if dram_only else global_dram | hot_tokens | pins | set(window)
        self.store = TokenStore(keys, values, resident)
        self.capacity = max(len(resident), len(pins) + len(window), budget_tokens or 0)
        center_ids = torch.tensor([int(cid) for cid in self.cids], dtype=torch.long, device=keys.device)
        self.centers = keys.detach().index_select(2, center_ids).cpu() if not dram_only else None
        if not self.cids and not dram_only:
            raise ValueError('SSD inference needs at least one cluster')
        for token in tokens:
            if token not in resident and not self.locations[token]:
                raise ValueError(f'Token {token} has neither DRAM data nor an SSD location')

    def cluster_members(self, index):
        return self.members[index] | self.dynamic[index] if self.dynamic[index] else self.members[index]

    def candidates(self, token):
        return self.locations[token] if token < len(self.locations) else self.online_locations.get(token, ())

    def add_location(self, request):
        token, _, _, disk, _ = request
        candidates = tuple(sorted((*self.candidates(token), request), key=lambda item: item[3]))
        if token < len(self.locations):
            self.locations[token] = candidates
        else:
            self.online_locations[token] = candidates

    def scores(self):
        return self.frequency * (self.alpha / self.sizes + 1)

    def victim_heap(self):
        if self.heap_dirty:
            scores = self.scores()
            self.heap = [(float(scores[i]), self.cids[i], i) for i in self.hot]
            heapq.heapify(self.heap)
            self.heap_dirty = False
        return list(self.heap)
