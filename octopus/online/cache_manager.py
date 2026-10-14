import heapq
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

try:
    if __package__:
        from . import octopus_io
    else:
        import octopus_io
except ImportError:
    octopus_io = None
if __package__:
    from .cache_state import LayerState
    from .debug_utils import debug_log, is_debug_enabled
else:
    from cache_state import LayerState
    from debug_utils import debug_log, is_debug_enabled


class CacheManager:
    def __init__(self, offline_dir, full_cache, num_layers, local_window, num_disks,
                 tau=.05, alpha=10., reward=1., ban_cache_update=False, dram_only=False,
                 attention_windows=None):
        self.num_layers, self.local_window, self.num_disks = num_layers, local_window, num_disks
        self.past_len = full_cache.get_seq_length()
        self.attention_windows = tuple(attention_windows or [None] * num_layers)
        if len(self.attention_windows) != num_layers:
            raise ValueError('Attention windows must match the layer count')
        if not 1 <= local_window <= self.past_len or num_disks < 1 or num_layers != len(full_cache.layers):
            raise ValueError('Invalid layer, window or disk configuration')
        if not 0 <= tau <= 1 or not np.isfinite(alpha) or alpha < 0 or not np.isfinite(reward) or reward < 0:
            raise ValueError('Invalid cache scoring parameters')
        self.tau, self.R = tau, reward
        self.ban_cache_update, self.dram_only = ban_cache_update, dram_only
        self.promote_min_margin = 1.0
        self.current_token_id = self.past_len - 1
        self.step = -1
        self.window = deque(range(self.past_len - local_window, self.past_len))
        self.window_tokens = set(self.window)
        self.append_offsets = [0] * num_disks
        root = Path(offline_dir)
        routing = json.loads((root / 'routing_index.json').read_text())
        budget_tokens = None
        if (root / 'storage.json').is_file():
            storage = json.loads((root / 'storage.json').read_text())
            disk_base = int(storage['disk_base'])
            budget_tokens = storage.get('dram_budget_tokens')
            if budget_tokens is not None and (not isinstance(budget_tokens, int) or budget_tokens < 0):
                raise ValueError('Invalid DRAM budget in storage metadata')
            if 'local_window' in storage and storage['local_window'] != local_window:
                raise ValueError('local-window must match offline storage metadata')
        else:
            disks = [loc['disk_idx'] for layer in routing.values() for cid, info in layer.items()
                     if cid != 'global_dram_tokens' for loc in info.get('physical_locations', [])]
            disk_base = 0 if 0 in disks else 1
        if disk_base not in (0, 1):
            raise ValueError('Disk numbering must start at zero or one')
        tokens = tuple(range(self.past_len))
        self.layers = [LayerState(routing[str(i)], layer.keys, layer.values, tokens, self.window,
                                  num_disks, disk_base, alpha, dram_only, budget_tokens)
                       for i, layer in enumerate(full_cache.layers)]

    def _protected(self, layer, token):
        return (self.dram_only or token in layer.pinned or token in self.window_tokens
                or token in layer.pending or layer.refs[token] > 0)

    def _write_locations(self, layer, assignments):
        requests, updates = [], []
        for token, disks in assignments:
            existing = {r[3] for r in layer.candidates(token)}
            key, value = layer.store.get(token)
            for disk in disks:
                if disk in existing:
                    continue
                existing.add(disk)
                offset = self.append_offsets[disk - 1]
                self.append_offsets[disk - 1] += layer.bytes_per_kv
                requests.append((disk, offset, key, value))
                updates.append((token, offset, layer.bytes_per_kv, disk, 1))
        if requests:
            if octopus_io is None:
                raise RuntimeError('octopus_io is required to evict KV data to SSD')
            octopus_io.async_append_kv(requests)
            for request in updates:
                layer.add_location(request)

    def _retire(self, layer, tokens, preferred=None):
        tokens = [t for t in tokens if t in layer.store and not self._protected(layer, t)]
        assignments = []
        for token in tokens:
            if not layer.candidates(token):
                cid = preferred if preferred is not None else layer.primary[token]
                if cid < 0:
                    raise RuntimeError(f'Token {token} has no cluster for writeback')
                disk = layer.next_disk[cid]
                layer.next_disk[cid] = disk % self.num_disks + 1
                assignments.append((token, (disk,)))
        self._write_locations(layer, assignments)
        for token in tokens:
            layer.store.remove(token)

    def _demote(self, layer, cid):
        layer.hot.remove(cid)
        members = layer.cluster_members(cid)
        for token in members:
            layer.refs[token] -= 1
            if not layer.refs[token]:
                del layer.refs[token]
        layer.heap_dirty = True
        self._retire(layer, members, preferred=cid)

    def _balance_capacity(self, layer, reserve=0):
        while len(layer.store) + reserve > layer.capacity + len(layer.pending):
            heap = layer.victim_heap()
            if not heap:
                raise RuntimeError('Protected KV data exceeds the DRAM capacity')
            self._demote(layer, heapq.heappop(heap)[2])

    def _assign_dynamic(self, layer, token, targets):
        layer.pending.pop(token, None)
        disks = []
        for cid in targets:
            if token not in layer.members[cid] and token not in layer.dynamic[cid]:
                layer.dynamic[cid].add(token)
                layer.sizes[cid] += 1
                if cid in layer.hot:
                    layer.refs[token] += 1
            if cid not in layer.hot:
                disks.append(layer.next_disk[cid])
                layer.next_disk[cid] = layer.next_disk[cid] % self.num_disks + 1
        self._write_locations(layer, [(token, disks)] if disks else [])
        layer.heap_dirty = True
        self._retire(layer, [token], preferred=int(targets[0]))

    def _process_pending(self, layer):
        for token, (born, baseline) in list(layer.pending.items()):
            counts = layer.hits.total - baseline
            distance = 1.0 - counts / (self.step - born)
            targets = np.flatnonzero(distance <= self.tau).tolist()
            if targets:
                self._assign_dynamic(layer, token, targets)
        while len(layer.pending) > self.local_window:
            token = next(iter(layer.pending))
            _, baseline = layer.pending[token]
            nearest = int(np.argmax(layer.hits.total - baseline))
            self._assign_dynamic(layer, token, [nearest])

    def step_start(self, new_token_idx):
        if new_token_idx != self.current_token_id + 1:
            raise ValueError('Decode token positions must advance by one')
        if self.step >= 0 and any(self.current_token_id not in layer.store for layer in self.layers):
            raise RuntimeError('Previous decode step has incomplete KV data')
        if self.step >= 0 and not self.dram_only:
            if any(layer.last_recorded_step != self.step for layer in self.layers):
                raise RuntimeError('Previous decode step has incomplete layer observations')
        self.current_token_id = new_token_idx
        self.step += 1
        expired = self.window.popleft()
        self.window_tokens.remove(expired)
        self.window.append(new_token_idx)
        self.window_tokens.add(new_token_idx)
        if self.dram_only:
            return
        for layer in self.layers:
            if expired < self.past_len:
                self._retire(layer, [expired])
            else:
                counts = layer.hits.recent
                targets = np.flatnonzero(1.0 - counts / self.local_window <= self.tau).tolist()
                if targets:
                    self._assign_dynamic(layer, expired, targets)
                else:
                    layer.pending[expired] = (self.step - self.local_window,
                                              (layer.hits.total - counts).astype(np.min_scalar_type(self.step - self.local_window)))
            self._process_pending(layer)
            self._balance_capacity(layer, reserve=int(new_token_idx not in layer.store))

    def add_online_kv_for_layer(self, layer_idx, k_state, v_state):
        layer = self.layers[layer_idx]
        expected = (1, layer.store.heads, 1, layer.store.dim)
        if tuple(k_state.shape) != expected or tuple(v_state.shape) != expected:
            raise ValueError('Online cache accepts one token with matching KV geometry per layer')
        if k_state.dtype != layer.store.dtype or v_state.dtype != layer.store.dtype:
            raise ValueError('Online KV dtype must match the prefix cache')
        if self.step < 0:
            raise RuntimeError('Call step_start before adding online KV data')
        if k_state.device.type == 'cpu':
            layer.store.put(self.current_token_id, k_state, v_state)
        else:
            pair = torch.stack((k_state[0, :, 0], v_state[0, :, 0])).detach().cpu()
            layer.store.put(self.current_token_id, pair[0], pair[1])

    def predict_hits_for_layer(self, layer_idx, query, top_k=50):
        layer = self.layers[layer_idx]
        if self.dram_only or not layer.cids or top_k <= 0:
            return []
        centers = layer.centers
        query = query.detach().to(device=centers.device, dtype=centers.dtype)
        batch, q_heads, q_len, dim = query.shape
        k_heads = centers.shape[1]
        if q_heads >= k_heads and q_heads % k_heads == 0:
            grouped = query.reshape(batch, k_heads, (q_heads // k_heads) * q_len, dim)
            scores = torch.matmul(grouped, centers.transpose(-1, -2)).reshape(batch, q_heads, q_len, -1)
        elif k_heads % q_heads == 0:
            scores = torch.matmul(query.repeat_interleave(k_heads // q_heads, dim=1), centers.transpose(-1, -2))
        else:
            raise ValueError('Query and KV heads must have an integral GQA ratio')
        mean_scores = scores.mean(dim=(0, 1, 2))
        indices = torch.topk(mean_scores, min(top_k, len(layer.cids))).indices.tolist()
        return [layer.cids[i] for i in indices]

    def update_cache(self, layer_idx, hit_cids):
        layer = self.layers[layer_idx]
        if layer.last_recorded_step == self.step:
            raise RuntimeError('Cluster observations were already recorded for this layer and step')
        indices = np.asarray([layer.index[cid] for cid in dict.fromkeys(hit_cids)], dtype=np.intp)
        layer.hits.record(indices)
        layer.last_recorded_step = self.step
        layer.pending_admission = None
        if self.ban_cache_update or self.dram_only or self.attention_windows[layer_idx]:
            return
        inactive = list(layer.hot - set(indices))
        layer.frequency[inactive] = np.maximum(layer.frequency[inactive] - 1, 0)
        layer.frequency[indices] += self.R
        layer.heap_dirty = True
        scores = layer.scores()
        layer.pending_admission = sorted((int(i) for i in indices if i not in layer.hot),
                                          key=lambda i: (-scores[i], layer.cids[i]))

    def generate_io_plan(self, layer_idx, hit_cids):
        if self.dram_only or not hit_cids:
            return []
        layer = self.layers[layer_idx]
        targets = set()
        for cid in hit_cids:
            targets.update(layer.cluster_members(layer.index[cid]))
        targets.difference_update(layer.store.slots)
        window = self.attention_windows[layer_idx]
        if window:
            targets = {t for t in targets if t > self.current_token_id - window}
        degrees = [[] for _ in range(self.num_disks + 1)]
        for token in sorted(targets):
            candidates = layer.candidates(token)
            if not candidates:
                raise RuntimeError(f'Layer {layer_idx}, token {token} has no SSD location')
            degrees[len(candidates)].append(candidates)
        loads = [0] * (self.num_disks + 1)
        buckets = [[] for _ in range(self.num_disks * 2)]
        for group in degrees[1:]:
            for candidates in group:
                best = candidates[0]
                for candidate in candidates[1:]:
                    if loads[candidate[3]] < loads[best[3]]:
                        best = candidate
                loads[best[3]] += 1
                buckets[(best[3] - 1) * 2 + best[4]].append(best)
        if is_debug_enabled():
            debug_log('cache_manager', f'layer={layer_idx} requests={len(targets)} disk_loads={loads[1:]}')
        return [bucket for bucket in buckets if bucket]

    def _admission_victims(self, layer, incoming):
        members = layer.cluster_members(incoming)
        missing = members - layer.store.slots.keys()
        available = layer.capacity + len(layer.pending) - len(layer.store)
        scores = layer.scores()
        heap = layer.victim_heap()
        removed_refs, victims = {}, []
        while available < len(missing):
            if not heap:
                return None
            score, _, victim = heapq.heappop(heap)
            if scores[incoming] <= score * self.promote_min_margin:
                return None
            victims.append(victim)
            for token in layer.cluster_members(victim):
                removed_refs[token] = removed_refs.get(token, 0) + 1
                if (removed_refs[token] == layer.refs[token] and token not in members
                        and token not in layer.pinned and token not in self.window_tokens
                        and token not in layer.pending):
                    available += 1
        return victims

    def complete_fetch(self, layer_idx, ssd_k, ssd_v, ssd_ids):
        layer = self.layers[layer_idx]
        pending, layer.pending_admission = layer.pending_admission, None
        for incoming in pending or ():
            victims = self._admission_victims(layer, incoming)
            if victims is None:
                continue
            members = layer.cluster_members(incoming)
            needed = sorted(members - layer.store.slots.keys())
            incoming_data = None
            if needed:
                positions = {token: i for i, token in enumerate(ssd_ids)}
                if ssd_k is None or any(token not in positions for token in needed):
                    raise RuntimeError('Promotion is missing fetched KV data')
                index = torch.tensor([positions[t] for t in needed], device=ssd_k.device)
                incoming_data = torch.stack((ssd_k[0].index_select(1, index).transpose(0, 1),
                                             ssd_v[0].index_select(1, index).transpose(0, 1)), dim=1).cpu()
            layer.hot.add(incoming)
            layer.refs.update(members)
            layer.heap_dirty = True
            for victim in victims:
                self._demote(layer, victim)
            for i, token in enumerate(needed):
                layer.store.put(token, incoming_data[i, 0], incoming_data[i, 1])
            if len(layer.store) > layer.capacity + len(layer.pending):
                raise RuntimeError('Cache admission exceeded the DRAM capacity')
            break

    def get_dram_packed_for_layer(self, layer_idx):
        return self.layers[layer_idx].store.packed()

    def get_dram_kv_for_layer(self, layer_idx):
        packed, ids = self.get_dram_packed_for_layer(layer_idx)
        if not ids:
            return None, None, []
        packed = packed.clone()
        return (packed[:, 0].permute(1, 0, 2).unsqueeze(0),
                packed[:, 1].permute(1, 0, 2).unsqueeze(0), list(ids))
