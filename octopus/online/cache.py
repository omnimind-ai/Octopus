from typing import Any, Optional, Tuple

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

try:
    if __package__:
        from . import octopus_io
    else:
        import octopus_io
except ImportError:
    octopus_io = None
if __package__:
    from .cache_manager import CacheManager
    from .shared_state import OCTOPUS_GLOBAL_Q
else:
    from cache_manager import CacheManager
    from shared_state import OCTOPUS_GLOBAL_Q


class OctopusCache(DynamicCache):
    def __init__(self, manager: CacheManager, device: torch.device, ban_prefetch: bool = False):
        super().__init__()
        self.manager = manager
        self.device = torch.device(device)
        self.num_layers = manager.num_layers
        self.ban_prefetch = ban_prefetch or manager.dram_only
        self.prefetches = [None] * self.num_layers
        self.current_sparse_lengths = [0] * self.num_layers
        self.logical_lengths = [manager.past_len] * self.num_layers

    def _plan(self, layer, query):
        hits = self.manager.predict_hits_for_layer(layer, query)
        self.manager.update_cache(layer, hits)
        return self.manager.generate_io_plan(layer, hits)

    def _assemble_kv(self, layer, key_states, ssd_k, ssd_v, ssd_ids):
        packed, dram_ids = self.manager.get_dram_packed_for_layer(layer)
        window = self.manager.attention_windows[layer]
        if window:
            positions = [i for i, token in enumerate(dram_ids) if token > self.manager.current_token_id - window]
            packed = packed.index_select(0, torch.tensor(positions, dtype=torch.long))
            dram_ids = [dram_ids[i] for i in positions]
        # CPU slots can change at the next step; finish H2D before lending them again.
        packed_gpu = packed.to(device=key_states.device, dtype=key_states.dtype)
        dram_k = packed_gpu[:, 0].permute(1, 0, 2).unsqueeze(0)
        dram_v = packed_gpu[:, 1].permute(1, 0, 2).unsqueeze(0)
        if not ssd_ids and self.manager.layers[layer].store.ordered:
            if key_states.device.type == 'cpu':
                dram_k, dram_v = dram_k.clone(), dram_v.clone()
            return dram_k, dram_v
        return self.assemble_and_sort(dram_k, dram_v, dram_ids, ssd_k, ssd_v, ssd_ids)

    def assemble_and_sort(self, dram_k, dram_v, dram_ids, ssd_k, ssd_v, ssd_ids):
        sources = [(k, v, ids) for k, v, ids in ((dram_k, dram_v, dram_ids), (ssd_k, ssd_v, ssd_ids)) if len(ids)]
        if not sources:
            raise RuntimeError('Attention has no KV entries')
        for k, v, ids in sources:
            if k is None or v is None or k.shape != v.shape or k.shape[2] != len(ids):
                raise RuntimeError('KV tensor and token ID lengths differ')
        ids = np.asarray([token for _, _, tokens in sources for token in tokens], dtype=np.int64)
        order = np.argsort(ids, kind='stable')
        if len(ids) > 1 and np.any(ids[order][1:] == ids[order][:-1]):
            raise RuntimeError('DRAM and SSD inputs contain duplicate tokens')
        destination = np.empty(len(ids), dtype=np.int64)
        destination[order] = np.arange(len(ids))
        first = sources[0][0]
        index = torch.from_numpy(destination).to(first.device)
        output = torch.empty((2, first.shape[0], first.shape[1], len(ids), first.shape[3]),
                             dtype=first.dtype, device=first.device)
        start = 0
        for k, v, tokens in sources:
            stop = start + len(tokens)
            output[0].index_copy_(2, index[start:stop], k)
            output[1].index_copy_(2, index[start:stop], v)
            start = stop
        return output[0], output[1]

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
               cache_kwargs: Optional[dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self.manager.add_online_kv_for_layer(layer_idx, key_states, value_states)
        ssd_k, ssd_v, ssd_ids = None, None, []
        if not self.manager.dram_only:
            if not OCTOPUS_GLOBAL_Q:
                raise RuntimeError('No query captured for sparse KV retrieval')
            if not self._can_prefetch(layer_idx):
                plan = self._plan(layer_idx, OCTOPUS_GLOBAL_Q[-1])
                if plan:
                    ssd_k, ssd_v, ssd_ids = octopus_io.sync_read_clusters(layer_idx, plan)
            else:
                pending, self.prefetches[layer_idx] = self.prefetches[layer_idx], None
                if pending is None or pending[0] != self.manager.current_token_id:
                    raise RuntimeError('Missing or stale layer prefetch')
                if pending[1] is not None:
                    ssd_k, ssd_v, ssd_ids = octopus_io.wait_and_get(pending[1])
        full_k, full_v = self._assemble_kv(layer_idx, key_states, ssd_k, ssd_v, ssd_ids)
        self.manager.complete_fetch(layer_idx, ssd_k, ssd_v, ssd_ids)
        self.current_sparse_lengths[layer_idx] = full_k.shape[2]
        self.logical_lengths[layer_idx] = self.manager.current_token_id + 1
        next_layer = layer_idx + 1
        if next_layer < self.num_layers and self._can_prefetch(next_layer):
            if self.prefetches[next_layer] is not None:
                raise RuntimeError('Previous prefetch was not consumed')
            plan = self._plan(next_layer, OCTOPUS_GLOBAL_Q[-1])
            handle = octopus_io.async_read_clusters(next_layer, plan) if plan else None
            self.prefetches[next_layer] = (self.manager.current_token_id, handle)
        return full_k, full_v

    def _can_prefetch(self, layer):
        windows = self.manager.attention_windows
        return not self.ban_prefetch and layer > 0 and windows[layer] == windows[layer - 1]

    def get_seq_length(self, layer_idx=0):
        return self.logical_lengths[layer_idx]

    def get_max_cache_shape(self, layer_idx=0):
        return -1
