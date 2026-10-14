from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class KVLayout:
    kv_heads: int
    head_dim: int

    def __post_init__(self):
        if self.kv_heads < 1 or self.head_dim < 1 or self.block_bytes > 1024 * 1024:
            raise ValueError('KV geometry must fit in a positive, at most 1 MiB record')

    @property
    def payload_bytes(self):
        return 4 * self.kv_heads * self.head_dim

    @property
    def block_bytes(self):
        return (self.payload_bytes + 4095) // 4096 * 4096

    def metadata(self):
        return {**asdict(self), 'dtype': 'bfloat16', 'block_size': self.block_bytes}

    @classmethod
    def from_cache(cls, cache):
        if not cache.layers:
            raise ValueError('KV cache has no layers')
        first = cache.layers[0].keys
        layout = cls(first.shape[1], first.shape[-1])
        for layer in cache.layers:
            if (layer.keys.shape != first.shape or layer.values.shape != first.shape
                    or first.ndim != 4 or first.shape[0] != 1
                    or layer.keys.dtype != torch.bfloat16 or layer.values.dtype != torch.bfloat16):
                raise ValueError('Expected uniform batch-one BF16 [1, kv_heads, tokens, head_dim] layers')
        return layout

    def pack(self, key, value):
        data = torch.stack((key, value)).contiguous().view(torch.uint8).numpy().tobytes()
        if len(data) != self.payload_bytes:
            raise ValueError('KV data does not match the configured geometry')
        return data + bytes(self.block_bytes - len(data))
