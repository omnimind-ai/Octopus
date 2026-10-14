import importlib
from contextlib import contextmanager

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from octopus.layout import KVLayout


FAMILIES = {'qwen2': 'qwen2', 'qwen3': 'qwen3', 'llama': 'llama',
            'gemma': 'gemma', 'gemma2': 'gemma2', 'gemma3_text': 'gemma3'}


class ModelAdapter:
    def __init__(self, config):
        self.config = config
        self.model_type = config.model_type
        if self.model_type not in FAMILIES:
            raise ValueError(f'Unsupported model_type={self.model_type}; choose {", ".join(FAMILIES)}')
        heads = config.num_attention_heads
        self.layout = KVLayout(config.num_key_value_heads, getattr(config, 'head_dim', None)
                               or config.hidden_size // heads)
        if heads % self.layout.kv_heads:
            raise ValueError('Attention heads must be divisible by KV heads')
        kinds = getattr(config, 'layer_types', None) or ['full_attention'] * config.num_hidden_layers
        if len(kinds) != config.num_hidden_layers or set(kinds) - {'full_attention', 'sliding_attention'}:
            raise ValueError('Unsupported attention layer types')
        self.windows = tuple(getattr(config, 'sliding_window', None) if kind == 'sliding_attention'
                             else None for kind in kinds)
        if any(kind == 'sliding_attention' and (window is None or window < 1)
               for kind, window in zip(kinds, self.windows)):
            raise ValueError('Sliding layers need a positive sliding_window')
        if getattr(config, 'use_bidirectional_attention', False):
            raise ValueError('Use a causal text model configuration')

    @classmethod
    def from_path(cls, path):
        return cls(AutoConfig.from_pretrained(path, local_files_only=True))

    def metadata(self):
        return {'model_type': self.model_type, 'num_layers': self.config.num_hidden_layers,
                'attention_windows': self.windows, **self.layout.metadata()}

    def check_context(self, prefix, steps):
        if prefix + steps > self.config.max_position_embeddings:
            raise ValueError('Prefix plus decoding exceeds the configured model context limit')

    def decode_mask(self, device, dtype):
        if self.model_type in ('llama', 'gemma'):
            # A broadcast zero mask avoids constructing a mask for logical positions
            # after the cache has selected a different number of physical KV rows.
            return torch.zeros((1, 1, 1, 1), device=device, dtype=dtype)
        return {'full_attention': None, 'sliding_attention': None}

    @contextmanager
    def capture_queries(self, target):
        module = importlib.import_module(f'transformers.models.{FAMILIES[self.model_type]}.modeling_{FAMILIES[self.model_type]}')
        original = module.apply_rotary_pos_emb

        def capture(*args, **kwargs):
            query, key = original(*args, **kwargs)
            target.append(query.detach())
            return query, key

        module.apply_rotary_pos_emb = capture
        try:
            yield
        finally:
            module.apply_rotary_pos_emb = original
            target.clear()


def load_model(path, device, *, profiling=False):
    adapter = ModelAdapter.from_path(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    # Gemma 2's attention-logit soft cap requires its eager implementation.
    attention = 'eager' if profiling or adapter.model_type == 'gemma2' else 'sdpa'
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, device_map=device,
        dtype=torch.bfloat16, attn_implementation=attention).eval()
    return model, tokenizer, adapter
