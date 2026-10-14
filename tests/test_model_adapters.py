import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, DynamicCache, Qwen2Config, Qwen3Config, LlamaConfig
from transformers import GemmaConfig, Gemma2Config, Gemma3TextConfig

from octopus.models import ModelAdapter
from octopus.layout import KVLayout
from octopus.online.cache_manager import CacheManager
from octopus.online.cache import OctopusCache
from octopus.online.shared_state import OCTOPUS_GLOBAL_Q


CONFIGS = [(Qwen2Config, 2, 64), (Qwen3Config, 1, 128), (LlamaConfig, 2, 64),
           (GemmaConfig, 1, 256), (Gemma2Config, 2, 256), (Gemma3TextConfig, 1, 256)]


def tiny_config(cls, heads, dim):
    return cls(vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
               num_attention_heads=4, num_key_value_heads=heads, head_dim=dim,
               max_position_embeddings=128, sliding_window=8,
               layer_types=['sliding_attention', 'full_attention'] if cls in (Gemma2Config, Gemma3TextConfig)
               else ['full_attention', 'full_attention'],
               query_pre_attn_scalar=dim, attn_implementation='eager',
               bos_token_id=1, eos_token_id=2, pad_token_id=0)


class ModelTests(unittest.TestCase):
    def test_dense_equivalence_across_families_and_sliding_boundaries(self):
        for cls, heads, dim in CONFIGS:
            with self.subTest(model=cls.model_type), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(72)
                config = tiny_config(cls, heads, dim)
                model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).eval()
                adapter = ModelAdapter(config)
                ids = torch.randint(3, 64, (1, 16))
                with torch.inference_mode():
                    prefix = model(ids, past_key_values=DynamicCache(), use_cache=True)
                full = prefix.past_key_values
                self.assertEqual(KVLayout.from_cache(full), adapter.layout)
                # Even sliding layers retain the full offline prefix for token-ID routing.
                self.assertEqual([layer.keys.shape[2] for layer in full.layers], [16, 16])
                dense_cache = copy.deepcopy(full)
                route = {str(i): {'0': {'type': 'PINNED_DRAM', 'routes': [{'token': t} for t in range(16)],
                                      'center_status': 'PINNED_CENTER'}, 'global_dram_tokens': list(range(16))}
                         for i in range(2)}
                root = Path(directory)
                (root / 'routing_index.json').write_text(json.dumps(route))
                manager = CacheManager(root, full, 2, 4, 1, dram_only=True, attention_windows=adapter.windows)
                cache = OctopusCache(manager, torch.device('cpu'))
                token = prefix.logits[:, -1].argmax(-1, keepdim=True)
                with torch.inference_mode(), adapter.capture_queries(OCTOPUS_GLOBAL_Q):
                    for step in range(12):
                        position = torch.tensor([16 + step])
                        dense = model(token, past_key_values=dense_cache, use_cache=True)
                        dense_cache = dense.past_key_values
                        OCTOPUS_GLOBAL_Q.clear()
                        manager.step_start(16 + step)
                        actual = model(token, past_key_values=cache, use_cache=True,
                                       attention_mask=adapter.decode_mask('cpu', model.dtype),
                                       position_ids=position.unsqueeze(0), cache_position=position)
                        torch.testing.assert_close(actual.logits, dense.logits, rtol=.025, atol=.015)
                        self.assertTrue(torch.equal(actual.logits.argmax(-1), dense.logits.argmax(-1)))
                        self.assertEqual(cache.get_seq_length(), 17 + step)
                        for layer, window in enumerate(adapter.windows):
                            if window:
                                self.assertEqual(cache.current_sparse_lengths[layer], window)
                        token = dense.logits[:, -1].argmax(-1, keepdim=True)
                self.assertFalse(OCTOPUS_GLOBAL_Q)

    def test_layout_alignment_and_padding(self):
        for heads, dim, expected in ((1, 64, 4096), (1, 256, 4096), (8, 128, 4096), (8, 256, 8192), (12, 256, 12288)):
            layout = KVLayout(heads, dim)
            self.assertEqual(layout.block_bytes, expected)
            payload = layout.pack(torch.ones((heads, dim), dtype=torch.bfloat16),
                                  torch.zeros((heads, dim), dtype=torch.bfloat16))
            self.assertEqual(len(payload), expected)
            self.assertEqual(payload[layout.payload_bytes:], bytes(expected - layout.payload_bytes))

    def test_unknown_family_is_rejected(self):
        config = LlamaConfig()
        config.model_type = 'unknown'
        with self.assertRaisesRegex(ValueError, 'Unsupported model_type'):
            ModelAdapter(config)


if __name__ == '__main__':
    unittest.main()
