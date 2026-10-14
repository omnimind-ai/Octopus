import os
import json
import copy
import math
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DynamicCache
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from octopus.models import load_model
from octopus.data import add_dataset_arguments, prepare_sample


def apply_activation_strategy(scores, strategy="quantile_mass", param=0.90):
    """Convert continuous attention scores into a 0/1 binary activation mask."""
    seq_len = len(scores)
    mask = np.zeros(seq_len, dtype=np.uint8)

    if strategy == "top_k":
        k = int(param)
        if k >= seq_len:
            k = seq_len - 1
        top_k_indices = np.argsort(scores)[-k:]
        mask[top_k_indices] = 1

    elif strategy == "threshold":
        mask[scores > param] = 1

    elif strategy == "quantile_mass":
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]
        cumulative_scores = np.cumsum(sorted_scores)
        cutoff_index = np.searchsorted(cumulative_scores, param)
        active_indices = sorted_indices[:cutoff_index + 1]
        mask[active_indices] = 1

    return mask


def clone_past_key_values(past_key_values):
    """Deep-clone past_key_values so each run starts from an identical prefix cache."""
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()

    if isinstance(past_key_values, (tuple, list)):
        cloned = []
        for layer in past_key_values:
            if isinstance(layer, tuple):
                cloned.append(tuple(
                    t.clone() if torch.is_tensor(t) else copy.deepcopy(t)
                    for t in layer
                ))
            else:
                cloned.append(layer.clone() if torch.is_tensor(layer) else copy.deepcopy(layer))
        return tuple(cloned) if isinstance(past_key_values, tuple) else cloned

    return copy.deepcopy(past_key_values)


def run_prefill(model, prefix_ids):
    """Run the deterministic prefill pass.

    Returns (prefill_logits at last position, past_key_values).
    """
    outputs = model(prefix_ids, past_key_values=DynamicCache(), output_attentions=False, use_cache=True)
    return outputs.logits[:, -1, :], outputs.past_key_values


def save_prefill_artifacts(past_key_values, logits, output_dir, sample, adapter):
    """Save full KV cache and generation metadata.

    Returns the greedy next_token_id (used as trigger token for online inference).
    """
    kv_cache_path = os.path.join(output_dir, "full_kv_cache.pt")
    meta_path = os.path.join(output_dir, "generation_metadata.json")
    next_token_id = torch.argmax(logits[0]).item()

    torch.save(past_key_values, kv_cache_path)
    with open(meta_path, "w") as f:
        json.dump({"next_token_id": next_token_id, "sample": sample, "model": adapter.metadata()}, f, indent=2)

    print(f"[Prefill] Saved full KV-Cache to {kv_cache_path}")
    print(f"[Prefill] Saved trigger token ({next_token_id}) to {meta_path}")
    return next_token_id


def run_decode_collection(model, base_past_key_values, prefill_logits, args, prefix_len, num_layers):
    """Run multi-run stochastic decode to collect per-layer binary activation observations.

    Returns observation_matrix of shape (collected_t, num_layers, prefix_len).
    """
    steps_per_run, extra_steps = divmod(args.total_t, args.runs)
    observation_matrix = np.zeros((args.total_t, num_layers, prefix_len), dtype=np.uint8)
    collected_t = 0

    for run_idx in tqdm(range(args.runs), desc="Total Runs"):
        if collected_t >= args.total_t:
            break

        torch.manual_seed(42 + run_idx)
        past_key_values = clone_past_key_values(base_past_key_values)

        logits = prefill_logits / args.temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        for step in tqdm(range(steps_per_run + (run_idx < extra_steps)), desc=f"Run {run_idx+1}/{args.runs} Decoding", leave=False):
            if collected_t >= args.total_t:
                break

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )

            past_key_values = outputs.past_key_values

            logits = outputs.logits[:, -1, :] / args.temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            for layer_idx in range(num_layers):
                layer_attn = outputs.attentions[layer_idx]
                attn_heads_mean = layer_attn[0, :, 0, :].mean(dim=0)

                attn_to_prefix = attn_heads_mean[:prefix_len].float().cpu().numpy()
                attn_sum = attn_to_prefix.sum()
                if attn_sum > 0:
                    attn_to_prefix /= attn_sum

                mask = apply_activation_strategy(attn_to_prefix, args.strategy, args.param)
                observation_matrix[collected_t, layer_idx, :] = mask

            collected_t += 1

    return observation_matrix[:collected_t]


def main():
    parser = argparse.ArgumentParser(description="Octopus: Layer-wise Co-activation Matrix Collector")
    parser.add_argument("--dataset_path", type=str)
    add_dataset_arguments(parser, internal=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to the local model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model")
    parser.add_argument("--prefix_length", type=int, default=3000, help="Max tokens for the prefix")
    parser.add_argument("--total_t", type=int, default=100, help="Total decode steps (T dimension)")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent decode runs")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--strategy", type=str, default="quantile_mass",
                        choices=["top_k", "threshold", "quantile_mass"])
    parser.add_argument("--param", type=float, default=0.90, help="Parameter for the activation strategy")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save outputs")
    args = parser.parse_args()
    if args.runs <= 0 or args.total_t < args.runs or args.prefix_length <= 0 or args.temperature <= 0:
        parser.error("Use positive prefix/temperature/runs and total_t >= runs")

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer, adapter = load_model(args.model_path, args.device, profiling=True)
    ids, sample = prepare_sample(tokenizer, args)
    prefix_ids = torch.tensor([ids], device=args.device)
    prefix_len = len(ids)
    adapter.check_context(prefix_len, math.ceil(args.total_t / args.runs))
    num_layers = model.config.num_hidden_layers

    print(f"Starting layer-wise decode collection... Target T={args.total_t} across {args.runs} runs")

    with torch.inference_mode():
        prefill_logits, past_key_values = run_prefill(model, prefix_ids)
        base_past_key_values = clone_past_key_values(past_key_values)
        save_prefill_artifacts(past_key_values, prefill_logits, args.output_dir, sample, adapter)

        M = run_decode_collection(model, base_past_key_values, prefill_logits, args, prefix_len, num_layers)

    print(f"\nCollection finished. Tensor shape (T, L, N): {M.shape}")

    np.save(os.path.join(args.output_dir, "activation_matrix.npy"), M)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({
            "dataset_path": args.dataset_path,
            "sample": sample, "model": adapter.metadata(),
            "model_path": args.model_path,
            "prefix_length": prefix_len,
            "num_layers": num_layers,
            "total_t": M.shape[0],
            "runs": args.runs,
            "temperature": args.temperature,
            "strategy": args.strategy,
            "param": args.param,
        }, f, indent=4)

    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
