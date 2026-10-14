import os
import json
import math
import torch
import argparse
import time
from pathlib import Path
import uuid
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from octopus.models import load_model as load_adapted_model
from octopus.layout import KVLayout
from octopus.data import score_prediction
import torch.nn.functional as F
try:
    if __package__:
        from . import octopus_io
    else:
        import octopus_io
except ImportError:
    octopus_io = None
if __package__:
    from .cache_manager import CacheManager
    from .cache import OctopusCache
    from .shared_state import OCTOPUS_GLOBAL_Q
    from .debug_utils import init_debug, debug_log
else:
    from cache_manager import CacheManager
    from cache import OctopusCache
    from shared_state import OCTOPUS_GLOBAL_Q
    from debug_utils import init_debug, debug_log


def synchronize_device(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def sample_logits(logits, temperature=0.7, top_k=50, top_p=0.9):
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1).unsqueeze(-1)
    logits = logits / temperature
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < top_k_values[..., -1, None]] = -float('Inf')
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


def setup_logging(args, octopus_root):
    if args.debug:
        log_dir = os.path.join(octopus_root, "log")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "inference_debug.log"), "w", encoding="utf-8"):
            pass
    log_path = init_debug(args.debug, octopus_root)
    debug_log("inference", f"start args={vars(args)} log_path={log_path}")
    if not args.dram_only and octopus_io is not None:
        octopus_io.set_debug_logging(args.debug, log_path)
    return log_path


def load_kv_and_metadata(kv_cache_path):
    past_key_values = torch.load(kv_cache_path, map_location="cpu", weights_only=False)
    k_sample = past_key_values.layers[0].keys
    num_layers = len(past_key_values.layers)

    meta_path = os.path.join(os.path.dirname(kv_cache_path), "generation_metadata.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    trigger_token_id = meta_data["next_token_id"]
    debug_log("inference", f"loaded kv metadata layers={num_layers} trigger_token_id={trigger_token_id}")

    return past_key_values, k_sample, num_layers, trigger_token_id


def build_cache_system(args, past_key_values, num_layers, adapter):
    cache_manager = CacheManager(
        args.offline_dir,
        past_key_values,
        num_layers,
        args.local_window,
        tau=args.tau,
        num_disks=args.num_disks,
        ban_cache_update=args.ban_cache_update,
        dram_only=args.dram_only,
        attention_windows=adapter.windows,
    )
    octopus_cache = OctopusCache(
        cache_manager, torch.device(args.device), ban_prefetch=args.ban_prefetch,
    )
    debug_log("inference", "cache manager and octopus cache initialized")
    return cache_manager, octopus_cache


def run_decode_loop(model, octopus_cache, cache_manager, input_ids, cache_position, args, tokenizer, adapter):
    past_len = cache_position[0].item()
    total_generated = 0
    scheduler_time_s = 0.0
    postprocess_time_s = 0.0
    token_ids, intervals = [], []

    synchronize_device(args.device)
    start_time = time.perf_counter()
    stamp = start_time

    mask = adapter.decode_mask(args.device, model.dtype)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            scheduler_start = time.perf_counter()
            OCTOPUS_GLOBAL_Q.clear()
            debug_log("inference", f"step={step} current_token_idx={past_len + step}")

            cache_manager.step_start(past_len + step)
            debug_log("inference", f"step={step} cache_manager_step_start token={past_len + step}")
            scheduler_time_s += time.perf_counter() - scheduler_start

            forward_start = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=cache_position.unsqueeze(0),
                past_key_values=octopus_cache,
                cache_position=cache_position,
                use_cache=True,
                logits_to_keep=1,
            )
            forward_time_s = time.perf_counter() - forward_start
            debug_log("inference", f"step={step} forward done logits_shape={tuple(outputs.logits.shape)}")

            post_start = time.perf_counter()
            next_token_id = sample_logits(outputs.logits[:, -1, :], temperature=args.temperature, top_k=20, top_p=0.8)
            token_val = next_token_id.item()
            total_generated += 1
            now = time.perf_counter()
            intervals.append((now - stamp) * 1000)
            stamp = now
            token_ids.append(token_val)

            if args.stream:
                print(tokenizer.decode([token_val]), end="", flush=True)
            postprocess_time_s += time.perf_counter() - post_start
            if token_val in eos_ids:
                break

            input_ids = next_token_id
            cache_position = cache_position + 1

    synchronize_device(args.device)
    inference_duration = time.perf_counter() - start_time
    forward_time_s = max(inference_duration - scheduler_time_s - postprocess_time_s, 0.0)
    safe_count = max(total_generated, 1)

    print(f"\n[Performance] TPS: {safe_count / inference_duration:.2f}, TPOT: {inference_duration / safe_count * 1000:.2f} ms")
    debug_log("inference", f"finished generated={total_generated} tps={safe_count / inference_duration:.4f} tpot_ms={inference_duration / safe_count * 1000:.4f}")
    debug_log(
        "inference",
        f"timing_summary forward_s={forward_time_s:.6f} "
        f"scheduler_s={scheduler_time_s:.6f} postprocess_s={postprocess_time_s:.6f} "
        f"decode_total_s={inference_duration:.6f} generated={total_generated}",
    )
    return {"generated_tokens": total_generated, "decode_seconds": inference_duration,
            "tokens_per_second": total_generated / inference_duration,
            "mean_tpot_ms": inference_duration * 1000 / safe_count,
            "inter_token_ms": intervals, "token_ids": token_ids,
            "metric_scope": "prefix continuation decoding; offline prefill excluded"}


def main():
    parser = argparse.ArgumentParser(description="Octopus: Online Sparse-Cache Inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--kv_cache", type=str, required=True)
    parser.add_argument("--offline_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--local_window", type=int, default=256)
    parser.add_argument("--num_disks", type=int, default=8)
    parser.add_argument("--disk_paths", nargs="+")
    parser.add_argument("--metrics_output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--stream", action="store_true", help="Print generated text during decoding")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ban_cache_update", action="store_true")
    parser.add_argument("--ban_prefetch", action="store_true")
    parser.add_argument("--dram_only", action="store_true",
                        help="Load all tokens into DRAM; skip SSD I/O (for testing without NVMe disks)")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.local_window < 1 or args.num_disks < 1 or not math.isfinite(args.temperature) or args.temperature < 0 or not 0 <= args.tau <= 1:
        parser.error("Token/window/disk counts must be positive; temperature must be nonnegative")
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        device_index = torch.device(args.device).index
        torch.cuda.set_device(0 if device_index is None else device_index)

    past_key_values, k_sample, num_layers, trigger_token_id = load_kv_and_metadata(args.kv_cache)
    layout = KVLayout.from_cache(past_key_values)
    model, tokenizer, adapter = load_adapted_model(args.model_path, args.device)
    if layout != adapter.layout or num_layers != model.config.num_hidden_layers:
        raise ValueError('Offline KV geometry does not match the model')
    metadata = json.loads((Path(args.offline_dir) / 'generation_metadata.json').read_text())
    if 'model' in metadata and metadata['model'] != json.loads(json.dumps(adapter.metadata())):
        raise ValueError('Offline model configuration differs from the requested model')
    storage = json.loads((Path(args.offline_dir) / 'storage.json').read_text())
    if 'local_window' in storage and args.local_window not in (storage['local_window'], storage.get('requested_local_window')):
        parser.error('local-window must match offline preparation')
    args.local_window = storage.get('local_window', args.local_window)
    if storage.get('block_size', layout.block_bytes) != layout.block_bytes:
        raise ValueError('Storage record size differs from the model KV geometry')
    if args.dram_only:
        args.ban_prefetch = True
        args.ban_cache_update = True
    else:
        if not args.device.startswith("cuda"):
            parser.error("SSD inference requires a CUDA device")
        if octopus_io is None or not all(hasattr(octopus_io, name) for name in ("configure_storage", "synchronize")):
            parser.error("octopus_io is not built; run bash setup.sh")
        paths = [str(Path(p).expanduser().resolve()) for p in (args.disk_paths or [])]
        if len(paths) != args.num_disks or len(set(paths)) != args.num_disks:
            parser.error("Supply one distinct --disk_paths entry per disk")
        if any(not Path(p).is_file() or Path(p).is_relative_to('/dev') for p in paths):
            parser.error("KV paths must be existing regular files")
        storage = json.loads((Path(args.offline_dir) / "storage.json").read_text())
        if not storage["physical"] or storage["num_disks"] != args.num_disks or storage["disk_paths"] != paths:
            parser.error("Disk paths/order must match storage.json from offline preparation")
        tag = uuid.uuid4().hex
        append_paths = [p + ".append-" + tag for p in paths]
        if not args.ban_cache_update:
            for path in append_paths:
                with open(path, "xb"):
                    pass
        octopus_io.configure_storage(paths, append_paths, layout.kv_heads, layout.head_dim)

    setup_logging(args, args.offline_dir)

    past_len = past_key_values.get_seq_length()
    adapter.check_context(past_len, args.max_new_tokens)
    cache_manager, octopus_cache = build_cache_system(args, past_key_values, num_layers, adapter)
    del past_key_values, k_sample

    input_ids = torch.tensor([[trigger_token_id]], device=args.device)
    cache_position = torch.arange(past_len, past_len + input_ids.shape[1], device=args.device, dtype=torch.long)

    print(f"\nStarting generation (Max new tokens: {args.max_new_tokens})...")
    if args.stream:
        print(tokenizer.decode([trigger_token_id]), end="", flush=True)

    try:
        with adapter.capture_queries(OCTOPUS_GLOBAL_Q):
            metrics = run_decode_loop(model, octopus_cache, cache_manager, input_ids, cache_position, args, tokenizer, adapter)
    finally:
        if not args.dram_only and octopus_io is not None:
            octopus_io.synchronize()
    generated_ids = [trigger_token_id, *metrics['token_ids']]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    metrics.update(generated_text=text, continuation_token_ids=generated_ids,
                   sample=metadata.get('sample'), model=adapter.metadata())
    metrics['sample_score'] = score_prediction(text, metadata.get('sample', {}))
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"[Metrics] {args.metrics_output}")


if __name__ == "__main__":
    main()
