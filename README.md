# Octopus: Unlocking SSD Parallelism for Sparse KVCache Retrieval

## Requirement

- **GPU:** An NVIDIA GPU with enough memory to load the model.
- **Storage:** **8 NVMe SSDs recommended**, mounted before running.
- **System:** Linux, Python 3.10+, a C++17 compiler, liburing, and a CUDA toolkit compatible with your CUDA-enabled PyTorch build.

## Setup

```bash
git clone https://github.com/omnimind-ai/Octopus.git
cd Octopus

# Ubuntu system dependencies
sudo apt install build-essential liburing-dev python3-venv

python3 -m venv .venv
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda
bash setup.sh
```

Adjust `CUDA_HOME` to your toolkit installation. If Python dependencies are
already installed, use `bash setup.sh --skip-deps`.

## Run

Use `evaluation/run.py` for all models, datasets, and execution modes. Run the
commands below from the repository root in Bash.

### Models

| Family | Example checkpoints |
|---|---|
| Qwen2 / Qwen2.5 | Qwen2.5-0.5B |
| Qwen3 | Qwen3-0.6B, Qwen3-8B |
| Llama | Llama-3.2-1B, SmolLM2-135M |
| Gemma | Gemma-2B |
| Gemma2 | Gemma-2-2B |
| Gemma3 | Gemma-3-270M |

Set `--model-path` to a local Hugging Face checkpoint directory containing its
configuration, tokenizer, and weights. 

### Datasets

| `--dataset` | Hugging Face source | `--dataset-config` | Local record fields |
|---|---|---|---|
| `context` | Local input | — | `context` |
| `mmlu` | `cais/mmlu` | `all` (default), or a subject such as `abstract_algebra` | `question`, `choices`, `answer` |
| `gsm8k` | `openai/gsm8k` | `main` (default) | `question`, `answer` |
| `wikitext` | `Salesforce/wikitext` | `wikitext-2-raw-v1` (default), or `wikitext-103-raw-v1` | `text` |
| `longbench` | `zai-org/LongBench` | Task name, e.g. `qasper` or `gov_report` | `context`, `input`, `answers` |
| `infinitebench` | `xinrongzhang2022/InfiniteBench` | Task name, e.g. `passkey` or `longbook_qa_eng` | `context`, `input`, `answer`, optional `options` |

Omit `--dataset-path` to load a named dataset from Hugging Face. LongBench and
InfiniteBench require `--dataset-config TASK` and use the test split. For MMLU,
GSM8K, and WikiText, select a split with `--dataset-split` (default: `test`).
Each run uses the record selected by `--sample-index` (default: `0`).

To use local data, add `--dataset-path` and select its format with `--dataset`.
Supported files are JSONL, JSON, Parquet, and directories created by
`datasets.save_to_disk`; WikiText also accepts `.txt`. MMLU `choices` contains
four strings, and `answer` is an index from 0 to 3 or a letter from A to D.
For local LongBench or InfiniteBench records without an explicit `input`, set
the task name with `--dataset-config`.

A custom `context` JSONL file contains one record per line:

```json
{"context": "Your input document goes here."}
```

Input length is controlled by `--prefix-length`:

- `context` uses the first requested number of tokens from the selected record;
  the record must contain at least that many tokens.
- WikiText joins consecutive nonempty rows to fill the requested prefix.
- MMLU and GSM8K use the complete question, up to the specified limit. Increase
  the limit if a question is longer.
- LongBench and InfiniteBench preserve the question and trim the middle of the
  context when necessary.

For shorter questions, the local KV window is capped at the actual input
length. Keep the prefix and continuation within the model's configured context
length. Hugging Face downloads use `HF_HOME` / `HF_HUB_CACHE` when set.

### Configure a run

```bash
model_path=/path/to/Qwen2.5-0.5B
work_dir=/path/to/new-run
dataset_args=(--dataset wikitext --dataset-config wikitext-2-raw-v1)

disk_paths=()
for i in {1..8}; do
    disk_paths+=("/mnt/kv_disk${i}/octopus-demo.bin")
done
```

Choose one of the following definitions to replace `dataset_args`:

```bash
# MMLU
dataset_args=(--dataset mmlu --dataset-config abstract_algebra --sample-index 0)

# GSM8K
dataset_args=(--dataset gsm8k --dataset-split test --sample-index 0)

# LongBench
dataset_args=(--dataset longbench --dataset-config qasper)

# InfiniteBench
dataset_args=(--dataset infinitebench --dataset-config passkey)

# Local GSM8K records
dataset_args=(--dataset gsm8k --dataset-path /path/to/gsm8k.jsonl)

# Custom context
dataset_args=(--dataset context --dataset-path /path/to/contexts.jsonl)
```

Offline preparation requires a new or empty work directory and new KV filenames.
Use different paths for each new model or input. Choose end-to-end execution,
or run offline followed by online.

### End-to-end

```bash
python evaluation/run.py \
    --model-path "$model_path" \
    "${dataset_args[@]}" \
    --work-dir "$work_dir" \
    --disk-paths "${disk_paths[@]}" \
    --device cuda:0 \
    --prefix-length 2048 --decode-tokens 32
```

### Offline only

```bash
python evaluation/run.py --stage offline \
    --model-path "$model_path" \
    "${dataset_args[@]}" \
    --work-dir "$work_dir" \
    --disk-paths "${disk_paths[@]}" \
    --device cuda:0 \
    --prefix-length 2048 --profile-steps 64 --profile-runs 4
```

### Online only

Reuse the model, work directory, disk paths/order, and local-window setting from
offline preparation.

```bash
python evaluation/run.py --stage inference \
    --model-path "$model_path" \
    --work-dir "$work_dir" \
    --disk-paths "${disk_paths[@]}" \
    --device cuda:0 \
    --decode-tokens 32
```

### DRAM-only

Run offline preparation and decoding with KV data in CPU memory:

```bash
python evaluation/run.py --dram-only \
    --model-path "$model_path" \
    "${dataset_args[@]}" \
    --work-dir "${work_dir}-dram" \
    --device cuda:0 \
    --prefix-length 2048 --decode-tokens 32
```

Use `--device cpu` for CPU execution. To reuse existing offline artifacts in
DRAM-only mode, add `--stage inference` and keep the original work directory.

### Common options

| Option | Default | Purpose |
|---|---|---|
| `--prefix-length` | `2048` | Input token length or limit, according to the dataset |
| `--profile-steps` / `--profile-runs` | `64` / `4` | Total offline profiling steps and independent runs |
| `--decode-tokens` | `32` | Online decoding steps |
| `--local-window` | `128` | Recent tokens retained in the local KV window |
| `--dram-ratio` | `0.2` | Offline DRAM placement budget ratio |
| `--temperature` | `0.7` | Online sampling temperature; `0` selects greedy decoding |
| `--workers` | `1` | Offline clustering workers |
| `--disable-prefetch` | Off | Use synchronous SSD retrieval |
| `--disable-cache-update` | Off | Freeze adaptive cache admission and scoring |
| `--stream` | Off | Print generated text |
| `--dry-run` | Off | Print the commands without running them |

Run `python evaluation/run.py --help` for all options.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
