#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
SKIP_DEPS=false
CHECK_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --skip-deps) SKIP_DEPS=true ;;
        --check) CHECK_ONLY=true ;;
        -h|--help)
            echo "Usage: bash setup.sh [--skip-deps] [--check]"
            echo "Install into the active Python environment and build octopus_io."
            echo "--skip-deps: use already installed Python dependencies"
            echo "--check: inspect prerequisites without installing or building"
            echo "Environment: PYTHON, CUDA_HOME, CXX"
            exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done
if [[ "$(uname -s)" != Linux ]]; then
    echo "Octopus's io_uring/CUDA backend requires Linux." >&2
    exit 1
fi
command -v "$PYTHON" >/dev/null
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 is required"'
command -v "${CXX:-c++}" >/dev/null || { echo "Install a C++17 compiler." >&2; exit 1; }
PROBE_DIR="$(mktemp -d)"
trap 'rm -rf -- "$PROBE_DIR"' EXIT
if ! printf '#include <liburing.h>\nint main(){io_uring r; return io_uring_queue_init(1, &r, 0);}\n' | "${CXX:-c++}" -x c++ - -o "$PROBE_DIR/liburing-check" -luring; then
    echo "Install liburing development headers and library (Ubuntu: sudo apt install liburing-dev)." >&2
    exit 1
fi
if ! $CHECK_ONLY && ! $SKIP_DEPS; then
    "$PYTHON" -m pip install -r "$ROOT/requirements.txt"
fi
"$PYTHON" - <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import torch
from torch.utils.cpp_extension import CUDA_HOME
for name in ("torch", "transformers", "numpy", "accelerate", "tqdm"):
    print(f"{name}: {metadata.version(name)}")
if torch.version.cuda is None:
    raise SystemExit("Install a CUDA-enabled PyTorch build before building octopus_io.")
if CUDA_HOME is None or not (Path(CUDA_HOME) / "include/cuda_runtime.h").is_file():
    raise SystemExit("CUDA toolkit headers were not found; set CUDA_HOME to the toolkit directory.")
print(f"PyTorch CUDA: {torch.version.cuda}; CUDA_HOME: {CUDA_HOME}")
PY
if $CHECK_ONLY; then exit 0; fi
(
    cd "$ROOT/octopus/online"
    "$PYTHON" setup.py build_ext --inplace
    "$PYTHON" -c 'import torch, octopus_io; print("octopus_io:", octopus_io.__file__)'
)
echo "Ready. Run: $PYTHON evaluation/run.py --help"
