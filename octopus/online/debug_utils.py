import os
import threading
from datetime import datetime


_DEBUG_ENABLED = False
_DEBUG_LOG_PATH = ""
_LOCK = threading.Lock()


def init_debug(enabled: bool, octopus_root: str) -> str:
    global _DEBUG_ENABLED, _DEBUG_LOG_PATH
    _DEBUG_ENABLED = bool(enabled)
    if not _DEBUG_ENABLED:
        _DEBUG_LOG_PATH = ""
        return _DEBUG_LOG_PATH

    log_dir = os.path.join(octopus_root, "log")
    os.makedirs(log_dir, exist_ok=True)
    _DEBUG_LOG_PATH = os.path.join(log_dir, "inference_debug.log")
    with _LOCK:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().isoformat()}] ==== Debug Session Start ====\n")
    return _DEBUG_LOG_PATH


def is_debug_enabled() -> bool:
    return _DEBUG_ENABLED


def get_debug_log_path() -> str:
    return _DEBUG_LOG_PATH


def debug_log(component: str, message: str) -> None:
    if not _DEBUG_ENABLED or not _DEBUG_LOG_PATH:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    line = f"[{ts}] [{component}] {message}\n"
    with _LOCK:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
