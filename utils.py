                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          # utils.py
"""
Utilities para CrypTrader
- atomic_write: escritura atómica de JSON/texto para evitar corrupciones
- hash_payload: hash para dedupe de payloads
- dedupe_check_and_add: cache en memoria para evitar replays
- set_global_seed: fija semilla para reproducibilidad
- safe_save_json: wrapper que intenta atomic_write y cae a write normal
"""

import os
import json
import tempfile
import hashlib
import time
from pathlib import Path
from typing import Any

# -------------------------
# Atomic write
# -------------------------
def atomic_write(path: str, data: Any, encoding: str = "utf-8") -> None:
    """
    Escribe de forma atómica en 'path'. Crea un tmp file y luego hace os.replace().
    data: dict/list/str -> si es dict/list hace json.dumps() automáticamente.
    """
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    if isinstance(data, (dict, list)):
        text = json.dumps(data, ensure_ascii=False, indent=2)
    else:
        text = str(data)
    # ensure parent exists
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    # atomic rename
    os.replace(tmp, p)

def safe_save_json(path: str, data: Any):
    """Wrapper: intenta atomic_write, si falla hace write normal."""
    try:
        atomic_write(path, data)
    except Exception:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# -------------------------
# Payload dedupe
# -------------------------
def hash_payload(payload: dict) -> str:
    """
    Hash SHA256 de payload JSON (ordenado) — útil para dedupe.
    """
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# in-memory dedupe cache
_DEDUPE_CACHE = {}

def dedupe_check_and_add(key: str, ttl_seconds: int = 30) -> bool:
    """
    Comprueba si el key está en cache; si no, lo añade y devuelve True.
    Si ya existe devuelve False.
    TTL: tiempo en segundos para olvidar keys antiguos.
    """
    now = time.time()
    # cleanup
    for k, t in list(_DEDUPE_CACHE.items()):
        if now - t > ttl_seconds:
            _DEDUPE_CACHE.pop(k, None)
    if key in _DEDUPE_CACHE:
        return False
    _DEDUPE_CACHE[key] = now
    return True

# -------------------------
# Seed / reproducibilidad
# -------------------------
def set_global_seed(seed: int):
    """
    Fija semillas para random, numpy y, si está, para otras librerías.
    """
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

# -------------------------
# Small helpers
# -------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
