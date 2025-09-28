# tvhook.py
"""
TradingView webhook receiver.
- Recibe POST /tvhook?token=...
- Body JSON esperado: {"symbol":"BINANCE:BTCUSDT", "signal":"LONG"|"SHORT"|"BUY"|"SELL", "time":"ISO8601", "extra": {...}}
- Dedupe por hash (usa utils.hash_payload si existe)
- Guarda en alerts.json (lista) y mantiene latest_alerts.json (últimas N)
- Opcional: si APP_NOTIFY_URL está configurada, hace POST a esa URL con la alerta para notificar a la UI
"""

import os
import json
import time
from typing import Any
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
import hashlib
from pathlib import Path

# si tienes utils.py con hash_payload y dedupe_check_and_add usa eso, sino usamos local
try:
    from utils import hash_payload, dedupe_check_and_add
except Exception:
    def hash_payload(payload: dict) -> str:
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    _DEDUPE = {}
    def dedupe_check_and_add(key: str, ttl_seconds: int = 30) -> bool:
        now = time.time()
        for k,t in list(_DEDUPE.items()):
            if now - t > ttl_seconds:
                _DEDUPE.pop(k, None)
        if key in _DEDUPE:
            return False
        _DEDUPE[key] = now
        return True

app = FastAPI()
STORAGE = Path("alerts.json")
LATEST = Path("latest_alerts.json")
TOKEN_ENV = os.environ.get("TV_HOOK_TOKEN", None)
APP_NOTIFY_URL = os.environ.get("APP_NOTIFY_URL", None)  # opcional: URL para notificar UI

def _read_alerts():
    if STORAGE.exists():
        try:
            return json.loads(STORAGE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _write_alerts(arr):
    STORAGE.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")

def _append_alert(alert: dict, keep_latest=50):
    arr = _read_alerts()
    arr.append(alert)
    _write_alerts(arr)
    # write latest snapshot
    latest = arr[-keep_latest:]
    LATEST.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")

@app.post("/tvhook")
async def tvhook(request: Request, token: str = Query(None)):
    # token check
    if TOKEN_ENV is None and token is None:
        # if nobody configured token we still allow but warn
        pass
    else:
        if token is None or token != TOKEN_ENV:
            raise HTTPException(status_code=401, detail="Invalid token")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # normalize expected fields
    symbol = payload.get("symbol") or payload.get("ticker") or payload.get("s") or "UNKNOWN"
    signal = payload.get("signal") or payload.get("type") or payload.get("action") or payload.get("alert") or ""
    timestamp = payload.get("time") or payload.get("timestamp") or payload.get("date") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    alert = {
        "symbol": str(symbol).upper(),
        "signal": str(signal).upper(),
        "time": timestamp,
        "raw": payload
    }

    # dedupe
    key = hash_payload(alert)
    if not dedupe_check_and_add(key, ttl_seconds=10):
        return JSONResponse({"ok": False, "reason": "duplicate"}, status_code=200)

    # save
    _append_alert(alert)

    # optional notify UI (best effort)
    if APP_NOTIFY_URL:
        try:
            import requests
            requests.post(APP_NOTIFY_URL, json=alert, timeout=5)
        except Exception:
            # no hacemos nada si falla
            pass

    return {"ok": True, "alert": alert}
