# tvhook.py (actualizado)
import os, json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from pathlib import Path

from utils import atomic_write, hash_payload, dedupe_check_and_add

app = FastAPI()
DATA_FILE = Path("alerts.json")
TV_HOOK_TOKEN = os.environ.get("TV_HOOK_TOKEN", "changeme")
DEDUPE_TTL = int(os.environ.get("TV_DEDUPE_TTL", "30"))  # segundos

def append_alert_atomic(payload: dict):
    payload["_received_at"] = datetime.utcnow().isoformat()
    # load existing
    if DATA_FILE.exists():
        with DATA_FILE.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
    else:
        data = []
    data.insert(0, payload)
    data = data[:1000]
    atomic_write(str(DATA_FILE), data)

@app.post("/tvhook")
async def tvhook(request: Request):
    q = dict(request.query_params)
    token_q = q.get("token")
    try:
        payload = await request.json()
    except Exception:
        text = (await request.body()).decode(errors="ignore")
        payload = {"raw": text}

    token_body = payload.get("token") if isinstance(payload, dict) else None
    token = token_q or token_body
    if token != TV_HOOK_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid token")

    # compute dedupe hash
    phash = hash_payload(payload)
    if not dedupe_check_and_add(phash, ttl_seconds=DEDUPE_TTL):
        return JSONResponse({"status":"duplicate","received": False})

    payload["_remote_addr"] = request.client.host if request.client else None
    append_alert_atomic(payload)
    return JSONResponse({"status":"ok","received": True, "timestamp": datetime.utcnow().isoformat()})
