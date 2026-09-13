# KeyGuard AI

Real-time continuous user authentication using keystroke dynamics, a multi-user
LSTM with unknown rejection, temporal risk smoothing, and optional Gemini verification.

## Architecture

```
index.html  →  server.py  →  model.py
                  ↓
            risk_engine.py
                  ↓
              users.py
```

- **`model.py`** — 6-feature extraction, multi-user LSTM, profile fallback, evaluation metrics.
- **`risk_engine.py`** — temporal smoothing, auth state machine, clipboard isolation.
- **`users.py`** — JSON persistence, 5-session enrollment with type tags.
- **`server.py`** — FastAPI REST + WebSocket orchestration.
- **`index.html`** — auth, enrollment wizard, protected workspace, activity log side panel.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env
python train.py              # optional: train multi-user LSTM
uvicorn server:app --reload  # http://127.0.0.1:8000
python test_flow.py          # E2E tests (server must be running)
```

## Training

```bash
python train.py
```

Session-based split: sessions 1–3 (normal/different/free) → train, session 4 (fast) → val,
session 5 (slow) → test. Outputs `model_multiuser.pt` + `model_multiuser_meta.json` and
prints FAR / FRR / EER.

## Adding a new detection signal (e.g. mouse-movement biometrics)

This is the one extension point the whole system is built around. In
`model.py`:

```python
def predict_mouse_anomaly(features: np.ndarray) -> float:
    ...  # your scoring logic
    return score  # 0-1

SIGNALS["mouse"] = predict_mouse_anomaly
```

That's it — no changes to `server.py` or the frontend. `server.py`'s
`score_signals()` already loops over `SIGNALS` generically (`for name, fn in
SIGNALS.items()`), with no per-signal `if/else`, and each call is wrapped in
its own isolated `try/except` so one signal failing doesn't take down the
others. The aggregate score used to decide whether to consult Gemini is the
mean across every registered signal, so a new signal is folded in
automatically.

## Message contract (`/ws/typing`)

**Client → server**, sent every N keystrokes or M ms:
```json
{"events": [{"key": "a", "event_type": "keydown", "timestamp": 1699999999123.4}, ...]}
```

**Server → client**, after each evaluated window:
```json
{
  "type": "score_update",
  "score": 0.82,
  "signal_scores": {"keystroke": 0.82},
  "action": "silent" | "verify" | "lock",
  "message": "",
  "confidence_reasoning": "",
  "timestamp": 1699999999.5
}
```

## Notes

- No classes anywhere — plain functions only, per spec.
- Gemini's API key is loaded via `.env` + `python-dotenv` (the non-Streamlit
  equivalent of `st.secrets`) — never hardcoded.
- `call_gemini` and each signal function each have their own isolated
  `try/except` with a safe fallback (`"action": "silent"`), so a broken
  signal or a Gemini outage degrades gracefully instead of crashing the
  pipeline.
