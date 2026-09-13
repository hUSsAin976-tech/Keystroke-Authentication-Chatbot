"""
server.py
---------
Orchestration, WebSocket API, user authentication, and biometric verification
for KeyGuard AI.
"""

from __future__ import annotations

import os
import json
import time
from collections import deque

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import extract_features, predict_user, learn_user_profile, invalidate_model_cache, SEQUENCE_LENGTH
from risk_engine import make_engine, decide_action, STATE_IDENTITY_CHANGED, STATE_UNKNOWN
from users import (
    register_user,
    authenticate_user,
    get_user,
    save_enrollment_session,
    get_enrollment_sessions,
    get_enrollment_session_defs,
    ENROLLMENT_SESSIONS,
    delete_user_profile,
    delete_all_profiles,
    get_registered_profiles,
)

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "40"))

app = FastAPI(title="KeyGuard AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class UserAuthRequest(BaseModel):
    username: str
    password: str


class EnrollSubmitRequest(BaseModel):
    username: str
    session_index: int
    session_type: str = "normal"
    events: list[dict]


# ---------------------------------------------------------------------------
# Gemini (optional verification messaging)
# ---------------------------------------------------------------------------

def build_gemini_prompt(score: float, context: dict) -> str:
    signal_scores = context.get("signal_scores", {})
    session_id = context.get("session_id", "unknown")
    recent_avg = context.get("recent_avg_score", score)
    state = context.get("state", "SUSPICIOUS")

    return f"""You are a calm, professional security assistant for KeyGuard AI.
Current authenticity score: {score:.3f} (1.0 = genuine user).
Recent rolling average: {recent_avg:.3f}.
Risk state: {state}.
Per-signal scores: {json.dumps(signal_scores)}.
Session ID: {session_id}.

Decide exactly one action:
  - "silent": typing is within normal range.
  - "verify": mildly suspicious; ask one short, friendly verification question.
  - "lock": strongly anomalous; session should be locked.

Respond with STRICT JSON only:
{{"action": "silent" | "verify" | "lock", "message": "<string>", "confidence_reasoning": "<one sentence>"}}
"""


def call_gemini(prompt: str) -> dict:
    fallback = {
        "action": "silent",
        "message": "",
        "confidence_reasoning": "Fallback: Gemini unavailable.",
    }
    if not GEMINI_API_KEY or "your_" in GEMINI_API_KEY:
        return {
            "action": "verify",
            "message": "KeyGuard AI: Unusual typing rhythm detected. Please confirm your activity.",
            "confidence_reasoning": "Local decision (GEMINI_API_KEY not set).",
        }
    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        response = httpx.post(GEMINI_URL, json=payload, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
        parsed.setdefault("message", "")
        parsed.setdefault("confidence_reasoning", "")
        return parsed
    except Exception as exc:
        fallback["confidence_reasoning"] = f"Fallback: {type(exc).__name__}."
        return fallback


# ---------------------------------------------------------------------------
# Window evaluation via risk engine
# ---------------------------------------------------------------------------

def _auth_action_to_legacy(auth_event: dict) -> str:
    action = auth_event.get("action", "continue")
    state = auth_event.get("state", "")
    if action in ("reauth", "notify") or state == STATE_UNKNOWN:
        return "notify"
    if action in ("warn", "notify") or state == STATE_IDENTITY_CHANGED:
        return "notify"
    if state in ("SUSPICIOUS", "INITIALIZING"):
        return "verify"
    return "silent"


def evaluate_window(
    events: list[dict],
    engine: dict,
    session_id: str = "unknown",
    recent_scores: deque | None = None,
) -> dict:
    """
    Feature extraction → multi-user LSTM → risk engine → unified payload.
    """
    features = extract_features(events)
    keydown_count = sum(1 for e in events if e.get("event_type") == "keydown")

    enrolled = engine.get("enrolled_username", "")
    prediction = predict_user(features, target_user=enrolled)
    prediction["keystroke_count"] = keydown_count

    auth_event = decide_action(engine, prediction=prediction)

    enrolled_score = auth_event.get("all_scores", {}).get(enrolled, 0.0)
    overall_score = enrolled_score if enrolled else auth_event.get("confidence", 0.0)
    anomaly_confidence = round(1.0 - overall_score, 4) if enrolled else round(
        1.0 - max(auth_event.get("all_scores", {}).values(), default=0.0), 4
    )

    if recent_scores is not None:
        recent_scores.append(overall_score)
    recent_avg = (sum(recent_scores) / len(recent_scores)) if recent_scores else overall_score

    legacy_action = _auth_action_to_legacy(auth_event)

    payload = {
        "type": "score_update",
        "score": round(overall_score, 4),
        "anomaly_confidence": anomaly_confidence,
        "signal_scores": auth_event.get("all_scores", {}),
        "action": legacy_action,
        "message": auth_event.get("message", ""),
        "confidence_reasoning": auth_event.get("message", ""),
        "auth_event": auth_event,
        "prediction": {
            "predicted_user": prediction.get("predicted_user"),
            "confidence": prediction.get("confidence"),
            "method": prediction.get("method"),
        },
        "username": enrolled,
        "timestamp": time.time(),
    }

    if legacy_action == "verify" and GEMINI_API_KEY and "your_" not in GEMINI_API_KEY:
        prompt = build_gemini_prompt(
            overall_score,
            {
                "signal_scores": auth_event.get("all_scores", {}),
                "session_id": session_id,
                "recent_avg_score": recent_avg,
                "state": auth_event.get("state"),
            },
        )
        decision = call_gemini(prompt)
        if decision.get("message"):
            payload["message"] = decision["message"]
        payload["confidence_reasoning"] = decision.get("confidence_reasoning", payload["confidence_reasoning"])
        if decision.get("action") == "lock":
            payload["action"] = "lock"

    return payload


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/register")
async def api_register(req: UserAuthRequest):
    try:
        user = register_user(req.username, req.password)
        return {"status": "ok", "user": user}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/login")
async def api_login(req: UserAuthRequest):
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {"status": "ok", "user": user}


@app.get("/api/enroll/prompts")
async def api_enroll_prompts():
    return {
        "sessions": get_enrollment_session_defs(),
        "prompts": [s["prompt"] for s in ENROLLMENT_SESSIONS if s.get("prompt")],
        "required_sessions": len(ENROLLMENT_SESSIONS),
    }


def _trigger_background_training(epochs: int = 35):
    """Run model training in a background thread with full traceback logging on failure."""
    def _run():
        try:
            from train import train
            print(f"[KeyGuard AI] Background retraining triggered for {epochs} epochs...")
            train(epochs=epochs)
            print("[KeyGuard AI] Background retraining completed successfully.")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[KeyGuard AI] ERROR in background training: {e}\n{tb}", flush=True)

    import threading
    threading.Thread(target=_run, daemon=True).start()


@app.post("/api/enroll/submit")
async def api_enroll_submit(req: EnrollSubmitRequest):
    try:
        count = save_enrollment_session(
            req.username,
            req.session_index,
            req.events,
            session_type=req.session_type,
        )
        if count >= len(ENROLLMENT_SESSIONS):
            sessions = get_enrollment_sessions(req.username)
            profile = learn_user_profile(req.username, sessions)
            invalidate_model_cache()
            _trigger_background_training(epochs=35)
            return {
                "status": "completed",
                "sessions_completed": count,
                "enrolled": True,
                "profile": profile,
            }
        return {
            "status": "in_progress",
            "sessions_completed": count,
            "required_sessions": len(ENROLLMENT_SESSIONS),
            "enrolled": False,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/user/status")
async def api_user_status(username: str):
    user = get_user(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return {
        "username": user["username"],
        "enrolled": user.get("enrolled", False),
        "sessions_completed": len(user.get("enrollment_sessions", [])),
    }


@app.get("/api/profiles")
async def api_list_profiles():
    """Return all registered biometric profiles."""
    profiles = get_registered_profiles()
    return {"status": "ok", "profiles": profiles, "count": len(profiles)}


@app.delete("/api/profile/{username}")
async def api_delete_profile(username: str):
    """Delete a registered profile and retrain/update cache."""
    success = delete_user_profile(username)
    if not success:
        raise HTTPException(status_code=404, detail=f"Profile '{username}' not found.")
    invalidate_model_cache()
    _trigger_background_training(epochs=35)
    return {"status": "ok", "message": f"Profile '{username}' deleted successfully."}


class DeleteProfileRequest(BaseModel):
    username: str


@app.post("/api/profile/delete")
async def api_delete_profile_post(req: DeleteProfileRequest):
    """POST endpoint alternative to delete a registered profile."""
    return await api_delete_profile(req.username)


@app.post("/api/profiles/clear")
async def api_clear_all_profiles():
    """Remove all registered biometric profiles."""
    removed = delete_all_profiles()
    invalidate_model_cache()
    return {"status": "ok", "removed": removed, "message": "All registered profiles removed."}



# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/typing")
async def typing_socket(websocket: WebSocket):
    await websocket.accept()
    query_params = dict(websocket.query_params)
    active_username = query_params.get("username", "")

    session_id = f"session-{id(websocket)}"
    event_buffer: list[dict] = []
    recent_scores: deque = deque(maxlen=10)
    engine = make_engine(enrolled_username=active_username)

    async def push(event: dict):
        await websocket.send_text(json.dumps(event))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                batch = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = batch.get("type", "")

            if msg_type == "set_user":
                active_username = batch.get("username", active_username)
                engine["enrolled_username"] = active_username
                continue

            if msg_type == "clipboard":
                clip = batch.get("clip", batch.get("clipboard", {}))
                auth_event = decide_action(engine, clipboard_event=clip)
                await push({"type": "auth_event", **auth_event})
                continue

            new_events = batch.get("events", [])
            event_buffer.extend(new_events)

            kd_events = [e for e in event_buffer if e.get("event_type") == "keydown"]

            if len(kd_events) >= SEQUENCE_LENGTH:
                window_kd = kd_events[-SEQUENCE_LENGTH:]
                window_events = list(window_kd)
                used_keyup_ids = set()

                for kd in window_kd:
                    k = kd.get("key")
                    t = kd.get("timestamp", 0)
                    for e in event_buffer:
                        if (
                            e.get("event_type") == "keyup"
                            and e.get("key") == k
                            and e.get("timestamp", 0) >= t
                            and id(e) not in used_keyup_ids
                        ):
                            used_keyup_ids.add(id(e))
                            window_events.append(e)
                            break

                window_events.sort(key=lambda e: e.get("timestamp", 0))

                result = evaluate_window(
                    window_events,
                    engine=engine,
                    session_id=session_id,
                    recent_scores=recent_scores,
                )
                await push(result)

                if result.get("auth_event"):
                    await push({"type": "auth_event", **result["auth_event"]})

                # Slide window by stride
                STRIDE = 15
                if len(kd_events) >= SEQUENCE_LENGTH + STRIDE:
                    cut_kd = kd_events[-SEQUENCE_LENGTH + STRIDE]
                    cut_idx = next((i for i, e in enumerate(event_buffer) if e is cut_kd), 0)
                    event_buffer = event_buffer[cut_idx:]

    except WebSocketDisconnect:
        pass


@app.get("/health")
async def health():
    from model import load_model
    model, meta = load_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "enrolled_users": meta.get("enrolled_users", []) if meta else [],
    }


@app.get("/")
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path)
