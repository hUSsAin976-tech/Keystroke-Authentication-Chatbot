"""
risk_engine.py
--------------
Pure risk-assessment logic for KeyGuard AI.
No ML, no web-framework, no UI code.

Responsibilities
----------------
1. Accept a stream of per-window LSTM predictions.
2. Apply exponential-weighted temporal smoothing over a configurable
   rolling history so that one anomalous window does not cause
   an immediate authentication failure.
3. Maintain a per-session state machine:
       INITIALIZING → VERIFIED
                    → SUSPICIOUS
                    → IDENTITY_CHANGED
                    → UNKNOWN
4. Detect clipboard events and log them without altering auth state.
5. Return structured activity-event dicts for the frontend activity log.

Thresholds (all configurable at engine-creation time)
------------------------------------------------------
smoothing_window        int     Number of past windows used for smoothing  (default 6)
confidence_threshold    float   Smoothed score ≥ this → VERIFIED            (default 0.65)
unknown_threshold       float   Smoothed top-1 < this → UNKNOWN             (default 0.50)
consecutive_mismatch    int     Windows before escalating to IDENTITY_CHANGED/UNKNOWN (default 4)
suspicious_threshold    float   Smoothed score < confidence_threshold but ≥ this → SUSPICIOUS (default 0.45)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

STATE_INITIALIZING    = "INITIALIZING"
STATE_VERIFIED        = "VERIFIED"
STATE_SUSPICIOUS      = "SUSPICIOUS"
STATE_IDENTITY_CHANGED = "IDENTITY_CHANGED"
STATE_UNKNOWN         = "UNKNOWN"

# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

_D_SMOOTH_WIN   = 6
_D_CONF         = 0.65
_D_UNKNOWN      = 0.50
_D_CONSEC       = 4
_D_SUSPICIOUS   = 0.45


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_engine(
    enrolled_username: str = "",
    smoothing_window: int   = _D_SMOOTH_WIN,
    confidence_threshold: float = _D_CONF,
    unknown_threshold: float    = _D_UNKNOWN,
    consecutive_mismatch: int   = _D_CONSEC,
    suspicious_threshold: float = _D_SUSPICIOUS,
) -> dict[str, Any]:
    """
    Create a fresh risk-engine state dict for one WebSocket session.
    Supports both single-user verification and open continuous identification.
    """
    return {
        "enrolled_username":   enrolled_username,
        "last_identified_user": enrolled_username or "",
        "smoothing_window":    smoothing_window,
        "confidence_threshold": confidence_threshold,
        "unknown_threshold":   unknown_threshold,
        "consecutive_mismatch": consecutive_mismatch,
        "suspicious_threshold": suspicious_threshold,
        # Rolling history of raw LSTM predictions
        "history": deque(maxlen=smoothing_window),
        "state":           STATE_INITIALIZING,
        "mismatch_count":  0,
        "windows_analyzed": 0,
    }


def temporal_smoother(history: list[dict]) -> dict:
    """Exponential-weighted average of per-user LSTM scores."""
    return _smooth(history)


def identity_change_detector(engine: dict[str, Any], top_user: str, top_conf: float) -> str:
    """Return next auth state after a mismatched window."""
    enrolled = engine["enrolled_username"]
    if top_conf < engine["unknown_threshold"]:
        engine["mismatch_count"] += 1
        return (
            STATE_UNKNOWN
            if engine["mismatch_count"] >= engine["consecutive_mismatch"]
            else STATE_SUSPICIOUS
        )
    if top_user == enrolled and top_conf >= engine["confidence_threshold"]:
        engine["mismatch_count"] = 0
        return STATE_VERIFIED
    if top_user == enrolled and top_conf >= engine["suspicious_threshold"]:
        engine["mismatch_count"] = max(0, engine["mismatch_count"] - 1)
        return STATE_SUSPICIOUS
    engine["mismatch_count"] += 1
    if engine["mismatch_count"] >= engine["consecutive_mismatch"]:
        return STATE_IDENTITY_CHANGED
    return STATE_SUSPICIOUS


def decide_action(
    engine: dict[str, Any],
    prediction: dict[str, Any] | None = None,
    clipboard_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Main entry: feed LSTM prediction or clipboard event → activity event dict."""
    return process_prediction(engine, prediction or {}, clipboard_event)


def process_prediction(
    engine: dict[str, Any],
    prediction: dict[str, Any],
    clipboard_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Feed one LSTM window result through the risk engine.

    prediction (from model.predict_user)
    ─────────────────────────────────────
    {
        "predicted_user": str,          # top-1 username or "unknown"
        "confidence":     float,        # raw top-1 softmax score
        "all_scores":     {user: float},
        "keystroke_count": int,
        "method":         "lstm" | "profile" | "none",
    }

    clipboard_event (optional)
    ────────────────────────────
    {
        "type":       "paste" | "copy" | "cut",
        "char_count": int,
    }

    Returns activity_event dict (see _make_event).
    """
    ts = time.time()

    # Clipboard events are logged without affecting auth state
    if clipboard_event:
        return _make_clipboard_event(clipboard_event, engine, ts)

    engine["windows_analyzed"] += 1
    engine["history"].append(prediction)
    history = list(engine["history"])

    # Not enough data yet
    if len(history) < 2:
        return _make_event(
            engine, STATE_INITIALIZING,
            engine["enrolled_username"], 0.0, ts,
            all_scores=prediction.get("all_scores", {}),
            keystroke_count=prediction.get("keystroke_count", 0),
        )

    smoothed = _smooth(history)
    top_user = smoothed["top_user"]
    top_conf = smoothed["top_confidence"]
    enrolled = engine.get("enrolled_username", "")
    last_id  = engine.get("last_identified_user", "")

    # ── Rejection gate ──────────────────────────────────────────────────────
    # If no enrolled user clears the unknown_threshold → UNKNOWN / UNAUTHORIZED
    if top_conf < engine["unknown_threshold"] or top_user == "unknown":
        engine["mismatch_count"] += 1
        new_state = (
            STATE_UNKNOWN
            if engine["mismatch_count"] >= engine["consecutive_mismatch"]
            else STATE_SUSPICIOUS
        )
        engine["state"] = new_state
        msg = (
            "Unauthorized typist detected — pattern does not match any enrolled user"
            if new_state == STATE_UNKNOWN
            else "Typing cadence divergence — analyzing typist pattern…"
        )
        return _make_event(
            engine, new_state, "Unauthorized Typist", top_conf, ts,
            all_scores=smoothed["all_scores"],
            keystroke_count=prediction.get("keystroke_count", 0),
            is_authorized=False,
            custom_message=msg,
        )

    # ── OPEN IDENTIFICATION MODE (No single user pre-selected) ─────────────
    if not enrolled:
        # Check if typist shifted from previously identified user
        if last_id and last_id != top_user:
            engine["mismatch_count"] += 1
            if engine["mismatch_count"] >= engine["consecutive_mismatch"]:
                engine["state"] = STATE_IDENTITY_CHANGED
                engine["last_identified_user"] = top_user
                msg = f"Identity shift detected: active typist matches @{top_user}"
                return _make_event(
                    engine, STATE_IDENTITY_CHANGED, top_user, top_conf, ts,
                    all_scores=smoothed["all_scores"],
                    keystroke_count=prediction.get("keystroke_count", 0),
                    is_authorized=True,
                    custom_message=msg,
                )
            else:
                engine["state"] = STATE_SUSPICIOUS
                return _make_event(
                    engine, STATE_SUSPICIOUS, top_user, top_conf, ts,
                    all_scores=smoothed["all_scores"],
                    keystroke_count=prediction.get("keystroke_count", 0),
                    is_authorized=True,
                    custom_message=f"Possible typist transition towards @{top_user}…",
                )

        # High confidence match to an enrolled user
        if top_conf >= engine["confidence_threshold"]:
            engine["mismatch_count"] = 0
            engine["state"] = STATE_VERIFIED
            engine["last_identified_user"] = top_user
            return _make_event(
                engine, STATE_VERIFIED, top_user, top_conf, ts,
                all_scores=smoothed["all_scores"],
                keystroke_count=prediction.get("keystroke_count", 0),
                is_authorized=True,
                custom_message=f"Authorized user verified: @{top_user}",
            )

        # Borderline confidence
        if top_conf >= engine["suspicious_threshold"]:
            engine["mismatch_count"] = max(0, engine["mismatch_count"] - 1)
            engine["state"] = STATE_SUSPICIOUS
            return _make_event(
                engine, STATE_SUSPICIOUS, top_user, top_conf, ts,
                all_scores=smoothed["all_scores"],
                keystroke_count=prediction.get("keystroke_count", 0),
                is_authorized=True,
                custom_message=f"Typing rhythm resembles @{top_user} with moderate confidence",
            )

    # ── TARGETED ENROLLED USER MODE ─────────────────────────────────────────
    if top_user == enrolled and top_conf >= engine["confidence_threshold"]:
        engine["mismatch_count"] = 0
        engine["state"] = STATE_VERIFIED
        return _make_event(
            engine, STATE_VERIFIED, enrolled, top_conf, ts,
            all_scores=smoothed["all_scores"],
            keystroke_count=prediction.get("keystroke_count", 0),
            is_authorized=True,
        )

    if top_user == enrolled and top_conf >= engine["suspicious_threshold"]:
        engine["mismatch_count"] = max(0, engine["mismatch_count"] - 1)
        engine["state"] = STATE_SUSPICIOUS
        return _make_event(
            engine, STATE_SUSPICIOUS, enrolled, top_conf, ts,
            all_scores=smoothed["all_scores"],
            keystroke_count=prediction.get("keystroke_count", 0),
            is_authorized=True,
        )

    engine["mismatch_count"] += 1
    if engine["mismatch_count"] >= engine["consecutive_mismatch"]:
        engine["state"] = STATE_IDENTITY_CHANGED
        return _make_event(
            engine, STATE_IDENTITY_CHANGED, top_user, top_conf, ts,
            all_scores=smoothed["all_scores"],
            keystroke_count=prediction.get("keystroke_count", 0),
            is_authorized=True,
            custom_message=f"Identity shift: cadence resembles @{top_user} instead of @{enrolled}",
        )
    else:
        engine["state"] = STATE_SUSPICIOUS
        return _make_event(
            engine, STATE_SUSPICIOUS, top_user, top_conf, ts,
            all_scores=smoothed["all_scores"],
            keystroke_count=prediction.get("keystroke_count", 0),
            is_authorized=True,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _smooth(history: list[dict]) -> dict:
    """
    Exponential-weighted average of per-user scores across the history window.
    More recent windows are weighted higher (decay = 0.85).
    """
    all_users: set[str] = set()
    for pred in history:
        all_users.update(pred.get("all_scores", {}).keys())
        pu = pred.get("predicted_user", "")
        if pu and pu != "unknown":
            all_users.add(pu)

    if not all_users:
        return {"top_user": "unknown", "top_confidence": 0.0, "all_scores": {}}

    n = len(history)
    decay = 0.85
    smoothed: dict[str, float] = {}

    for user in all_users:
        w_sum, w_total = 0.0, 0.0
        for i, pred in enumerate(history):
            w = decay ** (n - 1 - i)          # older windows have lower weight
            score = pred.get("all_scores", {}).get(user, 0.0)
            w_sum   += w * score
            w_total += w
        smoothed[user] = w_sum / w_total if w_total > 0 else 0.0

    top_user = max(smoothed, key=lambda u: smoothed[u])
    return {
        "top_user":        top_user,
        "top_confidence":  round(smoothed[top_user], 4),
        "all_scores":      {u: round(v, 4) for u, v in smoothed.items()},
    }


# Action and message maps
_ACTIONS = {
    STATE_INITIALIZING:    "continue",
    STATE_VERIFIED:        "continue",
    STATE_SUSPICIOUS:      "continue",
    STATE_IDENTITY_CHANGED: "notify",
    STATE_UNKNOWN:         "notify",
}

_MESSAGES = {
    STATE_INITIALIZING:    "Analyzing keystroke cadence…",
    STATE_VERIFIED:        "Authorized user verified",
    STATE_SUSPICIOUS:      "Cadence variance detected — evaluating next window",
    STATE_IDENTITY_CHANGED: "Identity change detected",
    STATE_UNKNOWN:         "Unauthorized typist — pattern does not match any registered user",
}

_EVENT_TYPES = {
    STATE_INITIALIZING:    "init",
    STATE_VERIFIED:        "verify",
    STATE_SUSPICIOUS:      "suspicious",
    STATE_IDENTITY_CHANGED: "identity_change",
    STATE_UNKNOWN:         "unknown",
}


def _make_event(
    engine: dict,
    state: str,
    display_user: str,
    confidence: float,
    ts: float,
    all_scores: dict | None = None,
    keystroke_count: int = 0,
    is_authorized: bool = True,
    custom_message: str | None = None,
) -> dict:
    return {
        "type":             "auth_event",
        "state":            state,
        "display_user":     display_user,
        "is_authorized":    is_authorized,
        "confidence":       round(confidence, 4),
        "windows_analyzed": engine["windows_analyzed"],
        "mismatch_count":   engine["mismatch_count"],
        "action":           _ACTIONS.get(state, "continue"),
        "event_type":       _EVENT_TYPES.get(state, "info"),
        "message":          custom_message or _MESSAGES.get(state, ""),
        "timestamp":        ts,
        "all_scores":       all_scores or {},
        "keystroke_count":  keystroke_count,
        "enrolled_username": engine.get("enrolled_username", ""),
        "last_identified_user": engine.get("last_identified_user", ""),
    }


def _make_clipboard_event(
    clip: dict,
    engine: dict,
    ts: float,
) -> dict:
    clip_type  = clip.get("type", "clipboard")
    char_count = clip.get("char_count", 0)
    labels = {"paste": "📋 Paste", "copy": "📋 Copy", "cut": "✂ Cut"}
    label  = labels.get(clip_type, "📋 Clipboard")
    return {
        "type":             "auth_event",
        "state":            engine["state"],   # auth state UNCHANGED by clipboard
        "display_user":     engine["enrolled_username"],
        "confidence":       None,
        "windows_analyzed": engine["windows_analyzed"],
        "mismatch_count":   engine["mismatch_count"],
        "action":           "continue",
        "event_type":       "clipboard",
        "message":          f"{label} detected — {char_count} characters",
        "timestamp":        ts,
        "all_scores":       {},
        "keystroke_count":  0,
        "enrolled_username": engine["enrolled_username"],
    }
