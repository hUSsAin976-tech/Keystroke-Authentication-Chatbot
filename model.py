"""
model.py
--------
Pure ML logic for KeyGuard AI.
No web-framework imports, no chatbot code, no knowledge of server.py or the frontend.

This module owns:
  1. Feature extraction from raw keystroke events (6 features)
  2. Sliding-window sequence construction for training
  3. Normalization (fit-on-train, apply everywhere)
  4. Model loading / caching (thread-safe)
  5. Multi-user LSTM inference with unknown-rejection gate
  6. Statistical-profile fallback when no LSTM model is trained yet
  7. Per-user profile learning (used as the fallback scorer)
  8. The SIGNALS registry for backward-compat with server.py

─────────────────────────────────────────────────────────────────
6-FEATURE VECTOR (per keystroke / keydown event)
─────────────────────────────────────────────────────────────────
  [0] dwell_time        keyup(A) − keydown(A)                    (seconds)
  [1] flight_time       keydown(B) − keyup(A)                    (seconds, can be < 0 for overlapping keys)
  [2] transition_time   keydown(B) − keydown(A) = dwell+flight   (seconds)
  [3] pause_duration    flight if flight > PAUSE_THRESHOLD else 0 (seconds)
  [4] correction_flag   1.0 if key is Backspace/Delete, 0.0 otherwise
  [5] rhythm_variability rolling std of last RHYTHM_WINDOW dwell times (seconds)

Features [0]–[2] capture absolute timing; the normalisation scaler
brings all users onto the same numeric range while preserving
inter-user behavioral differences (the LSTM learns those differences).
Feature [5] captures whether a user is a consistent or erratic typist —
a behavioral trait that is largely independent of raw typing speed.

─────────────────────────────────────────────────────────────────
SEQUENCE FORMAT
─────────────────────────────────────────────────────────────────
  Shape: (SEQUENCE_LENGTH, FEATURE_DIM) = (50, 6)
  Shorter sequences are zero-padded at the front.
  Longer sequences keep the most recent 50 rows.

  Sliding window (for training / continuous inference):
    stride = SEQUENCE_LENGTH − OVERLAP  (default: 25 keystrokes)

─────────────────────────────────────────────────────────────────
MULTI-USER LSTM
─────────────────────────────────────────────────────────────────
  Input  : (batch, 50, 6)
  LSTM   : 2 layers, 128 hidden units, dropout 0.35
  Output : (batch, num_enrolled_users)  → softmax
  Rejection gate: if max(softmax) < UNKNOWN_REJECTION_THRESHOLD → "unknown"

  Model checkpoint   : model_multiuser.pt   (state_dict only)
  Metadata           : model_multiuser_meta.json
      {
          "enrolled_users": ["alice", "bob", ...],
          "scaler": {"mean": [...], "std": [...]},
          "sequence_length": 50,
          "feature_dim": 6,
      }

  When the checkpoint does not exist, predict_user() falls back to
  per-user statistical profile matching (z-score distance).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 50          # keystrokes per window
FEATURE_DIM     = 6           # features per keystroke
PAUSE_THRESHOLD = 0.80        # seconds: flight time above this is a "pause"
RHYTHM_WINDOW   = 5           # number of recent dwell times used for rolling std

MULTIUSER_MODEL_PATH = os.environ.get("MULTIUSER_MODEL_PATH", "model_multiuser.pt")
MULTIUSER_META_PATH  = os.environ.get("MULTIUSER_META_PATH",  "model_multiuser_meta.json")
UNKNOWN_REJECTION_THRESHOLD = float(os.environ.get("UNKNOWN_REJECTION_THRESHOLD", "0.50"))

_CORRECTION_KEYS = {"Backspace", "Delete"}


# ---------------------------------------------------------------------------
# 1. Feature extraction
# ---------------------------------------------------------------------------

def extract_features(keystroke_events: list[dict]) -> np.ndarray:
    """
    Convert a raw list of {key, event_type, timestamp} dicts into a
    fixed-length feature matrix of shape (SEQUENCE_LENGTH, FEATURE_DIM).

    event_type must be "keydown" or "keyup".
    timestamp is in milliseconds (float).

    The resulting matrix is zero-padded at the front when fewer than
    SEQUENCE_LENGTH keystrokes are present, and truncated to the most
    recent SEQUENCE_LENGTH keystrokes when there are more.
    """
    if not keystroke_events:
        return np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

    events = sorted(keystroke_events, key=lambda e: e.get("timestamp", 0))

    # Per-key stacks for matching keydown ↔ keyup
    pending_keydowns: dict[str, list[float]] = {}
    # Ordered list of keydown timestamps (one per row in `rows`)
    keydown_ts_list: list[float] = []
    last_keyup_ts:   float | None = None
    last_keydown_ts: float | None = None
    dwell_history:   list[float]  = []   # for rhythm_variability

    # rows[i] = [dwell, flight, transition, pause, correction, rhythm_var]
    rows: list[list[float]] = []

    for event in events:
        key   = event.get("key", "")
        etype = event.get("event_type", "")
        ts    = float(event.get("timestamp", 0.0)) / 1000.0   # ms → s

        if etype == "keydown":
            pending_keydowns.setdefault(key, []).append(ts)
            keydown_ts_list.append(ts)

            # Feature 1 (flight_time)
            flight_time = (ts - last_keyup_ts) if last_keyup_ts is not None else 0.0

            # Feature 2 (transition_time)
            transition_time = (ts - last_keydown_ts) if last_keydown_ts is not None else 0.0

            # Feature 3 (pause_duration)
            pause_duration = max(flight_time, 0.0) if flight_time > PAUSE_THRESHOLD else 0.0

            # Feature 4 (correction_flag)
            correction_flag = 1.0 if key in _CORRECTION_KEYS else 0.0

            # Feature 5 (rhythm_variability — rolling std of recent dwells)
            if len(dwell_history) >= 2:
                recent = dwell_history[-RHYTHM_WINDOW:]
                rhythm_var = float(np.std(recent)) if len(recent) >= 2 else 0.0
            else:
                rhythm_var = 0.0

            last_keydown_ts = ts

            # Feature 0 (dwell_time) is filled once the matching keyup arrives.
            rows.append([0.0, flight_time, transition_time,
                         pause_duration, correction_flag, rhythm_var])

        elif etype == "keyup":
            stack = pending_keydowns.get(key)
            if stack:
                last_keyup_ts = ts
                down_ts    = stack.pop(0)
                dwell_time = max(ts - down_ts, 0.0)
                dwell_history.append(dwell_time)

                # Back-fill the dwell_time in the matching row.
                # Match by keydown timestamp stored in keydown_ts_list.
                for row_idx in range(len(rows) - 1, -1, -1):
                    if (rows[row_idx][0] == 0.0 and
                            row_idx < len(keydown_ts_list) and
                            abs(keydown_ts_list[row_idx] - down_ts) < 1e-9):
                        rows[row_idx][0] = dwell_time
                        break

    features = np.array(rows, dtype=np.float32)

    if features.shape[0] == 0:
        return np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

    # Pad (front) or truncate (keep tail)
    if features.shape[0] < SEQUENCE_LENGTH:
        pad = np.zeros(
            (SEQUENCE_LENGTH - features.shape[0], FEATURE_DIM), dtype=np.float32
        )
        features = np.vstack([pad, features])
    else:
        features = features[-SEQUENCE_LENGTH:]

    return features


# ---------------------------------------------------------------------------
# 2. Sliding-window sequence builder (for training)
# ---------------------------------------------------------------------------

def build_sequences(
    sessions_events: list[list[dict]],
    sequence_length: int = SEQUENCE_LENGTH,
    overlap: int = 35,
) -> np.ndarray:
    """
    Build sliding-window sequences from multiple sessions of raw events.

    Returns np.ndarray of shape (N, sequence_length, FEATURE_DIM).
    N depends on how many keystrokes each session contains.

    The sliding window advances by (sequence_length − overlap) keystrokes
    at a time, so each window shares `overlap` keystrokes with the next.
    Window construction pairs each of the exactly sequence_length keydowns
    with its matching keyup, preventing extra keystrokes from leaking into
    the window.
    """
    stride    = max(sequence_length - overlap, 1)
    sequences = []

    for session_events in sessions_events:
        if not session_events:
            continue

        events = sorted(session_events, key=lambda e: e.get("timestamp", 0))

        # Separate keydown events — each one becomes a row in extract_features
        keydown_events = [e for e in events if e.get("event_type") == "keydown"]

        if len(keydown_events) < sequence_length:
            # Session too short — use all keystrokes as one padded sequence
            sequences.append(extract_features(session_events))
            continue

        # Slide over keydown events
        for start in range(0, len(keydown_events) - sequence_length + 1, stride):
            window_kd = keydown_events[start: start + sequence_length]
            window_events = list(window_kd)
            used_keyup_ids = set()
            for kd in window_kd:
                k = kd.get("key")
                t = kd.get("timestamp", 0)
                for e in events:
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
            sequences.append(extract_features(window_events))

    if not sequences:
        return np.zeros((0, sequence_length, FEATURE_DIM), dtype=np.float32)

    return np.array(sequences, dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. Normalization (fit on training data; apply consistently everywhere)
# ---------------------------------------------------------------------------

def fit_scaler(X: np.ndarray) -> dict:
    """
    Compute per-feature mean and std from X of shape (N, seq_len, features).

    Only non-zero rows (non-padding rows) contribute to the statistics
    so that zero-padding does not skew the normalization.

    Returns a dict suitable for JSON serialization.
    """
    flat = X.reshape(-1, X.shape[-1])   # (N*seq_len, features)
    # Mask out padding rows (all zeros)
    non_pad = flat[np.any(flat != 0, axis=1)]
    if non_pad.shape[0] == 0:
        non_pad = flat
    mean = non_pad.mean(axis=0)
    std  = non_pad.std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)   # prevent division-by-zero
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_scaler(X: np.ndarray, scaler: dict) -> np.ndarray:
    """
    Apply a pre-fitted StandardScaler to X.
    Preserves zero rows (padding) by masking them out.
    """
    mean = np.array(scaler["mean"], dtype=np.float32)
    std  = np.array(scaler["std"],  dtype=np.float32)
    norm = (X - mean) / std
    # Re-zero the padding rows
    pad_mask = np.all(X == 0, axis=-1, keepdims=True)
    return np.where(pad_mask, 0.0, norm).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. PyTorch LSTM definition
# ---------------------------------------------------------------------------

def _build_lstm(num_users: int,
                input_size:  int   = FEATURE_DIM,
                hidden_size: int   = 128,
                num_layers:  int   = 2,
                dropout:     float = 0.35):
    """
    Returns an uninitialized KeystrokeLSTM nn.Module.
    Defined as a local class so the function is importable without PyTorch
    being required at module-load time.
    """
    import torch.nn as nn

    class KeystrokeLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size  = input_size,
                hidden_size = hidden_size,
                num_layers  = num_layers,
                batch_first = True,
                dropout     = dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_size, num_users)

        def forward(self, x):
            out, _ = self.lstm(x)           # (batch, seq, hidden)
            out    = self.dropout(out[:, -1, :])  # last time-step
            return self.fc(out)             # raw logits (batch, num_users)

    return KeystrokeLSTM()


# ---------------------------------------------------------------------------
# 5. Model cache (thread-safe)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}
_META_CACHE:  dict = {}
_CACHE_LOCK         = threading.Lock()


def _load_multiuser_model():
    """Load and cache the multi-user LSTM + metadata. Returns (model, meta) or (None, None)."""
    import torch

    mp = MULTIUSER_MODEL_PATH
    jp = MULTIUSER_META_PATH

    if not (os.path.exists(mp) and os.path.exists(jp)):
        return None, None

    with _CACHE_LOCK:
        if mp in _MODEL_CACHE:
            return _MODEL_CACHE[mp], _META_CACHE[jp]

        with open(jp, "r", encoding="utf-8") as f:
            meta = json.load(f)

        num_users = len(meta["enrolled_users"])
        model     = _build_lstm(num_users,
                                input_size  = meta.get("feature_dim",     FEATURE_DIM),
                                hidden_size = meta.get("hidden_size",     128),
                                num_layers  = meta.get("num_layers",      2),
                                dropout     = meta.get("dropout",         0.35))
        try:
            state = torch.load(mp, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(mp, map_location="cpu")

        model.load_state_dict(state)
        model.eval()

        _MODEL_CACHE[mp] = model
        _META_CACHE[jp]  = meta
        return model, meta


def invalidate_model_cache():
    """Call after retraining so the next inference reloads from disk."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        _META_CACHE.clear()


# ---------------------------------------------------------------------------
# 6. Inference — multi-user LSTM
# ---------------------------------------------------------------------------

def predict_user(features: np.ndarray, target_user: str | None = None) -> dict:
    """
    Run continuous keystroke verification on a (SEQUENCE_LENGTH, FEATURE_DIM) window.
    Supports both multi-user LSTM inference and statistical profile matching.

    Parameters
    ----------
    features : np.ndarray
        Shape (SEQUENCE_LENGTH, FEATURE_DIM).
    target_user : str | None
        If provided, ensures this enrolled user's score is evaluated against their
        enrolled profile even if not yet in the active LSTM checkpoint.

    Returns
    -------
    {
        "predicted_user": str,          # username or "unknown"
        "confidence":     float,        # top-1 smoothed score [0, 1]
        "all_scores":     {user: float},
        "method":         "lstm" | "profile" | "none",
    }
    """
    active = features[np.any(features != 0, axis=1)]
    if len(active) < 4:
        return {
            "predicted_user": "unknown",
            "confidence": 0.0,
            "all_scores": {},
            "method": "none",
        }

    try:
        from users import get_all_enrolled_users
        enrolled_db = get_all_enrolled_users()
    except Exception:
        enrolled_db = {}

    model, meta = _load_multiuser_model()
    if model is not None and meta is not None:
        return _predict_lstm(features, model, meta, enrolled_db=enrolled_db, target_user=target_user)
    return _predict_statistical(features, enrolled_db=enrolled_db, target_user=target_user)


def _predict_lstm(
    features: np.ndarray,
    model,
    meta: dict,
    enrolled_db: dict | None = None,
    target_user: str | None = None,
) -> dict:
    import torch
    import torch.nn.functional as F

    active = features[np.any(features != 0, axis=1)]
    enrolled_lstm = meta["enrolled_users"]
    scaler = meta.get("scaler")

    feats = apply_scaler(features[np.newaxis], scaler)[0] if scaler else features
    batch = torch.from_numpy(feats[np.newaxis]).float()   # (1, seq_len, 6)

    with torch.no_grad():
        logits = model(batch)                              # (1, num_users)
        probs  = F.softmax(logits, dim=-1).squeeze(0)     # (num_users,)

    all_scores = {}
    for i, u in enumerate(enrolled_lstm):
        all_scores[u] = round(float(probs[i]), 4)

    # Supplement with statistical profile ONLY for users in DB not in the LSTM model
    if enrolled_db:
        for u, profile in enrolled_db.items():
            if not profile:
                continue
            if u not in all_scores and len(active) >= 4:
                all_scores[u] = round(_profile_similarity(active, profile), 4)

    if not all_scores:
        return {
            "predicted_user": "unknown",
            "confidence": 0.0,
            "all_scores": {},
            "method": "lstm",
        }

    top_user = max(all_scores, key=all_scores.get)
    top_conf = all_scores[top_user]
    method = "lstm" if top_user in enrolled_lstm else "profile"

    # Secondary verification: Impostor / Anomaly Rejection
    # 1. Softmax confidence below rejection threshold (model is uncertain)
    is_rejected = (top_conf < UNKNOWN_REJECTION_THRESHOLD)

    # 2. Impostor / Anomaly verification against predicted user's profile
    if not is_rejected and len(active) >= 4:
        active_dwells = active[active[:, 0] > 0, 0]
        active_flights = active[active[:, 1] > 0, 1]
        med_dwell = float(np.median(active_dwells)) if len(active_dwells) > 0 else 0.0
        med_flight = float(np.median(active_flights)) if len(active_flights) > 0 else 0.0

        # Physical impossibility check (rejects automated bots & mechanical script injection)
        if med_dwell < 0.025 or med_dwell > 2.0 or med_flight < 0.015:
            is_rejected = True
        elif enrolled_db and top_user in enrolled_db and enrolled_db[top_user]:
            # Verify that the active cadence fits the predicted user's biometric profile
            user_sim = _profile_similarity(active, enrolled_db[top_user])
            if user_sim < 0.35:
                is_rejected = True

    if is_rejected:
        return {
            "predicted_user": "unknown",
            "confidence":     round(top_conf, 4),
            "all_scores":     {u: round(v, 4) for u, v in all_scores.items()},
            "method":         method,
        }

    return {
        "predicted_user": top_user,
        "confidence":     round(top_conf, 4),
        "all_scores":     {u: round(v, 4) for u, v in all_scores.items()},
        "method":         method,
    }


def _predict_statistical(
    features: np.ndarray,
    enrolled_db: dict | None = None,
    target_user: str | None = None,
) -> dict:
    """
    Fallback scorer when no trained LSTM checkpoint exists.
    Computes z-score distance between the incoming window and each
    enrolled user's statistical profile.
    """
    if enrolled_db is None:
        try:
            from users import get_all_enrolled_users
            enrolled_db = get_all_enrolled_users()
        except Exception:
            enrolled_db = {}

    if not enrolled_db:
        return {"predicted_user": "unknown", "confidence": 0.0,
                "all_scores": {}, "method": "none"}

    active = features[np.any(features != 0, axis=1)]
    if len(active) < 4:
        return {"predicted_user": "unknown", "confidence": 0.0,
                "all_scores": {}, "method": "profile"}

    all_scores = {}
    for username, profile in enrolled_db.items():
        if profile:
            all_scores[username] = _profile_similarity(active, profile)

    if not all_scores:
        return {"predicted_user": "unknown", "confidence": 0.0,
                "all_scores": {}, "method": "profile"}

    top_user = max(all_scores, key=all_scores.get)
    top_conf = all_scores[top_user]

    if top_conf < UNKNOWN_REJECTION_THRESHOLD:
        return {
            "predicted_user": "unknown",
            "confidence":     round(top_conf, 4),
            "all_scores":     {u: round(v, 4) for u, v in all_scores.items()},
            "method":         "profile",
        }

    return {
        "predicted_user": top_user,
        "confidence":     round(top_conf, 4),
        "all_scores":     {u: round(v, 4) for u, v in all_scores.items()},
        "method":         "profile",
    }


def _profile_similarity(active: np.ndarray, profile: dict) -> float:
    """Robust Gaussian similarity score between current window and a user's profile."""
    cur_dwells  = active[active[:, 0] > 0, 0]
    cur_flights = active[active[:, 1] > 0, 1]

    if len(cur_dwells) == 0 or len(cur_flights) == 0:
        return 0.5

    cur_dwell  = float(np.median(cur_dwells))
    cur_flight = float(np.median(cur_flights))

    p_dwell  = profile.get("dwell_median", profile.get("dwell_mean", 0.12))
    p_flight = profile.get("flight_median", profile.get("flight_mean", 0.35))

    p_dwell_scale  = max(profile.get("dwell_std", 0.04), 0.03) * 2.0
    p_flight_scale = max(profile.get("flight_std", 0.25), 0.20) * 2.0

    z_dwell  = abs(cur_dwell  - p_dwell)  / p_dwell_scale
    z_flight = abs(cur_flight - p_flight) / p_flight_scale

    dist_sq = float(z_dwell**2 + z_flight**2)
    score = float(np.exp(-0.5 * dist_sq))
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 7. Profile learning (statistical fallback; also stored for display)
# ---------------------------------------------------------------------------

def learn_user_profile(username: str, sessions_events: list[list[dict]]) -> dict:
    """
    Compute and persist a statistical biometric profile from the user's
    five enrollment sessions. Used as the inference fallback until the
    multi-user LSTM is retrained.
    """
    from users import complete_enrollment

    all_dwells:      list[float] = []
    all_flights:     list[float] = []
    all_transitions: list[float] = []
    all_pauses:      list[float] = []
    correction_count = 0
    total_keys       = 0

    for events in sessions_events:
        if not events:
            continue
        feats  = extract_features(events)
        active = feats[np.any(feats != 0, axis=1)]
        if len(active) == 0:
            continue

        dwells      = active[active[:, 0] > 0, 0]
        flights     = active[:, 1]   # flight can be negative
        transitions = active[active[:, 2] > 0, 2]
        pauses      = active[active[:, 3] > 0, 3]
        corrections = active[:, 4]

        all_dwells.extend(dwells.tolist())
        all_flights.extend(flights.tolist())
        all_transitions.extend(transitions.tolist())
        all_pauses.extend(pauses.tolist())
        correction_count += int(corrections.sum())
        total_keys       += len(active)

    def _arr(lst, fallback):
        return np.array(lst, dtype=np.float64) if lst else np.array([fallback])

    dw_arr  = _arr(all_dwells,      0.10)
    fl_arr  = _arr(all_flights,     0.14)
    tr_arr  = _arr(all_transitions, 0.24)

    profile = {
        # Dwell
        "dwell_mean":        float(np.mean(dw_arr)),
        "dwell_std":         float(max(np.std(dw_arr),  0.018)),
        "dwell_median":      float(np.median(dw_arr)),
        # Flight
        "flight_mean":       float(np.mean(fl_arr)),
        "flight_std":        float(max(np.std(fl_arr),  0.025)),
        "flight_median":     float(np.median(fl_arr)),
        # Transition
        "transition_mean":   float(np.mean(tr_arr)),
        "transition_std":    float(max(np.std(tr_arr),  0.030)),
        # Pause
        "pause_rate":        len(all_pauses) / max(total_keys, 1),
        "pause_mean":        float(np.mean(np.array(all_pauses))) if all_pauses else 0.0,
        # Corrections
        "correction_rate":   correction_count / max(total_keys, 1),
        # Meta
        "samples_count":     len(all_dwells),
    }

    scaler = fit_scaler(build_sequences(sessions_events))
    complete_enrollment(username, profile, scaler=scaler)
    return profile


# ---------------------------------------------------------------------------
# 8. Aliases & evaluation
# ---------------------------------------------------------------------------

normalize_features = apply_scaler
fit_normalize = fit_scaler


def load_model():
    """Return (model, meta) or (None, None)."""
    return _load_multiuser_model()


def predict(features: np.ndarray, target_user: str | None = None) -> dict:
    """Alias for predict_user()."""
    return predict_user(features, target_user=target_user)


def evaluate_model(
    X: np.ndarray,
    y_labels: list[str],
    enrolled_users: list[str],
    unknown_threshold: float = UNKNOWN_REJECTION_THRESHOLD,
) -> dict:
    """
    Compute FAR, FRR, and approximate EER on labeled sequences.
    y_labels[i] is the true username (or 'unknown' for impostor samples).
    """
    if len(X) == 0:
        return {"far": 0.0, "frr": 0.0, "eer": 0.0, "accuracy": 0.0, "n": 0}

    genuine_total = impostor_total = 0
    false_accepts = false_rejects = correct = 0
    scores_genuine: list[float] = []
    scores_impostor: list[float] = []

    for i, feats in enumerate(X):
        pred = predict_user(feats)
        true_user = y_labels[i]
        pred_user = pred.get("predicted_user", "unknown")
        conf = pred.get("confidence", 0.0)
        enrolled_match = pred.get("all_scores", {}).get(true_user, 0.0)

        if true_user == "unknown" or true_user not in enrolled_users:
            impostor_total += 1
            scores_impostor.append(conf if pred_user != "unknown" else 0.0)
            if pred_user != "unknown" and conf >= unknown_threshold:
                false_accepts += 1
            else:
                correct += 1
        else:
            genuine_total += 1
            scores_genuine.append(enrolled_match)
            if pred_user == true_user and enrolled_match >= unknown_threshold:
                correct += 1
            else:
                false_rejects += 1

    far = false_accepts / impostor_total if impostor_total else 0.0
    frr = false_rejects / genuine_total if genuine_total else 0.0
    eer = _approx_eer(scores_genuine, scores_impostor)
    n = len(X)

    return {
        "far": round(far, 4),
        "frr": round(frr, 4),
        "eer": round(eer, 4),
        "accuracy": round(correct / n, 4) if n else 0.0,
        "n": n,
        "genuine_n": genuine_total,
        "impostor_n": impostor_total,
    }


def _approx_eer(genuine_scores: list[float], impostor_scores: list[float]) -> float:
    """Approximate EER from score distributions at 101 threshold steps."""
    if not genuine_scores or not impostor_scores:
        return (1.0 - np.mean(genuine_scores)) if genuine_scores else 0.5

    best_t, best_diff = 0.5, 1.0
    for t in np.linspace(0.0, 1.0, 101):
        frr = sum(1 for s in genuine_scores if s < t) / len(genuine_scores)
        far = sum(1 for s in impostor_scores if s >= t) / len(impostor_scores)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff, best_t = diff, float(t)
    return best_t


# ---------------------------------------------------------------------------
# 9. SIGNALS registry (backward compatibility with server.py score_signals())
# ---------------------------------------------------------------------------

SIGNALS: dict[str, Callable[[np.ndarray], float]] = {
    "keystroke": lambda features: predict_user(features).get("confidence", 0.5),
}
