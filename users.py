"""
users.py
--------
User management, authentication, and biometric enrollment storage for KeyGuard AI.
Persists registered users, password hashes (salted SHA-256), enrollment progress,
and learned biometric profiles in `users_db.json`.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from typing import Any

DB_FILE = os.path.join(os.path.dirname(__file__), "users_db.json")
_DB_LOCK = threading.Lock()

# Five distinct enrollment session types (spec order)
SESSION_TYPES = ("normal", "different", "free", "fast", "slow")

ENROLLMENT_SESSIONS = [
    {
        "session_index": 1,
        "session_type": "normal",
        "title": "Session 1 — Standard pace",
        "instruction": "Read and type this paragraph at your natural, comfortable pace.",
        "prompt": (
            "Security is not just about a strong password, but the unique natural cadence "
            "with which every individual interacts with their keyboard."
        ),
        "mode": "guided",
        "timer_seconds": None,
    },
    {
        "session_index": 2,
        "session_type": "different",
        "title": "Session 2 — Different text, normal pace",
        "instruction": "Type this different paragraph at your normal pace.",
        "prompt": (
            "Artificial intelligence analyzes subtle timing differences between key presses "
            "to build an invisible shield protecting your personal workspace."
        ),
        "mode": "guided",
        "timer_seconds": None,
    },
    {
        "session_index": 3,
        "session_type": "free",
        "title": "Session 3 — Free typing",
        "instruction": "Type freely about anything for 30–60 seconds. Do not copy-paste.",
        "prompt": "",
        "mode": "free",
        "timer_seconds": 45,
    },
    {
        "session_index": 4,
        "session_type": "fast",
        "title": "Session 4 — Type fast",
        "instruction": "Type this paragraph noticeably faster than usual while staying accurate.",
        "prompt": (
            "Every fingertip motion carries tiny micro-variations that distinguish the genuine "
            "authorized user from an imposter or automated script."
        ),
        "mode": "guided",
        "timer_seconds": None,
    },
    {
        "session_index": 5,
        "session_type": "slow",
        "title": "Session 5 — Type slowly",
        "instruction": "Type this paragraph slowly and deliberately, pausing between phrases.",
        "prompt": (
            "Confidence in digital identity grows when biometric verification seamlessly "
            "confirms your unique rhythm with high statistical precision."
        ),
        "mode": "guided",
        "timer_seconds": None,
    },
]

# Backward-compatible flat prompt list
ENROLLMENT_PARAGRAPHS = [s["prompt"] for s in ENROLLMENT_SESSIONS if s["prompt"]]


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return hashed, salt


def _load_db() -> dict[str, Any]:
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_db(data: dict[str, Any]) -> None:
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def register_user(username: str, password: str) -> dict[str, Any]:
    username = username.strip().lower()
    if not username or len(username) < 3:
        raise ValueError("Username must be at least 3 characters long.")
    if not password or len(password) < 4:
        raise ValueError("Password must be at least 4 characters long.")

    with _DB_LOCK:
        db = _load_db()
        if username in db:
            raise ValueError(f"Username '{username}' already exists. Please log in.")

        hashed, salt = _hash_password(password)
        user_record = {
            "username": username,
            "password_hash": hashed,
            "salt": salt,
            "enrolled": False,
            "enrollment_sessions": [],
            "profile": None,
            "scaler": None,
        }
        db[username] = user_record
        _save_db(db)
        return {
            "username": username,
            "enrolled": False,
            "sessions_completed": 0,
        }


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    username = username.strip().lower()
    with _DB_LOCK:
        db = _load_db()
        user = db.get(username)
        if not user:
            return None

        hashed, _ = _hash_password(password, user["salt"])
        if hashed != user["password_hash"]:
            return None

        return {
            "username": username,
            "enrolled": user.get("enrolled", False),
            "sessions_completed": len(user.get("enrollment_sessions", [])),
        }


def get_user(username: str) -> dict[str, Any] | None:
    username = username.strip().lower()
    with _DB_LOCK:
        db = _load_db()
        return db.get(username)


def get_all_enrolled_users() -> dict[str, dict[str, Any]]:
    """Return {username: profile} for every fully enrolled user."""
    with _DB_LOCK:
        db = _load_db()
        return {
            u: data["profile"]
            for u, data in db.items()
            if data.get("enrolled") and data.get("profile")
        }


def get_enrollment_session_defs() -> list[dict[str, Any]]:
    return list(ENROLLMENT_SESSIONS)


def save_enrollment_session(
    username: str,
    session_index: int,
    events: list[dict],
    session_type: str = "normal",
) -> int:
    username = username.strip().lower()
    with _DB_LOCK:
        db = _load_db()
        user = db.get(username)
        if not user:
            raise ValueError(f"User '{username}' not found.")

        sessions = user.setdefault("enrollment_sessions", [])
        sessions.append({
            "session_index": session_index,
            "session_type": session_type,
            "events_count": len(events),
            "events": events,
        })
        _save_db(db)
        return len(sessions)


def get_enrollment_sessions(username: str) -> list[list[dict]]:
    username = username.strip().lower()
    with _DB_LOCK:
        db = _load_db()
        user = db.get(username, {})
        sessions = user.get("enrollment_sessions", [])
        return [s.get("events", []) for s in sessions]


def get_enrollment_sessions_meta(username: str) -> list[dict[str, Any]]:
    """Full session records including session_type tags."""
    username = username.strip().lower()
    with _DB_LOCK:
        db = _load_db()
        user = db.get(username, {})
        return list(user.get("enrollment_sessions", []))


def complete_enrollment(
    username: str,
    profile_data: dict[str, Any],
    scaler: dict[str, Any] | None = None,
) -> bool:
    username = username.strip().lower()
    with _DB_LOCK:
        db = _load_db()
        user = db.get(username)
        if not user:
            return False

        user["enrolled"] = True
        user["profile"] = profile_data
        if scaler is not None:
            user["scaler"] = scaler
        _save_db(db)
        return True


def get_user_profile(username: str) -> dict[str, Any] | None:
    username = username.strip().lower()
    with _DB_LOCK:
        db = _load_db()
        user = db.get(username, {})
        return user.get("profile")


def get_user_scaler(username: str) -> dict[str, Any] | None:
    username = username.strip().lower()
    with _DB_LOCK:
        db = _load_db()
        user = db.get(username, {})
        return user.get("scaler")


def get_all_training_data() -> dict[str, list[dict[str, Any]]]:
    """All enrolled users with full session metadata for train.py."""
    with _DB_LOCK:
        db = _load_db()
        out: dict[str, list[dict[str, Any]]] = {}
        for username, data in db.items():
            if data.get("enrolled") and data.get("enrollment_sessions"):
                out[username] = list(data["enrollment_sessions"])
        return out


def reset_enrollment(username: str) -> None:
    username = username.strip().lower()
    with _DB_LOCK:
        db = _load_db()
        user = db.get(username)
        if user:
            user["enrolled"] = False
            user["enrollment_sessions"] = []
            user["profile"] = None
            user["scaler"] = None
            _save_db(db)


def delete_user_profile(username: str) -> bool:
    """Remove a registered user and their biometric profile completely from DB."""
    username = username.strip().lower()
    with _DB_LOCK:
        db = _load_db()
        if username in db:
            del db[username]
            _save_db(db)
            return True
        return False


def delete_all_profiles() -> list[str]:
    """Remove all registered biometric profiles from DB."""
    with _DB_LOCK:
        db = _load_db()
        removed = list(db.keys())
        db.clear()
        _save_db(db)
        return removed


def get_registered_profiles() -> list[dict[str, Any]]:
    """Return summary list of all enrolled profiles for UI display and management."""
    with _DB_LOCK:
        db = _load_db()
        profiles = []
        for uname, data in db.items():
            if data.get("enrolled"):
                prof = data.get("profile") or {}
                profiles.append({
                    "username": uname,
                    "sessions_count": len(data.get("enrollment_sessions", [])),
                    "samples_count": prof.get("samples_count", 0),
                    "dwell_mean_ms": round(float(prof.get("dwell_mean", 0.0)) * 1000, 1),
                    "flight_mean_ms": round(float(prof.get("flight_mean", 0.0)) * 1000, 1),
                })
        return profiles
