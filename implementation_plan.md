# KeyGuard AI — Complete System Overhaul

Real-Time Continuous User Authentication Using Keystroke Dynamics and LSTM.

## Overview

This is a ground-up redesign of the existing system. The current system uses a single anomaly-scoring LSTM with only 4 features. The new system implements a full **multi-user identification + rejection** model with 6 rich behavioral features, a separate risk engine, temporal smoothing, and a live activity-monitor UI panel.

---

## Architecture

```
Browser (index.html)
  ├── Keystroke Capture (keydown/keyup events)
  ├── Clipboard Monitor (copy/paste/cut)
  └── WebSocket → server.py

server.py  (FastAPI orchestration)
  ├── REST: /api/register, /api/login, /api/enroll/submit, /api/user/status
  ├── WS:  /ws/typing  → risk engine → activity log → UI push
  └── calls → model.py, users.py, risk_engine.py

model.py  (pure ML — no web code)
  ├── extract_features()        6-feature vector per keystroke
  ├── build_sequences()         sliding-window dataset builder
  ├── load_model() / predict()  multi-user LSTM + rejection
  └── learn_user_profile()      per-user stat profile for threshold

users.py   (persistence — JSON DB)
  ├── register_user / authenticate_user
  ├── save_enrollment_session / get_enrollment_sessions
  └── complete_enrollment / get_user_profile

risk_engine.py  (NEW — completely separate from LSTM)
  ├── temporal_smoother()       rolling window of LSTM outputs
  ├── identity_change_detector()
  └── decide_action() → VERIFIED / SUSPICIOUS / IDENTITY_CHANGED / UNKNOWN

train.py   (offline training script — run once per user set)
  └── session-split train/val/test → FAR/FRR/EER report

index.html (frontend)
  ├── Auth panel (login/register tabs)
  ├── Enrollment wizard (5 sessions with specific instructions)
  ├── Protected workspace (typing area + live metrics)
  └── Side panel: Current-User Card + Activity Log
```

---

## Files Changed / Created

### [MODIFY] model.py
- **6 features per keystroke**: dwell_time, flight_time, transition_time, pause_duration, correction_indicator, timing_variability
- Multi-user LSTM output (one score per enrolled user + unknown rejection)
- `build_sequences()` sliding-window builder with configurable overlap
- `normalize_features()` fit-on-train-only scaler
- `predict_user()` returns `{user: str, confidence: float, all_scores: dict}`
- `learn_user_profile()` stores per-user mean/std for threshold calibration
- `evaluate_model()` FAR/FRR/EER report

### [MODIFY] users.py
- Store enrollment sessions with full raw event data + session type tags (normal / different / free / fast / slow)
- Store learned scaler params alongside biometric profile
- `get_all_enrolled_users()` for multi-user LSTM

### [NEW] risk_engine.py
- `RiskEngine` — pure functions, no web imports
- Rolling deque of last N LSTM window predictions
- Temporal smoothing (exponential weighted average)
- State machine: VERIFIED → SUSPICIOUS → IDENTITY_CHANGED / UNKNOWN
- Configurable thresholds: `consecutive_mismatch_threshold`, `confidence_threshold`, `unknown_rejection_threshold`
- Returns rich events for activity log

### [MODIFY] server.py
- WebSocket handler uses `risk_engine.decide_action()` instead of inline logic
- Push structured activity events to frontend
- Clipboard event handler (separate from LSTM path)
- `evaluate_window()` calls multi-user `predict_user()` not binary anomaly score

### [MODIFY] index.html
- **Zero-Login Autonomous Identification**: Login section removed; user immediately enters the main workspace on load. The AI autonomously identifies who is typing and whether they are authorized.
- **5-Session Enrollment Wizard Modal** ("+ Enroll New Profile"):
  - Step 0: User identifier (e.g. `alice`, `bob`)
  - Session 1: Standard paragraph, normal pace
  - Session 2: Different paragraph, normal pace
  - Session 3: Free text, 30–60 s countdown timer
  - Session 4: Guided paragraph, type fast
  - Session 5: Guided paragraph, type slowly
  - Auto-trains and returns straight to workspace without any login required.
- **Side panel**:
  - Active Typist Card (Identified user, status badge: `AUTHORIZED` / `UNAUTHORIZED` / `SUSPICIOUS`, match confidence %, risk level %, windows analyzed)
  - Registered Profiles Breakdown (live probability bars for all enrolled profiles)
  - Live Activity Log (chronological event stream with icons & timestamps)
- **Non-blocking Security Notification**:
  - Replaced hard modal lockout with an alert banner (`🚨 Security Alert: Unauthorized typist detected…`).
  - Textarea remains enabled so the user can continue typing and view real-time biometric adjustments.
- **Clipboard Monitoring**:
  - `paste`, `copy`, `cut` events are logged to the activity stream without affecting biometric scoring.

### [MODIFY] train.py
- Session-based split (sessions 1–3 train, session 4 val, session 5 test)
- Multi-user dataset builder
- FAR / FRR / EER evaluation
- Saves scaler alongside model checkpoint

---

## Key Design Decisions

> [!IMPORTANT]
> **Zero-Login Autonomous Identification**: Users do not need to log in to start typing. The multi-user model dynamically determines who is at the keyboard based purely on keystroke timing dynamics.

> [!IMPORTANT]
> **Non-Blocking Security Alerts**: Unauthorized typists trigger a high-visibility security alert notification banner and red status badge, but the input is never blocked so continuous monitoring and recovery testing can proceed.

> [!IMPORTANT]
> **Speed must NOT dominate**: The 6-feature vector explicitly separates dwell, flight, and transition timing. Fast/slow enrollment sessions teach the model intra-user speed variance.

> [!IMPORTANT]
> **Rejection gate**: If `max(scores) < unknown_rejection_threshold`, the typist is flagged as `Unauthorized Typist` / `UNKNOWN`.

> [!IMPORTANT]
> **Clipboard events are isolated**: Paste/copy/cut are logged to the activity stream but never fed into the LSTM feature pipeline.

---

## Verification Plan

### Automated
```bash
python train.py          # trains multi-user model, prints FAR/FRR/EER
python test_flow.py      # end-to-end: register → 5-session enroll → verify → unauthorized detection
```

### Manual
1. Open http://127.0.0.1:8000 in browser (loads directly into typing workspace, no login required).
2. Click "+ Enroll New Profile", register "alice", and complete the 5 sessions (normal, different, free, fast, slow).
3. Click "+ Enroll New Profile", register "bob", and complete the 5 sessions.
4. Type as alice in the main workspace → side panel identifies `@alice` with `AUTHORIZED` green badge.
5. Have someone else type (or type with an artificial cadence) → top banner alerts `Unauthorized typist detected`, badge shows `UNAUTHORIZED`, but typing is NOT blocked.
6. Paste text into the area → Activity log logs `📋 Paste detected`, with no biometric penalty.
7. Switch typing style between alice and bob → side panel displays live probability shifts across enrolled profiles.
