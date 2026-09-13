"""Quick inspection of user enrollment data."""
import json
import numpy as np

with open("users_db.json", "r", encoding="utf-8") as f:
    db = json.load(f)

users = list(db.keys())
print(f"Total users in DB: {len(users)}")
print(f"Users: {users}\n")

for u in users:
    rec = db[u]
    enrolled = rec.get("enrolled", False)
    sessions = rec.get("enrollment_sessions", [])
    profile = rec.get("profile")
    print(f"--- {u} ---")
    print(f"  enrolled: {enrolled}")
    print(f"  sessions: {len(sessions)}")
    if profile:
        print(f"  profile dwell_mean: {profile.get('dwell_mean', '?')}")
        print(f"  profile flight_mean: {profile.get('flight_mean', '?')}")
    for i, s in enumerate(sessions):
        events = s.get("events", [])
        stype = s.get("session_type", "?")
        sidx = s.get("session_index", "?")
        keydowns = [e for e in events if e.get("event_type") == "keydown"]
        keyups = [e for e in events if e.get("event_type") == "keyup"]
        
        # Compute actual dwell/flight timing
        if keydowns and keyups:
            # Sample first few dwells
            timestamps = sorted([e["timestamp"] for e in events])
            ts_range = (timestamps[-1] - timestamps[0]) / 1000.0  # seconds
            
            # Actual dwell: find matched keydown/keyup for same key
            dwells = []
            pending = {}
            for e in sorted(events, key=lambda x: x["timestamp"]):
                if e["event_type"] == "keydown":
                    pending.setdefault(e["key"], []).append(e["timestamp"])
                elif e["event_type"] == "keyup":
                    stack = pending.get(e["key"])
                    if stack:
                        dts = stack.pop(0)
                        dwells.append((e["timestamp"] - dts) / 1000.0)
            
            # Actual flights between consecutive keydowns
            kd_ts = sorted([e["timestamp"] for e in keydowns])
            flights = [(kd_ts[j+1] - kd_ts[j]) / 1000.0 for j in range(len(kd_ts)-1)]
            
            print(f"  Session {sidx} ({stype}): {len(keydowns)} keydowns, {len(keyups)} keyups, span={ts_range:.1f}s")
            if dwells:
                print(f"    dwell: mean={np.mean(dwells):.4f}s std={np.std(dwells):.4f}s")
            if flights:
                print(f"    inter-key: mean={np.mean(flights):.4f}s std={np.std(flights):.4f}s")
        else:
            print(f"  Session {sidx} ({stype}): {len(events)} total events")
    print()
