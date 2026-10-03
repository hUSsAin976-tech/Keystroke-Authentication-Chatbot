"""
test_train_accuracy.py
----------------------
Automated test suite verifying the "Train to Improve Accuracy" feature:
1. /api/train/verify-prediction returns real-time predictions, unauthorized flags, and cadence telemetry
2. /api/train/feedback updates the biometric profile and training dataset for the specific user
3. /api/train/retrain triggers speed-augmented neural LSTM retraining
"""

import asyncio
import time
import httpx

from server import app
from users import get_user_profile, get_all_enrolled_users, get_registered_profiles


def make_keystrokes(text: str, dwell_ms: float = 110.0, flight_ms: float = 135.0):
    events = []
    now = time.time() * 1000
    for i, ch in enumerate(text):
        down = now + i * (dwell_ms + flight_ms)
        up = down + dwell_ms
        events.append({"key": ch, "event_type": "keydown", "timestamp": down})
        events.append({"key": ch, "event_type": "keyup", "timestamp": up})
    return events


async def test_train_accuracy_feature():
    print("\n=======================================================")
    print(" [TEST] Train to Improve Accuracy Feature Verification ")
    print("=======================================================\n")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Test /api/profiles
        resp = await client.get("/api/profiles")
        assert resp.status_code == 200, f"Profiles failed: {resp.text}"
        profiles = resp.json().get("profiles", [])
        print(f"[1] Fetched registered profiles: {len(profiles)} found.")
        for p in profiles:
            print(f"    - @{p['username']}: {p['sessions_count']} sessions, dwell: {p['dwell_mean_ms']}ms")

        # 2. Test /api/train/verify-prediction
        events = make_keystrokes("The quick brown fox jumps over the lazy dog and tests continuous accuracy in real time.", dwell_ms=115.0, flight_ms=145.0)
        resp = await client.post("/api/train/verify-prediction", json={"events": events, "target_user": None})
        assert resp.status_code == 200, f"Verify prediction failed: {resp.text}"
        data = resp.json()
        print("\n[2] Real-time 1-Second Prediction Result:")
        print(f"    - Predicted User: {data.get('predicted_user')}")
        print(f"    - Confidence:     {data.get('confidence')}")
        print(f"    - Unauthorized:   {data.get('is_unauthorized')}")
        print(f"    - Telemetry:      {data.get('telemetry')}")
        print(f"    - All Scores:     {data.get('all_scores')}")
        assert "telemetry" in data
        assert "predicted_user" in data
        assert "all_scores" in data

        # 3. Test /api/train/feedback for a specific user
        target_user = profiles[0]["username"] if profiles else "alice"
        feedback_events = make_keystrokes("Improving accuracy by typing natural pattern sequences and verifying the ground truth.", dwell_ms=112.0, flight_ms=138.0)
        resp = await client.post("/api/train/feedback", json={
            "username": target_user,
            "events": feedback_events,
            "was_correct": True,
            "predicted_user": target_user,
            "retrain": False
        })
        assert resp.status_code == 200, f"Feedback failed: {resp.text}"
        fb_data = resp.json()
        print(f"\n[3] Pattern Learned Feedback Result for @{target_user}:")
        print(f"    - Sessions Count: {fb_data.get('sessions_count')}")
        print(f"    - Profile:        {fb_data.get('profile')}")
        print(f"    - Message:        {fb_data.get('message')}")
        assert fb_data["status"] == "ok"
        assert fb_data["sessions_count"] >= 1

        # 4. Test /api/train/retrain
        resp = await client.post("/api/train/retrain", json={"epochs": 10})
        assert resp.status_code == 200, f"Retrain failed: {resp.text}"
        retrain_data = resp.json()
        print("\n[4] Retraining Trigger Result:")
        print(f"    - Status:  {retrain_data.get('status')}")
        print(f"    - Message: {retrain_data.get('message')}")
        assert retrain_data["status"] == "ok"

    print("\n=======================================================")
    print(" [PASSED] All Accuracy Training API tests succeeded! ")
    print("=======================================================\n")


if __name__ == "__main__":
    asyncio.run(test_train_accuracy_feature())


