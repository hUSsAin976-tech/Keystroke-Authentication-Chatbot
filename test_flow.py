"""
test_flow.py
------------
End-to-end automated testing for KeyGuard AI:
1. User Registration & Login
2. 5-Session Biometric Enrollment (all session types)
3. Real-time WebSocket verification of the enrolled user
4. Imposter detection triggering lockout via risk engine
"""

import asyncio
import json
import time
import httpx
import websockets

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/typing"

SESSION_TYPES = ["normal", "different", "free", "fast", "slow"]


def simulate_typing_events(text: str, dwell_ms: float = 100.0, flight_ms: float = 140.0):
    events = []
    now = time.time() * 1000
    for i, char in enumerate(text):
        down = now + i * (dwell_ms + flight_ms)
        up = down + dwell_ms
        events.append({"key": char, "event_type": "keydown", "timestamp": down})
        events.append({"key": char, "event_type": "keyup", "timestamp": up})
    return events


async def run_tests():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
        username = f"alice_{int(time.time())}"
        password = "secretpassword123"
        print(f"\n[Test 1] Registering user: {username}...")
        resp = await client.post("/api/register", json={"username": username, "password": password})
        assert resp.status_code == 200, f"Register failed: {resp.text}"
        print(" -> Registered:", resp.json()["user"])

        print(f"\n[Test 2] Logging in as {username}...")
        resp = await client.post("/api/login", json={"username": username, "password": password})
        assert resp.status_code == 200

        print("\n[Test 3] Fetching enrollment session definitions...")
        resp = await client.get("/api/enroll/prompts")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) == 5

        print("\n[Test 4] Submitting 5 enrollment sessions (normal/different/free/fast/slow)...")
        for i, sess in enumerate(sessions):
            stype = sess["session_type"]
            prompt = sess.get("prompt") or "Free typing sample for biometric enrollment today."
            if stype == "fast":
                dwell, flight = 55.0, 75.0
            elif stype == "slow":
                dwell, flight = 190.0, 230.0
            else:
                dwell, flight = 105.0, 145.0
            events = simulate_typing_events(prompt * (2 if stype == "free" else 1), dwell, flight)
            resp = await client.post("/api/enroll/submit", json={
                "username": username,
                "session_index": i + 1,
                "session_type": stype,
                "events": events,
            })
            assert resp.status_code == 200, f"Session {i+1} failed: {resp.text}"
            res_data = resp.json()
            print(f" -> Session {i+1} ({stype}): status={res_data['status']}, enrolled={res_data['enrolled']}")

        assert res_data["enrolled"] is True
        print(f" -> Enrolled! Profile keys: {list(res_data['profile'].keys())[:5]}...")

    print("\n[Test 5] WebSocket -- genuine user typing...")
    uri = f"{WS_URL}?username={username}"
    async with websockets.connect(uri) as ws:
        # Generate a long text to ensure we exceed WINDOW_SIZE (40 keydowns)
        long_text = (
            "This is Alice typing naturally in her protected secure workspace with normal rhythm. "
            "She types with confidence and consistency as the AI analyzes her unique biometric pattern."
        )
        events = simulate_typing_events(long_text, dwell_ms=105.0, flight_ms=145.0)
        verified = False
        # Send in multiple batches to fill the window
        for batch_num in range(5):
            await ws.send(json.dumps({"events": events}))
            # Drain all available messages
            for _ in range(5):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=8.0)
                    payload = json.loads(msg)
                    state = payload.get("auth_event", {}).get("state") or payload.get("state", "")
                    if payload.get("score") is not None:
                        print(f" -> Score={payload.get('score')}, action={payload.get('action')}, state={state}")
                    if state == "VERIFIED":
                        verified = True
                        break
                except (asyncio.TimeoutError, TimeoutError):
                    break
            if verified:
                break
        print(f" -> Genuine user test: {'VERIFIED [OK]' if verified else 'Not verified (may need more data)'}")
        assert verified, f"Expected VERIFIED state for genuine enrolled user {username}"

    print("\n[Test 6] WebSocket -- imposter bot typing...")
    async with websockets.connect(uri) as ws:
        bot_text = (
            "Hacker injecting commands rapidly into the session with machine-like precision and speed "
            "attempting to bypass all security measures and gain unauthorized access to protected system."
        )
        bot_events = simulate_typing_events(bot_text, dwell_ms=1.0, flight_ms=1.0)
        locked = False
        for batch_num in range(8):
            await ws.send(json.dumps({"events": bot_events}))
            for _ in range(5):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=8.0)
                    payload = json.loads(msg)
                    state = payload.get("state") or payload.get("auth_event", {}).get("state", "")
                    event_type = payload.get("type", "")
                    if event_type == "auth_event" or state:
                        print(f" -> state={state}, msg={payload.get('message', '')[:60]}")
                        if state in ("IDENTITY_CHANGED", "UNKNOWN"):
                            locked = True
                            break
                    if payload.get("action") in ("lock", "notify"):
                        locked = True
                        print(f" -> Notified/Locked: {payload.get('message', '')[:60]}")
                        break
                except (asyncio.TimeoutError, TimeoutError):
                    break
                except websockets.exceptions.ConnectionClosed:
                    locked = True
                    break
            if locked:
                break
        assert locked, "Expected lockout or identity change for imposter"

    print("\n==========================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==========================================\n")


if __name__ == "__main__":
    asyncio.run(run_tests())
