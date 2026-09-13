"""
test_multiusers.py
------------------
End-to-End Test Suite for Continuous Multi-User Keystroke Biometric Identification.
Tests the exact requirements:
  - Test A: Ali genuine typing -> Current User = ali musa raza
  - Test B: Hussain genuine typing -> Current User = hussain afroz
  - Test C: Switch users during same session (Ali -> Hussain) -> transitions after consistent windows
  - Test D: Ali types faster -> still Ali
  - Test E: Ali types slower -> still Ali
  - Test F: Ali typing with natural pauses -> still Ali
  - Test G: Unknown typist -> Unknown / Unauthorized Typist
"""

import asyncio
import json
import time
import numpy as np
import websockets

WS_URL = "ws://127.0.0.1:8000/ws/typing"


def load_user_events(username: str, session_index: int = 0) -> list[dict]:
    with open("users_db.json", "r", encoding="utf-8") as f:
        db = json.load(f)
    return db[username]["enrollment_sessions"][session_index]["events"]


def synthesize_stranger_events(dwell_ms: float = 300.0, flight_ms: float = 800.0, count: int = 70) -> list[dict]:
    evts = []
    now = time.time() * 1000.0
    text = "The unknown impostor is submitting keystrokes trying to evade continuous biometric classification."
    rng = np.random.RandomState(99)
    for i in range(count):
        ch = text[i % len(text)]
        d = max(20.0, dwell_ms + rng.randn() * 15.0)
        f = max(20.0, flight_ms + rng.randn() * 30.0)
        down = now + i * (d + f)
        up = down + d
        evts.append({"key": ch, "event_type": "keydown", "timestamp": down})
        evts.append({"key": ch, "event_type": "keyup", "timestamp": up})
    return evts


def get_user_from_msg(r: dict) -> str:
    if "prediction" in r and r["prediction"].get("predicted_user"):
        return r["prediction"]["predicted_user"]
    if "auth_event" in r and r["auth_event"].get("display_user"):
        return r["auth_event"]["display_user"]
    if r.get("display_user"):
        return r["display_user"]
    return ""


async def send_and_collect(ws, events: list[dict], wait_sec: float = 2.0) -> list[dict]:
    await ws.send(json.dumps({"events": events}))
    results = []
    end_time = time.time() + wait_sec
    while time.time() < end_time:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=0.6)
            data = json.loads(msg)
            results.append(data)
        except (asyncio.TimeoutError, TimeoutError):
            break
    return results


async def run_all_tests():
    print("=" * 65)
    print(" [KeyGuard AI] Running Continuous Multi-User Identification Tests")
    print("=" * 65)

    test_results = {}

    # -----------------------------------------------------------------------
    # Test A: Ali genuine typing -> ali musa raza
    # -----------------------------------------------------------------------
    print("\n--- [Test A] Ali Musa Raza Typing (Session 1: Normal) ---")
    ali_events = load_user_events("ali musa raza", session_index=0)
    async with websockets.connect(WS_URL) as ws:
        res = await send_and_collect(ws, ali_events[:160], wait_sec=2.5)
        predicted_users = [get_user_from_msg(r) for r in res if get_user_from_msg(r)]
        print(f"  Received updates: {len(res)}")
        print(f"  Predicted user identities: {predicted_users}")
        passed = any("ali" in str(p).lower() for p in predicted_users)
        test_results["Test A (Ali)"] = "PASSED" if passed else "FAILED"
        print(f"  Result: {'PASSED [OK]' if passed else 'FAILED'}")

    # -----------------------------------------------------------------------
    # Test B: Hussain genuine typing -> hussain afroz
    # -----------------------------------------------------------------------
    print("\n--- [Test B] Hussain Afroz Typing (Session 1: Normal) ---")
    hus_events = load_user_events("hussain afroz", session_index=0)
    async with websockets.connect(WS_URL) as ws:
        res = await send_and_collect(ws, hus_events[:160], wait_sec=2.5)
        predicted_users = [get_user_from_msg(r) for r in res if get_user_from_msg(r)]
        print(f"  Received updates: {len(res)}")
        print(f"  Predicted user identities: {predicted_users}")
        passed = any("hussain" in str(p).lower() for p in predicted_users)
        test_results["Test B (Hussain)"] = "PASSED" if passed else "FAILED"
        print(f"  Result: {'PASSED [OK]' if passed else 'FAILED'}")

    # -----------------------------------------------------------------------
    # Test C: Switch users during same session (Ali -> Hussain)
    # -----------------------------------------------------------------------
    print("\n--- [Test C] Switching Users Live During Session (Ali -> Hussain) ---")
    async with websockets.connect(WS_URL) as ws:
        # 1. Ali starts typing
        print("  Step 1: Ali typing on keyboard...")
        r_ali = await send_and_collect(ws, ali_events[:160], wait_sec=2.0)
        ali_pred = [get_user_from_msg(r) for r in r_ali if get_user_from_msg(r)]
        print(f"    Current User while Ali types: {ali_pred[-1] if ali_pred else 'None'}")

        # 2. Hussain takes over keyboard and types
        print("  Step 2: Hussain takes keyboard and types...")
        r_hus = await send_and_collect(ws, hus_events[:200], wait_sec=2.5)
        hus_pred = [get_user_from_msg(r) for r in r_hus if get_user_from_msg(r)]
        print(f"    Current User after Hussain continues typing: {hus_pred[-1] if hus_pred else 'None'}")

        ali_detected = any("ali" in str(p).lower() for p in ali_pred)
        hus_detected = any("hussain" in str(p).lower() for p in hus_pred)
        passed = ali_detected and hus_detected
        test_results["Test C (User Switch)"] = "PASSED" if passed else "FAILED"
        print(f"  Result: {'PASSED [OK]' if passed else 'FAILED'}")

    # -----------------------------------------------------------------------
    # Test D: Ali types faster -> still Ali
    # -----------------------------------------------------------------------
    print("\n--- [Test D] Ali Typing Faster (Session 4: Fast) ---")
    ali_fast = load_user_events("ali musa raza", session_index=3)
    async with websockets.connect(WS_URL) as ws:
        res = await send_and_collect(ws, ali_fast[:160], wait_sec=2.5)
        preds = [get_user_from_msg(r) for r in res if get_user_from_msg(r)]
        print(f"  Predicted user identities: {preds}")
        passed = any("ali" in str(p).lower() for p in preds)
        test_results["Test D (Ali Fast)"] = "PASSED" if passed else "FAILED"
        print(f"  Result: {'PASSED [OK]' if passed else 'FAILED'}")

    # -----------------------------------------------------------------------
    # Test E: Ali types slower -> still Ali
    # -----------------------------------------------------------------------
    print("\n--- [Test E] Ali Typing Slower (Session 5: Slow) ---")
    ali_slow = load_user_events("ali musa raza", session_index=4)
    async with websockets.connect(WS_URL) as ws:
        res = await send_and_collect(ws, ali_slow[:160], wait_sec=2.5)
        preds = [get_user_from_msg(r) for r in res if get_user_from_msg(r)]
        print(f"  Predicted user identities: {preds}")
        passed = any("ali" in str(p).lower() for p in preds)
        test_results["Test E (Ali Slow)"] = "PASSED" if passed else "FAILED"
        print(f"  Result: {'PASSED [OK]' if passed else 'FAILED'}")

    # -----------------------------------------------------------------------
    # Test F: Ali pauses -> still Ali
    # -----------------------------------------------------------------------
    print("\n--- [Test F] Ali Typing with Pauses (Session 2: Different with pauses) ---")
    ali_diff = load_user_events("ali musa raza", session_index=1)
    async with websockets.connect(WS_URL) as ws:
        res = await send_and_collect(ws, ali_diff[:160], wait_sec=2.5)
        preds = [get_user_from_msg(r) for r in res if get_user_from_msg(r)]
        print(f"  Predicted user identities: {preds}")
        passed = any("ali" in str(p).lower() for p in preds)
        test_results["Test F (Ali Pauses)"] = "PASSED" if passed else "FAILED"
        print(f"  Result: {'PASSED [OK]' if passed else 'FAILED'}")

    # -----------------------------------------------------------------------
    # Test G: Unknown human / Bot -> Unknown / Unauthorized Typist
    # -----------------------------------------------------------------------
    print("\n--- [Test G] Unknown Human Stranger Typing ---")
    stranger_events = synthesize_stranger_events(dwell_ms=300.0, flight_ms=800.0, count=70)
    async with websockets.connect(WS_URL) as ws:
        res = await send_and_collect(ws, stranger_events, wait_sec=2.5)
        preds = [get_user_from_msg(r) for r in res if get_user_from_msg(r)]
        states = [
            r.get("auth_event", {}).get("state") or r.get("state")
            for r in res if r.get("auth_event") or r.get("state")
        ]
        print(f"  Predicted user identities: {preds}")
        print(f"  Security states: {states}")
        passed = any("unknown" in str(p).lower() or "unauthorized" in str(p).lower() for p in preds) or any(s in ("UNKNOWN", "SUSPICIOUS") for s in states)
        test_results["Test G (Unknown)"] = "PASSED" if passed else "FAILED"
        print(f"  Result: {'PASSED [OK]' if passed else 'FAILED'}")

    print("\n" + "=" * 65)
    print(" [Summary of Test Results]")
    print("=" * 65)
    all_ok = True
    for tname, status in test_results.items():
        print(f"  {tname:32s}: {status}")
        if status != "PASSED":
            all_ok = False

    print("=" * 65)
    print(f" Final Test Status: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print("=" * 65 + "\n")
    return all_ok


if __name__ == "__main__":
    asyncio.run(run_all_tests())
