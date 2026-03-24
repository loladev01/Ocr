"""
pusher.py — runs on BOTH PC 1 and PC 2 alongside the OCR script
===============================================================
Two jobs only:
  1. Push local match_state.json to the hub every second
  2. Pull tournament_config.json from hub every few seconds
     so manual overrides set on the dashboard work locally

Install:  pip install requests
Run:      python pusher.py

Edit the three lines under "Settings" before running.
"""

import hashlib
import json
import os
import time

import requests

# ── Settings — edit these before running ─────────────────────────
HUB_URL = "https://ocr-hub.onrender.com"  # ← your Render or Railway URL
PC_ID   = "pc1"                             # ← "pc1"  OR  "pc2"
API_KEY = "bgmiocr2026secretkey"                        # ← same as API_KEY env var on the hub
# ─────────────────────────────────────────────────────────────────

STATE_PATH  = "match_state.json"
CONFIG_PATH = "tournament_config.json"

PUSH_INTERVAL   = 1.0    # push state every second
CONFIG_POLL_SEC = 1.0    # check for config updates every 3 seconds
REQUEST_TIMEOUT = 4.0    # give up on a request after 4 seconds

# How many consecutive failures before printing "unreachable".
# Stops the noisy flip-flop when Render is briefly between requests.
# 3 failures = ~3 seconds of actual silence before alerting.
FAIL_THRESHOLD = 3

PUSH_URL   = f"{HUB_URL.rstrip('/')}/api/push/{PC_ID}"
CONFIG_URL = f"{HUB_URL.rstrip('/')}/api/config"
HEADERS    = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def file_md5(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02 * (attempt + 1))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    try:
        os.remove(tmp)
    except OSError:
        pass


def config_is_meaningful(cfg: dict) -> bool:
    """
    Returns True only if the config from hub has real tournament data.

    Why this matters:
    Render free tier restarts the container when it wakes from sleep.
    On a fresh start tournament_config.json does not exist yet on the
    hub's ephemeral filesystem so /api/config returns {}.
    Without this guard pusher would overwrite the local working config
    (which has all team names, slots, ranges) with that empty {}.
    We ignore any hub config that has no tournaments or no slots.
    """
    if not cfg:
        return False
    has_tournaments = bool(cfg.get("tournaments"))
    has_slots       = bool(cfg.get("slots"))
    has_name        = bool(cfg.get("tournament", "").strip())
    return has_tournaments or has_slots or has_name


def main():
    print("=" * 52)
    print(f"  BGMI PUSHER  —  {PC_ID.upper()}")
    print("=" * 52)
    print(f"  Hub     : {HUB_URL}")
    print(f"  State   : {os.path.abspath(STATE_PATH)}")
    print(f"  Config  : {os.path.abspath(CONFIG_PATH)}")
    print()

    if "your-hub" in HUB_URL:
        print("  ⚠  HUB_URL is still the placeholder.")
        print("     Edit pusher.py with your real Render URL.")
        print("     Press Ctrl-C, edit, then re-run.\n")

    if API_KEY == "changeme":
        print("  ⚠  API_KEY is still 'changeme'.")
        print("     Set the same key here and as an env var on the hub.\n")

    last_config_poll  = 0.0
    last_config_hash  = file_md5(CONFIG_PATH)
    was_connected     = False
    push_count        = 0
    consec_fail_count = 0      # consecutive failure counter
    last_fail_reason  = ""

    while True:
        t0  = time.time()
        now = t0

        # ── 1. Push state to hub ──────────────────────────────────
        state = read_json(STATE_PATH)
        try:
            r = requests.post(
                PUSH_URL,
                json=state,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            if r.status_code == 200:
                # Reset failure streak on any success
                if not was_connected:
                    # Only announce reconnect if we were truly marked
                    # offline (past threshold) or this is the first push
                    if consec_fail_count >= FAIL_THRESHOLD or push_count == 0:
                        print(f"[PUSHER] ✓  Connected to hub  ({time.strftime('%H:%M:%S')})")
                    was_connected     = True
                    consec_fail_count = 0

                consec_fail_count = 0
                push_count += 1
                if push_count % 30 == 0:
                    teams = len(state.get("Teams", []))
                    print(f"[PUSHER] ↑  {teams} teams pushed  {time.strftime('%H:%M:%S')}")
            else:
                raise Exception(f"HTTP {r.status_code}: {r.text[:80]}")

        except Exception as exc:
            consec_fail_count += 1
            last_fail_reason   = str(exc)

            # Only flip to offline and print after FAIL_THRESHOLD
            # consecutive failures — ignores brief 404 blips from
            # Render waking between requests
            if consec_fail_count == FAIL_THRESHOLD:
                print(
                    f"[PUSHER] ✗  Hub unreachable ({FAIL_THRESHOLD} consecutive)"
                    f" — {last_fail_reason}  ({time.strftime('%H:%M:%S')})"
                )
                was_connected = False

            # After threshold crossed, log again every 15 s so you
            # know it is still down without spamming every second
            elif consec_fail_count > FAIL_THRESHOLD:
                was_connected = False
                if consec_fail_count % 15 == 0:
                    down_sec = int(consec_fail_count * PUSH_INTERVAL)
                    print(
                        f"[PUSHER] ✗  Still unreachable — {down_sec}s down"
                        f"  ({time.strftime('%H:%M:%S')})"
                    )

        # ── 2. Pull config updates from hub ──────────────────────
        if now - last_config_poll >= CONFIG_POLL_SEC:
            last_config_poll = now
            try:
                r = requests.get(CONFIG_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
                if r.status_code == 200:
                    remote_cfg = r.json()

                    # GUARD — never overwrite local config with an empty
                    # hub response caused by a Render restart / cold start.
                    # An empty {} means hub lost its filesystem — keep
                    # the local config with all team names intact.
                    if not config_is_meaningful(remote_cfg):
                        pass   # silently skip — protect local config
                    else:
                        remote_hash = hashlib.md5(
                            json.dumps(remote_cfg, sort_keys=True).encode()
                        ).hexdigest()
                        if remote_hash != last_config_hash:
                            atomic_write_json(CONFIG_PATH, remote_cfg)
                            last_config_hash = remote_hash
                            print(
                                f"[PUSHER] ↓  Config updated from hub"
                                f"  ({time.strftime('%H:%M:%S')})"
                            )
            except Exception:
                pass   # config pull failure is non-critical

        elapsed   = time.time() - t0
        sleep_for = max(0.0, PUSH_INTERVAL - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[PUSHER] Stopped.")