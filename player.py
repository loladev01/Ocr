"""
player.py
---------
Player alive / dead detection using bright-pixel-ratio on player name ROIs.

Map-aware dead-lock (anti-fluctuation):
  - Erangel / Miramar  → no recall.  Once a player is confirmed DEAD they are
                          permanently locked DEAD for the rest of the match.
  - Rondo              → has recall.  DEAD → ALIVE transition is allowed, so
                          no lock is applied.

The lock lives in a module-level dict so it persists across player_loop calls
without needing to store it in main.py's state dict.
"""

import cv2
import numpy as np

from config import PIXEL_THR, ALIVE_RATIO, DEAD_RATIO, RECALL_MAPS, DEAD_CONFIRM_FRAMES

# ── Per-player permanent-dead lock  {(slot, player_key): True} ──────────────
# Only populated for non-recall maps.
_dead_locked: dict[tuple, bool] = {}

# Per-player consecutive-DEAD frame counter for confirmation before locking.
# { (slot, player_key): int }
_dead_counter: dict[tuple, int] = {}


def reset_player_locks():
    """Call this from main.py reset_state() so locks clear on match reset."""
    _dead_locked.clear()
    _dead_counter.clear()


# ---------------------------------------------------------------------------
# Core detection helpers
# ---------------------------------------------------------------------------

def bright_pixel_ratio(frame_bgr, roi):
    """
    Compute the fraction of pixels above PIXEL_THR in the given ROI.

    Parameters
    ----------
    frame_bgr : np.ndarray   Full BGR frame (absolute coordinates).
    roi       : list/tuple   [x, y, w, h] in absolute screen pixels.

    Returns
    -------
    ratio : float   Fraction of bright pixels (0.0–1.0).
    """
    x, y, w, h = [int(v) for v in roi]
    fh, fw = frame_bgr.shape[:2]
    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))
    w = max(1, min(w, fw - x))
    h = max(1, min(h, fh - y))

    crop_img = frame_bgr[y:y + h, x:x + w]
    if crop_img.size == 0:
        return 0.0

    gray   = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    bright = int(np.sum(gray > PIXEL_THR))
    return bright / gray.size


def raw_player_status(ratio):
    """
    Convert a bright-pixel ratio to a raw status string (no locking applied).

    Returns
    -------
    "ALIVE" | "DEAD" | "UNSURE"
    """
    if ratio >= ALIVE_RATIO:
        return "ALIVE"
    elif ratio <= DEAD_RATIO:
        return "DEAD"
    return "UNSURE"


# ---------------------------------------------------------------------------
# Per-card detection  (map-aware)
# ---------------------------------------------------------------------------

def detect_players_card(frame_bgr, card_players_rois, player_count=4,
                        slot=None, map_name=""):
    """
    Detect alive/dead status for every player on one card.

    Parameters
    ----------
    frame_bgr        : np.ndarray   Full BGR frame.
    card_players_rois: dict         {"player1": [x,y,w,h], ...}
                                    Absolute coords from merged_pageN.json.
    player_count     : int          How many players to check (1-4).
    slot             : str|None     Slot string e.g. "07". Required for locking.
    map_name         : str          Current map name (e.g. "erangel", "rondo").

    Returns
    -------
    statuses : dict   {"player1": "ALIVE"|"DEAD"|"UNSURE", ...}
    ratios   : dict   {"player1": float, ...}
    all_dead : bool   True when every active player is DEAD.
    """
    # Recall maps: DEAD is not permanent (player can be revived / recalled).
    # Non-recall maps: once confirmed DEAD → lock forever this match.
    map_key    = map_name.strip().lower()
    has_recall = map_key in RECALL_MAPS

    statuses = {}
    ratios   = {}

    for pi in range(1, player_count + 1):
        key = f"player{pi}"
        roi = card_players_rois.get(key)
        if roi is None:
            continue

        lock_key = (slot, key)

        # If already permanently locked dead on a no-recall map → stay DEAD
        if not has_recall and _dead_locked.get(lock_key, False):
            statuses[key] = "DEAD"
            ratios[key]   = 0.0
            continue

        ratio  = bright_pixel_ratio(frame_bgr, roi)
        status = raw_player_status(ratio)

        if not has_recall and slot is not None:
            if status == "DEAD":
                # Increment consecutive-dead counter
                _dead_counter[lock_key] = _dead_counter.get(lock_key, 0) + 1
                if _dead_counter[lock_key] >= DEAD_CONFIRM_FRAMES:
                    # Confirmed dead enough times → lock permanently
                    _dead_locked[lock_key] = True
            else:
                # Any non-DEAD reading resets counter (but NOT the lock —
                # lock is permanent once set)
                _dead_counter[lock_key] = 0

        # Apply UNSURE → keep previous status to reduce flicker.
        # If no previous status exists and status is UNSURE, treat as ALIVE.
        if status == "UNSURE":
            prev = statuses.get(key)   # won't exist yet this iteration —
            # use dead_locked as fallback
            if _dead_locked.get(lock_key, False):
                status = "DEAD"
            else:
                status = "ALIVE"   # default assumption while uncertain

        statuses[key] = status
        ratios[key]   = ratio

    alive_count = sum(1 for s in statuses.values() if s == "ALIVE")
    dead_count  = sum(1 for s in statuses.values() if s == "DEAD")
    all_dead    = len(statuses) > 0 and dead_count == len(statuses)

    return statuses, ratios, all_dead


# ---------------------------------------------------------------------------
# Page-level detection  (called from main.py's player_loop)
# ---------------------------------------------------------------------------

def detect_players_page(frame_bgr, merged_rois, stable_slots,
                        players_for_slot_fn, cfg, max_cards=9,
                        map_name=""):
    """
    Run player alive/dead detection for all cards on the current page.

    Parameters
    ----------
    frame_bgr          : np.ndarray   Full BGR frame.
    merged_rois        : dict         Contents of merged_page1.json or merged_page2.json.
    stable_slots       : list[str]    Slot strings per card index.
    players_for_slot_fn: callable     players_for_slot(slot, cfg) → int.
    cfg                : dict         Tournament config.
    max_cards          : int          Max cards to process.
    map_name           : str          Current map ("erangel" / "miramar" / "rondo").

    Returns
    -------
    player_statuses_by_slot : dict
        {
            "07": {
                "player1": "ALIVE",
                "player2": "DEAD",
                ...
                "ratios":      {"player1": 0.23, ...},
                "all_dead":    False,
                "alive_count": 3,
            },
            ...
        }
    """
    result = {}
    cards  = merged_rois.get("cards", [])[:max_cards]

    for i, card in enumerate(cards):
        slot = stable_slots[i] if i < len(stable_slots) else None
        if not slot:
            continue

        player_rois  = card.get("players", {})
        player_count = players_for_slot_fn(slot, cfg)

        statuses, ratios, all_dead = detect_players_card(
            frame_bgr, player_rois, player_count,
            slot=slot, map_name=map_name,
        )

        alive_count = sum(1 for s in statuses.values() if s == "ALIVE")

        result[slot] = {
            **statuses,
            "ratios":      ratios,
            "all_dead":    all_dead,
            "alive_count": alive_count,
        }

    return result