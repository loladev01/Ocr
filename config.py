import os
import json
import time
import copy

# ===================== CONFIG =====================
MONITOR_INDEX = 2

ROIS_PAGE1 = "rois_page1.json"
ROIS_PAGE2 = "rois_page2.json"
SLOT_TEMPL_DIR = "assets/slot_digits"  # 0.png..9.png + samples/

TESSERACT_CMD = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# ── Tournament config (written by dashboard) ──────────────────────
TOURNAMENT_CONFIG = "tournament_config.json"
# ─────────────────────────────────────────────────────────────────

REF_W, REF_H = 1920, 1080

CAPTURE_FPS = 30

SLOT_HZ  = 8.0
BADGE_HZ = 15.0
KILLS_HZ = 8.0
JSON_HZ  = 8.0

OCR_WORKERS    = 10
PIXEL_DIFF_SKIP = 0.02

# --- Kill accuracy fixes ---
KILL_SMOOTH   = 5
TOTAL_CONFIRM = 2
MAX_KILL_JUMP = 5

# Playzone banner suppression
PLAYZONE_YELLOW_RATIO = 0.015

DISPLAY_WIDTH = 1500

# slot templates
DIG_W, DIG_H   = 24, 40
SLOT_MIN_SCORE = 0.55

# Lock slot/page after stable for N seconds
SLOT_LOCK_SEC      = 60
MIN_VALID_FOR_LOCK = 7

# badge dead
EDGE_THR     = 0.030
BADGE_CONFIRM = 1
BADGE_DECAY   = 1

# dead -> final scan window -> lock team
FINAL_KILL_SCAN_DELAY_SEC = 30

# How often (seconds) the badge loop re-reads tournament_config.json
CFG_RELOAD_INTERVAL = 2.0

OUT_JSON = "match_state.json"

# ── Merged ROI files (include player name ROIs per card) ─────────────────────
MERGED_PAGE1 = "merged_page1.json"
MERGED_PAGE2 = "merged_page2.json"

# ── Player alive / dead detection thresholds ─────────────────────────────────
PIXEL_THR   = 100    # Grayscale cutoff — pixels above this count as bright text
ALIVE_RATIO = 0.15   # >= 15% bright pixels → ALIVE
DEAD_RATIO  = 0.07   # <=  7% bright pixels → DEAD

# How often player alive/dead is re-evaluated
PLAYER_HZ = 8.0

# ── Map-aware dead-lock (anti-fluctuation) ────────────────────────────────────
# Maps listed here support recall / revival — DEAD → ALIVE is possible.
# All other maps (erangel, miramar, sanhok, vikendi, etc.) have no recall,
# so once a player is confirmed dead their status is permanently locked DEAD.
RECALL_MAPS = {"rondo"}   # lowercase — add more map names if needed

# How many consecutive DEAD readings before the lock is applied.
# Higher = more resistant to brief dark-frame false positives.
# At PLAYER_HZ=8:  3 frames ≈ 0.4 s,  5 frames ≈ 0.6 s
DEAD_CONFIRM_FRAMES = 3

# Current map — set this before starting a match.
# Used by player_loop to decide whether to apply the dead-lock.
# Valid values (case-insensitive): "erangel", "miramar", "rondo", "sanhok", etc.
MAP_NAME = "erangel" 


GAME_RECT_PATH          = "game_rect.json"
USE_GAME_RECT_IF_EXISTS = True

SHOTS_DIR = "shots"


# ===================== TOURNAMENT CONFIG LOADER =====================
VALID_MAPS = {"erangel", "miramar", "rondo"}   # lowercase canonical names

_CONFIG_DEFAULTS = {
    "tournament":  "Unknown",
    "total_teams": 25,
    "page1Range":  [7, 15],
    "page2Range":  [16, 25],
    "map":         "erangel",
    "slots": {},
}


def _parse_config(cfg):
    total = max(1, min(25, int(cfg.get("total_teams", 25))))
    cfg["total_teams"] = total
    cfg.setdefault("page1Range", [7, 15])
    cfg.setdefault("page2Range", [16, 25])
    # ── map field ──────────────────────────────────────────────────
    raw_map = str(cfg.get("map", "erangel")).strip().lower()
    cfg["map"] = raw_map if raw_map in VALID_MAPS else "erangel"
    # ──────────────────────────────────────────────────────────────
    slots = cfg.get("slots", {})
    clean = {}
    for k, v in slots.items():
        slot = str(k).zfill(2)
        entry = {
            "team":    str(v.get("team", "")),
            "players": max(1, min(4, int(v.get("players", 4)))),
        }
        mk = v.get("manual_kills")
        if mk is not None:
            entry["manual_kills"] = max(0, int(mk))
        md = v.get("manual_dead")
        if md is not None:
            entry["manual_dead"] = bool(md)
        clean[slot] = entry
    cfg["slots"] = clean
    return cfg


def load_tournament_config(path=TOURNAMENT_CONFIG, retries=8, retry_delay=0.4):
    last_exc = None
    for attempt in range(retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()

            if not raw:
                raise ValueError("file is empty (mid-write?)")

            cfg = json.loads(raw)
            cfg = _parse_config(cfg)
            print(f"[CONFIG] Loaded: tournament='{cfg.get('tournament')}' "
                  f"map='{cfg.get('map','erangel')}' "
                  f"teams={cfg['total_teams']} slots={list(cfg['slots'].keys())[:6]}...")
            return cfg

        except FileNotFoundError:
            if attempt == 0:
                print(f"[CONFIG] {path} not found — waiting for dashboard...")
            last_exc = "not found"

        except (json.JSONDecodeError, ValueError) as exc:
            if attempt == 0:
                print(f"[CONFIG] {path} not ready yet ({exc}) — retrying...")
            last_exc = exc

        except Exception as exc:
            print(f"[CONFIG] Unexpected error reading {path}: {exc}")
            last_exc = exc

        time.sleep(retry_delay)

    print(f"[CONFIG] Could not load {path} after {retries} attempts "
          f"(last error: {last_exc}) — using defaults")
    return copy.deepcopy(_CONFIG_DEFAULTS)


def players_for_slot(slot, cfg):
    entry = cfg["slots"].get(slot)
    if entry is None:
        return 4
    return entry["players"]


def map_for_tournament(cfg):
    """Return the current map name (lowercase). Defaults to 'erangel'."""
    return str(cfg.get("map", "erangel")).strip().lower()


def manual_kills_for_slot(slot, cfg):
    entry = cfg["slots"].get(slot)
    if entry is None:
        return None
    mk = entry.get("manual_kills")
    return None if mk is None else max(0, int(mk))


def manual_dead_for_slot(slot, cfg):
    entry = cfg["slots"].get(slot)
    if entry is None:
        return None
    v = entry.get("manual_dead")
    return None if v is None else bool(v)


def clear_manual_dead_in_config(slot, path=TOURNAMENT_CONFIG):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[CFG] clear_manual_dead: could not read {path}: {exc}")
        return

    slot_key = str(slot).zfill(2)

    if "slots" in data and slot_key in data["slots"]:
        data["slots"][slot_key].pop("manual_dead", None)

    for tid, t in data.get("tournaments", {}).items():
        if "slots" in t and slot_key in t["slots"]:
            t["slots"][slot_key].pop("manual_dead", None)

    tmp = path + ".tmp"
    try:
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
    except Exception as exc:
        print(f"[CFG] clear_manual_dead: write failed for {path}: {exc}")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def active_card_count(cfg):
    return cfg["total_teams"]