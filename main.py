import os
import json
import time
import threading
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pytesseract
from mss import mss

from config import (
    MONITOR_INDEX, REF_W, REF_H, CAPTURE_FPS, DISPLAY_WIDTH,
    ROIS_PAGE1, ROIS_PAGE2, SLOT_TEMPL_DIR, TESSERACT_CMD,
    TOURNAMENT_CONFIG, OUT_JSON, GAME_RECT_PATH, SHOTS_DIR,
    USE_GAME_RECT_IF_EXISTS,
    SLOT_HZ, BADGE_HZ, KILLS_HZ, JSON_HZ,
    OCR_WORKERS, PIXEL_DIFF_SKIP,
    KILL_SMOOTH, TOTAL_CONFIRM, MAX_KILL_JUMP,
    PLAYZONE_YELLOW_RATIO, SLOT_LOCK_SEC, MIN_VALID_FOR_LOCK,
    EDGE_THR, BADGE_CONFIRM, BADGE_DECAY,
    FINAL_KILL_SCAN_DELAY_SEC, CFG_RELOAD_INTERVAL,
    MERGED_PAGE1, MERGED_PAGE2,          # NEW
    PLAYER_HZ,                           # NEW
    load_tournament_config, active_card_count,
    players_for_slot, manual_kills_for_slot, manual_dead_for_slot,
    clear_manual_dead_in_config,
    map_for_tournament,                  # NEW
)
from utils import (
    atomic_write_json, crop, scale_rect,
    resize_for_display, draw_text_bg, most_common_non_none,
)
from slots import (
    load_templates_multi, read_slots_page, page_score, repair_slot_sequence,
)
from badge import badge_edge_ratio, is_playzone_banner
from kills import ocr_one_card, parse_ocr_int
from player import detect_players_page, reset_player_locks       # NEW


def build_payload(state, cfg):
    team_total_by_slot    = state["team_total_by_slot"]
    dead_by_slot          = state["dead_by_slot"]
    locked_by_slot        = state["locked_by_slot"]
    death_time_by_slot    = state.get("death_time_by_slot", {})
    player_status_by_slot = state.get("player_status_by_slot", {})  # NEW

    active_slots = sorted(cfg["slots"].keys()) if cfg["slots"] else [
        f"{n:02d}" for n in range(1, cfg["total_teams"] + 1)
    ]

    p1lo, p1hi = cfg["page1Range"]
    p2lo, p2hi = cfg["page2Range"]
    active_slots = [
        s for s in active_slots
        if p1lo <= int(s) <= p1hi or p2lo <= int(s) <= p2hi
    ]

    total_teams = cfg["total_teams"]
    dead_sorted = sorted(death_time_by_slot.items(), key=lambda x: x[1])
    elim_rank = {slot: total_teams - i for i, (slot, _) in enumerate(dead_sorted)}

    out = {
        "Meta": {
            "SlotsLocked": bool(state["slots_locked"]),
            "Page": state["page"],
            "Tournament": cfg.get("tournament", ""),
            "TotalTeams": cfg["total_teams"],
        },
        "Teams": []
    }
    for slot in active_slots:
        n    = int(slot)
        ocr_dead  = bool(dead_by_slot.get(slot, False))
        md        = manual_dead_for_slot(slot, cfg)
        dead      = md if md is not None else ocr_dead
        mk             = manual_kills_for_slot(slot, cfg)
        ocr_total      = int(team_total_by_slot.get(slot, 0))
        total          = max(mk, ocr_total) if mk is not None else ocr_total
        is_manual_kill = mk is not None
        is_manual_dead = md is not None

        # ── Player alive/dead status ───────────────────────────────── NEW
        pinfo = player_status_by_slot.get(slot, {})
        player_status_out = {
            k: v for k, v in pinfo.items() if k.startswith("player")
        }
        alive_count = pinfo.get("alive_count", None)
        # ──────────────────────────────────────────────────────────────

        out["Teams"].append({
            "Slot No":        n,
            "Team":           cfg["slots"].get(slot, {}).get("team", ""),
            "Players":        players_for_slot(slot, cfg),
            "Dead":           dead,
            "Alive":          not dead,
            "Locked":         bool(locked_by_slot.get(slot, False)),
            "TeamTotal":      total,
            "ManualOverride": is_manual_kill,
            "ManualDead":     is_manual_dead,
            "ElimRank":       elim_rank.get(slot, None),
            "DeathTime":      death_time_by_slot.get(slot, None),
            "PlayerStatus":   player_status_out,    # NEW  e.g. {"player1":"ALIVE","player2":"DEAD",...}
            "AliveCount":     alive_count,           # NEW  e.g. 3
        })
    return out


def snapshot(state):
    snap = [int(state["slots_locked"]), -1 if state["page"] is None else int(state["page"])]
    for n in range(1, 26):
        slot = f"{n:02d}"
        snap.append(int(state["team_total_by_slot"].get(slot, 0)))
        snap.append(1 if state["dead_by_slot"].get(slot, False)   else 0)
        snap.append(1 if state["locked_by_slot"].get(slot, False) else 0)
        snap.append(1 if slot in state.get("death_time_by_slot", {}) else 0)
    return tuple(snap)


def main():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    os.makedirs(SHOTS_DIR, exist_ok=True)

    # ── Load tournament config from dashboard ─────────────────────
    cfg = load_tournament_config(TOURNAMENT_CONFIG)
    PAGE1_RANGE   = tuple(cfg["page1Range"])
    PAGE2_RANGE   = tuple(cfg["page2Range"])
    SWITCH_MARGIN = 0.5
    MAX_CARDS     = active_card_count(cfg)

    print(f"[CONFIG] page1Range={PAGE1_RANGE}  page2Range={PAGE2_RANGE}  max_cards={MAX_CARDS}")
    # ─────────────────────────────────────────────────────────────

    rois1      = json.load(open(ROIS_PAGE1,   "r", encoding="utf-8"))
    rois2      = json.load(open(ROIS_PAGE2,   "r", encoding="utf-8"))
    merged1    = json.load(open(MERGED_PAGE1, "r", encoding="utf-8"))   # NEW
    merged2    = json.load(open(MERGED_PAGE2, "r", encoding="utf-8"))   # NEW
    slot_tmpls = load_templates_multi(SLOT_TEMPL_DIR)

    game_rect = None
    if USE_GAME_RECT_IF_EXISTS and os.path.exists(GAME_RECT_PATH):
        game_rect = json.load(open(GAME_RECT_PATH, "r", encoding="utf-8"))["gameRect"]
        print("Using game_rect:", game_rect)

    ocr_pool = ThreadPoolExecutor(max_workers=OCR_WORKERS)

    latest_lock = threading.Lock()
    latest = {"img": None, "w": 0, "h": 0}

    state_lock = threading.Lock()
    state = {
        "armed":                 False,
        "page":                  None,
        "rois":                  rois1,
        "stable_slots":          [None] * 9,
        "slot_scores":           [(0.0, 0.0)] * 9,
        "slots_locked":          False,
        "edge_ratios":           [None] * 9,
        "kills_by_card":         [[None, None, None, None] for _ in range(9)],
        "raw_ocr_by_card":       [[None, None, None, None] for _ in range(9)],
        "team_total_by_slot":    {},
        "dead_by_slot":          {},
        "locked_by_slot":        {},
        "lock_due_at":           {},
        "death_time_by_slot":    {},
        "player_status_by_slot": {},    # NEW  slot -> {player1:ALIVE, player2:DEAD, ...}
    }

    stop = threading.Event()

    slot_hist = {
        1: [deque(maxlen=10) for _ in range(9)],
        2: [deque(maxlen=10) for _ in range(9)],
    }

    slots_locked  = False
    locked_page   = None
    locked_slots  = None
    lock_since    = None
    prev_sig      = None
    last_page     = 2

    pending_total       = {}
    badge_seen          = {}
    prev_kill_rois      = {}
    kill_val_hist       = {}
    kill_confirmed      = {}
    manual_kills_synced = {}

    last_json_snap = None

    try:
        with open(OUT_JSON, "r", encoding="utf-8") as _f:
            _prev = json.load(_f)
        for _team in _prev.get("Teams", []):
            _slot = str(_team.get("Slot No", 0)).zfill(2)
            if _team.get("Dead"):
                state["dead_by_slot"][_slot] = True
            if _team.get("Locked"):
                state["locked_by_slot"][_slot] = True
            _total = _team.get("TeamTotal", 0)
            if _total:
                state["team_total_by_slot"][_slot] = int(_total)
            _rank = _team.get("ElimRank")
            if _rank is not None and _team.get("Dead"):
                state["death_time_by_slot"][_slot] = float(1000000 - _rank)
        print(f"[STATE] Restored previous match_state.json")
    except Exception:
        with state_lock:
            atomic_write_json(OUT_JSON, build_payload(state, cfg))

    last_json_snap = snapshot(state)

    def reset_state():
        nonlocal slots_locked, locked_page, locked_slots, lock_since, prev_sig, last_page
        nonlocal pending_total, badge_seen, last_json_snap
        slots_locked = False; locked_page = None; locked_slots = None
        lock_since   = None;  prev_sig    = None; last_page    = 2
        pending_total = {}; badge_seen = {}
        prev_kill_rois.clear(); kill_val_hist.clear(); kill_confirmed.clear()
        manual_kills_synced.clear()
        reset_player_locks()   # clear per-player dead locks on match reset

        with state_lock:
            state["page"]                  = None
            state["rois"]                  = rois1
            state["stable_slots"]          = [None] * 9
            state["slot_scores"]           = [(0.0, 0.0)] * 9
            state["slots_locked"]          = False
            state["edge_ratios"]           = [None] * 9
            state["kills_by_card"]         = [[None, None, None, None] for _ in range(9)]
            state["raw_ocr_by_card"]       = [[None, None, None, None] for _ in range(9)]
            state["team_total_by_slot"]    = {}
            state["dead_by_slot"]          = {}
            state["locked_by_slot"]        = {}
            state["lock_due_at"]           = {}
            state["death_time_by_slot"]    = {}
            state["player_status_by_slot"] = {}    # NEW

        reset_player_locks()   # clear dead-locks on match reset
        atomic_write_json(OUT_JSON, build_payload(state, cfg))
        last_json_snap = snapshot(state)

    def capture_loop():
        period = 1.0 / CAPTURE_FPS
        with mss() as sct:
            mon = sct.monitors[MONITOR_INDEX]
            while not stop.is_set():
                t0  = time.time()
                img = np.array(sct.grab(mon))[:, :, :3]
                if game_rect:
                    gx, gy, gw, gh = game_rect
                    img = img[gy:gy+gh, gx:gx+gw]
                h, w = img.shape[:2]
                with latest_lock:
                    latest["img"] = img
                    latest["w"]   = w
                    latest["h"]   = h
                dt = time.time() - t0
                if period - dt > 0:
                    time.sleep(period - dt)

    def slot_loop():
        nonlocal slots_locked, locked_page, locked_slots, lock_since, prev_sig, last_page

        period = 1.0 / SLOT_HZ
        while not stop.is_set():
            t0 = time.time()

            with state_lock:
                if not state["armed"]:
                    time.sleep(0.05); continue

            with latest_lock:
                img = None if latest["img"] is None else latest["img"].copy()
                w   = latest["w"]; h = latest["h"]
            if img is None:
                time.sleep(0.05); continue

            sx, sy = w / REF_W, h / REF_H
            now    = time.time()

            if slots_locked:
                page         = locked_page
                rois         = rois1 if page == 1 else rois2
                stable_slots = locked_slots
                slot_scores  = [(0.0, 0.0)] * 9
            else:
                slots1, scores1 = read_slots_page(
                    img, rois1, sx, sy, slot_tmpls, PAGE1_RANGE, max_cards=MAX_CARDS)
                slots2, scores2 = read_slots_page(
                    img, rois2, sx, sy, slot_tmpls, PAGE2_RANGE, max_cards=MAX_CARDS)

                s1 = page_score(slots1, scores1, *PAGE1_RANGE)
                s2 = page_score(slots2, scores2, *PAGE2_RANGE)

                if last_page == 1:
                    last_page = 2 if (s2 > s1 + SWITCH_MARGIN) else 1
                else:
                    last_page = 1 if (s1 > s2 + SWITCH_MARGIN) else 2

                if last_page == 1:
                    page, rois, slots, slot_scores = 1, rois1, slots1, scores1
                else:
                    page, rois, slots, slot_scores = 2, rois2, slots2, scores2

                stable_slots = []
                for i in range(len(slots)):
                    if slots[i] is not None:
                        slot_hist[page][i].append(slots[i])
                    stable_slots.append(
                        most_common_non_none(slot_hist[page][i]) or slots[i])

                stable_slots = repair_slot_sequence(
                    stable_slots,
                    PAGE1_RANGE if page == 1 else PAGE2_RANGE,
                )

                sig       = (page, tuple(stable_slots))
                valid_cnt = sum(1 for s in stable_slots if s is not None)

                if sig == prev_sig and valid_cnt >= MIN_VALID_FOR_LOCK:
                    if lock_since is None:
                        lock_since = now
                    elif (now - lock_since) >= SLOT_LOCK_SEC:
                        slots_locked  = True
                        locked_page   = page
                        locked_slots  = list(stable_slots)
                        print(f"[LOCK] page={locked_page} slots={locked_slots}")
                else:
                    prev_sig   = sig
                    lock_since = None

            with state_lock:
                state["page"]         = page
                state["rois"]         = rois
                state["stable_slots"] = stable_slots
                state["slot_scores"]  = slot_scores
                state["slots_locked"] = slots_locked

            dt = time.time() - t0
            if period - dt > 0:
                time.sleep(period - dt)

    def badge_loop():
        period = 1.0 / BADGE_HZ
        last_cfg_reload = 0.0
        while not stop.is_set():
            t0 = time.time()

            if t0 - last_cfg_reload >= CFG_RELOAD_INTERVAL:
                try:
                    new_cfg = load_tournament_config(TOURNAMENT_CONFIG)
                    cfg.update(new_cfg)
                except Exception as exc:
                    print(f"[CFG] reload failed: {exc}")
                last_cfg_reload = t0

            with state_lock:
                armed          = state["armed"]
                page           = state["page"]
                rois           = state["rois"]
                stable_slots   = list(state["stable_slots"])
                dead_by_slot   = dict(state["dead_by_slot"])
                locked_by_slot = dict(state["locked_by_slot"])
                lock_due_at    = dict(state["lock_due_at"])

            if not armed or page is None:
                time.sleep(0.05); continue

            with latest_lock:
                img = None if latest["img"] is None else latest["img"].copy()
                w   = latest["w"]; h = latest["h"]
            if img is None:
                time.sleep(0.05); continue

            sx, sy = w / REF_W, h / REF_H
            now    = time.time()
            edge_ratios = [None] * 9

            cards = rois["cards"][:MAX_CARDS]
            for i, c in enumerate(cards):
                slot = stable_slots[i] if i < len(stable_slots) else None
                if not slot:
                    continue

                md_val = manual_dead_for_slot(slot, cfg)
                if md_val is not None:
                    if md_val:
                        dead_by_slot[slot] = True
                        with state_lock:
                            if slot not in state["death_time_by_slot"]:
                                state["death_time_by_slot"][slot] = now
                        continue
                    else:
                        dead_by_slot[slot] = False
                        locked_by_slot[slot] = False
                        lock_due_at.pop(slot, None)
                        with state_lock:
                            state["death_time_by_slot"].pop(slot, None)

                if locked_by_slot.get(slot, False):
                    continue

                card_abs = scale_rect(c["card"], sx, sy)
                card_img = crop(img, card_abs)

                er             = badge_edge_ratio(card_img, c["badge"])
                edge_ratios[i] = er
                present        = (er >= EDGE_THR)
                key            = (page, i)

                if not dead_by_slot.get(slot, False):
                    if present:
                        badge_seen[key] = badge_seen.get(key, 0) + 1
                    else:
                        badge_seen[key] = max(badge_seen.get(key, 0) - BADGE_DECAY, 0)

                    if badge_seen.get(key, 0) >= BADGE_CONFIRM:
                        dead_by_slot[slot]  = True
                        lock_due_at[slot]   = now + FINAL_KILL_SCAN_DELAY_SEC
                        locked_by_slot.setdefault(slot, False)

                        with state_lock:
                            if slot not in state["death_time_by_slot"]:
                                state["death_time_by_slot"][slot] = now

                        if manual_dead_for_slot(slot, cfg) is not None:
                            cfg["slots"].get(slot, {}).pop("manual_dead", None)
                            threading.Thread(
                                target=clear_manual_dead_in_config,
                                args=(slot,),
                                daemon=True,
                            ).start()
                            print(f"[OVERRIDE] Slot {slot} — manual ALIVE cancelled: OCR confirmed DEAD")

            with state_lock:
                state["edge_ratios"]    = edge_ratios
                state["dead_by_slot"]   = dead_by_slot
                state["locked_by_slot"] = locked_by_slot
                state["lock_due_at"]    = lock_due_at

            dt = time.time() - t0
            if period - dt > 0:
                time.sleep(period - dt)

    def kills_loop():
        period = 1.0 / KILLS_HZ
        while not stop.is_set():
            t0 = time.time()

            with state_lock:
                armed          = state["armed"]
                page           = state["page"]
                rois           = state["rois"]
                stable_slots   = list(state["stable_slots"])
                team_total     = dict(state["team_total_by_slot"])
                dead_by_slot   = dict(state["dead_by_slot"])
                locked_by_slot = dict(state["locked_by_slot"])
                lock_due_at    = dict(state["lock_due_at"])

            for _slot in list(stable_slots):
                if not _slot:
                    continue
                _md = manual_dead_for_slot(_slot, cfg)
                if _md is not None:
                    dead_by_slot[_slot] = _md
                    if not _md:
                        locked_by_slot[_slot] = False
                        lock_due_at.pop(_slot, None)

            kills_by_card   = [[None, None, None, None] for _ in range(9)]
            raw_ocr_by_card = [[None, None, None, None] for _ in range(9)]

            if not armed or page is None:
                time.sleep(0.05); continue

            with latest_lock:
                img = None if latest["img"] is None else latest["img"].copy()
                w   = latest["w"]; h = latest["h"]
            if img is None:
                time.sleep(0.05); continue

            sx, sy = w / REF_W, h / REF_H
            now    = time.time()

            cards = rois["cards"][:MAX_CARDS]

            futures = {
                ocr_pool.submit(
                    ocr_one_card,
                    (
                        i, c, img, sx, sy,
                        stable_slots, locked_by_slot, prev_kill_rois,
                        players_for_slot(stable_slots[i] or "00", cfg),
                    )
                ): i
                for i, c in enumerate(cards)
            }

            for fut in as_completed(futures):
                try:
                    i, slot, raw_vals = fut.result()
                except Exception as exc:
                    print(f"[OCR] card {futures[fut]} error: {exc}")
                    continue

                if not slot or raw_vals is None:
                    continue

                n_players     = players_for_slot(slot, cfg)
                smoothed_vals = []
                raw_ocr_by_card[i] = list(raw_vals)

                for pi, raw_v in enumerate(raw_vals):
                    hkey = (slot, pi)
                    if hkey not in kill_val_hist:
                        kill_val_hist[hkey]  = deque(maxlen=KILL_SMOOTH)
                    if hkey not in kill_confirmed:
                        kill_confirmed[hkey] = 0
                    dq   = kill_val_hist[hkey]
                    conf = kill_confirmed[hkey]
                    raw_v = parse_ocr_int(raw_v)

                    if pi >= n_players:
                        smoothed_vals.append(None)
                        continue

                    if raw_v is not None:
                        if raw_v < conf - 1:
                            dq.clear()
                            kill_confirmed[hkey] = raw_v
                            conf = raw_v
                        elif conf > 0 and raw_v > conf + MAX_KILL_JUMP:
                            raw_v = None

                        if raw_v is not None:
                            dq.append(raw_v)

                    if dq:
                        counts        = Counter(dq)
                        twice_or_more = {v: c for v, c in counts.items() if c >= 2}
                        smoothed      = (max(twice_or_more.keys()) if twice_or_more
                                         else most_common_non_none(dq))
                    else:
                        smoothed = None

                    if smoothed is not None and smoothed > kill_confirmed[hkey]:
                        kill_confirmed[hkey] = smoothed

                    smoothed_vals.append(smoothed)

                kills_by_card[i] = smoothed_vals

                mk_now = manual_kills_for_slot(slot, cfg)
                if mk_now is not None:
                    if manual_kills_synced.get(slot) != mk_now:
                        team_total[slot] = mk_now
                        manual_kills_synced[slot] = mk_now
                        pending_total.pop(slot, None)
                        for _pi in range(4):
                            kill_val_hist.pop((slot, _pi), None)
                elif slot in manual_kills_synced:
                    del manual_kills_synced[slot]

                cur_total  = sum(v for v in smoothed_vals[:n_players] if v is not None)
                prev_total = team_total.get(slot, 0)

                if cur_total != prev_total:
                    cand, cnt = pending_total.get(slot, (cur_total, 0))
                    if cand != cur_total:
                        cand, cnt = cur_total, 1
                    else:
                        cnt += 1
                    pending_total[slot] = (cand, cnt)

                    if cnt >= TOTAL_CONFIRM:
                        direction    = "\U0001F4C8" if cur_total > prev_total else "\U0001F4C9"
                        player_lines = ""
                        for _pi in range(n_players):
                            _v    = smoothed_vals[_pi] if _pi < len(smoothed_vals) else None
                            _vstr = str(_v) if _v is not None else "??"
                            player_lines += f"  P{_pi+1}: {_vstr}\n"
                        team_name = cfg["slots"].get(slot, {}).get("team", slot)
                        team_total[slot] = cur_total
                        pending_total.pop(slot, None)
                else:
                    pending_total.pop(slot, None)

            for slot, due in list(lock_due_at.items()):
                if (dead_by_slot.get(slot, False)
                        and not locked_by_slot.get(slot, False)
                        and manual_dead_for_slot(slot, cfg) is None
                        and now >= due):
                    locked_by_slot[slot] = True

            with state_lock:
                state["kills_by_card"]      = kills_by_card
                state["raw_ocr_by_card"]    = raw_ocr_by_card
                state["team_total_by_slot"] = team_total
                state["locked_by_slot"]     = locked_by_slot
                state["lock_due_at"]        = lock_due_at

            dt = time.time() - t0
            if period - dt > 0:
                time.sleep(period - dt)

    # ── NEW: player_loop ──────────────────────────────────────────────────────
    def player_loop():
        period = 1.0 / PLAYER_HZ
        while not stop.is_set():
            t0 = time.time()

            with state_lock:
                armed        = state["armed"]
                page         = state["page"]
                stable_slots = list(state["stable_slots"])

            if not armed or page is None:
                time.sleep(0.05); continue

            with latest_lock:
                img = None if latest["img"] is None else latest["img"].copy()

            if img is None:
                time.sleep(0.05); continue

            # Use merged_page1 or merged_page2 (they have "players" ROIs per card)
            merged_rois = merged1 if page == 1 else merged2

            player_status_by_slot = detect_players_page(
                img,
                merged_rois,
                stable_slots,
                players_for_slot,
                cfg,
                max_cards=MAX_CARDS,
                map_name=map_for_tournament(cfg),   # NEW — drives dead-lock
            )

            with state_lock:
                state["player_status_by_slot"] = player_status_by_slot

            dt = time.time() - t0
            if period - dt > 0:
                time.sleep(period - dt)
    # ─────────────────────────────────────────────────────────────────────────

    def json_loop():
        nonlocal last_json_snap
        period = 1.0 / JSON_HZ
        while not stop.is_set():
            t0 = time.time()
            with state_lock:
                snap = snapshot(state)
            if snap != last_json_snap:
                with state_lock:
                    atomic_write_json(OUT_JSON, build_payload(state, cfg))
                last_json_snap = snap
            dt = time.time() - t0
            if period - dt > 0:
                time.sleep(period - dt)

    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=slot_loop,    daemon=True).start()
    threading.Thread(target=badge_loop,   daemon=True).start()
    threading.Thread(target=kills_loop,   daemon=True).start()
    threading.Thread(target=player_loop,  daemon=True).start()   # NEW
    threading.Thread(target=json_loop,    daemon=True).start()

    cv2.namedWindow("BGMI OCR (q quit)", cv2.WINDOW_NORMAL)

    try:
        while True:
            with latest_lock:
                img = None if latest["img"] is None else latest["img"].copy()
                w   = latest["w"]; h = latest["h"]
            if img is None:
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            sx, sy = w / REF_W, h / REF_H
            with state_lock:
                armed                 = state["armed"]
                page                  = state["page"]
                rois                  = state["rois"]
                stable_slots          = list(state["stable_slots"])
                slot_scores           = list(state["slot_scores"])
                edge_ratios           = list(state["edge_ratios"])
                kills_by_card         = [list(x) for x in state["kills_by_card"]]
                raw_ocr_by_card       = [list(x) for x in state["raw_ocr_by_card"]]
                team_total            = dict(state["team_total_by_slot"])
                dead_by_slot          = dict(state["dead_by_slot"])
                locked_by_slot        = dict(state["locked_by_slot"])
                slots_locked_flag     = state["slots_locked"]
                player_status_by_slot = dict(state["player_status_by_slot"])  # NEW

            vis    = img.copy()
            header = "DETECTING PAGE..." if page is None else f"PAGE {page}"
            if slots_locked_flag:
                header += " | SLOTS LOCKED"
            header += f" | {cfg.get('tournament','?')} ({MAX_CARDS} teams)"
            header += " | RUNNING" if armed else " | PAUSED (P/SPACE)"
            draw_text_bg(
                vis, header, 20, 40,
                color=(0, 255, 0) if armed else (0, 0, 255),
                scale=1.0, thick=2,
            )

            if page is not None:
                cards = rois["cards"][:MAX_CARDS]
                for i, c in enumerate(cards):
                    cx, cy, cw, ch = scale_rect(c["card"], sx, sy)
                    cv2.rectangle(vis, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)

                    slot  = stable_slots[i] if i < len(stable_slots) else "??"
                    slot  = slot or "??"
                    s1, s2 = slot_scores[i] if i < len(slot_scores) else (0.0, 0.0)
                    dead   = dead_by_slot.get(slot, False)
                    locked = locked_by_slot.get(slot, False)
                    total  = team_total.get(slot, 0)
                    n_players = players_for_slot(slot, cfg)

                    sr = scale_rect(c["slot"], sx, sy)
                    ax, ay = cx + sr[0], cy + sr[1]
                    cv2.rectangle(vis, (ax, ay), (ax+sr[2], ay+sr[3]), (255, 0, 0), 2)
                    ty       = min(ay + sr[3] + 22, vis.shape[0]-10)
                    slot_col = (0, 255, 255) if slots_locked_flag else (255, 0, 0)
                    team_name = cfg["slots"].get(slot, {}).get("team", "")
                    label = f"{slot}{'('+team_name+')' if team_name else ''} ({s1:.2f},{s2:.2f})"
                    draw_text_bg(vis, label, ax, ty, color=slot_col, scale=0.55, thick=2)

                    br = scale_rect(c["badge"], sx, sy)
                    bx, by = cx + br[0], cy + br[1]
                    cv2.rectangle(vis, (bx, by), (bx+br[2], by+br[3]), (0, 255, 255), 2)
                    er = edge_ratios[i] if i < len(edge_ratios) else None
                    if er is not None:
                        draw_text_bg(vis, f"{er:.3f}", bx, max(by-5, 10),
                                     color=(0, 255, 255), scale=0.6, thick=2)

                    vals = kills_by_card[i] if i < len(kills_by_card) else [None]*4
                    raws = raw_ocr_by_card[i] if i < len(raw_ocr_by_card) else [None]*4
                    for p in range(4):
                        kr = scale_rect(c["kills"][p], sx, sy)
                        kx, ky = cx + kr[0], cy + kr[1]
                        if p < n_players:
                            cv2.rectangle(vis, (kx, ky), (kx+kr[2], ky+kr[3]), (255, 0, 255), 1)
                            v       = vals[p]
                            raw_txt = raws[p] if p < len(raws) else None
                            col     = (0, 0, 255) if v is None else (255, 0, 255)
                            draw_text_bg(vis, raw_txt, kx, max(ky-4, 10), color=col, scale=0.55, thick=2)
                        else:
                            cv2.rectangle(vis, (kx, ky), (kx+kr[2], ky+kr[3]), (60, 60, 60), 1)
                            draw_text_bg(vis, "—", kx, max(ky-4, 10),
                                         color=(80, 80, 80), scale=0.5, thick=1)

                    # ── NEW: draw player ROI overlay (rectangle + label) ──
                    pinfo = player_status_by_slot.get(slot, {})
                    STATUS_COLOR = {
                        "ALIVE":  (0, 255, 0),
                        "DEAD":   (0, 0, 255),
                        "UNSURE": (0, 165, 255),
                    }
                    # merged_rois_display holds the merged JSON for current page
                    merged_rois_display = merged1 if page == 1 else merged2
                    merged_cards_display = merged_rois_display.get("cards", [])
                    merged_card = merged_cards_display[i] if i < len(merged_cards_display) else {}
                    merged_players = merged_card.get("players", {})

                    for pi in range(1, n_players + 1):
                        pkey  = f"player{pi}"
                        pstat = pinfo.get(pkey)
                        roi   = merged_players.get(pkey)
                        if roi is None:
                            continue

                        col = STATUS_COLOR.get(pstat, (128, 128, 128))

                        # Player ROI coords are absolute — scale them
                        prx, pry, prw, prh = scale_rect(roi, sx, sy)

                        # Coloured rectangle around the player name ROI
                        cv2.rectangle(vis,
                                      (prx, pry),
                                      (prx + prw, pry + prh),
                                      col, 2)

                        # Label: "P1 ALIVE" / "P2 DEAD" etc. above the ROI
                        label_txt = f"P{pi} {pstat or '?'}"
                        draw_text_bg(vis, label_txt,
                                     prx, max(pry - 2, 10),
                                     color=col, scale=0.45, thick=1)
                    # ──────────────────────────────────────────────────────

                    status  = "LOCKED" if locked else ("DEAD" if dead else "ALIVE")
                    col     = (255, 255, 0) if locked else ((0, 0, 255) if dead else (0, 255, 0))
                    mk_flag = " [M]" if manual_kills_for_slot(slot, cfg) is not None else ""
                    alive_count = pinfo.get("alive_count", "?")          # NEW
                    draw_text_bg(
                        vis,
                        f"{status} total={total}{mk_flag} p={n_players} alive={alive_count}",  # NEW
                        cx+5, cy+25, color=col, scale=0.65, thick=2,
                    )

            disp = resize_for_display(vis, DISPLAY_WIDTH)
            cv2.imshow("BGMI OCR (q quit)", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("p"), ord(" ")):
                with state_lock:
                    state["armed"] = not state["armed"]
                    armed_now      = state["armed"]
                if armed_now:
                    new_cfg = load_tournament_config(TOURNAMENT_CONFIG)
                    cfg.update(new_cfg)
                    PAGE1_RANGE = tuple(cfg["page1Range"])
                    PAGE2_RANGE = tuple(cfg["page2Range"])
                    reset_state()
                    print(f"[ARM] Match started — tournament='{cfg.get('tournament')}' teams={cfg['total_teams']}")
                else:
                    print(f"[ARM] Match paused at {time.strftime('%H:%M:%S')}")
            if key == ord("r"):
                reset_state()
            if key == ord("s"):
                ts   = time.strftime("%Y%m%d_%H%M%S")
                outp = os.path.join(SHOTS_DIR, f"shot_{ts}.png")
                cv2.imwrite(outp, vis)
                print("saved:", outp)

    finally:
        stop.set()
        ocr_pool.shutdown(wait=False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()