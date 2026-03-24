import cv2
import numpy as np
import pytesseract

from config import PIXEL_DIFF_SKIP
from utils import crop, scale_rect
from badge import is_playzone_banner


def preprocess_digits(bgr, scale=2):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (g.shape[1]*scale, g.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = 255 - th
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return th


def ocr_digits(img_bin):
    cfg = r"--oem 1 --psm 8"
    txt = pytesseract.image_to_string(img_bin, config=cfg).strip()

    replacements_multi = {
        "QO": "0",
        "0)": "0",
        "Q)": "0",
        "Q]": "0",
        "O": "0",
        "D": "0",
        "01": "0"
    }
    for k, v in replacements_multi.items():
        txt = txt.replace(k, v)
    replacements_single = {
        "{": "1",
        "|": "1",
        "]": "1",
        "?": "2",
        "Q": "0",
        "Z": "2",
    }
    txt = "".join(replacements_single.get(ch, ch) for ch in txt)
    txt = "".join(ch for ch in txt if ch.isdigit())
    return txt


def parse_ocr_int(raw_txt):
    if raw_txt is None:
        return None
    if isinstance(raw_txt, int):
        return raw_txt
    txt = "".join(ch for ch in str(raw_txt) if ch.isdigit())
    return int(txt) if txt else None


def ocr_kill_value(roi_bgr):
    th = preprocess_digits(roi_bgr, scale=2)
    return ocr_digits(th)


def ocr_one_card(args):
    i, c, img, sx, sy, stable_slots, locked_by_slot, prev_rois, player_count = args

    slot = stable_slots[i]
    if not slot or locked_by_slot.get(slot, False):
        return i, slot, [None] * 4

    card_abs = scale_rect(c["card"], sx, sy)
    card_img = crop(img, card_abs)

    if is_playzone_banner(card_img):
        return i, slot, None

    vals = [None] * 4

    for pi in range(player_count):
        rr  = scale_rect(c["kills"][pi], sx, sy)
        roi = crop(card_img, rr)

        key  = (i, pi)
        prev = prev_rois.get(key)

        if prev is not None and roi.shape == prev.shape:
            diff = np.mean(np.abs(roi.astype(np.int16) - prev.astype(np.int16))) / 255.0
            if diff < PIXEL_DIFF_SKIP:
                cached_val = prev_rois.get(("val", key))
                if cached_val is not None:
                    vals[pi] = cached_val
                    continue

        prev_rois[key] = roi.copy()
        v              = ocr_kill_value(roi)
        prev_rois[("val", key)] = v
        vals[pi] = v

    return i, slot, vals
