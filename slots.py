import os

import cv2
import numpy as np

from config import DIG_W, DIG_H, SLOT_MIN_SCORE
from utils import crop, scale_rect, most_common_non_none


def fit_digit(bin_img, pad=2):
    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(bin_img.shape[1]-1, x1 + pad)
    y1 = min(bin_img.shape[0]-1, y1 + pad)
    roi = bin_img[y0:y1+1, x0:x1+1]
    roi = cv2.resize(roi, (DIG_W, DIG_H), interpolation=cv2.INTER_NEAREST)
    _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return roi


def _load_template(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    _, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return fit_digit(im)


def load_templates_multi(dir_path):
    tmpls = {str(d): [] for d in range(10)}
    for d in range(10):
        p  = os.path.join(dir_path, f"{d}.png")
        im = _load_template(p)
        if im is not None:
            tmpls[str(d)].append(im)

    samples_dir = os.path.join(dir_path, "samples")
    if os.path.isdir(samples_dir):
        for fn in os.listdir(samples_dir):
            if not fn.lower().endswith(".png"):
                continue
            digit = fn[0]
            if digit not in tmpls:
                continue
            im = _load_template(os.path.join(samples_dir, fn))
            if im is not None:
                tmpls[digit].append(im)
    return tmpls


def match_digit_multi_xor(dimg, templates):
    if dimg is None:
        return None, 0.0
    dig = (dimg > 0).astype(np.uint8)
    best_d, best_s = None, -1.0
    for d, lst in templates.items():
        for tmpl in lst:
            t = (tmpl > 0).astype(np.uint8)
            s = 1.0 - float(np.mean(dig ^ t))
            if s > best_s:
                best_d, best_s = d, s
    return best_d, best_s


def preprocess_slot_roi(slot_roi_bgr):
    g = cv2.cvtColor(slot_roi_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = 255 - th
    if th.shape[0] > 4 and th.shape[1] > 4:
        th[0:1, :] = 0; th[-1:, :] = 0
        th[:, 0:1] = 0; th[:, -1:] = 0
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return th


def split_slot_lr_masks(th):
    H, W = th.shape[:2]
    mid  = W // 2
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    left  = np.zeros_like(th)
    right = np.zeros_like(th)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 25:
            continue
        cx = x + w / 2.0
        if cx < mid:
            cv2.drawContours(left,  [c], -1, 255, -1)
        else:
            cv2.drawContours(right, [c], -1, 255, -1)

    if np.sum(left > 0) == 0 or np.sum(right > 0) == 0:
        return fit_digit(th[:, :mid]), fit_digit(th[:, mid:])
    return fit_digit(left), fit_digit(right)


def read_slot_tm(slot_roi_bgr, slot_tmpls):
    th = preprocess_slot_roi(slot_roi_bgr)
    l, r = split_slot_lr_masks(th)
    d1, s1 = match_digit_multi_xor(l, slot_tmpls)
    d2, s2 = match_digit_multi_xor(r, slot_tmpls)

    if s1 < SLOT_MIN_SCORE or s2 < SLOT_MIN_SCORE:
        return None, (s1, s2)

    v = int(d1 + d2)
    if 1 <= v <= 25:
        return f"{v:02d}", (s1, s2)
    return None, (s1, s2)


def repair_slot_sequence(slots, page_range):
    lo, hi = page_range
    n = len(slots)

    def to_int(s):
        try:
            return int(s)
        except (TypeError, ValueError):
            return None

    vals    = [to_int(s) for s in slots]
    changed = True

    while changed:
        changed = False
        for i in range(n):
            left_v  = vals[i - 1] if i > 0     else None
            right_v = vals[i + 1] if i < n - 1 else None

            left_says  = (left_v  + 1) if left_v  is not None else None
            right_says = (right_v - 1) if right_v is not None else None

            if left_says is not None and right_says is not None:
                if left_says == right_says:
                    if vals[i] != left_says and lo <= left_says <= hi:
                        vals[i] = left_says
                        changed = True
            elif left_says is not None:
                if lo <= left_says <= hi:
                    fits_right = (right_v is None or vals[i] == right_v - 1)
                    if vals[i] != left_says and not fits_right:
                        vals[i] = left_says
                        changed = True
            elif right_says is not None:
                if lo <= right_says <= hi:
                    fits_left = (left_v is None or vals[i] == left_v + 1)
                    if vals[i] != right_says and not fits_left:
                        vals[i] = right_says
                        changed = True

    return [
        f"{v:02d}" if (v is not None and lo <= v <= hi) else None
        for v in vals
    ]


def read_slots_page(frame, rois, sx, sy, slot_tmpls, page_range, max_cards=9):
    slots, scores = [], []
    cards = rois["cards"][:max_cards]
    for card in cards:
        card_abs = scale_rect(card["card"], sx, sy)
        card_img = crop(frame, card_abs)
        slot_r   = scale_rect(card["slot"], sx, sy)
        slot_img = crop(card_img, slot_r)
        s, sc    = read_slot_tm(slot_img, slot_tmpls)
        slots.append(s)
        scores.append(sc)
    slots = repair_slot_sequence(slots, page_range)
    return slots, scores


def page_score(slots, scores, lo, hi):
    s = 0.0
    for slot, (a, b) in zip(slots, scores):
        if slot is None:
            continue
        v = int(slot)
        if lo <= v <= hi:
            s += min(a, b)
    return s
