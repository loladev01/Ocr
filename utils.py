import os
import json
import time
from collections import Counter

import cv2
import numpy as np


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

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        print(f"[JSON] write failed for {path}: {exc}")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def crop(img, r):
    x, y, w, h = r
    return img[y:y+h, x:x+w]


def scale_rect(r, sx, sy):
    x, y, w, h = r
    return [int(x*sx), int(y*sy), int(w*sx), int(h*sy)]


def resize_for_display(img, target_w):
    h, w = img.shape[:2]
    sc = target_w / w
    return cv2.resize(img, (target_w, int(h * sc)), interpolation=cv2.INTER_AREA)


def draw_text_bg(img, text, x, y, color=(255, 255, 255), scale=0.7, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def most_common_non_none(dq):
    c = Counter([x for x in dq if x is not None])
    return c.most_common(1)[0][0] if c else None
