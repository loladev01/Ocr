import cv2
import numpy as np

from config import PLAYZONE_YELLOW_RATIO, EDGE_THR


def badge_edge_ratio(card_img, badge_roi):
    x, y, w, h = badge_roi
    roi = card_img[y:y+h, x:x+w]
    g   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    e   = cv2.Canny(g, 50, 150)
    return float(np.mean(e > 0))


def is_playzone_banner(card_img_bgr):
    hsv   = cv2.cvtColor(card_img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([18, 160, 160], dtype=np.uint8)
    upper = np.array([38, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    ratio = float(np.count_nonzero(mask)) / max(mask.size, 1)
    return ratio >= PLAYZONE_YELLOW_RATIO
