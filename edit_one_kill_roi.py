import json
import cv2

# ---- change these ----
FRAME_PATH = "Screenshot 2026-03-23 22-12-45.png"   # or frame_page2.png
ROIS_PATH = "rois_page1.json"    # or rois_page2.json
CARD_INDEX = 6 # 1..9 (which card in the 3x3 grid)
PLAYER_IDX = 3 # 1..4 (which player row)
SCALE = 5
# ----------------------


def crop(img, r):
    x, y, w, h = r
    return img[y:y+h, x:x+w]


def select_roi_on_scaled(img, title, scale):
    h, w = img.shape[:2]

    disp = cv2.resize(
        img,
        (w * scale, h * scale),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, disp.shape[1], disp.shape[0])

    roi = cv2.selectROI(title, disp, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)

    x, y, rw, rh = map(int, roi)

    if rw == 0 or rh == 0:
        raise RuntimeError("Cancelled")

    return [
        int(x / scale),
        int(y / scale),
        int(rw / scale),
        int(rh / scale)
    ]


def main():
    frame = cv2.imread(FRAME_PATH)

    if frame is None:
        raise FileNotFoundError(FRAME_PATH)

    with open(ROIS_PATH, "r", encoding="utf-8") as f:
        rois = json.load(f)

    card = rois["cards"][CARD_INDEX - 1]

    card_img = crop(frame, card["card"])

    old = card["kills"][PLAYER_IDX - 1]
    print("Old ROI:", old)

    new_roi = select_roi_on_scaled(
        card_img,
        f"Card {CARD_INDEX} - select NEW kills ROI for P{PLAYER_IDX}",
        SCALE
    )

    print("New ROI:", new_roi)

    card["kills"][PLAYER_IDX - 1] = new_roi

    with open(ROIS_PATH, "w", encoding="utf-8") as f:
        json.dump(rois, f, indent=2)

    print("Saved:", ROIS_PATH)


if __name__ == "__main__":
    main()