"""
Generates a synthetic parking lot image for demo purposes.
Run once: python create_sample.py
"""

import cv2
import numpy as np

def draw_car(img, x1, y1, x2, y2, body_color):
    cx = (x1 + x2) // 2
    pad_x = (x2 - x1) // 6
    pad_y = (y2 - y1) // 8

    # Body
    cv2.rectangle(img, (x1 + pad_x, y1 + pad_y), (x2 - pad_x, y2 - pad_y), body_color, -1)

    # Windshields
    win_color = (210, 230, 215)
    win_h = (y2 - y1) // 5
    cv2.rectangle(img,
                  (x1 + pad_x + 6, y1 + pad_y + 6),
                  (x2 - pad_x - 6, y1 + pad_y + win_h),
                  win_color, -1)
    cv2.rectangle(img,
                  (x1 + pad_x + 6, y2 - pad_y - win_h),
                  (x2 - pad_x - 6, y2 - pad_y - 6),
                  win_color, -1)

    # Wheels (dark circles at corners)
    r = max((x2 - x1) // 10, 4)
    for wx in [x1 + pad_x + r, x2 - pad_x - r]:
        for wy in [y1 + pad_y + r, y2 - pad_y - r]:
            cv2.circle(img, (wx, wy), r, (18, 18, 18), -1)

    # Roof highlight
    roof_w = (x2 - x1) // 3
    roof_h = (y2 - y1) // 3
    bright = tuple(min(c + 60, 255) for c in body_color)
    cv2.rectangle(img,
                  (cx - roof_w // 2, (y1 + y2) // 2 - roof_h // 2),
                  (cx + roof_w // 2, (y1 + y2) // 2 + roof_h // 2),
                  bright, -1)


def create_parking_lot(out_path="sample_parking.jpg"):
    H, W = 620, 1080
    rng = np.random.default_rng(42)

    # ── Asphalt background ────────────────────────────────────────────────────
    base = rng.integers(48, 68, (H, W, 3), dtype=np.uint8)
    img  = base.copy()

    # Subtle cracks / texture
    for _ in range(120):
        x0 = int(rng.integers(0, W))
        y0 = int(rng.integers(0, H))
        x1 = x0 + int(rng.integers(-80, 80))
        y1 = y0 + int(rng.integers(-40, 40))
        cv2.line(img, (x0, y0), (x1, y1), (40, 40, 40), 1)

    # ── Layout: 2 rows × 6 spots ──────────────────────────────────────────────
    spot_w, spot_h = 155, 95
    margin_x = 60
    n_cols   = 6

    row_configs = [
        {
            "y0"    : 80,
            "cars"  : [True, False, True, True, False, True],
            "colors": [
                (30,  30, 180),   # blue
                None,
                (20,  90, 200),   # teal
                (55,   0, 110),   # purple
                None,
                (10, 110,  50),   # green
            ],
        },
        {
            "y0"    : 420,
            "cars"  : [False, True, False, True, True, False],
            "colors": [
                None,
                (0,   50, 180),   # red
                None,
                (0,  100, 170),   # orange
                (80,  80,  80),   # silver
                None,
            ],
        },
    ]

    for row in row_configs:
        y0 = row["y0"]
        y1 = y0 + spot_h

        # Lot surface (slightly lighter strip)
        cv2.rectangle(img,
                      (margin_x - 5, y0 - 5),
                      (margin_x + n_cols * spot_w + 5, y1 + 5),
                      (62, 62, 62), -1)

        # White parking lines
        for i in range(n_cols + 1):
            x = margin_x + i * spot_w
            cv2.line(img, (x, y0), (x, y1), (210, 210, 210), 3)
        cv2.line(img, (margin_x, y0), (margin_x + n_cols * spot_w, y0), (210, 210, 210), 3)
        cv2.line(img, (margin_x, y1), (margin_x + n_cols * spot_w, y1), (210, 210, 210), 3)

        # Cars
        for i, (has_car, color) in enumerate(zip(row["cars"], row["colors"])):
            if has_car and color is not None:
                cx1 = margin_x + i * spot_w + 6
                cy1 = y0 + 6
                cx2 = margin_x + (i + 1) * spot_w - 6
                cy2 = y1 - 6
                draw_car(img, cx1, cy1, cx2, cy2, color)

    # ── Driveway lane (horizontal stripe between rows) ─────────────────────────
    lane_y0, lane_y1 = 175, 420
    lane_color = (58, 58, 58)
    cv2.rectangle(img, (0, lane_y0), (W, lane_y1), lane_color, -1)

    # Lane centre dashes
    for x in range(100, W - 100, 80):
        cv2.line(img, (x, (lane_y0 + lane_y1) // 2),
                 (x + 45, (lane_y0 + lane_y1) // 2),
                 (200, 200, 100), 2)

    # ── Outer boundary ────────────────────────────────────────────────────────
    cv2.rectangle(img, (10, 10), (W - 10, H - 10), (150, 150, 150), 4)

    # ── Timestamp watermark (makes it look like CCTV footage) ─────────────────
    cv2.putText(img, "CAM-01  2024-11-08  10:32:47",
                (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 180), 1)

    cv2.imwrite(out_path, img)
    print(f"[OK] Sample image saved: {out_path}  ({W}x{H})")


if __name__ == "__main__":
    create_parking_lot()
