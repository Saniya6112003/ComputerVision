"""
=============================================================
  Pixel-Count Methodology for Automated Parking Detection
  Using YOLO - SRM Institute of Science and Technology
  Department: DSBS | Guide: Dr. Anna Anhumozhi
=============================================================

Phase 1: Setup & Manual ROI
Phase 2: YOLO Detection Pipeline
Phase 3: Occupancy Logic (Intersection over Union)
"""

import cv2
import numpy as np
import pickle
import json
import os
import time
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
OVERLAP_THRESHOLD = 0.40          # 40–50% IoU → spot is Occupied
MODEL_NAME        = "yolov8n.pt"  # lightweight; swap to yolov8s/m/l as needed
ROI_PICKLE_PATH   = "parking_rois.pkl"
LOG_JSON_PATH     = "occupancy_log.json"


# ─────────────────────────────────────────────
# PHASE 1 — Manual ROI Setup
# ─────────────────────────────────────────────
class ParkingROISelector:
    """
    Interactive tool: click to define parking-spot polygons.
    Left-click  → add point to current polygon
    Right-click → close & save current polygon
    'z'         → undo last point
    'c'         → clear all
    's'         → save ROIs to pickle and exit
    'q'         → quit without saving
    """

    def __init__(self, image_path: str):
        self.image_path   = image_path
        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")
        self.rois          : list[list[tuple[int,int]]] = []  # completed polygons
        self.current_pts   : list[tuple[int,int]]       = []  # in-progress polygon
        self.window_name   = "Parking ROI Selector"

    # ── mouse callback ──────────────────────────────────────────────
    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_pts.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_pts) >= 3:
                self.rois.append(self.current_pts.copy())
                print(f"[ROI] Spot #{len(self.rois)} saved with {len(self.current_pts)} pts")
            self.current_pts = []

    # ── drawing ──────────────────────────────────────────────────────
    def _draw(self) -> np.ndarray:
        canvas = self.original_img.copy()

        # finished ROIs
        for i, roi in enumerate(self.rois):
            pts = np.array(roi, np.int32)
            cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
            cx = int(np.mean([p[0] for p in roi]))
            cy = int(np.mean([p[1] for p in roi]))
            cv2.putText(canvas, f"S{i+1}", (cx-10, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # in-progress polygon
        for pt in self.current_pts:
            cv2.circle(canvas, pt, 4, (0, 0, 255), -1)
        if len(self.current_pts) > 1:
            cv2.polylines(canvas, [np.array(self.current_pts, np.int32)], False, (0, 0, 255), 1)

        # HUD
        cv2.putText(canvas, f"Spots defined: {len(self.rois)}  |  Current pts: {len(self.current_pts)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(canvas, "LClick=add pt  RClick=close  z=undo  c=clear  s=save  q=quit",
                    (10, canvas.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return canvas

    # ── public ───────────────────────────────────────────────────────
    def run(self) -> list[list[tuple[int,int]]]:
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

        while True:
            cv2.imshow(self.window_name, self._draw())
            key = cv2.waitKey(20) & 0xFF

            if key == ord('z') and self.current_pts:
                self.current_pts.pop()
            elif key == ord('c'):
                self.rois.clear(); self.current_pts.clear()
            elif key == ord('s'):
                cv2.destroyAllWindows()
                return self.rois
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return []

    @staticmethod
    def save(rois: list, path: str = ROI_PICKLE_PATH):
        with open(path, "wb") as f:
            pickle.dump(rois, f)
        print(f"[ROI] {len(rois)} parking spots saved → {path}")

    @staticmethod
    def load(path: str = ROI_PICKLE_PATH) -> list:
        if not os.path.exists(path):
            return []
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────
def bbox_to_polygon(x1, y1, x2, y2) -> np.ndarray:
    """Convert YOLO bbox corners to a 4-pt polygon."""
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.float32)


def polygon_iou(poly_a: list[tuple], bbox: tuple[int,int,int,int]) -> float:
    """
    Intersection-over-Union between a parking-spot polygon and a bounding box.
    Uses pixel-mask approach (robust to irregular spot shapes).
    """
    x1, y1, x2, y2 = bbox
    # bounding box of the union area (clipped canvas)
    xs = [p[0] for p in poly_a] + [x1, x2]
    ys = [p[1] for p in poly_a] + [y1, y2]
    mx, Mx = int(min(xs)), int(max(xs))
    my, My = int(min(ys)), int(max(ys))
    w, h   = max(Mx - mx + 1, 1), max(My - my + 1, 1)

    # draw each shape on a tiny canvas
    mask_roi = np.zeros((h, w), np.uint8)
    mask_box = np.zeros((h, w), np.uint8)

    shifted_roi = [(p[0]-mx, p[1]-my) for p in poly_a]
    cv2.fillPoly(mask_roi, [np.array(shifted_roi, np.int32)], 255)
    cv2.rectangle(mask_box, (x1-mx, y1-my), (x2-mx, y2-my), 255, -1)

    inter = cv2.bitwise_and(mask_roi, mask_box)
    union = cv2.bitwise_or(mask_roi, mask_box)

    inter_area = np.count_nonzero(inter)
    union_area = np.count_nonzero(union)
    return inter_area / union_area if union_area > 0 else 0.0


# ─────────────────────────────────────────────
# PHASE 2 & 3 — Detection + Occupancy
# ─────────────────────────────────────────────
class ParkingDetector:
    """
    Runs YOLO on each frame, determines per-spot occupancy via IoU,
    and annotates the output frame.
    """

    def __init__(self, model_path: str = MODEL_NAME,
                 rois: list | None = None,
                 threshold: float = OVERLAP_THRESHOLD):
        print(f"[MODEL] Loading {model_path} …")
        self.model     = YOLO(model_path)
        self.rois      = rois or []
        self.threshold = threshold
        self.log       : list[dict] = []          # frame-level log
        self._frame_id = 0

    # ── core ─────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        self._frame_id += 1
        h, w = frame.shape[:2]

        # ── Phase 2: YOLO inference ──────────────────────────────────
        results    = self.model(frame, verbose=False)[0]
        detections = []                                   # (x1,y1,x2,y2,cls,conf)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2, cls_id, conf))

        # ── Phase 3: IoU occupancy logic ─────────────────────────────
        spot_status: list[dict] = []
        for i, roi in enumerate(self.rois):
            best_iou   = 0.0
            best_det   = None
            for det in detections:
                x1, y1, x2, y2, cls_id, conf = det
                iou = polygon_iou(roi, (x1, y1, x2, y2))
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            occupied = best_iou >= self.threshold
            spot_status.append({
                "spot"    : i + 1,
                "occupied": occupied,
                "iou"     : round(best_iou, 3),
                "vehicle" : VEHICLE_CLASS_IDS.get(best_det[4], "?") if best_det else None
            })

        # ── Annotate frame ───────────────────────────────────────────
        annotated = self._annotate(frame.copy(), detections, spot_status)

        # stats
        total    = len(spot_status)
        occupied = sum(1 for s in spot_status if s["occupied"])
        free     = total - occupied

        stats = {
            "frame_id": self._frame_id,
            "timestamp": time.strftime("%H:%M:%S"),
            "total"   : total,
            "occupied": occupied,
            "free"    : free,
            "spots"   : spot_status
        }
        self.log.append(stats)
        return annotated, stats

    # ── drawing helpers ───────────────────────────────────────────────
    def _annotate(self, frame, detections, spot_status) -> np.ndarray:
        # draw parking spots
        for s in spot_status:
            roi    = self.rois[s["spot"]-1]
            pts    = np.array(roi, np.int32)
            color  = (0, 60, 220) if s["occupied"] else (0, 200, 60)
            cv2.polylines(frame, [pts], True, color, 2)
            cx = int(np.mean([p[0] for p in roi]))
            cy = int(np.mean([p[1] for p in roi]))
            label = f"S{s['spot']} {'OCC' if s['occupied'] else 'FREE'}"
            cv2.putText(frame, label, (cx-28, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # draw vehicle boxes
        for x1, y1, x2, y2, cls_id, conf in detections:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,165,0), 2)
            cv2.putText(frame,
                        f"{VEHICLE_CLASS_IDS[cls_id]} {conf:.2f}",
                        (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,165,0), 2)

        # HUD overlay
        h, w = frame.shape[:2]
        total    = len(spot_status)
        occupied = sum(1 for s in spot_status if s["occupied"])
        free     = total - occupied
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, 0), (260, 75), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, f"Total : {total}", (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(255,255,255),2)
        cv2.putText(frame, f"Occupied: {occupied}", (10,44), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0,80,255),2)
        cv2.putText(frame, f"Free    : {free}", (10,66), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0,220,80),2)
        return frame

    # ── persistence ───────────────────────────────────────────────────
    def save_log(self, path: str = LOG_JSON_PATH):
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"[LOG] {len(self.log)} frames saved → {path}")


# ─────────────────────────────────────────────
# ENTRY POINTS
# ─────────────────────────────────────────────
def setup_rois(image_path: str):
    """Run the interactive ROI selector and save results."""
    selector = ParkingROISelector(image_path)
    rois     = selector.run()
    if rois:
        ParkingROISelector.save(rois)
    else:
        print("[ROI] No spots saved.")


def run_on_video(video_path: str = 0,      # 0 = webcam
                 output_path: str | None = None):
    """Run full parking detection on a video file or camera feed."""
    rois = ParkingROISelector.load()
    if not rois:
        print("[WARN] No ROIs found. Run setup_rois() first.")

    detector = ParkingDetector(rois=rois)
    cap      = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # optional writer
    writer = None
    if output_path:
        fps  = cap.get(cv2.CAP_PROP_FPS) or 25
        fw   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, four, fps, (fw, fh))

    print("[RUN] Press 'q' to quit, 's' to save log snapshot …")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, stats = detector.process_frame(frame)
            cv2.imshow("Parking Detection — YOLO", annotated)

            if writer:
                writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                detector.save_log()

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        detector.save_log()


def run_on_image(image_path: str, output_path: str | None = None):
    """Run detection on a single static image."""
    rois     = ParkingROISelector.load()
    detector = ParkingDetector(rois=rois)
    frame    = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")

    annotated, stats = detector.process_frame(frame)
    print(json.dumps(stats, indent=2))

    out = output_path or image_path.replace(".", "_detected.")
    cv2.imwrite(out, annotated)
    print(f"[OUT] Saved → {out}")
    cv2.imshow("Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Pixel-Count Parking Detection — YOLO")
    ap.add_argument("mode", choices=["setup", "video", "image"],
                    help="setup=define ROIs | video=process video | image=process image")
    ap.add_argument("--source", default="0",
                    help="Image/video path, or '0' for webcam")
    ap.add_argument("--output", default=None,
                    help="Optional output path for annotated result")
    args = ap.parse_args()

    if args.mode == "setup":
        setup_rois(args.source)

    elif args.mode == "video":
        src = int(args.source) if args.source == "0" else args.source
        run_on_video(src, args.output)

    elif args.mode == "image":
        run_on_image(args.source, args.output)
