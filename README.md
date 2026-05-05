# Pixel-Count Methodology for Automated Parking Detection — YOLOv8
**SRM Institute of Science and Technology | DSBS | Guide: Dr. Anna Anhumozhi**

---

## Project Structure

```
parking_detection/
├── parking_detector.py   ← core Python backend (all 3 phases)
├── requirements.txt      ← dependencies
├── dashboard.html        ← live monitoring UI (open in browser)
├── parking_rois.pkl      ← auto-generated after ROI setup
└── occupancy_log.json    ← auto-generated detection log
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
YOLOv8 weights (`yolov8n.pt`) are downloaded automatically on first run.

---

### Phase 1 — Define Parking Spots (Manual ROI)
Point it at any parking-lot image:
```bash
python parking_detector.py setup --source parking_lot.jpg
```

**Controls in the window:**
| Key | Action |
|-----|--------|
| Left-click | Add polygon vertex |
| Right-click | Close & save current spot |
| `z` | Undo last point |
| `c` | Clear everything |
| `s` | Save ROIs to `parking_rois.pkl` and exit |
| `q` | Quit without saving |

---

### Phase 2 & 3 — Run Detection

**On a video file:**
```bash
python parking_detector.py video --source parking_lot.mp4 --output out.mp4
```

**On a live webcam:**
```bash
python parking_detector.py video --source 0
```

**On a single image:**
```bash
python parking_detector.py image --source test_frame.jpg
```

While video is running press:
- `q` — quit
- `s` — save JSON log snapshot

---

## How It Works

```
Frame → YOLO Inference → Filter Vehicle Classes → Bounding Boxes
                                                        ↓
                          Parking ROIs ──── IoU Calculation
                                                        ↓
                                            IoU ≥ 0.40 → OCCUPIED
                                            IoU <  0.40 → FREE
```

### Phase 1 — Setup & Manual ROI
- Polygons are clicked manually per parking spot
- Saved to `parking_rois.pkl` for persistence across sessions

### Phase 2 — YOLO Detection Pipeline
- Runs YOLOv8 on the full frame in one pass
- Filters detections to vehicle classes only: car(2), motorcycle(3), bus(5), truck(7)
- Returns bounding boxes with confidence scores

### Phase 3 — Occupancy Logic (IoU)
- For each parking spot ROI, computes pixel-mask IoU against every vehicle bounding box
- If the best IoU ≥ threshold (default 40%) → spot is **OCCUPIED**
- Threshold adjustable: `OVERLAP_THRESHOLD = 0.40` in config

---

## Configuration
Edit the top of `parking_detector.py`:
```python
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
OVERLAP_THRESHOLD = 0.40       # 40% IoU → occupied
MODEL_NAME        = "yolov8n.pt"   # swap to yolov8s/m/l for more accuracy
```

---

## Dashboard
Open `dashboard.html` in any browser. It simulates the live occupancy feed
and auto-updates every 3 seconds. Click any spot to toggle it manually.

To connect it to real data, your backend can write `occupancy_log.json` and
the dashboard can poll it via `fetch()`.

---

## Literature Referenced
| Author | Technique | Gap Addressed |
|--------|-----------|---------------|
| Samoud Rafique et al. | YOLOv5 + MS COCO | Real-time speed |
| Janzhe Fang et al. | YOLOv11 + Special Conv Kernel | Detection time |
| Noura Alatawi et al. | YOLO + OpenCV | Data variability |
