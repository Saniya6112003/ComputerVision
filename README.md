# 🅿️ Pixel-Count Parking Detection — Classical CV

> **21CSE524T — Computer Vision Techniques**  
> SRM Institute of Science and Technology | Dept. of Data Science & Business Systems  
> Guide: **Dr. Anna Anhumozhi**

---

## Overview

A real-time parking occupancy detection system built entirely with **classical computer vision** — no deep learning. Upload any parking lot image and the app walks through each CV stage step-by-step, showing exactly how the occupancy decision is made.

---

## CV Pipeline

```
Input Image
    │
    ▼
Stage 1 — Colour Constancy (Gray World)      [Unit 1, T2]
    │  Removes lighting bias by equalising R/G/B channel means
    ▼
Stage 2 — Gaussian Filtering                 [Unit 1]
    │  Smooths noise before edge detection
    ▼
Stage 3 — Canny Edge Detection               [Unit 3, T5]
    │  Gradient → NMS → double threshold → hysteresis
    ▼
Stage 4 — Hough Transform (Line Detection)   [Unit 2/3, T6, T7]
    │  Detects parking bay markings in (ρ, θ) space
    ▼
Stage 5 — Occupancy Decision                 [Unit 1, Unit 2]
    │  Choose one of three methods:
    │   • Colour Saturation — HSV (best for coloured cars)
    │   • Edge Density      — Canny pixel ratio
    │   • Pixel Count       — Otsu binary threshold
    ▼
Stage 6 — Connected-Component Analysis       [Unit 2]
    │  Labels blobs, counts components, filters by size
    ▼
Annotated Output  →  FREE 🟢 / OCCUPIED 🔴
```

---

## Features

- **8 interactive tabs** — see every stage's output side-by-side
- **Live parameter tuning** — adjust thresholds, grid layout, and detection method from the sidebar with instant feedback
- **Per-row grid positioning** — independently set the Y position of each parking row (handles lots with uneven row spacing)
- **Bonus views** — Gaussian pyramid (T4) and Mean Shift segmentation (T8)
- **Built-in demo image** — works out of the box with no uploads needed

---

## Project Structure

```
ComputerVision/
├── app.py               ← Streamlit app (complete CV pipeline)
├── create_sample.py     ← Script to regenerate the demo image
├── sample_parking.jpg   ← Built-in demo parking lot image
├── requirements.txt     ← Python dependencies
└── README.md
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Saniya6112003/ComputerVision.git
cd ComputerVision
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** and loads the built-in sample image automatically.

---

## Usage

1. **Upload** your own parking lot image from the sidebar, or use the built-in sample
2. **Tune the grid** — set X offset, width, spots per row, and the Y start of each parking row
3. **Pick a detection method** in Stage 5:
   - *Colour Saturation* — works best for coloured cars on grey asphalt
   - *Edge Density* — works for all car colours including silver/white
   - *Pixel Count* — classic Otsu thresholding approach
4. **Explore each tab** to see how every CV stage contributes to the final result

---

## Configuration

All parameters are adjustable live from the sidebar — no code editing needed:

| Parameter | What it controls |
|---|---|
| Colour Constancy | Toggle Gray World correction on/off |
| Gaussian kernel size | Smoothing before edge detection |
| Canny low / high | Edge sensitivity thresholds |
| Hough vote threshold, min length, max gap | Line detection sensitivity |
| X offset, Width | Horizontal extent of the parking grid |
| Spot height (%) | Height of each bay's analysis zone |
| Row N Y start (%) | Vertical position of each parking row |
| Saturation / Edge / Pixel threshold | Occupied vs Free decision boundary |

---

## Literature Referenced

| Author | Technique | Gap Addressed |
|---|---|---|
| Samoud Rafique et al. | YOLOv5 + MS COCO | Real-time speed |
| Janzhe Fang et al. | YOLOv11 + Special Conv Kernel | Detection time |
| Noura Alatawi et al. | YOLO + OpenCV | Data variability |
