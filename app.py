"""
Pixel-Count Methodology for Automated Parking Detection
21CSE524T — Computer Vision Techniques | SRMIST | Dept. DSBS
Guide: Dr. Anna Anhumozhi

Classical CV pipeline (no deep learning):
  Stage 1 — Colour Constancy / Gray World  (Unit 1, T2)
  Stage 2 — Gaussian filtering             (Unit 1 – classical filtering)
  Stage 3 — Canny edge detector            (Unit 3, T5)
  Stage 4 — Hough Transform line detection (Unit 2/3, T6, T7)
  Stage 5 — Binary thresholding + pixel-count occupancy  (Unit 1, Unit 2)
  Stage 6 — Connected-component / shape analysis         (Unit 2)
  Bonus   — Gaussian pyramid multi-scale view            (T4)
           — Mean shift segmentation view                (T8)
"""

from __future__ import annotations

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os

st.set_page_config(
    page_title="Parking Detection — CV Techniques",
    page_icon="🅿️",
    layout="wide",
)

# ─── helpers ──────────────────────────────────────────────────────────────────

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def stag(text: str):
    st.markdown(f"> 📖 **Syllabus ref:** {text}")

# ─── Stage 1: Colour Constancy — Gray World (Unit 1, T2) ──────────────────────

def gray_world(img_bgr: np.ndarray) -> np.ndarray:
    """
    Assumption: spatial average of reflectance across the scene is achromatic.
    Scale each channel so its mean equals the global mean.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    mr, mg, mb = rgb[:, :, 0].mean(), rgb[:, :, 1].mean(), rgb[:, :, 2].mean()
    mu = (mr + mg + mb) / 3.0
    out = np.clip(rgb * [mu / (mr + 1e-9), mu / (mg + 1e-9), mu / (mb + 1e-9)], 0, 255)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)

# ─── Stage 2 + 3: Gaussian blur + Canny edges (Unit 1 / Unit 3, T5) ──────────

def preprocess_and_edges(img_bgr: np.ndarray, blur_k: int, low: int, high: int):
    blurred = cv2.GaussianBlur(img_bgr, (blur_k, blur_k), 0)
    gray    = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges   = cv2.Canny(gray, low, high)
    return blurred, gray, edges

# ─── Stage 4: Hough Transform line detection (Unit 2/3, T6, T7) ──────────────

def hough_lines(edges: np.ndarray, base_img: np.ndarray,
                threshold: int, min_len: int, max_gap: int):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                             minLineLength=min_len, maxLineGap=max_gap)
    result = base_img.copy()
    count  = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
        count = len(lines)
    return result, count

# ─── Stage 5: Per-spot occupancy via pixel count (Unit 1 / Unit 2) ────────────

def is_occupied(spot_bgr: np.ndarray, spot_gray: np.ndarray, spot_edges: np.ndarray,
                method: str, edge_pct: float, px_pct: float,
                sat_pct: float = 0.10) -> tuple[bool, float]:
    """
    Edge density (T5): count Canny edge pixels → high density = car present.
    Pixel count     : Otsu threshold → count foreground pixels.
    Colour Saturation (T2): mean HSV-S → coloured cars on grey asphalt.
    """
    if spot_gray.size == 0:
        return False, 0.0

    if method == "edge":
        ratio = np.count_nonzero(spot_edges) / spot_edges.size
        return ratio > edge_pct, ratio
    elif method == "color":
        hsv   = cv2.cvtColor(spot_bgr, cv2.COLOR_BGR2HSV)
        ratio = float(hsv[:, :, 1].mean()) / 255.0
        return ratio > sat_pct, ratio
    else:
        _, binary = cv2.threshold(spot_gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ratio = np.count_nonzero(binary) / binary.size
        return ratio > px_pct, ratio

# ─── Stage 6: Connected-component shape analysis (Unit 2) ─────────────────────

def component_analysis(spot_gray: np.ndarray):
    """Return (num_components, largest_area) for one spot ROI."""
    blur = cv2.GaussianBlur(spot_gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return 0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return n - 1, int(areas.max())

# ─── Grid helper ──────────────────────────────────────────────────────────────

def make_grid(h: int, w: int,
              rx: int, rows_y: list, rw: int, rh_pct: int,
              cols: int) -> list[tuple]:
    x0 = int(rx / 100 * w)
    bw = int(rw / 100 * w)
    cw = max(bw // cols, 1)
    ch = max(int(rh_pct / 100 * h), 1)
    return [
        (x0 + c * cw, int(ry / 100 * h),
         x0 + (c + 1) * cw, int(ry / 100 * h) + ch)
        for ry in rows_y for c in range(cols)
    ]

# ─── Annotate final frame ─────────────────────────────────────────────────────

def annotate(img: np.ndarray,
             spots: list[tuple],
             results: list[tuple[bool, float]]) -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2), (occ, score) in zip(spots, results):
        color  = (0, 40, 220) if occ else (0, 200, 60)
        label  = "OCC" if occ else "FREE"
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                      (0, 0, 160) if occ else (0, 140, 30), -1)
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (cx - 20, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(out, f"{score*100:.0f}%", (cx - 14, cy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

    total    = len(results)
    occupied = sum(1 for o, _ in results if o)
    free     = total - occupied

    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (250, 75), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.6, out, 0.4, 0, out)
    cv2.putText(out, f"Total   : {total}",    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(out, f"Occupied: {occupied}", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
    cv2.putText(out, f"Free    : {free}",     (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 80), 2)
    return out

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🅿️ Parking Detection")
        st.caption("21CSE524T · SRMIST · Dept. DSBS")
        st.markdown("**Guide:** Dr. Anna Anhumozhi")
        st.divider()

        uploaded = st.file_uploader("📁 Upload Parking Lot Image",
                                    type=["jpg", "jpeg", "png", "bmp"])

        SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_parking.jpg")
        use_sample  = False
        if uploaded is None and os.path.exists(SAMPLE_PATH):
            use_sample = st.toggle("▶ Use built-in sample image", value=True)

        st.divider()

        st.subheader("Stage 1 · Preprocessing")
        use_cc = st.toggle("Colour Constancy — Gray World (T2)", value=True)
        blur_k = st.slider("Gaussian kernel size", 1, 15, 5, step=2,
                           help="Classical filtering — Unit 1")

        st.subheader("Stage 2 · Canny Edge Detector (T5)")
        canny_low  = st.slider("Low threshold",  10, 200, 50)
        canny_high = st.slider("High threshold", 50, 400, 150)

        st.subheader("Stage 3 · Hough Lines (T6, T7)")
        hough_thr = st.slider("Vote threshold",  10, 300,  60)
        min_len   = st.slider("Min line length", 10, 300,  60)
        max_gap   = st.slider("Max line gap",     1, 100,  15)

        st.subheader("Stage 4 · Parking Grid")
        rx     = st.slider("X offset (%)",    0, 80,  5)
        rw     = st.slider("Width (%)",      10, 100, 89)
        gcols  = st.slider("Spots per row",   1,  20,  6)
        rh_pct = st.slider("Spot height (%)", 3,  40, 16,
                           help="Height of each parking bay as % of image height.")
        n_rows = st.slider("Number of rows",  1,   5,  2)
        _row_defaults = [13, 68]
        rows_y = [
            st.slider(f"Row {i+1} Y start (%)", 0, 95,
                      _row_defaults[i] if i < len(_row_defaults) else min(13 + i * 30, 90),
                      key=f"ry{i}")
            for i in range(n_rows)
        ]

        st.subheader("Stage 5 · Occupancy Method")
        method = st.radio("Detection method", [
            "Colour Saturation — HSV (Unit 1, T2)",
            "Edge Density — Canny (Unit 3, T5)",
            "Pixel Count — Thresholding (Unit 1)",
        ])
        sat_pct  = st.slider("Saturation threshold (%)",  1, 60, 10,
                             help="Mean HSV-S of spot / 255. Coloured cars >> grey asphalt.") / 100
        edge_pct = st.slider("Edge threshold (%)",        1, 30,  4) / 100
        px_pct   = st.slider("Pixel threshold (%)",      10, 90, 35) / 100

    # ── No image and sample not available ─────────────────────────────────────
    if uploaded is None and not use_sample:
        st.title("🅿️ Pixel-Count Parking Detection — Classical CV")
        st.info("Upload a parking lot image from the sidebar to begin.")
        return

    # ── Load & resize ─────────────────────────────────────────────────────────
    if uploaded is not None:
        img_pil = Image.open(uploaded)
    else:
        img_pil = Image.open(SAMPLE_PATH)

    raw  = pil_to_bgr(img_pil)
    h, w = raw.shape[:2]
    if w > 1280:
        raw = cv2.resize(raw, (1280, int(h * 1280 / w)))
        h, w = raw.shape[:2]

    # ── Run pipeline ──────────────────────────────────────────────────────────
    cc_img   = gray_world(raw) if use_cc else raw.copy()
    proc, gray_full, edges_full = preprocess_and_edges(cc_img, blur_k, canny_low, canny_high)
    hough_img, line_count       = hough_lines(edges_full, proc, hough_thr, min_len, max_gap)
    spots    = make_grid(h, w, rx, rows_y, rw, rh_pct, gcols)

    method_key = ("color" if "Colour" in method
                  else "edge" if "Edge" in method
                  else "pixel")
    results    = []
    comp_stats = []
    for (x1, y1, x2, y2) in spots:
        s_bgr   = cc_img[y1:y2, x1:x2]
        s_gray  = gray_full[y1:y2, x1:x2]
        s_edges = edges_full[y1:y2, x1:x2]
        occ, score = is_occupied(s_bgr, s_gray, s_edges, method_key, edge_pct, px_pct, sat_pct)
        n_comp, big = component_analysis(s_gray)
        results.append((occ, score))
        comp_stats.append((n_comp, big))

    total    = len(results)
    occupied = sum(1 for o, _ in results if o)
    free     = total - occupied

    final = annotate(raw, spots, results)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📷 Original",
        "1 · Preprocessing",
        "2 · Edge Detection",
        "3 · Hough Lines",
        "4 · Spot Grid",
        "5 · Shape Analysis",
        "📊 Extra Views",
        "🅿️ Final Result",
    ])

    # ── Tab 0: Original ───────────────────────────────────────────────────────
    with tabs[0]:
        st.image(bgr_to_pil(raw), caption="Original Upload", use_container_width=True)

    # ── Tab 1: Preprocessing ──────────────────────────────────────────────────
    with tabs[1]:
        stag("Unit 1 — T2: Colour Constancy (Gray World); Classical Gaussian filtering")
        st.markdown(
            "**Gray World assumption:** The spatial average of reflectance over the scene is constant "
            "(achromatic/grey). Each R, G, B channel is scaled so its mean equals the global mean. "
            "This removes colour casts from uneven outdoor lighting before edge analysis.\n\n"
            "**Gaussian blur** then smooths noise so Canny doesn't fire on texture."
        )
        c1, c2 = st.columns(2)
        c1.image(bgr_to_pil(raw),  caption="Original",                        use_container_width=True)
        c2.image(bgr_to_pil(proc), caption="After Gray World + Gaussian blur", use_container_width=True)

        raw_rgb  = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(float)
        proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB).astype(float)
        st.markdown(f"""
| Channel | Original mean | After correction |
|---------|--------------|-----------------|
| R | {raw_rgb[:,:,0].mean():.1f} | {proc_rgb[:,:,0].mean():.1f} |
| G | {raw_rgb[:,:,1].mean():.1f} | {proc_rgb[:,:,1].mean():.1f} |
| B | {raw_rgb[:,:,2].mean():.1f} | {proc_rgb[:,:,2].mean():.1f} |
        """)

    # ── Tab 2: Canny edges ────────────────────────────────────────────────────
    with tabs[2]:
        stag("Unit 3 — T5: Canny edge detector")
        st.markdown(
            "**Canny pipeline:** Gaussian smooth → gradient magnitude & direction "
            "(Sobel) → non-maximum suppression → double thresholding "
            f"({canny_low} / {canny_high}) → hysteresis edge tracking.\n\n"
            "Parked cars produce **dense edge clusters** (body lines, windows, bumpers). "
            "Empty spots produce **sparse edges** (only lot markings)."
        )
        c1, c2 = st.columns(2)
        c1.image(bgr_to_pil(proc),           caption="Preprocessed",    use_container_width=True)
        c2.image(Image.fromarray(edges_full), caption="Canny Edges",     use_container_width=True)
        st.metric("Total edge pixels", int(np.count_nonzero(edges_full)))

    # ── Tab 3: Hough lines ────────────────────────────────────────────────────
    with tabs[3]:
        stag("Unit 2/3 — Hough Transform for line detection (T6, T7): detect and count lot markings")
        st.markdown(
            "Each Canny edge pixel votes in **(ρ, θ) parameter space**. "
            "Peaks in the accumulator correspond to straight lot-marking lines. "
            "Probabilistic HT (HoughLinesP) finds line segments efficiently."
        )
        c1, c2 = st.columns(2)
        c1.image(Image.fromarray(edges_full), caption="Canny Edges (input to HT)", use_container_width=True)
        c2.image(bgr_to_pil(hough_img),       caption=f"Hough Lines — {line_count} detected", use_container_width=True)
        st.metric("Lines detected (T7 — counting lines)", line_count)

    # ── Tab 4: Spot grid & occupancy ─────────────────────────────────────────
    with tabs[4]:
        stag("Unit 1 — Thresholding | Unit 3 — Canny edge density (T5) | Unit 2 — Binary shape analysis")
        st.markdown(
            f"**Method:** {method}\n\n"
            + (
                "Mean **HSV saturation** of each spot is computed. "
                "Coloured cars have high saturation; grey asphalt has near-zero saturation. "
                f"**Saturation > {sat_pct*100:.0f}% → OCCUPIED**."
                if method_key == "color" else
                "Each spot ROI is passed to Canny. The ratio of edge pixels to total pixels "
                f"is computed. **Edge density > {edge_pct*100:.0f}% → OCCUPIED**."
                if method_key == "edge" else
                "Each spot ROI is binarised with Otsu's threshold. The ratio of foreground pixels "
                f"to total pixels is computed. **Foreground ratio > {px_pct*100:.0f}% → OCCUPIED**."
            )
        )
        grid_vis = raw.copy()
        for (x1, y1, x2, y2) in spots:
            cv2.rectangle(grid_vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
        st.image(bgr_to_pil(grid_vis), caption="Parking grid (cyan boxes)", use_container_width=True)

        scores = [s * 100 for _, s in results]
        thresh = (edge_pct if method_key == "edge" else px_pct) * 100
        df_scores = pd.DataFrame({
            "Score (%)": scores,
            "Threshold": [thresh] * len(scores),
        }, index=[f"S{i+1}" for i in range(len(scores))])
        st.bar_chart(df_scores)
        st.caption(f"Spots above {thresh:.0f}% threshold are OCCUPIED")

    # ── Tab 5: Connected-component / shape analysis ───────────────────────────
    with tabs[5]:
        stag("Unit 2 — Binary shape analysis, connectedness, object labeling and counting, size filtering")
        st.markdown(
            "For each parking spot, Otsu's threshold creates a binary mask. "
            "**Connected-components analysis** labels each separate blob. "
            "A large number of significant components (vehicles have many parts) "
            "and large maximum blob area both indicate an occupied spot."
        )

        df_comp = pd.DataFrame([
            {
                "Spot": f"S{i+1}",
                "Components": n,
                "Largest area (px²)": a,
                "Status": "OCCUPIED" if occ else "FREE",
            }
            for i, ((occ, _), (n, a)) in enumerate(zip(results, comp_stats))
        ])
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # show one example spot's binary mask
        if spots:
            x1, y1, x2, y2 = spots[0]
            sample_gray = gray_full[y1:y2, x1:x2]
            if sample_gray.size > 0:
                _, sample_bin = cv2.threshold(sample_gray, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                n, lbl, sts, _ = cv2.connectedComponentsWithStats(sample_bin, connectivity=8)
                color_lbl = np.zeros((*sample_bin.shape, 3), dtype=np.uint8)
                rng = np.random.default_rng(7)
                for j in range(1, n):
                    color_lbl[lbl == j] = rng.integers(80, 255, 3)
                c1, c2 = st.columns(2)
                c1.image(Image.fromarray(sample_bin),  caption="S1 Binary mask", use_container_width=True)
                c2.image(Image.fromarray(color_lbl),   caption=f"S1 Components ({n-1})", use_container_width=True)

    # ── Tab 6: Extra views ────────────────────────────────────────────────────
    with tabs[6]:
        view = st.radio("Extra technique view", [
            "Gaussian Pyramid (T4)",
            "Mean Shift Segmentation (T8)",
        ], horizontal=True)

        if view.startswith("Gaussian"):
            stag("T4: Write a program that produces a Gaussian pyramid from an image")
            st.markdown(
                "A Gaussian pyramid is built by repeatedly blurring and halving the resolution (pyrDown). "
                "Multi-scale analysis at lower levels can detect large vehicles; "
                "finer detail at higher resolution catches compact cars."
            )
            levels = st.slider("Levels", 2, 5, 4, key="pyr_lv")
            pyr = [raw]
            cur = raw.copy()
            for _ in range(levels - 1):
                cur = cv2.pyrDown(cur)
                pyr.append(cur)
            cols = st.columns(levels)
            for i, (col, lv) in enumerate(zip(cols, pyr)):
                ph, pw = lv.shape[:2]
                col.image(bgr_to_pil(lv), caption=f"Level {i}  {pw}×{ph}", use_container_width=True)

        else:
            stag("Unit 4 — T8: Mean shift segmenter | Mean shift and mode finding")
            st.markdown(
                "Mean shift iteratively moves each pixel toward the **mode (density peak)** "
                "in joint spatial-colour space, producing flat-colour regions. "
                "This segments the parking area cleanly, making vehicle blobs more uniform "
                "before pixel-count analysis."
            )
            c1, c2 = st.columns(2)
            sp = c1.slider("Spatial bandwidth", 5, 80, 21)
            sr = c2.slider("Colour bandwidth",  5, 80, 31)
            small = cv2.resize(raw, (320, 240))
            seg   = cv2.pyrMeanShiftFiltering(small, sp, sr)
            seg   = cv2.resize(seg, (w, h))
            cc1, cc2 = st.columns(2)
            cc1.image(bgr_to_pil(raw), caption="Original",              use_container_width=True)
            cc2.image(bgr_to_pil(seg), caption="Mean Shift Segmented",  use_container_width=True)

    # ── Tab 7: Final result ───────────────────────────────────────────────────
    with tabs[7]:
        stag(
            "Full pipeline: Gray World (T2) → Gaussian → Canny (T5) "
            "→ Hough (T6/T7) → Pixel-count occupancy → Shape analysis (Unit 2)"
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Spots",  total)
        m2.metric("Occupied 🔴", occupied)
        m3.metric("Free 🟢",     free)

        st.image(bgr_to_pil(final), caption="Annotated Parking Lot", use_container_width=True)

        df_out = pd.DataFrame([
            {
                "Spot":       f"S{i+1}",
                "Status":     "OCCUPIED" if occ else "FREE",
                "Score (%)":  f"{score*100:.1f}",
                "Components": comp_stats[i][0],
                "Largest blob (px²)": comp_stats[i][1],
            }
            for i, (occ, score) in enumerate(results)
        ])
        st.dataframe(df_out, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
