"""
CCTV Surveillance — Streamlit Inference App
============================================
Test videos sourced from:
  SmartCity-CCTV-Violence-Detection-Dataset (SCVD)
  https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd

Model trained on:
  Real-Time Anomaly Detection in CCTV Surveillance
  https://www.kaggle.com/datasets/mateohervas/dcsass-dataset
"""

import os, time, tempfile, threading, queue
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_DIR = Path(__file__).parent

# Default test clips (SCVD)
DEFAULT_CLIPS = {
    "t_v009 — Violence sample (SCVD)":  str(_DIR / "t_v009.mp4"),
}

# Default pkl files (009-series from training set)
DEFAULT_PKLS = [
    "Robbery009_x264.pkl",
    "Normal_Videos009_x264.pkl",
    "Abuse009_x264.pkl",
    "Fighting009_x264.pkl",
    "Vandalism009_x264.pkl",
]

DISPLAY_EVERY_N = 3   # render 1 in 3 frames to keep the browser smooth
PERSON_DET_CONF = 0.25  # lower threshold works better on low-res CCTV clips

# ─────────────────────────────────────────────────────────────────────────────
# Cached model loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ResNet50 feature extractor…")
def load_base_model():
    # Keras 3 + Streamlit cache can leave stale scope state between reruns.
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    tf.keras.backend.clear_session()
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")

@st.cache_resource(show_spinner="Loading BiLSTM classifier…")
def load_lstm_model():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    for candidate in [
        _DIR / "best_model.h5",
        _DIR / "best_model.keras",
        _DIR / "models" / "lstm_model.keras",
        _DIR / "models" / "lstm_model.h5",
    ]:
        if candidate.exists():
            return tf.keras.models.load_model(str(candidate))
    st.error("Model not found. Place `best_model.h5` next to `app.py`.")
    st.stop()

@st.cache_resource(show_spinner="Loading YOLOv8 person detector…")
def load_yolo():
    from ultralytics import YOLO
    pt_path = _DIR / "models" / "yolov8n.pt"
    return YOLO(str(pt_path) if pt_path.exists() else "yolov8n.pt")

# ─────────────────────────────────────────────────────────────────────────────
# Label config
# ─────────────────────────────────────────────────────────────────────────────
LABEL2ID = {"normal": 0, "theft": 1, "violence": 2, "property_damage": 3}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

LABEL_CSS_COLOR = {
    "normal":          "#22c55e",
    "violence":        "#ef4444",
    "theft":           "#f59e0b",
    "property_damage": "#a855f7",
    "unknown":         "#6b7280",
    "error":           "#f97316",
}
LABEL_BG_COLOR = {
    "normal":          "#052e16",
    "violence":        "#450a0a",
    "theft":           "#451a03",
    "property_damage": "#3b0764",
    "unknown":         "#1f2937",
    "error":           "#431407",
}
LABEL_COLORS_BGR = {
    "violence":        (0,   0, 235),
    "normal":          (0, 200,  50),
    "theft":           (0, 160, 245),
    "property_damage": (235,  0, 210),
    "buffering":       (160, 160, 160),
    "error":           (0,  120, 249),
}
DEFAULT_BGR = (255, 255, 0)

LABEL_BADGE = {
    "violence":        "🔴 VIOLENCE",
    "normal":          "🟢 NORMAL",
    "theft":           "🟡 THEFT",
    "property_damage": "🟣 PROPERTY DAMAGE",
    "error":           "⚠️ ERROR",
}
LABEL_ICON = {
    "violence":        "⚠️",
    "normal":          "✅",
    "theft":           "🔍",
    "property_damage": "🔥",
    "error":           "❗",
}

PREFIX_TO_LABEL: dict[str, str] = {
    "Abuse":          "violence",
    "Arson":          "property_damage",
    "Assault":        "violence",
    "Burglary":       "theft",
    "Fighting":       "violence",
    "Normal_Videos_": "normal",
    "Normal_Videos":  "normal",
    "Robbery":        "theft",
    "Shooting":       "violence",
    "Shoplifting":    "theft",
    "Stealing":       "theft",
    "Vandalism":      "property_damage",
}
CATEGORY_TO_LABEL: dict[str, str] = {
    "Abuse":         "violence",
    "Arson":         "property_damage",
    "Assault":       "violence",
    "Burglary":      "theft",
    "Fighting":      "violence",
    "Normal_Videos": "normal",
    "Robbery":       "theft",
    "Shooting":      "violence",
    "Shoplifting":   "theft",
    "Stealing":      "theft",
    "Vandalism":     "property_damage",
}

BUFFER_SIZE     = 64
SEQUENCE_LENGTH = 16
INFER_EVERY_N   = 8
SMOOTH_HISTORY_N = 5

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
* { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background-color: #080c14 !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed; top:0; left:0; right:0; bottom:0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px,
        rgba(0,0,0,0.025) 2px, rgba(0,0,0,0.025) 4px);
    pointer-events: none; z-index: 9999;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1321 0%,#0a0f1c 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}
.page-header {
    background: linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#0f172a 100%);
    border: 1px solid rgba(99,102,241,0.25); border-radius: 16px;
    padding: 1.5rem 2rem; margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.page-header::before {
    content: ""; position: absolute; top:-50%; left:-50%; width:200%; height:200%;
    background: radial-gradient(ellipse at 60% 50%,rgba(99,102,241,0.08) 0%,transparent 60%);
    pointer-events: none;
}
.page-header h1 {
    font-size: 1.9rem; font-weight: 700; margin: 0 0 0.3rem;
    background: linear-gradient(135deg,#e2e8f0 0%,#a5b4fc 50%,#818cf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.page-header p { color:#64748b; font-size:0.88rem; margin:0; }
.glass-card {
    background: rgba(15,23,42,0.8); backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.15); border-radius:14px;
    padding:1.2rem 1.4rem; transition: border-color .25s, box-shadow .25s;
}
.glass-card:hover { border-color:rgba(99,102,241,0.4); box-shadow:0 0 24px rgba(99,102,241,0.1); }
.glass-card .card-value { font-size:2rem; font-weight:700; line-height:1.1;
    font-family:'JetBrains Mono',monospace; }
.glass-card .card-label { font-size:0.72rem; color:#475569; text-transform:uppercase;
    letter-spacing:0.1em; margin-top:0.35rem; }
.verdict-card {
    background: linear-gradient(135deg,rgba(15,23,42,.9) 0%,rgba(30,27,75,.6) 100%);
    backdrop-filter:blur(16px); border:2px solid rgba(99,102,241,.2); border-radius:18px;
    padding:2rem; text-align:center; margin-top:1rem; position:relative; overflow:hidden;
}
.verdict-card::before { content:""; position:absolute; inset:0;
    background:radial-gradient(ellipse at 50% 0%,rgba(99,102,241,.08) 0%,transparent 70%); pointer-events:none; }
.verdict-title { font-size:0.75rem; color:#475569; text-transform:uppercase;
    letter-spacing:0.12em; margin-bottom:0.6rem; }
.badge { display:inline-flex; align-items:center; gap:.3em; padding:.3em 1em;
    border-radius:999px; font-weight:600; font-size:.82rem; letter-spacing:.05em;
    text-transform:uppercase; white-space:nowrap; }
.badge-normal          { background:#052e16; color:#22c55e; border:1.5px solid #22c55e; }
.badge-violence        { background:#450a0a; color:#ef4444; border:1.5px solid #ef4444;
    box-shadow:0 0 12px rgba(239,68,68,.25); }
.badge-theft           { background:#451a03; color:#f59e0b; border:1.5px solid #f59e0b; }
.badge-property_damage { background:#3b0764; color:#a855f7; border:1.5px solid #a855f7; }
.badge-unknown         { background:#1f2937; color:#6b7280; border:1.5px solid #6b7280; }
.badge-error           { background:#431407; color:#f97316; border:1.5px solid #f97316; }
.conf-bar-wrap { background:rgba(255,255,255,.06); border-radius:6px; height:10px;
    overflow:hidden; margin:4px 0 10px; }
.conf-bar-fill { height:100%; border-radius:6px; }
.live-panel { background:rgba(15,23,42,.85); backdrop-filter:blur(12px);
    border:1px solid rgba(99,102,241,.18); border-radius:14px; padding:1.2rem; }
.live-label { font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700;
    text-align:center; margin-bottom:.5rem; }
.live-sub { font-size:.7rem; color:#475569; text-transform:uppercase;
    letter-spacing:.1em; text-align:center; }
[data-baseweb="tab-list"] { gap:6px; background:transparent !important; }
[data-baseweb="tab"] {
    background:rgba(15,23,42,.6) !important; border:1px solid rgba(99,102,241,.12) !important;
    border-radius:10px 10px 0 0 !important; padding:.55rem 1.4rem !important;
    font-weight:500 !important; color:#64748b !important; transition:all .2s !important;
}
[data-baseweb="tab"]:hover { border-color:rgba(99,102,241,.35) !important; color:#a5b4fc !important; }
[aria-selected="true"][data-baseweb="tab"] {
    background:rgba(30,27,75,.8) !important; border-color:rgba(99,102,241,.5) !important;
    color:#a5b4fc !important; box-shadow:0 -2px 12px rgba(99,102,241,.15) !important;
}
[data-testid="stButton"] > button[kind="primary"] {
    background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%); border:none;
    border-radius:10px; font-weight:600; letter-spacing:.04em;
    box-shadow:0 4px 16px rgba(79,70,229,.3); transition:all .2s;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow:0 6px 24px rgba(79,70,229,.5); transform:translateY(-1px);
}
[data-testid="stButton"] > button:not([kind="primary"]) {
    background:rgba(30,27,75,.6); border:1px solid rgba(99,102,241,.25);
    border-radius:10px; color:#a5b4fc; font-weight:500; transition:all .2s;
}
[data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color:rgba(99,102,241,.5); background:rgba(30,27,75,.9);
}
[data-testid="stProgressBar"] > div {
    background:linear-gradient(90deg,#4f46e5,#7c3aed) !important; border-radius:999px !important;
}
[data-testid="stProgressBar"] { background:rgba(255,255,255,.06) !important; border-radius:999px !important; }
.status-pill { display:inline-flex; align-items:center; gap:.4em; padding:.3em .9em;
    border-radius:999px; font-size:.82rem; font-weight:600; }
.status-ok  { background:#052e16; color:#22c55e; border:1px solid #22c55e; }
.status-err { background:#450a0a; color:#ef4444; border:1px solid #ef4444; }
.legend-item { display:flex; align-items:center; gap:.6em; padding:.35rem .5rem;
    border-radius:8px; margin-bottom:4px; font-size:.88rem; font-weight:500; transition:background .15s; }
.legend-item:hover { background:rgba(255,255,255,.04); }
.legend-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.attrib-card { background:rgba(15,23,42,.7); border:1px solid rgba(99,102,241,.12);
    border-radius:10px; padding:.8rem 1rem; font-size:.78rem; color:#475569; margin-top:.5rem; }
.attrib-card a { color:#818cf8; text-decoration:none; }
.attrib-card a:hover { text-decoration:underline; }
hr { border-color:rgba(99,102,241,.15) !important; margin:1.2rem 0 !important; }
[data-testid="stAlert"] { background:rgba(15,23,42,.7) !important; border-radius:10px !important; }
[data-testid="stFileUploader"] { background:rgba(15,23,42,.6);
    border:2px dashed rgba(99,102,241,.25); border-radius:12px; transition:border-color .2s; }
[data-testid="stFileUploader"]:hover { border-color:rgba(99,102,241,.5); }
[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Frame preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(frame_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def get_densest_cluster_coords(frame_shape, people_bboxes):
    bboxes  = np.array(people_bboxes)
    centers = bboxes[:, :2] + bboxes[:, 2:] / 2.0
    h_f, w_f = frame_shape[:2]
    threshold = 0.3 * np.sqrt(h_f**2 + w_f**2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    densest_idx = int(np.argmax(np.sum(dist_matrix < threshold, axis=1)))
    cluster = bboxes[dist_matrix[densest_idx] < threshold]
    x1, y1 = np.min(cluster[:, 0]), np.min(cluster[:, 1])
    x2, y2 = np.max(cluster[:, 0] + cluster[:, 2]), np.max(cluster[:, 1] + cluster[:, 3])
    pad = 0.15
    w_c, h_c = x2 - x1, y2 - y1
    return (int(max(0, x1 - w_c*pad)), int(max(0, y1 - h_c*pad)),
            int(min(w_f, x2 + w_c*pad)), int(min(h_f, y2 + h_c*pad)))


def crop_to_people_with_info(frame_bgr: np.ndarray, yolo_model) -> tuple[np.ndarray, dict]:
    clahe_bgr = apply_clahe(frame_bgr)
    detect_source = "original"
    results = yolo_model.predict(
        frame_bgr, classes=[0], conf=PERSON_DET_CONF, verbose=False, imgsz=640
    )[0]
    boxes = results.boxes.xywh.cpu().numpy()
    if len(boxes) == 0:
        detect_source = "clahe"
        results = yolo_model.predict(
            clahe_bgr, classes=[0], conf=PERSON_DET_CONF, verbose=False, imgsz=640
        )[0]
        boxes = results.boxes.xywh.cpu().numpy()

    people_bboxes = [[b[0] - b[2] / 2, b[1] - b[3] / 2, b[2], b[3]] for b in boxes]
    frame_h, frame_w = clahe_bgr.shape[:2]
    crop_reason = "no_people_fallback"
    crop_was_applied = False
    crop_box = (0, 0, frame_w, frame_h)

    if len(people_bboxes) == 1:
        x, y, w, h = people_bboxes[0]
        p = 0.2
        x1 = int(max(0, x - w * p))
        y1 = int(max(0, y - h * p))
        x2 = int(min(frame_w, x + w * (1 + p)))
        y2 = int(min(frame_h, y + h * (1 + p)))
        crop_box = (x1, y1, x2, y2)
        crop_reason = "single_person"
        crop_was_applied = True
    elif len(people_bboxes) > 1:
        crop_box = get_densest_cluster_coords(clahe_bgr.shape, people_bboxes)
        crop_reason = "densest_cluster"
        crop_was_applied = True

    x1, y1, x2, y2 = crop_box
    cropped_bgr = clahe_bgr[y1:y2, x1:x2]
    if crop_was_applied and cropped_bgr.size == 0:
        crop_box = (0, 0, frame_w, frame_h)
        crop_reason = "invalid_crop_fallback"
        crop_was_applied = False
        cropped_bgr = clahe_bgr

    return cropped_bgr, {
        "clahe_bgr": clahe_bgr,
        "people_bboxes": people_bboxes,
        "people_count": len(people_bboxes),
        "crop_box": crop_box,
        "crop_reason": crop_reason,
        "crop_was_applied": crop_was_applied,
        "detect_source": detect_source,
    }


def crop_to_people(frame_bgr: np.ndarray, yolo_model) -> np.ndarray:
    cropped_bgr, _ = crop_to_people_with_info(frame_bgr, yolo_model)
    return cropped_bgr


def preprocess_frame(frame_bgr: np.ndarray, yolo_model) -> np.ndarray:
    frame_bgr = crop_to_people(frame_bgr, yolo_model)
    resized   = cv2.resize(frame_bgr, (224, 224))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32")


def select_motion_from_array(frames: np.ndarray, sequence_length: int = 16) -> np.ndarray:
    n = len(frames)
    if n <= sequence_length:
        result = list(frames)
        while len(result) < sequence_length:
            result.append(frames[-1])
        return np.array(result)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t    = torch.from_numpy(frames).float().to(device)
        gray = 0.299*t[:,:,:,0] + 0.587*t[:,:,:,1] + 0.114*t[:,:,:,2]
        scores = torch.sum(torch.abs(gray[1:] - gray[:-1]), dim=(1, 2)).cpu().numpy()
    except Exception:
        gray   = 0.299*frames[:,:,:,0] + 0.587*frames[:,:,:,1] + 0.114*frames[:,:,:,2]
        scores = np.sum(np.abs(gray[1:] - gray[:-1]), axis=(1, 2))
    top_rel = np.argsort(scores)[-sequence_length:]
    indices = sorted([int(i) + 1 for i in top_rel if int(i) + 1 < n])
    while len(indices) < sequence_length:
        indices.append(indices[-1] if indices else n - 1)
    return frames[indices[:sequence_length]]


def get_smoothed_label(prediction_history, new_label: str, new_conf: float) -> str:
    prediction_history.append((new_label, new_conf))
    scores: dict[str, float] = {}
    for label, conf in prediction_history:
        scores[label] = scores.get(label, 0.0) + conf
    return max(scores, key=scores.get)


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_ground_truth(pkl_name: str) -> str:
    stem = Path(pkl_name).stem
    for prefix, label in sorted(PREFIX_TO_LABEL.items(), key=lambda x: -len(x[0])):
        if stem.startswith(prefix):
            return label
    return "unknown"


@st.cache_data(show_spinner="Building pipeline breakdown examples…")
def _get_pipeline_breakdown_assets(video_path: str) -> dict | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    sampled_bgr: list[np.ndarray] = []
    frame_i = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Sample up to ~96 frames across the clip for motion selection previews.
    sample_stride = max(total_frames // 96, 1) if total_frames > 0 else 1
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_i % sample_stride == 0:
            sampled_bgr.append(frame_bgr.copy())
            if len(sampled_bgr) >= 96:
                break
        frame_i += 1
    cap.release()

    if not sampled_bgr:
        return None

    ref_bgr = sampled_bgr[len(sampled_bgr) // 2]
    yolo_model = load_yolo()
    cropped_bgr, crop_info = crop_to_people_with_info(ref_bgr, yolo_model)
    clahe_bgr = crop_info["clahe_bgr"]

    detect_vis = clahe_bgr.copy()
    crop_box_vis = clahe_bgr.copy()
    people_bboxes = crop_info["people_bboxes"]
    frame_h, frame_w = clahe_bgr.shape[:2]

    for x, y, bw, bh in people_bboxes:
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + bw), int(y + bh)
        cv2.rectangle(detect_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    crop_reason = crop_info["crop_reason"]
    crop_was_applied = crop_info["crop_was_applied"]
    crop_box = crop_info["crop_box"]
    if len(people_bboxes) > 1:
        x1, y1, x2, y2 = crop_box
        cv2.rectangle(detect_vis, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(
            detect_vis,
            "Densest cluster",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    crop_color = (255, 0, 255) if crop_was_applied else (255, 191, 0)
    label_text = "Crop region" if crop_was_applied else "Fallback: full frame"
    bx1, by1, bx2, by2 = crop_box
    cv2.rectangle(crop_box_vis, (bx1, by1), (bx2, by2), crop_color, 3)
    cv2.putText(
        crop_box_vis,
        label_text,
        (bx1, max(18, by1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        crop_color,
        2,
        cv2.LINE_AA,
    )

    resnet_input_bgr = cv2.resize(cropped_bgr, (224, 224))
    sampled_rgb = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sampled_bgr], dtype=np.uint8)
    selected_rgb = select_motion_from_array(sampled_rgb, SEQUENCE_LENGTH)
    processed_frames: list[np.ndarray] = []
    per_frame_crop_reasons: list[str] = []
    for frame_rgb in selected_rgb:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cropped_pf, crop_info_pf = crop_to_people_with_info(frame_bgr, yolo_model)
        per_frame_crop_reasons.append(crop_info_pf["crop_reason"])
        resized_pf = cv2.resize(cropped_pf, (224, 224))
        processed_frames.append(cv2.cvtColor(resized_pf, cv2.COLOR_BGR2RGB).astype(np.uint8))

    processed_rgb = np.array(processed_frames, dtype=np.uint8)
    per_frame_crop_applied = [r in ("single_person", "densest_cluster") for r in per_frame_crop_reasons]
    applied_count = int(sum(per_frame_crop_applied))
    fallback_frames = [i + 1 for i, applied in enumerate(per_frame_crop_applied) if not applied]

    def _grid_16(frames_rgb: np.ndarray, tile_w: int = 220, tile_h: int = 124) -> np.ndarray:
        tiles: list[np.ndarray] = []
        for idx in range(SEQUENCE_LENGTH):
            frame = frames_rgb[idx] if idx < len(frames_rgb) else frames_rgb[-1]
            tile = cv2.resize(frame, (tile_w, tile_h))
            cv2.putText(
                tile,
                f"F{idx+1}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            tiles.append(tile)
        rows = []
        for r in range(4):
            rows.append(np.hstack(tiles[r * 4:(r + 1) * 4]))
        return np.vstack(rows)

    return {
        "original_rgb": cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB),
        "clahe_rgb": cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB),
        "detect_rgb": cv2.cvtColor(detect_vis, cv2.COLOR_BGR2RGB),
        "crop_box_preview_rgb": cv2.cvtColor(crop_box_vis, cv2.COLOR_BGR2RGB),
        "cropped_rgb": cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB),
        "resnet_input_preview_rgb": cv2.cvtColor(resnet_input_bgr, cv2.COLOR_BGR2RGB),
        "selected_grid_rgb": _grid_16(selected_rgb),
        "processed_grid_rgb": _grid_16(processed_rgb),
        "people_count": len(people_bboxes),
        "crop_box": crop_box,
        "crop_was_applied": crop_was_applied,
        "crop_reason": crop_reason,
        "sampled_frames": len(sampled_bgr),
        "selected_frames": SEQUENCE_LENGTH,
        "per_frame_crop_reasons": per_frame_crop_reasons,
        "per_frame_crop_applied_count": applied_count,
        "per_frame_fallback_frames": fallback_frames,
        "detect_source": crop_info["detect_source"],
    }


def _render_live_panel(ph, label: str, conf: float, all_probs: list[float],
                       frame_count: int, total_frames: int,
                       recent_labels: list[str],
                       error_msg: str = "") -> None:
    color = LABEL_CSS_COLOR.get(label, "#6b7280")
    badge_text = LABEL_BADGE.get(label, label.upper())
    icon       = LABEL_ICON.get(label, "●")

    if error_msg:
        error_html = (
            f"<div style='margin-top:.6rem;padding:.5rem .8rem;background:#431407;"
            f"border:1px solid #f97316;border-radius:8px;font-size:.72rem;color:#f97316'>"
            f"<b>Worker error:</b><br><code style='word-break:break-all'>{error_msg[:200]}</code>"
            f"</div>"
        )
    else:
        error_html = ""

    bars_html = ""
    for idx in sorted(ID2LABEL):
        cls = ID2LABEL[idx]
        p = all_probs[idx] if idx < len(all_probs) else 0.0
        c = LABEL_CSS_COLOR.get(cls, "#6b7280")
        w = int(p * 100)
        bars_html += (
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
            f"<span style='font-size:.7rem;color:#475569;width:110px;flex-shrink:0'>"
            f"{cls.replace('_',' ').title()}</span>"
            f"<div class='conf-bar-wrap' style='flex:1'>"
            f"<div class='conf-bar-fill' style='width:{w}%;background:{c}'></div>"
            f"</div>"
            f"<span style='font-size:.7rem;color:{c};width:36px;text-align:right;"
            f"font-family:JetBrains Mono,monospace'>{p:.0%}</span>"
            f"</div>"
        )

    history_html = "".join(
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
        f"background:{LABEL_CSS_COLOR.get(lbl, '#6b7280')};margin:2px'></span>"
        for lbl in recent_labels[-12:]
    )

    ph.markdown(
        f"<div class='live-panel'>"
        f"<div class='live-sub'>Current Prediction</div>"
        f"<div class='live-label' style='color:{color}'>{icon} {badge_text}</div>"
        f"<div style='text-align:center;margin-bottom:.8rem'>"
        f"<span style='font-family:JetBrains Mono,monospace;font-size:1.1rem;"
        f"font-weight:600;color:{color}'>{conf:.1%}</span>"
        f"<span style='color:#475569;font-size:.7rem'> confidence</span>"
        f"</div>"
        f"{error_html}"
        f"<div style='border-top:1px solid rgba(255,255,255,.06);padding-top:.8rem'>"
        f"<div style='font-size:.7rem;color:#475569;text-transform:uppercase;"
        f"letter-spacing:.08em;margin-bottom:8px'>Class Probabilities</div>"
        f"{bars_html}</div>"
        f"<div style='border-top:1px solid rgba(255,255,255,.06);padding-top:.8rem;margin-top:.4rem'>"
        f"<div style='font-size:.7rem;color:#475569;text-transform:uppercase;"
        f"letter-spacing:.08em;margin-bottom:4px'>Frame</div>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:.88rem;color:#94a3b8'>"
        f"{frame_count} / {total_frames}</div></div>"
        f"<div style='border-top:1px solid rgba(255,255,255,.06);padding-top:.8rem;margin-top:.8rem'>"
        f"<div style='font-size:.7rem;color:#475569;text-transform:uppercase;"
        f"letter-spacing:.08em;margin-bottom:6px'>Prediction History</div>"
        f"<div>{history_html}</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_summary(metrics_ph, predictions: list[str]) -> None:
    if not predictions:
        metrics_ph.warning("No predictions — video may be too short.")
        return
    counts = Counter(predictions)
    final  = counts.most_common(1)[0][0]
    total  = len(predictions)
    final_color = LABEL_CSS_COLOR.get(final, "#888")

    with metrics_ph.container():
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Session Summary")
        cols = st.columns(max(len(counts), 1))
        for col, (label, count) in zip(cols, counts.most_common()):
            lbl_color = LABEL_CSS_COLOR.get(label, "#ccc")
            lbl_badge = LABEL_BADGE.get(label, label.upper())
            badge_cls = f"badge-{label}"
            col.markdown(
                f"<div class='glass-card'>"
                f"<div class='card-value' style='color:{lbl_color}'>{count}/{total}</div>"
                f"<div class='card-label'>"
                f"<span class='badge {badge_cls}'>{lbl_badge}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        verdict_badge = LABEL_BADGE.get(final, final.upper())
        st.markdown(
            f"<div class='verdict-card'>"
            f"<div class='verdict-title'>Final Verdict</div>"
            f"<div style='font-size:2rem;font-weight:700;color:{final_color};margin:.4rem 0'>"
            f"{verdict_badge}</div>"
            f"<div style='color:#475569;font-size:.82rem'>"
            f"Most frequent prediction across {total} prediction events"
            f"</div></div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core video inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_on_video(
    video_path: str,
    frame_ph,
    live_ph,
    metrics_ph,
    progress_bar,
    stop_event: threading.Event,
    smooth_n: int = 5,
    infer_every: int = INFER_EVERY_N,
) -> None:
    from tensorflow.keras.applications.resnet50 import preprocess_input

    # ── Open video — show a friendly error if cv2 can't open it ──────────────
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"cv2.VideoCapture could not open: {video_path}\n"
                          "Check the file path and that the codec is supported "
                          "(AVI/MP4/MOV/MKV). On Windows, try re-encoding with FFmpeg.")
    except Exception as e:
        metrics_ph.error(f"**Cannot open video:** {e}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25

    base_model = load_base_model()
    lstm_model = load_lstm_model()
    yolo_model = load_yolo()

    frame_buffer      = deque(maxlen=64)
    prediction_history = deque(maxlen=smooth_n)
    predictions: list[str] = []
    recent_labels: list[str] = []

    current_label     = "buffering"
    current_conf      = 0.0
    current_all_probs: list[float] = [0.25, 0.25, 0.25, 0.25]
    current_error     = ""
    frame_count       = 0

    infer_q: queue.Queue = queue.Queue(maxsize=1)
    result_q: queue.Queue = queue.Queue(maxsize=1)

    def inference_worker():
        while True:
            item = infer_q.get()
            if item is None:
                break
            selected_rgb = item
            try:
                prepped = np.array(
                    [preprocess_frame(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), yolo_model)
                     for f in selected_rgb],
                    dtype="float32",
                )
                features = base_model.predict(
                    preprocess_input(prepped), verbose=0)[np.newaxis, ...]
                probs    = lstm_model.predict(features, verbose=0)[0]
                pred_id  = int(np.argmax(probs))
                conf     = float(probs[pred_id])
                label    = ID2LABEL[pred_id]
                result_q.put((label, conf, probs.tolist(), ""))
            except Exception as exc:
                result_q.put(("error", 0.0, [0.25, 0.25, 0.25, 0.25], str(exc)))

    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    try:
        while True:
            if stop_event.is_set():
                break
            ret, bgr = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_buffer.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            # Trigger inference every `infer_every` frames once buffer is full
            if (frame_count % infer_every == 0
                    and len(frame_buffer) >= SEQUENCE_LENGTH
                    and infer_q.empty()):
                selected = select_motion_from_array(
                    np.array(list(frame_buffer)), SEQUENCE_LENGTH)
                if selected is not None:
                    try:
                        infer_q.put_nowait(selected.copy())
                    except queue.Full:
                        pass

            # Consume new result if available (non-blocking)
            new_result = False
            try:
                raw_label, raw_conf, raw_probs, raw_err = result_q.get_nowait()
                if raw_label != "error":
                    current_label = get_smoothed_label(prediction_history, raw_label, raw_conf)
                    current_conf  = raw_conf
                    predictions.append(current_label)
                    recent_labels.append(current_label)
                    current_error = ""
                else:
                    current_error = raw_err
                current_all_probs = raw_probs
                new_result = True
            except queue.Empty:
                pass

            # ── Annotated frame (every DISPLAY_EVERY_N frames) ───────────
            if frame_count % DISPLAY_EVERY_N == 0:
                frame = bgr.copy()
                h, w  = frame.shape[:2]
                color = LABEL_COLORS_BGR.get(current_label, DEFAULT_BGR)

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 58), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

                cv2.putText(frame, current_label.upper(),
                            (12, 38), cv2.FONT_HERSHEY_DUPLEX, 1.05,
                            color, 2, cv2.LINE_AA)

                if current_conf > 0:
                    bx, by, bw_b, bh = w - 230, 16, 210, 24
                    cv2.rectangle(frame, (bx, by), (bx+bw_b, by+bh), (40, 40, 40), -1)
                    cv2.rectangle(frame, (bx, by),
                                  (bx + int(bw_b * current_conf), by+bh), color, -1)
                    cv2.rectangle(frame, (bx, by), (bx+bw_b, by+bh), (120, 120, 120), 1)
                    cv2.putText(frame, f"{current_conf:.0%}",
                                (bx + bw_b + 6, by + 17),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200, 200, 200), 1)

                cv2.putText(frame, f"FRAME {frame_count}/{total_frames}",
                            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (140, 140, 140), 1)

                frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               use_container_width=True)

            # ── Live metrics panel (only when prediction updates) ─────────
            if new_result:
                _render_live_panel(
                    live_ph, current_label, current_conf, current_all_probs,
                    frame_count, total_frames, recent_labels, current_error,
                )

            progress_bar.progress(min(frame_count / max(total_frames, 1), 1.0))

    finally:
        infer_q.put(None)
        worker.join(timeout=10)
        cap.release()

    _render_summary(metrics_ph, predictions)


# ─────────────────────────────────────────────────────────────────────────────
# Feature inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_on_features(pkl_path: str, metrics_ph) -> None:
    import pickle
    import pandas as pd

    lstm_model = load_lstm_model()

    with open(pkl_path, "rb") as f:
        features = pickle.load(f)

    features = np.array(features, dtype="float32")
    if features.ndim == 2 and features.shape[-1] == 2048:
        features = features[np.newaxis, ...]
    elif features.ndim != 3:
        metrics_ph.error(
            f"Unexpected shape `{features.shape}`. "
            "Expected `(n_predictions, 16, 2048)`."
        )
        return

    n_predictions = features.shape[0]
    ground_truth = _infer_ground_truth(Path(pkl_path).name)
    file_stem    = Path(pkl_path).stem

    prediction_labels: list[str]  = []
    prediction_confs:  list[float] = []

    prog = metrics_ph.progress(0.0, text="Running BiLSTM on 16-frame chunks…")
    for i in range(n_predictions):
        frame_chunk = features[i][np.newaxis, ...]   # shape (1, 16, 2048)
        try:
            probs   = lstm_model.predict(frame_chunk, verbose=0)[0]
            pred_id = int(np.argmax(probs))
            conf    = float(probs[pred_id])
            label   = ID2LABEL.get(pred_id, "unknown")
        except Exception as exc:
            label, conf = "error", 0.0
        prediction_labels.append(label)
        prediction_confs.append(conf)
        prog.progress((i + 1) / n_predictions,
                      text=f"Prediction {i+1}/{n_predictions} (16 frames) — {label} ({conf:.1%})")

    prog.empty()

    with metrics_ph.container():
        counts  = Counter(prediction_labels)
        verdict = counts.most_common(1)[0][0]
        match_icon = "✅" if ground_truth == verdict else "❌"

        st.markdown(
            f"<div class='verdict-card'>"
            f"<div style='font-size:.75rem;color:#475569;text-transform:uppercase;"
            f"letter-spacing:.12em;margin-bottom:1rem'>{file_stem}</div>"
            f"<div style='display:flex;justify-content:center;gap:3rem;flex-wrap:wrap'>"
            f"<div style='text-align:center'>"
            f"<div class='verdict-title'>Ground Truth</div>"
            f"<span class='badge badge-{ground_truth}'>"
            f"{LABEL_BADGE.get(ground_truth, ground_truth.upper())}"
            f"</span></div>"
            f"<div style='display:flex;align-items:center;font-size:1.8rem'>{match_icon}</div>"
            f"<div style='text-align:center'>"
            f"<div class='verdict-title'>Model Verdict</div>"
            f"<span class='badge badge-{verdict}'>"
            f"{LABEL_BADGE.get(verdict, verdict.upper())}"
            f"</span></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_chart, col_table = st.columns(2)

        with col_chart:
            st.markdown("**Class distribution across 16-frame predictions**")
            dist_rows = [
                {"Class": ID2LABEL[i].replace("_", " ").title(),
                 "Predictions": counts.get(ID2LABEL[i], 0),
                 "Color": LABEL_CSS_COLOR.get(ID2LABEL[i], "#888")}
                for i in sorted(ID2LABEL)
            ]
            dist_df = pd.DataFrame(dist_rows)
            try:
                import altair as alt
                chart = (
                    alt.Chart(dist_df)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("Class:N", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("Predictions:Q"),
                        color=alt.Color("Class:N",
                            scale=alt.Scale(
                                domain=[r["Class"] for r in dist_rows],
                                range=[r["Color"] for r in dist_rows]),
                            legend=None),
                        tooltip=["Class", "Predictions"],
                    )
                    .properties(background="transparent")
                    .configure_axis(gridColor="rgba(255,255,255,.05)",
                                    labelColor="#64748b", titleColor="#64748b")
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(chart, use_container_width=True)
            except ImportError:
                st.bar_chart(dist_df.set_index("Class")["Predictions"])
            st.dataframe(dist_df[["Class","Predictions"]], use_container_width=True, hide_index=True)

        with col_table:
            st.markdown("**Prediction timeline (16-frame chunks)**")
            timeline_df = pd.DataFrame({
                "Prediction #": list(range(1, n_predictions + 1)),
                "Prediction":   prediction_labels,
                "Confidence":   [f"{c:.1%}" for c in prediction_confs],
            })

            def _color_row(row):
                bg  = LABEL_BG_COLOR.get(row["Prediction"], "#1f2937")
                clr = LABEL_CSS_COLOR.get(row["Prediction"], "#6b7280")
                return [f"background-color:{bg};color:{clr}"] * len(row)

            st.dataframe(timeline_df.style.apply(_color_row, axis=1),
                         use_container_width=True, hide_index=True, height=420)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset stats (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Scanning dataset…")
def get_dataset_stats(pkl_dir: str) -> dict:
    pkl_dir_p = Path(pkl_dir)
    files     = list(pkl_dir_p.glob("*.pkl"))
    cat_counts: dict[str, int]  = {}
    class_counts: dict[str, int] = {k: 0 for k in LABEL2ID}

    for f in files:
        stem = f.stem
        matched_cat, matched_len = None, 0
        for cat in CATEGORY_TO_LABEL:
            if stem.startswith(cat) and len(cat) > matched_len:
                matched_cat, matched_len = cat, len(cat)
        if matched_cat:
            cat_counts[matched_cat] = cat_counts.get(matched_cat, 0) + 1
            class_counts[CATEGORY_TO_LABEL[matched_cat]] += 1

    return {"total_files": len(files), "cat_counts": cat_counts, "class_counts": class_counts}


# ═════════════════════════════════════════════════════════════════════════════
# Page setup
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="CCTV Surveillance", page_icon="🎥", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:.5rem 0 1rem'>"
        "<span style='font-size:2.5rem'>🎥</span>"
        "<div style='font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-top:.4rem'>"
        "CCTV Surveillance</div>"
        "<div style='font-size:.75rem;color:#475569;margin-top:.2rem'>"
        "ResNet50 + BiLSTM anomaly detection</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # Model status
    model_ok = (_DIR / "best_model.h5").exists() or (_DIR / "best_model.keras").exists()
    pkl_count = len(list((_DIR / "processed_motion_frames").glob("*.pkl"))) \
                if (_DIR / "processed_motion_frames").exists() else 0

    st.markdown(
        f"<span class='status-pill {'status-ok' if model_ok else 'status-err'}'>"
        f"{'● BiLSTM Ready' if model_ok else '● Model Not Found'}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span class='status-pill' style='background:#0f172a;color:#64748b;"
        f"border:1px solid #334155;margin-top:6px'>📦 {pkl_count} feature files</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:.72rem;color:#475569;text-transform:uppercase;"
        "letter-spacing:.1em;font-weight:600;margin-bottom:.6rem'>⚙️ Inference Settings</div>",
        unsafe_allow_html=True,
    )
    infer_every = st.slider(
        "Predict every N frames", 4, 32, INFER_EVERY_N, 4,
        help=(
            "Every N video frames, a 16-frame chunk is sampled (motion-selected) "
            "and sent to the BiLSTM for classification. "
            "Lower N = more frequent predictions but higher CPU/GPU load. "
            "Each prediction always uses exactly 16 frames."
        ),
    )
    INFER_EVERY_N = infer_every

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:.72rem;color:#475569;text-transform:uppercase;"
        "letter-spacing:.1em;font-weight:600;margin-bottom:.6rem'>🏷️ Class Legend</div>",
        unsafe_allow_html=True,
    )
    for cls, label_txt, icon_txt in [
        ("normal",          "Normal",          "🟢"),
        ("violence",        "Violence",        "🔴"),
        ("theft",           "Theft",           "🟡"),
        ("property_damage", "Property Damage", "🟣"),
    ]:
        c  = LABEL_CSS_COLOR[cls]
        bg = LABEL_BG_COLOR[cls]
        st.markdown(
            f"<div class='legend-item' style='background:{bg}22'>"
            f"<span class='legend-dot' style='background:{c}'></span>"
            f"<span style='color:{c}'>{icon_txt} {label_txt}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Attribution
    st.markdown(
        "<div class='attrib-card'>"
        "<b style='color:#a5b4fc'>Test clips</b><br>"
        "<a href='https://www.kaggle.com/datasets/toluwaniaremu/"
        "smartcity-cctv-violence-detection-dataset-scvd' target='_blank'>"
        "SmartCity-CCTV-Violence-Detection-Dataset (SCVD)</a><br><br>"
        "<b style='color:#a5b4fc'>Model trained on</b><br>"
        "<a href='https://www.kaggle.com/datasets/mateohervas/dcsass-dataset' target='_blank'>"
        "Real-Time Anomaly Detection in CCTV Surveillance</a>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Models cached for the session.")


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    "<div class='page-header'>"
    "<h1>🎥 CCTV Anomaly Detection</h1>"
    "<p>Pipeline: <strong style='color:#a5b4fc'>CLAHE</strong> → "
    "<strong style='color:#a5b4fc'>YOLOv8 crop</strong> → "
    "<strong style='color:#a5b4fc'>ResNet50</strong> → "
    "<strong style='color:#a5b4fc'>BiLSTM</strong>. "
    "Or skip extraction entirely with pre-extracted 16-frame feature chunks.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Two tabs ─────────────────────────────────────────────────────────────────
tab_infer, tab_dataset = st.tabs(["🎬 Video Inference", "📊 Dataset Explorer"])

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — VIDEO INFERENCE
# ════════════════════════════════════════════════════════════════════════
with tab_infer:

    source_mode = st.radio(
        "Video source",
        ["Default Clip (SCVD)", "Upload Video", "Pre-extracted Features (pkl)"],
        horizontal=True,
        help="Choose how to provide the video for inference.",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    video_path_to_run: str | None = None
    tmp_file_handle = None
    run_features_mode = False
    selected_pkl_path: str | None = None

    # ── Source: Default Clip ─────────────────────────────────────────────
    if source_mode == "Default Clip (SCVD)":
        st.markdown(
            "<div class='attrib-card' style='margin-bottom:1rem'>"
            "Test clips from the "
            "<a href='https://www.kaggle.com/datasets/toluwaniaremu/"
            "smartcity-cctv-violence-detection-dataset-scvd' target='_blank'>"
            "SmartCity-CCTV-Violence-Detection-Dataset (SCVD)</a>."
            "</div>",
            unsafe_allow_html=True,
        )

        available = {name: path for name, path in DEFAULT_CLIPS.items()
                     if Path(path).exists()}

        if not available:
            st.warning(
                "Default clips not found. Expected:\n"
                + "\n".join(f"- `{p}`" for p in DEFAULT_CLIPS.values())
            )
        else:
            clip_choice = st.selectbox("Choose clip", list(available.keys()))
            chosen_path = available[clip_choice]

            gt_hint = "violence" if "v009" in clip_choice else "normal"
            gt_color = LABEL_CSS_COLOR.get(gt_hint, "#888")
            st.markdown(
                f"Expected class: <span class='badge badge-{gt_hint}'>"
                f"{LABEL_BADGE.get(gt_hint, gt_hint.upper())}</span>",
                unsafe_allow_html=True,
            )
            st.video(chosen_path)
            video_path_to_run = chosen_path

    # ── Source: Upload ───────────────────────────────────────────────────
    elif source_mode == "Upload Video":
        uploaded = st.file_uploader(
            "Drop a CCTV video here",
            type=["avi", "mp4", "mov", "mkv"],
            help="Supported formats: AVI, MP4, MOV, MKV",
        )
        if uploaded:
            suffix = Path(uploaded.name).suffix
            tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.flush()
            tmp_file_handle   = tmp
            video_path_to_run = tmp.name
            st.success(f"Loaded: **{uploaded.name}**  ({uploaded.size / 1e6:.1f} MB)")
            st.video(video_path_to_run)

    # ── Source: Pre-extracted features ───────────────────────────────────
    else:
        run_features_mode = True
        pkl_dir = _DIR / "processed_motion_frames"
        pkl_files = sorted(pkl_dir.glob("*.pkl")) if pkl_dir.exists() else []

        if not pkl_files:
            st.warning("No `.pkl` files found in `processed_motion_frames/`.")
        else:
            st.markdown(
                "Each `.pkl` file holds pre-extracted ResNet50 features split into "
                "**16-frame chunks** — inference skips CLAHE/YOLO/ResNet50 and runs "
                "the **BiLSTM** directly on each chunk. "
                "Files are from the model's training dataset "
                "(**real-time-anomaly-detection-in-cctv-surveillance**)."
            )

            # Surface the 009-series first, then the rest
            pkl_names    = [p.name for p in pkl_files]
            default_pkls = [n for n in DEFAULT_PKLS if n in pkl_names]
            other_pkls   = [n for n in pkl_names if n not in DEFAULT_PKLS]
            ordered_pkls = default_pkls + other_pkls

            col_sel, col_badge = st.columns([3, 1])
            with col_sel:
                selected_pkl = st.selectbox(
                    "Feature file (009-series shown first)",
                    ordered_pkls,
                    help=(
                        "Files named <Category><number>_x264.pkl. "
                        "Each contains (n_predictions, 16, 2048) ResNet50 feature arrays. "
                        "The 009-series matches the t_v009/t_n009 test videos."
                    ),
                )
            with col_badge:
                gt_label = _infer_ground_truth(selected_pkl) if selected_pkl else "unknown"
                st.markdown("<div style='padding-top:1.6rem'>", unsafe_allow_html=True)
                st.markdown(
                    f"<span class='badge badge-{gt_label}'>"
                    f"{LABEL_BADGE.get(gt_label, gt_label.upper())}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            selected_pkl_path = str(pkl_dir / selected_pkl) if selected_pkl else None

    # ── Run / Stop controls ──────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)

    if "stop_event" not in st.session_state:
        st.session_state.stop_event = threading.Event()

    if run_features_mode:
        feat_metrics = st.empty()
        if st.button("▶ Run Feature Inference", type="primary", use_container_width=True,
                     disabled=(selected_pkl_path is None)):
            run_inference_on_features(selected_pkl_path, feat_metrics)

    else:
        col_run, col_stop = st.columns([3, 1])
        run_pressed  = col_run.button(
            "▶ Run Inference", type="primary", use_container_width=True,
            disabled=(video_path_to_run is None),
        )
        stop_pressed = col_stop.button("⏹ Stop", use_container_width=True)

        if stop_pressed:
            st.session_state.stop_event.set()

        if run_pressed and video_path_to_run:
            st.session_state.stop_event.clear()

            col_vid, col_live = st.columns([2, 1])
            frame_ph     = col_vid.empty()
            live_ph      = col_live.empty()
            progress_bar = st.progress(0.0)
            metrics_ph   = st.empty()

            run_inference_on_video(
                video_path   = video_path_to_run,
                frame_ph     = frame_ph,
                live_ph      = live_ph,
                metrics_ph   = metrics_ph,
                progress_bar = progress_bar,
                stop_event   = st.session_state.stop_event,
                smooth_n     = SMOOTH_HISTORY_N,
                infer_every  = INFER_EVERY_N,
            )
            progress_bar.progress(1.0)

            if tmp_file_handle is not None:
                try:
                    os.unlink(tmp_file_handle.name)
                except OSError:
                    pass


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET EXPLORER
# ════════════════════════════════════════════════════════════════════════
with tab_dataset:
    import pandas as pd

    st.markdown("### Dataset Explorer")
    with st.expander("Pipeline breakdown (default clip)", expanded=False):
        st.caption(
            "Detailed end-to-end breakdown of the default clip from raw frame to final BiLSTM classification."
        )
        default_clip_path = next(
            (p for p in DEFAULT_CLIPS.values() if Path(p).exists()),
            None,
        )
        if default_clip_path is None:
            st.warning("No default clip found for pipeline breakdown.")
        else:
            assets = _get_pipeline_breakdown_assets(default_clip_path)
            if not assets:
                st.warning("Could not build pipeline breakdown for this clip.")
            else:
                c1, c2 = st.columns(2)
                c1.image(assets["original_rgb"], caption="1) Original frame", width="stretch")
                c2.image(assets["clahe_rgb"], caption="2) After CLAHE", width="stretch")

                c3, c4 = st.columns(2)
                c3.image(
                    assets["detect_rgb"],
                    caption=f"3) YOLO people detection ({assets['people_count']} detected)",
                    width="stretch",
                )
                c4.image(
                    assets["crop_box_preview_rgb"],
                    caption="4) Crop box on frame",
                    width="stretch",
                )

                c5, c6 = st.columns(2)
                c5.image(
                    assets["cropped_rgb"],
                    caption="5) Actual cropped output",
                    width="stretch",
                )
                c6.image(
                    assets["resnet_input_preview_rgb"],
                    caption="6) ResNet50 input (resized to 224x224)",
                    width="stretch",
                )

                fallback_frame_text = (
                    ", ".join(f"F{idx}" for idx in assets["per_frame_fallback_frames"])
                    if assets["per_frame_fallback_frames"]
                    else "None"
                )
                st.markdown(
                    "#### Per-16-frame preprocessing summary\n"
                    f"- CLAHE applied to: `{assets['selected_frames']}/{assets['selected_frames']}` frames.\n"
                    f"- YOLO + crop successfully applied to: `{assets['per_frame_crop_applied_count']}/{assets['selected_frames']}` frames.\n"
                    f"- Full-frame fallback used on: `{len(assets['per_frame_fallback_frames'])}/{assets['selected_frames']}` frames.\n"
                    f"- Fallback frame IDs: `{fallback_frame_text}`."
                )

                st.image(
                    assets["selected_grid_rgb"],
                    caption="7) 16 motion-selected frames (used for one prediction event)",
                    width="stretch",
                )
                st.image(
                    assets["processed_grid_rgb"],
                    caption="8) Same 16 frames after CLAHE + YOLO crop + resize (actual ResNet50 inputs)",
                    width="stretch",
                )

                crop_reason_map = {
                    "single_person": "single person crop",
                    "densest_cluster": "densest-cluster crop",
                    "no_people_fallback": "no people detected, full CLAHE frame used",
                    "invalid_crop_fallback": "invalid crop region, full CLAHE frame used",
                }
                crop_reason_text = crop_reason_map.get(assets["crop_reason"], assets["crop_reason"])

                if not assets["crop_was_applied"]:
                    st.warning(
                        "Crop fallback active: no valid people crop was available, so the full CLAHE frame was used."
                    )

                st.markdown(
                    "#### Stage details\n"
                    f"- CLAHE: improves local contrast before detection and feature extraction.\n"
                    f"- YOLOv8 people detector: confidence threshold `{PERSON_DET_CONF}`; "
                    f"detected `{assets['people_count']}` people in the example frame "
                    f"(detection source: `{assets['detect_source']}`).\n"
                    f"- Densest-cluster rule: used when multiple people are detected to focus on the main activity group.\n"
                    f"- Crop-to-people output: `{crop_reason_text}`.\n"
                    f"- Motion selection: sampled `{assets['sampled_frames']}` frames from video, "
                    f"selected `{assets['selected_frames']}` highest-motion frames for one prediction.\n"
                    "- ResNet50 embeddings: each selected frame is converted to a 2048-d feature; "
                    "combined tensor shape per prediction is `(16, 2048)`.\n"
                    "- BiLSTM classifier: consumes `(16, 2048)` sequence, outputs class probabilities "
                    "for `normal`, `theft`, `violence`, and `property_damage`."
                )
                st.caption(
                    "Data flow: Raw frame -> CLAHE -> YOLO people detection -> crop/fallback -> resize (224x224) "
                    "-> ResNet50 per-frame embeddings -> shape (16, 2048) -> BiLSTM -> class probabilities + label."
                )

    st.markdown(
        "Overview of the **1,650 pre-extracted feature files** in `processed_motion_frames/`, "
        "sourced from the "
        "[real-time-anomaly-detection-in-cctv-surveillance](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset) "
        "training dataset."
    )

    pkl_dir_ds = _DIR / "processed_motion_frames"
    if not pkl_dir_ds.exists():
        st.warning("No `processed_motion_frames/` folder found.")
    else:
        stats  = get_dataset_stats(str(pkl_dir_ds))
        total_f = stats["total_files"]
        cat_c   = stats["cat_counts"]
        cls_c   = stats["class_counts"]

        # KPI row
        kpi_cols = st.columns(5)
        kpis = [
            ("Total Files",    str(total_f),                       "#a5b4fc"),
            ("Normal",         str(cls_c.get("normal", 0)),        LABEL_CSS_COLOR["normal"]),
            ("Violence",       str(cls_c.get("violence", 0)),      LABEL_CSS_COLOR["violence"]),
            ("Theft",          str(cls_c.get("theft", 0)),         LABEL_CSS_COLOR["theft"]),
            ("Prop. Damage",   str(cls_c.get("property_damage",0)), LABEL_CSS_COLOR["property_damage"]),
        ]
        for kc, (lbl, val, clr) in zip(kpi_cols, kpis):
            kc.markdown(
                f"<div class='glass-card' style='text-align:center'>"
                f"<div class='card-value' style='color:{clr}'>{val}</div>"
                f"<div class='card-label'>{lbl}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.markdown("**Files by model class**")
            cls_rows = [
                {"Class": k.replace("_", " ").title(), "Files": v,
                 "Color": LABEL_CSS_COLOR.get(k, "#888")}
                for k, v in cls_c.items() if v > 0
            ]
            cls_df = pd.DataFrame(cls_rows)
            try:
                import altair as alt
                pie = (
                    alt.Chart(cls_df)
                    .mark_arc(innerRadius=55, padAngle=0.03, cornerRadius=4)
                    .encode(
                        theta=alt.Theta("Files:Q"),
                        color=alt.Color("Class:N",
                            scale=alt.Scale(domain=[r["Class"] for r in cls_rows],
                                            range=[r["Color"] for r in cls_rows]),
                            legend=alt.Legend(orient="right", labelColor="#94a3b8",
                                              titleColor="#475569")),
                        tooltip=["Class", "Files"],
                    )
                    .properties(background="transparent", height=260)
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(pie, use_container_width=True)
            except ImportError:
                st.bar_chart(cls_df.set_index("Class")["Files"])

        with col_bar:
            st.markdown("**Files by raw video category**")
            cat_rows = sorted(cat_c.items(), key=lambda x: -x[1])
            cat_df   = pd.DataFrame(cat_rows, columns=["Category", "Files"])
            cat_df["Class"] = cat_df["Category"].map(CATEGORY_TO_LABEL).fillna("unknown")
            cat_df["Color"] = cat_df["Class"].map(LABEL_CSS_COLOR).fillna("#888")
            try:
                import altair as alt
                bar = (
                    alt.Chart(cat_df)
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                    .encode(
                        x=alt.X("Files:Q"),
                        y=alt.Y("Category:N", sort="-x",
                                axis=alt.Axis(labelColor="#94a3b8", titleColor="#475569")),
                        color=alt.Color("Category:N",
                            scale=alt.Scale(domain=list(cat_df["Category"]),
                                            range=list(cat_df["Color"])),
                            legend=None),
                        tooltip=["Category", "Files", "Class"],
                    )
                    .properties(background="transparent", height=320)
                    .configure_axis(gridColor="rgba(255,255,255,.05)",
                                    labelColor="#64748b", titleColor="#64748b")
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(bar, use_container_width=True)
            except ImportError:
                st.bar_chart(cat_df.set_index("Category")["Files"])

        with st.expander("Full category breakdown", expanded=False):
            tbl = cat_df[["Category", "Class", "Files"]].copy()
            tbl["Class"] = tbl["Class"].str.replace("_", " ").str.title()
            st.dataframe(tbl, use_container_width=True, hide_index=True)
