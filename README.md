# Real-Time Anomaly Detection in CCTV Surveillance

A deep learning system that watches surveillance footage and automatically flags threatening activity — theft, violence, and property damage — in real time.

---

## What It Does

https://github.com/user-attachments/assets/05728f18-76f9-41db-9522-7c1bf61501a8

Most CCTV systems record everything but catch nothing until a human reviews the footage after the fact. This project flips that — a model watches the stream live, classifies what's happening every few frames, and overlays the result directly on the video.

**Four output classes:**
- `normal` — no incident
- `theft` — shoplifting, robbery, burglary
- `violence` — assault, fighting, abuse
- `property_damage` — arson, vandalism

---

## Architecture

```
Video Frame
    │
    ▼
CLAHE contrast enhancement          ← improves visibility in dark CCTV footage
    │
    ▼
YOLOv8 person detection + crop      ← focuses the model on people, not background
    │
    ▼
Top-16 motion frame selection       ← picks the most action-rich frames from a 64-frame buffer
    │
    ▼
ResNet50 feature extraction         ← pretrained CNN, outputs 2048-dim vector per frame
    │
    ▼
BiLSTM classifier                   ← learns temporal patterns across the 16-frame sequence
    │
    ▼
4-class prediction + smoothing      ← rolling window reduces single-frame flicker
```

The same preprocessing pipeline runs identically during training and live inference, so the model never sees a different distribution at runtime.

---

## Results

Evaluated on a held-out stratified test split (15% of dataset).

| Class | Precision | Recall | F1 |
|---|---|---|---|
| normal | 0.9495 | 0.8710 | 0.9086 |
| theft | 0.8350 | 0.8750 | 0.8545 |
| violence | 0.8586 | 0.9111 | 0.8841 |
| property_damage | 0.8703 | 0.9244 | 0.8966 |


## Applications

- **Security operations** — automated pre-screening across multi-camera feeds
- **Retail loss prevention** — real-time theft flagging
- **Smart city CCTV** — large-scale public incident detection
- **Forensic review** — fast search through archived footage

---

## Tech Stack

| Role | Tool |
|---|---|
| Person detection | YOLOv8n (Ultralytics) |
| Feature extraction | ResNet50 (Keras, ImageNet) |
| Temporal model | Bidirectional LSTM (TensorFlow) |
| Video decoding | Decord |
| Frame processing | OpenCV |
| Motion scoring | PyTorch |

---

## Dataset

UCF Crime Dataset — real-world CCTV footage across 13 anomaly categories, consolidated into 4 classes for this project. Class imbalance is handled through per-class window oversampling and weighted loss during training.

---

*Kaggle notebook project — GPU (P100/T4) recommended for feature extraction and training.*
