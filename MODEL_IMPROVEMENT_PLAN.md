# Bouldering Hold Detection & Segmentation Improvement Plan

## 1. Executive Summary & Diagnostic Findings

This project uses Ultralytics YOLO to detect and segment climbing holds from indoor wall photos. After inspecting the codebase, training runs, annotations, and test outputs, several critical bottlenecks and opportunities for improvement were identified:

| Area | Current State | Bottleneck / Observation | Target State |
| :--- | :--- | :--- | :--- |
| **Annotation Format** | 8 coordinates per label line (4 corners) | Exported from Roboflow as **YOLO OBB (Oriented Bounding Box)**, but trained with `yolo11l-seg.pt` (Instance Segmentation). Mask loss forces the model to fit 4-corner boxes rather than organic hold curves. | Either migrate to **YOLO11 OBB** (`task=obb`) or generate true polygon masks via **SAM (Segment Anything)**. |
| **Dataset Volume** | 44 train, 7 val, 1 test image | High risk of overfitting with a large 25M-parameter backbone (`yolo11l`). Single test image provides unreliable evaluation metrics. | 200–500+ diverse images, 15–20 test images, plus 15+ negative background images (empty walls). |
| **Input Resolution** | `imgsz=800` | Small footholds, jibs, and crimps (often 20–40 px in full 12MP–48MP gym photos) shrink to 3–6 px, causing feature loss. | **SAHI (Sliced Aided Hyper Inference)** at inference time and training at `imgsz=1024` on RTX 4070 SUPER. |
| **Class Taxonomy** | `data.yaml` has 16 colors; `data_hold.yaml` has 1 class | 16-class detection directly from small data suffers severe class imbalance (rare colors have < 3 examples). | **Two-stage architecture**: Class-agnostic hold detector (`hold`) + downstream color clustering/classification. |
| **Augmentation** | `degrees=180.0`, default HSV | 180° rotation inverts gravity/gym context (mats on ceiling). Wall photos are predominantly upright. | `degrees=25.0`, elevated HSV jitter (`hsv_s=0.5, hsv_v=0.5`), and synthetic chalk/occlusion noise. |

---

## 2. Phase 1: Task Formulation Alignment

### The Core Problem
In `train/labels/*.txt`, every annotation follows the 4-corner oriented box format:
```text
0 x1 y1 x2 y2 x3 y4 x4 y4
```
When trained with `yolo11l-seg.pt`, YOLO interprets these 4 points as a 4-vertex polygon. The segmentation mask head is penalized whenever it detects the natural rounded shape of a hold, because the ground truth is a sharp box.

### Path A: Native Oriented Bounding Boxes (Recommended for Speed & Simplicity)
If rotated rectangles are sufficient for your application (capturing hold position, size, and rotation angle):
1. Switch task from `segment` to `obb`.
2. Use pretrained OBB weights: `yolo11s-obb.pt` or `yolo11m-obb.pt`.
3. YOLO OBB natively optimizes rotation angle loss and ProbIoU loss, which trains faster and converges more cleanly.

### Path B: True Organic Segmentation (Recommended for Clean Visual Overlays)
If pixel-accurate hold boundaries are required:
1. **Automated Re-labeling with SAM**: Use [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) or [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) in zero-shot mode. Pass your existing 4-point bounding boxes as box prompts to SAM to extract true organic polygon masks automatically without manual re-drawing.
2. Update the label files to true YOLO segmentation format (varying polygon point counts: `0 x1 y1 x2 y2 ... xn yn`).

---

## 3. Phase 2: Resolving the Small Foothold Bottleneck (SAHI)

On large gym wall photos (e.g. 4032x3024), small footholds disappear when downsampled to 800px.

### Sliced Aided Hyper Inference (SAHI)
SAHI tiles high-resolution images into overlapping slices, runs inference per slice, and resolves overlapping detections using Non-Maximum Suppression (NMS).

```python
# Installation: pip install sahi
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',  # Compatible with Ultralytics YOLO11
    model_path='runs/segment/training_runs/bouldering_holds_seg/weights/best.pt',
    confidence_threshold=0.22,
    device='cuda:0'
)

# Run sliced inference over high-res photos
result = get_sliced_prediction(
    image="test/images/wall_photo.jpg",
    detection_model=detection_model,
    slice_height=800,
    slice_width=800,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25
)
```

**Benefits**:
* Increases detection recall of distant, low-profile footholds and screw-ons by 20–40%.
* Requires zero retraining.

---

## 4. Phase 3: Dataset Quality & Negative Sample Strategy

### 1. Adding Hard Negatives (Empty Backgrounds)
Climbing walls contain screw t-nuts, empty bolt holes, chalk smudges, and wood grain that frequently trigger false positives.
* Collect 15–20 photos of:
  * Empty climbing wall panels with bolt holes.
  * Safety mats and gym floors.
  * Bare volumes without holds.
* Add these to `train/images/` with **empty `.txt` label files** in `train/labels/`.
* This penalizes false positives and teaches the model that holes $\neq$ holds.

### 2. Dataset Expansion
* Integrate open climbing datasets from Roboflow Universe (e.g., search `climbing holds`, `bouldering`, `moonboard`, `kilterboard`) and GitHub/Kaggle (e.g. DeepHold).
* Target at least 250–500 images across different gym wall angles (slab, vertical, overhang, cave/roof) and textures (wooden ply, fiberglass, painted concrete).

### 3. Reliable Validation & Test Split
* Current test set contains only 1 image.
* Split your dataset into at least:
  * **Train**: 75–80%
  * **Validation**: 10–15%
  * **Test**: 10% (at least 15–25 unique images from walls not seen in train).

---

## 5. Phase 4: Training Pipeline Optimization

### Recommended Hyperparameters for NVIDIA RTX 4070 SUPER (12GB VRAM)

```python
from ultralytics import YOLO

# 1. Model Selection
# yolo11m gives the optimal balance of capacity without severe overfitting on <1000 images
model = YOLO('yolo11m-seg.pt')  # or 'yolo11m-obb.pt'

model.train(
    data='data_hold.yaml',
    epochs=120,
    imgsz=1024,              # Higher resolution preserves small foot chips
    batch=8,                 # Fixed batch size with batch norm stability
    device=0,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    patience=30,
    cos_lr=True,
    freeze=10,               # Freeze backbone for initial epochs to preserve pre-trained features
    
    # Climbing-specific augmentations
    degrees=25.0,            # Avoid 180.0; preserves wall gravity context
    flipud=0.0,              # Don't flip walls upside down
    fliplr=0.5,              # Horizontal flip is physically realistic
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.15,
    hsv_h=0.02,              # Slight hue jitter
    hsv_s=0.5,               # Handles chalky vs clean holds
    hsv_v=0.4,               # Handles bright spotlights vs dim ambient gym light
    scale=0.5,
    translate=0.1,
    perspective=0.0005,      # Simulates steep wall perspective angles
    
    name='bouldering_holds_v2',
    project='training_runs',
    exist_ok=True
)
```

---

## 6. Phase 5: Two-Stage Route & Color Identification

Attempting to classify 16 hold colors in a single detector with limited images causes failure on rare colors. A two-stage pipeline separates detection from color recognition:

```
[Full Photo]
     │
     ▼
[Stage 1: Hold Detector] ───► Bounding Box / Contour for each hold
     │
     ▼
[Stage 2: Color Extraction]
     ├── 1. Crop hold region using detected mask
     ├── 2. Mask out specular highlights & white chalk (HSV saturation filter)
     ├── 3. Cluster remaining pixels via K-Means in CIELAB color space
     └── 4. Match dominant cluster center to nearest gym route color palette
```

### Color Extraction Snippet
```python
import cv2
import numpy as np

def get_hold_dominant_color(hold_crop_bgr, mask=None):
    hsv = cv2.cvtColor(hold_crop_bgr, cv2.COLOR_BGR2HSV)
    
    # Filter out pure white chalk and specular reflection
    # (High value, very low saturation = chalk)
    non_chalk = (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 30)
    if mask is not None:
        valid_pixels = non_chalk & (mask > 0)
    else:
        valid_pixels = non_chalk
        
    pixels = hold_crop_bgr[valid_pixels]
    if len(pixels) == 0:
        return (128, 128, 128)  # Fallback neutral
        
    # K-Means clustering (K=1 dominant color)
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    dominant_bgr = centers[0].astype(int)
    return tuple(dominant_bgr)
```

---

## 7. Phase 6: Domain-Specific Post-Processing

In climbing applications, hold geometry and orientation provide critical routing context:

1. **Grip Direction & Orientation**:
   * Calculate minimum area bounding rectangle using `cv2.minAreaRect(pts)`.
   * Extract angle and aspect ratio:
     * Elongated vertical hold $\rightarrow$ Sidepull / Gaston
     * Flat horizontal top $\rightarrow$ Incut crimp / Jug
     * Symmetrical circular $\rightarrow$ Sloper / Foothold chip
2. **Contour Regularization**:
   * Low-resolution masks can produce jagged stair-step contours.
   * Apply polygon approximation: `cv2.approxPolyDP(pts, epsilon=1.5, closed=True)` or morphological closing (`cv2.morphologyEx`) before saving/rendering.

---

## 8. Prioritized Implementation Checklist

- [ ] **Milestone 1 (Immediate - 0 Retraining Required)**:
  - Add SAHI sliced inference to [predict_seg.py](file:///c:/Users/Wojtek/PycharmProjects/PythonProject/predict_seg.py) to benchmark test recall on small holds.
  - Test test-time threshold adjustments (`conf=0.22`, `iou=0.6`).
- [ ] **Milestone 2 (Data & Annotation Fix)**:
  - Decide between YOLO OBB (`yolo11m-obb.pt`) vs. SAM-assisted polygon segmentation.
  - Add 15–20 empty wall background images (hard negative samples).
  - Expand test set from 1 image to 15+ images.
- [ ] **Milestone 3 (Model Training v2)**:
  - Train `yolo11m` at `imgsz=1024` with realistic rotations (`degrees=25.0`, `flipud=0.0`).
  - Benchmark mAP50 and mAP50-95 against the baseline run.
- [ ] **Milestone 4 (Product Features)**:
  - Implement Stage 2 hold color extraction for route grouping.
  - Implement hold orientation / grip direction calculation.
