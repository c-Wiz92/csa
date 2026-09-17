# Assistive Vision System — V1

Real-time computer vision prototype as an assistive system for blind/visually impaired users.

## V1 Scope

```
Webcam → Face Detection → Face Tracking → Visual Speaking Detection → Debug Display
```

## What's Implemented

- Project scaffolding with modular architecture
- Clean interfaces (abstract base classes) for all pipeline stages
- Configuration system (YAML with defaults)
- Video capture wrapper
- Visualization/debug display
- Pipeline orchestrator with real-time loop
- **Face recognition** — SFace ONNX model (128-dim embeddings) with identity management and quality gating

## What's NOT Implemented Yet

- Face detection model (YOLOv8-Face) — placeholder only
- Face tracking model (BotSORT) — placeholder only
- Visual speaking detection (MAR/temporal model) — placeholder only
- TTS / audio output — deferred

## Project Structure

```
CSA1/
├── README.md
├── config.yaml              # All configuration
├── requirements.txt         # Python dependencies
├── models/                  # Downloaded model weights (not in git)
├── src/
│   ├── __init__.py
│   ├── main.py              # Entry point
│   ├── config.py            # Config loader with defaults
│   ├── data_structures.py   # Shared types (TrackedFace, Detection, etc.)
│   ├── pipeline.py          # Pipeline orchestrator
│   ├── video_capture.py     # Webcam input
│   ├── face_detection.py    # FaceDetector interface + placeholder
│   ├── face_tracking.py     # FaceTracker interface + placeholder
│   ├── speaking_detection.py # SpeakingDetector interface + placeholder
│   ├── face_recognition.py  # FaceRecognizer interface (V2 stub)
│   └── visualization.py     # Debug display
└── tests/
    └── __init__.py
```

## Architecture

### Data Flow

```
Camera → Frame → FaceDetector.detect() → List[Detection]
       → FaceTracker.update() → List[TrackedFace]
       → FaceRecognizer.get_embedding() per face → embedding
       → IdentityManager.match_embedding() → identity
       → SpeakingDetector.detect() per face → SpeakingState
       → Visualizer.draw() → Display
```

### Key Design Decisions

1. **Abstract base classes** for each module — swap implementations without touching the pipeline.
2. **TrackedFace** is the central data structure — carries track_id, bbox, confidence, speaking state, and extension fields for future face recognition.
3. **Models loaded once at startup** — no per-frame model loading.
4. **CUDA preferred, CPU fallback** — `get_device()` checks availability at runtime.
5. **Face recognition as independent module** — `FaceRecognizer` interface defined in `face_recognition.py`, ready for V2 without pipeline changes.

### Extension Points

| Future Feature | Where | Status |
|---|---|---|
| TTS audio output | New `src/audio_output.py` | Not started |
| Better speaking model | `src/speaking_detection.py` | Placeholder |

## Face Recognition

The face recognition module uses the **SFace** model (`face_recognition_sface_2021dec.onnx`) from the OpenCV Model Zoo.

### Model Details

- **Architecture**: SFace (Squeeze-and-Excitation Face recognition)
- **Input**: 112×112 RGB face crop
- **Output**: 128-dim embedding vector (L2-normalized by the recognizer)
- **Runtime**: ONNX Runtime with CPUExecutionProvider
- **Performance**: ~20ms per inference on CPU (median)

### Preprocessing Pipeline

1. Resize to 112×112
2. BGR → RGB color conversion
3. Normalize to [-1, 1] range: `(pixel - 127.5) / 128.0`
4. HWC → CHW transpose
5. Add batch dimension: (1, 3, 112, 112)

### Components

| Component | File | Responsibility |
|---|---|---|
| `MobileFaceNetRecognizer` | `src/face_recognition.py` | ONNX inference, preprocessing, L2 normalization |
| `IdentityManager` | `src/identity_manager.py` | Identity registry, embedding matching, track association |
| `FaceQualityGate` | `src/face_quality.py` | Filters unsuitable crops (too small, too dark/bright) |
| `RecognitionScheduler` | `src/recognition_scheduler.py` | Controls which tracks get recognized and when |

### Configuration

Edit `config.yaml` → `face_recognition:` section:
- `enabled`: Enable/disable recognition
- `model_path`: Path to the ONNX model file
- `matching_threshold`: Cosine similarity threshold (default 0.99, validated)
- `recognition_interval`: Frames between re-verification of known identities
- `retry_interval`: Frames between retries for unknown tracks
- `max_per_frame`: Max recognition attempts per frame (rate limiting)

### Validation Results

Empirical validation performed using the **Olivetti faces dataset** (5 identities, 10 images each, 50 images total).

**Dataset source**: scikit-learn `fetch_olivetti_faces()` - public domain academic dataset

**Genuine (Same-Person) Similarities** (225 pairs):
| Stat | Value |
|------|-------|
| min | 0.9184 |
| max | 0.9992 |
| mean | 0.9716 |
| median | 0.9754 |
| std | 0.0206 |

**Impostor (Different-Person) Similarities** (1000 pairs):
| Stat | Value |
|------|-------|
| min | 0.8441 |
| max | 0.9871 |
| mean | 0.9348 |
| median | 0.9349 |
| std | 0.0260 |

**Separation (genuine_min - impostor_max)**: -0.0687 (negative = overlap)

**Threshold Evaluation**:
| Threshold | GAR | False Accepts | False Rejects |
|-----------|-----|---------------|---------------|
| 0.40 | 1.0000 | 1000 | 0 |
| 0.90 | 1.0000 | 916 | 0 |
| 0.95 | 0.8133 | 304 | 42 |
| 0.98 | 0.4311 | 23 | 128 |
| **0.99** | **0.2667** | **0** | **165** |
| 0.995 | 0.0667 | 0 | 210 |
| 1.00 | 0.0044 | 0 | 224 |

**Conclusion**: The previous threshold of 0.4 was **unsafe** (100% false accept rate). A threshold of **0.99** provides zero false accepts with 26.67% genuine acceptance rate. This is the recommended conservative threshold for this assistive application where false identity assignment is the most critical failure mode.

**Limitations**:
- Small validation dataset (5 identities)
- Olivetti dataset has controlled conditions (frontal faces, uniform background)
- Real-world performance may vary with pose, lighting, and expression changes
- Threshold may need adjustment with larger, more diverse datasets

**Validation script**: `scripts/evaluate_face_recognition.py`

### Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows

# 2. Install core dependencies (already have torch + opencv)
pip install -r requirements.txt

# 3. Install onnxruntime for face recognition
pip install onnxruntime

# 4. Download the SFace model (if not already present)
# models/face_recognition_sface_2021dec.onnx (~4MB)

# 5. Run
python -m src.main
```

## Configuration

Edit `config.yaml` to adjust:
- Camera device, resolution, FPS
- Detection confidence thresholds
- Tracking parameters
- Speaking detection thresholds
- Display options

## System Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support (optional, CPU fallback available)
- Webcam

### Tested On

- Python 3.13.7
- NVIDIA RTX 3050 4GB Laptop GPU
- PyTorch 2.11.0+cu128
- OpenCV 5.0.0
- Windows 11