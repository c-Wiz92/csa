# SFace Implementation Audit Report

## 1. Exact Model and Authoritative Source

**Model file:** `face_recognition_sface_2021dec.onnx`

**Authoritative source:** Official OpenCV SFace model, released with OpenCV's `cv.FaceRecognizerSF` module. Original work: zhongyy/SFace (TIP2021 paper "SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition").

**Model metadata from ONNX inspection:**
- Input name: `data`
- Input shape: `[1, 3, 112, 112]` (NCHW layout)
- Input type: `tensor(float)`
- Output: 512-dimensional embedding

**Key detail from SFace source code (MobileFaceNet backbone):**
```python
def forward(self, x):
    x = x - 127.5
    x = x * 0.078125  # Equivalent to /128.0
    ...
```

The model contains an **internal normalization step**: `(x - 127.5) * 0.078125`. This maps `[0, 255]` input to approximately `[-1, +1]`. The model expects **raw [0, 255] pixel values** as input.

## 2. Official Preprocessing Requirements

Per OpenCV `FaceRecognizerSF` documentation and source:

| Step | Official OpenCV |
|------|-----------------|
| **Crop** | `alignCrop(src, bbox)` — takes full image + face bounding box |
| **Alignment** | **YES — 5-landmark face alignment performed internally** |
| **Resize** | To 112×112 (inside alignCrop) |
| **Color** | BGR (OpenCV convention) |
| **Normalization** | Raw `[0, 255]` pixel values — model handles normalization internally |
| **Tensor layout** | NCHW (1, 3, 112, 112) |
| **Embedding handling** | Cosine similarity via `match()` |

**Critical:** `alignCrop()` performs **5-landmark face alignment** using a built-in landmark detector. It computes a similarity transform to canonical eye/nose/mouth positions, then crops and resizes to 112×112.

## 3. Current Preprocessing Implementation

The current `MobileFaceNetRecognizer._preprocess()` does:
1. Resize face crop to 112×112
2. Convert BGR→RGB
3. Cast to float32
4. Normalize: `(img - 127.5) / 128.0` → range ≈ [-1, +1]
5. Transpose HWC → CHW
6. Add batch dimension

## 4. Discrepancy Table

| Step | Official OpenCV | Current Implementation | Match? |
|------|-----------------|------------------------|--------|
| **Crop** | `alignCrop(image, bbox)` — uses full image + bbox | Receives pre-cropped face bounding box | ⚠️ Different input contract |
| **Alignment** | **YES — 5-landmark alignment** | **NONE — just resizes bbox crop** | ❌ **MAJOR DISCREPANCY** |
| **Resize** | To 112×112 | To 112×112 | ✅ |
| **Color** | BGR | BGR→RGB conversion | ⚠️ **Discrepancy** |
| **Normalization** | Raw [0, 255] (model handles internally) | (x - 127.5) / 128.0 → [-1, 1] | ⚠️ **Double normalization** |
| **Tensor layout** | NCHW (1, 3, 112, 112) | NCHW (1, 3, 112, 112) | ✅ |
| **Embedding handling** | Cosine similarity via match() | L2 normalization + dot product | ✅ (equivalent) |

## 5. Critical Issues Found

### Issue 1: Missing Face Alignment (MAJOR)

**This is the most likely cause of poor discrimination.**

The official SFace model expects **aligned faces** — geometrically normalized using 5-point landmarks. The model was trained on aligned faces.

Our implementation simply resizes the YOLO bounding box crop. This means:
- Faces at different poses/angles are not geometrically normalized
- Eye positions, nose position, mouth corners are not in consistent locations
- The model receives inputs significantly different from its training distribution

This directly explains:
- Genuine similarity mean only 0.97 (should be >0.99 for aligned)
- Genuine minimum drops to 0.918 (some genuine faces poorly aligned)
- Impostor maximum reaches 0.987 (some impostors similar when misaligned)
- Substantial overlap between distributions

### Issue 2: Incorrect Normalization (MODERATE)

The model already performs `(x - 127.5) * 0.078125` internally. Our preprocessing ALSO applies `(x - 127.5) / 128.0`:

- Our preprocessing maps `[0, 255]` → `[-0.996, +1.004]`
- The model then applies its internal normalization AGAIN to this already-normalized input
- Model receives values in range `[-1.003, +0.992]` instead of `[0, 255]`
- This double-normalization degrades feature quality

**Fix:** Remove our normalization step. Pass raw `[0, 255]` float32 values directly.

### Issue 3: Color Channel Order (MINOR)

Official OpenCV pipeline uses BGR. Our implementation converts BGR→RGB. Since the model was trained on data fed by OpenCV's BGR pipeline, we should keep BGR and NOT convert to RGB.


## 6. Fixes Required

### Fix 1: Remove Double Normalization

Remove the `(img - 127.5) / 128.0` line. The model handles its own normalization internally.

### Fix 2: Keep BGR Color Order

Remove the `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` call. The model expects BGR (OpenCV convention).

### Fix 3: Add Face Alignment (RECOMMENDED)

To properly use SFace, implement 5-landmark face alignment:
1. Detect 5 facial landmarks (eye centers, nose tip, mouth corners)
2. Compute similarity transform to canonical positions
3. Warp and crop the face

This requires a landmark detection model (e.g., OpenCV's Facemark or a dedicated landmark model).

**Note:** Without alignment, the model will continue to perform poorly on real-world data with pose variation, regardless of threshold tuning.

## 7. Before/After Statistics

### Before (Current Implementation)
```
Genuine:  mean=0.9716, min=0.9184, max=1.0000
Impostor: mean=0.9348, min=0.7412, max=0.9871

Threshold 0.40: 100% false accepts
Threshold 0.99: 0% false accepts, 26.67% genuine acceptance
```

### After Fixes
*To be measured after implementing the fixes above.*

## 8. Threshold Assessment

**0.99 remains reasonable as a provisional safety threshold** but is NOT optimal. The high threshold is compensating for:
1. Poor discrimination caused by missing alignment
2. Double normalization degrading feature quality

Once alignment is implemented:
- The genuine/impostor separation should improve significantly
- A lower threshold (e.g., 0.3-0.5) may become viable
- This must be measured, not assumed

## 9. Summary of Actions

| Priority | Action | Status |
|----------|--------|--------|
| HIGH | Remove double normalization (remove `(img - 127.5) / 128.0`) | Pending |
| HIGH | Remove BGR→RGB conversion (keep BGR) | Pending |
| HIGH | Add 5-landmark face alignment | Pending |
| LOW | Re-run real-face validation after fixes | Pending |
| LOW | Consider providing full image + bbox to recognizer (for alignCrop) | Pending |

## 10. Recommendation

The current implementation has **two definite bugs** (double normalization, wrong color order) and **one architectural gap** (missing alignment). All three contribute to poor discrimination.

**Do NOT tune the threshold to fix this.** The threshold is a symptom, not the cause.

After fixing the normalization and color bugs, measure again. If separation remains poor, alignment is confirmed as the root cause and must be implemented for real-world use.

The Olivetti dataset (5 identities, controlled frontal imagery) is sufficient to catch these implementation bugs but insufficient to establish optimal thresholds for real-world deployment.

---

*Report prepared by auditing:*
- *OpenCV FaceRecognizerSF documentation and source*
- *SFace original implementation (zhongyy/SFace)*
- *ONNX model input/output inspection*
- *Current MobileFaceNetRecognizer implementation*
