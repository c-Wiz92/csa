"""Face detection module.

Provides an abstract interface and implementations:
- YOLOv8FaceDetector: YOLOv8-Face model (ultralytics)
- PlaceholderFaceDetector: Returns no detections (for testing)

Interface:
    FaceDetector.detect(frame) -> List[Detection]
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from src.data_structures import Detection, BoundingBox


class FaceDetector(ABC):
    """Abstract base class for face detectors.

    Any face detector implementation must provide a `detect` method
    that takes a frame and returns a list of detections.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect faces in a frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection objects with bounding boxes and confidences.
        """
        pass

    @property
    def device(self) -> str:
        """Device used for inference (e.g., 'cuda', 'cpu')."""
        return "cpu"


class YOLOv8FaceDetector(FaceDetector):
    """YOLOv8-Face detector using ultralytics.

    Loads a YOLOv8 model trained for face detection and runs inference
    on each frame. Supports GPU acceleration via CUDA.

    Config keys:
        model_path: Path to .pt weights file
        confidence_threshold: Minimum confidence to keep a detection
        device: "cuda" or "cpu"
        img_size: Inference resolution (e.g., 640)
    """

    def __init__(self, config: dict):
        from ultralytics import YOLO

        self.confidence_threshold: float = config.get("confidence_threshold", 0.5)
        self._device: str = config.get("device", "cpu")
        self.img_size: int = config.get("img_size", 640)
        model_path: str = config.get("model_path", "models/yolov8n-face.pt")

        self.model = YOLO(model_path)
        # Warm up the model with a dummy inference
        self._warmup()

    def _warmup(self):
        """Run a dummy inference to initialize the model (avoids first-frame latency)."""
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        self.model.predict(
            dummy,
            device=self._device,
            imgsz=self.img_size,
            conf=self.confidence_threshold,
            verbose=False,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect faces in a frame using YOLOv8.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection objects with bounding boxes and confidences.
        """
        results = self.model.predict(
            frame,
            device=self._device,
            imgsz=self.img_size,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # boxes.xyxy is (N, 4) tensor, boxes.conf is (N,) tensor
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            bbox = BoundingBox(
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
            )
            detections.append(Detection(bbox=bbox, confidence=float(confs[i])))

        return detections

    @property
    def device(self) -> str:
        return self._device


class PlaceholderFaceDetector(FaceDetector):
    """Placeholder detector that returns no detections.

    Used for scaffolding and testing the pipeline without a real model.
    Replace with YOLOv8FaceDetector or similar when ready.
    """

    def __init__(self, config: dict):
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self._device = config.get("device", "cpu")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Returns empty list. Replace with real implementation."""
        return []

    @property
    def device(self) -> str:
        return self._device
