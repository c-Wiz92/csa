"""Facial landmark detection module.

Provides an abstract interface and a MediaPipe implementation.
The landmark component takes tracked faces and produces facial landmarks
for each face region.

Interface:
    FaceLandmarker.detect_landmarks(frame, tracked_faces) -> List[FaceLandmarks]
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from src.data_structures import TrackedFace, FaceLandmarks


class FaceLandmarker(ABC):
    """Abstract base class for facial landmark detectors.

    Any landmark detector must take a frame and a list of tracked faces,
    and return facial landmarks for each face.
    """

    @abstractmethod
    def detect_landmarks(
        self, frame: np.ndarray, tracked_faces: List[TrackedFace]
    ) -> None:
        """Detect facial landmarks for each tracked face.

        Updates each TrackedFace in-place with a FaceLandmarks object.
        If landmarks cannot be detected for a face, its landmarks field
        remains None.

        Args:
            frame: BGR image as numpy array (H, W, 3).
            tracked_faces: List of tracked faces to process.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset any internal state."""
        pass


class MediaPipeFaceLandmarker(FaceLandmarker):
    """MediaPipe Face Landmarker implementation.

    Uses the MediaPipe FaceLandmarker model to detect 478 facial landmarks
    for each tracked face. Runs on CPU (MediaPipe handles its own acceleration).

    Config keys:
        model_path: Path to the .task model file
        num_faces: Maximum number of faces to detect
        min_face_detection_confidence: Minimum confidence for face detection
        min_face_presence_confidence: Minimum confidence for face presence
        min_tracking_confidence: Minimum confidence for face tracking
    """

    def __init__(self, config: dict):
        from mediapipe import tasks
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.face_landmarker import (
            FaceLandmarker,
            FaceLandmarkerOptions,
        )
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )

        model_path: str = config.get("model_path", "models/face_landmarker.task")
        num_faces: int = config.get("num_faces", 5)
        min_face_detection_confidence: float = config.get(
            "min_face_detection_confidence", 0.5
        )
        min_face_presence_confidence: float = config.get(
            "min_face_presence_confidence", 0.5
        )
        min_tracking_confidence: float = config.get(
            "min_tracking_confidence", 0.5
        )

        base_options = BaseOptions(model_asset_path=model_path)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._landmarker = FaceLandmarker.create_from_options(options)
        self._num_faces = num_faces

    def detect_landmarks(
        self, frame: np.ndarray, tracked_faces: List[TrackedFace]
    ) -> None:
        """Detect facial landmarks for each tracked face using MediaPipe.

        Args:
            frame: BGR image as numpy array (H, W, 3).
            tracked_faces: List of tracked faces to process.
        """
        if not tracked_faces:
            return

        import mediapipe as mp

        # Convert BGR to RGB for MediaPipe
        frame_rgb = frame[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run MediaPipe Face Landmarker
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return

        # Associate landmarks with tracked faces by proximity
        # MediaPipe returns landmarks in the same order as detected faces
        # We match by finding the closest tracked face to each detected face
        h, w = frame.shape[:2]

        for face_landmarks_raw in result.face_landmarks:
            # Convert normalized landmarks to pixel coordinates for matching
            # Use the centroid of all landmarks to find the closest tracked face
            cx = sum(lm.x for lm in face_landmarks_raw) / len(face_landmarks_raw)
            cy = sum(lm.y for lm in face_landmarks_raw) / len(face_landmarks_raw)

            # Find the closest tracked face by centroid distance
            best_match: Optional[TrackedFace] = None
            best_dist: float = float("inf")

            for tracked in tracked_faces:
                if tracked.landmarks is not None:
                    continue  # Already has landmarks
                tcx, tcy = tracked.bbox.center
                dist = ((cx * w - tcx) ** 2 + (cy * h - tcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_match = tracked

            if best_match is not None:
                # Convert landmarks to list of (x, y, z) tuples
                landmarks_list = [
                    (lm.x, lm.y, lm.z) for lm in face_landmarks_raw
                ]
                best_match.landmarks = FaceLandmarks(landmarks=landmarks_list)

    def reset(self):
        """Reset the landmarker state."""
        # MediaPipe IMAGE mode is stateless per call, nothing to reset
        pass


class PlaceholderFaceLandmarker(FaceLandmarker):
    """Placeholder landmarker that does nothing.

    Used for testing the pipeline without landmark detection.
    """

    def __init__(self, config: dict):
        pass

    def detect_landmarks(
        self, frame: np.ndarray, tracked_faces: List[TrackedFace]
    ) -> None:
        """Does nothing. Landmarks remain None."""
        pass

    def reset(self):
        pass