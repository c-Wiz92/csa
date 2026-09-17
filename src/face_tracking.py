"""Face tracking module.

Provides an abstract interface and implementations:
- BotSORTFaceTracker: BotSORT tracker (boxmot library)
- PlaceholderFaceTracker: Sequential IDs without real tracking

Interface:
    FaceTracker.update(detections, frame) -> List[TrackedFace]
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.data_structures import (
    Detection,
    TrackedFace,
    SpeakingState,
    BoundingBox,
)


class FaceTracker(ABC):
    """Abstract base class for face trackers.

    Any tracker must consume detections and produce tracked faces
    with persistent IDs across frames.
    """

    @abstractmethod
    def update(
        self, detections: List[Detection], frame: np.ndarray
    ) -> List[TrackedFace]:
        """Update tracker with new detections.

        Args:
            detections: List of detections from the face detector.
            frame: Current BGR frame (for appearance features if needed).

        Returns:
            List of TrackedFace objects with persistent track IDs.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset all tracking state."""
        pass


class BotSORTFaceTracker(FaceTracker):
    """BotSORT face tracker using the boxmot library.

    Takes face detections from YOLOv8-Face and assigns persistent
    track IDs across frames using the BotSORT algorithm.

    Config keys:
        track_high_thresh: High confidence threshold for detections
        track_low_thresh: Low confidence threshold for detections
        new_track_thresh: Threshold for creating new tracks
        track_buffer: Frames to keep lost tracks before removing
        with_reid: Whether to use ReID (False for V1, no ReID model)
        use_cmc: Whether to use camera motion correction
        frame_rate: Frame rate for the tracker
    """

    def __init__(self, config: dict):
        from boxmot.trackers.bbox.botsort import BotSort

        self.track_high_thresh: float = config.get("track_high_thresh", 0.5)
        self.track_low_thresh: float = config.get("track_low_thresh", 0.1)
        self.new_track_thresh: float = config.get("new_track_thresh", 0.6)
        self.track_buffer: int = config.get("track_buffer", 30)
        with_reid: bool = config.get("with_reid", False)
        use_cmc: bool = config.get("use_cmc", False)
        frame_rate: int = config.get("frame_rate", 30)

        self.tracker = BotSort(
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            with_reid=with_reid,
            use_cmc=use_cmc,
            frame_rate=frame_rate,
        )

    def update(
        self, detections: List[Detection], frame: np.ndarray
    ) -> List[TrackedFace]:
        """Update tracker with new detections and return tracked faces.

        Args:
            detections: List of face detections from YOLOv8-Face.
            frame: Current BGR frame (used for camera motion correction).

        Returns:
            List of TrackedFace objects with persistent track IDs.
        """
        if not detections:
            # Pass empty array with correct shape (0, 6)
            dets_array = np.empty((0, 6), dtype=np.float32)
        else:
            # Convert detections to [x1, y1, x2, y2, conf, cls] format
            dets_array = np.zeros((len(detections), 6), dtype=np.float32)
            for i, det in enumerate(detections):
                dets_array[i, 0] = det.bbox.x1
                dets_array[i, 1] = det.bbox.y1
                dets_array[i, 2] = det.bbox.x2
                dets_array[i, 3] = det.bbox.y2
                dets_array[i, 4] = det.confidence
                dets_array[i, 5] = 0  # class 0 = face

        # Run BotSORT update
        results = self.tracker.update(dets_array, frame)

        # Convert results to TrackedFace objects
        tracked: List[TrackedFace] = []
        if results is None or len(results.id) == 0:
            return tracked

        xyxy = results.xyxy
        ids = results.id
        confs = results.conf

        for i in range(len(ids)):
            bbox = BoundingBox(
                x1=int(round(xyxy[i, 0])),
                y1=int(round(xyxy[i, 1])),
                x2=int(round(xyxy[i, 2])),
                y2=int(round(xyxy[i, 3])),
            )
            tracked.append(
                TrackedFace(
                    track_id=int(ids[i]),
                    bbox=bbox,
                    detection_confidence=float(confs[i]),
                    speaking_state=SpeakingState(is_speaking=False, confidence=0.0),
                    identity=None,  # V2: face recognition
                )
            )

        return tracked

    def reset(self):
        """Reset all tracking state."""
        self.tracker.reset()


class PlaceholderFaceTracker(FaceTracker):
    """Placeholder tracker that assigns sequential IDs without real tracking.

    Each detection gets a new ID every frame. This means no persistence
    across frames — purely for scaffolding. Replace with real tracker
    (BotSORT/DeepSORT) when ready.
    """

    def __init__(self, config: dict):
        self._next_id: int = 1

    def update(
        self, detections: List[Detection], frame: np.ndarray
    ) -> List[TrackedFace]:
        """Assign a new ID to each detection (no real tracking)."""
        tracked = []
        for det in detections:
            tracked.append(
                TrackedFace(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    detection_confidence=det.confidence,
                    speaking_state=SpeakingState(is_speaking=False, confidence=0.0),
                )
            )
            self._next_id += 1
        return tracked

    def reset(self):
        self._next_id = 1
