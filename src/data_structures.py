"""Shared data structures for the assistive vision pipeline.

These types are used across all modules to ensure consistent data flow
from detection through tracking, speaking detection, and visualization.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List

import numpy as np


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)

    def is_valid(self, frame_width: int = 0, frame_height: int = 0) -> bool:
        if self.x1 < 0 or self.y1 < 0 or self.x2 <= self.x1 or self.y2 <= self.y1:
            return False
        if frame_width > 0 and frame_height > 0:
            if self.x2 > frame_width or self.y2 > frame_height:
                return False
        return True


@dataclass
class Detection:
    """A single face detection from the face detector."""
    bbox: BoundingBox
    confidence: float


@dataclass
class FaceLandmarks:
    """Facial landmarks for a single face.

    Stores all 478 MediaPipe face mesh landmarks as normalized (x, y, z)
    coordinates. Also provides convenient access to mouth-specific landmarks
    needed for speaking detection.

    Attributes:
        landmarks: List of (x, y, z) tuples, normalized to [0, 1].
                   Index follows MediaPipe Face Mesh convention.
    """
    landmarks: list  # List of (x, y, z) tuples, normalized [0, 1]

    # Key landmark indices (MediaPipe Face Mesh 478-point model)
    # Mouth corners
    MOUTH_LEFT: int = 61
    MOUTH_RIGHT: int = 291
    # Upper lip top
    UPPER_LIP_TOP: int = 13
    # Lower lip bottom
    LOWER_LIP_BOTTOM: int = 14
    # Additional mouth landmarks for future use
    UPPER_LIP_INNER: int = 0
    LOWER_LIP_INNER: int = 17

    @property
    def mouth_left(self) -> tuple:
        return self.landmarks[self.MOUTH_LEFT]

    @property
    def mouth_right(self) -> tuple:
        return self.landmarks[self.MOUTH_RIGHT]

    @property
    def upper_lip(self) -> tuple:
        return self.landmarks[self.UPPER_LIP_TOP]

    @property
    def lower_lip(self) -> tuple:
        return self.landmarks[self.LOWER_LIP_BOTTOM]

    @property
    def num_landmarks(self) -> int:
        return len(self.landmarks)


@dataclass
class MouthObservation:
    """A single frame's mouth measurements derived from facial landmarks.

    Stored in the temporal buffer for each tracked face. Contains all data
    needed for current and future speaking-detection approaches.

    Attributes:
        frame_number: Frame index when this observation was captured
        timestamp: Wall-clock time when captured
        mouth_landmarks: Dict of landmark_index → (x, y) pixel coordinates
                         for key mouth points (corners, upper/lower lip)
        mouth_width: Horizontal distance between mouth corners (pixels)
        mouth_height: Vertical distance between upper and lower lip (pixels)
        mar: Mouth Aspect Ratio = mouth_height / mouth_width (0.0 if width=0)
        all_landmarks: Optional list of all 478 (x, y, z) normalized landmarks
    """
    frame_number: int
    timestamp: float
    mouth_landmarks: dict  # {landmark_idx: (x_px, y_px)}
    mouth_width: float
    mouth_height: float
    mar: float
    all_landmarks: Optional[list] = None  # All 478 landmarks (x, y, z) normalized


@dataclass
class SpeakingState:
    """Speaking detection result for a single face."""
    is_speaking: bool
    confidence: float  # 0.0 to 1.0 probability of speaking


@dataclass
class TrackedFace:
    """A tracked face with all associated metadata.

    This is the primary data structure passed through the pipeline.
    It carries detection, tracking, and speaking information for a single face.

    Extension points:
        identity / identity_confidence: For future FaceRecognizer module.
        metadata: Generic dict for additional per-face data.
    """
    track_id: int
    bbox: BoundingBox
    detection_confidence: float
    speaking_state: SpeakingState
    face_crop: Optional[np.ndarray] = None

    # --- Extension point for V2 face recognition ---
    identity: Optional[str] = None
    identity_confidence: Optional[float] = None
    # -------------------------------------------------

    # --- Extension point for facial landmarks ---
    landmarks: Optional[FaceLandmarks] = None
    # ---------------------------------------------

    # Generic metadata for future extensions
    metadata: Dict[str, Any] = field(default_factory=dict)


class MovementDirection(Enum):
    """Direction of speaker movement relative to camera."""
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STATIONARY = "STATIONARY"
    UNKNOWN = "UNKNOWN"


class AlignmentState(Enum):
    """Alignment state of the user relative to the active speaker."""
    ALIGNED = "ALIGNED"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    NO_TARGET = "NO_TARGET"
    UNKNOWN = "UNKNOWN"


@dataclass
class AlignmentResult:
    """Alignment decision produced by the alignment controller.

    Attributes:
        state: Current alignment state.
        track_id: Active speaker track ID (-1 if no target).
        bearing_degrees: Current (smoothed) bearing in degrees.
        command: Human-readable command string.
        timestamp: Wall-clock time.
        frame_number: Frame number.
        angular_velocity: Speaker angular velocity in deg/s (if available).
        confidence: Confidence in the alignment decision (0.0 to 1.0).
    """
    state: AlignmentState
    track_id: int
    bearing_degrees: float
    command: str
    timestamp: float = 0.0
    frame_number: int = 0
    angular_velocity: float = 0.0
    confidence: float = 0.0


@dataclass
class SpeakerMotion:
    """Motion state of a speaker derived from bearing measurements.

    Attributes:
        track_id: The tracked face ID.
        current_angle: Current bearing angle in degrees.
        previous_angle: Previous bearing angle in degrees.
        angular_velocity: Smoothed angular velocity in degrees/second.
        movement: Classified movement direction.
        timestamp: Wall-clock time of current measurement.
        frame_number: Frame number of current measurement.
    """
    track_id: int
    current_angle: float
    previous_angle: float
    angular_velocity: float
    movement: MovementDirection
    timestamp: float = 0.0
    frame_number: int = 0


@dataclass
class SpeakerBearing:
    """Horizontal bearing of a speaker relative to camera optical axis.

    Attributes:
        track_id: The tracked face ID of the speaker.
        angle_degrees: Horizontal angle in degrees. Positive = right of center,
                       negative = left of center, 0 = directly in front.
        center_x: Horizontal pixel coordinate of the face bounding box center.
        normalized_x: Horizontal position normalized to [-1, 1], where
                      -1 = far left, 0 = center, 1 = far right.
        frame_number: Frame number when this bearing was calculated.
        timestamp: Wall-clock time when captured.
    """
    track_id: int
    angle_degrees: float
    center_x: float
    normalized_x: float
    frame_number: int = 0
    timestamp: float = 0.0

    @property
    def direction(self) -> str:
        """Return human-readable direction string."""
        if abs(self.angle_degrees) < 5.0:
            return "CENTER"
        elif self.angle_degrees > 0:
            return "RIGHT"
        else:
            return "LEFT"


@dataclass
class FrameResult:
    """Complete result for a single processed frame."""
    frame: np.ndarray
    frame_number: int
    timestamp: float
    tracked_faces: List[TrackedFace]
    detection_count: int = 0
    processing_time_ms: float = 0.0
    active_speaker_bearing: Optional[SpeakerBearing] = None
    active_speaker_motion: Optional[SpeakerMotion] = None
    active_speaker_alignment: Optional[AlignmentResult] = None
