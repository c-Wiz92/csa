"""Speaker localization and bearing estimation module.

Provides an abstract interface and a camera-based bearing estimator.
The bearing estimator calculates the horizontal angle of a speaker
relative to the camera optical axis using a pinhole camera model.

Architecture:
    SpeakerLocalizer (ABC) ← CameraBearingEstimator

The bearing estimator is independent of target selection logic.
Target selection (which face is the active speaker) is handled separately
and supplies a TrackedFace to the estimator.

Future extensions:
    - Multi-speaker target selection
    - Motion estimation (bearing velocity)
    - Temporal filtering (Kalman, EMA)
    - IMU fusion
"""

import math
import time
from abc import ABC, abstractmethod
from typing import Optional, List

from src.data_structures import TrackedFace, SpeakerBearing


class SpeakerLocalizer(ABC):
    """Abstract base class for speaker localization.

    Implementations estimate the bearing (horizontal angle) of a speaker
    relative to the camera's forward direction.
    """

    @abstractmethod
    def compute_bearing(
        self, tracked_face: TrackedFace, frame_width: int, frame_number: int = 0
    ) -> SpeakerBearing:
        """Compute the horizontal bearing of a tracked face.

        Args:
            tracked_face: The tracked face to compute bearing for.
            frame_width: Width of the camera frame in pixels.
            frame_number: Current frame number for timestamping.

        Returns:
            SpeakerBearing with angle and position information.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset any internal state."""
        pass


class CameraBearingEstimator(SpeakerLocalizer):
    """Estimates speaker bearing using a pinhole camera model.

    Uses the horizontal center of the face bounding box to compute
    the angle relative to the camera optical axis.

    The bearing is computed as:
        angle = atan((center_x - image_center_x) / focal_length_pixels)

    This gives:
        - Positive angle: speaker is to the RIGHT of center
        - Negative angle: speaker is to the LEFT of center
        - Zero: speaker is directly in front

    Config keys:
        focal_length_pixels: Focal length in pixels (default: estimated from FOV)
        horizontal_fov_degrees: Horizontal field of view (used if focal_length not set)
        center_deadband_degrees: Angles within this range are reported as CENTER
    """

    def __init__(self, config: dict):
        self.focal_length_pixels: float = config.get("focal_length_pixels", 0.0)
        self.horizontal_fov_degrees: float = config.get(
            "horizontal_fov_degrees", 60.0
        )
        self.center_deadband_degrees: float = config.get(
            "center_deadband_degrees", 5.0
        )

        # If focal_length_pixels not provided, estimate from FOV
        if self.focal_length_pixels <= 0:
            # focal_length = (frame_width / 2) / tan(FOV / 2)
            # We'll compute per-frame since frame_width varies
            self._use_fov_fallback = True
        else:
            self._use_fov_fallback = False

    def compute_bearing(
        self, tracked_face: TrackedFace, frame_width: int, frame_number: int = 0
    ) -> SpeakerBearing:
        """Compute horizontal bearing using pinhole camera model.

        Args:
            tracked_face: The tracked face to compute bearing for.
            frame_width: Width of the camera frame in pixels.
            frame_number: Current frame number.

        Returns:
            SpeakerBearing with angle in degrees and position info.
        """
        # Calculate face center
        center_x = (tracked_face.bbox.x1 + tracked_face.bbox.x2) / 2.0
        image_center_x = frame_width / 2.0

        # Calculate focal length in pixels
        if self._use_fov_fallback:
            # focal_length = (frame_width / 2) / tan(FOV / 2)
            fov_rad = math.radians(self.horizontal_fov_degrees)
            focal_length = (frame_width / 2.0) / math.tan(fov_rad / 2.0)
        else:
            focal_length = self.focal_length_pixels

        # Calculate angle using pinhole model
        # angle = atan((center_x - image_center_x) / focal_length)
        displacement = center_x - image_center_x
        angle_rad = math.atan2(displacement, focal_length)
        angle_deg = math.degrees(angle_rad)

        # Normalized horizontal position [-1, 1]
        normalized_x = displacement / (frame_width / 2.0)
        normalized_x = max(-1.0, min(1.0, normalized_x))

        return SpeakerBearing(
            track_id=tracked_face.track_id,
            angle_degrees=round(angle_deg, 2),
            center_x=center_x,
            normalized_x=round(normalized_x, 4),
            frame_number=frame_number,
            timestamp=time.time(),
        )

    def reset(self):
        """No internal state to reset."""
        pass


def select_active_speaker(
    tracked_faces: List[TrackedFace],
) -> Optional[TrackedFace]:
    """Select the active speaker from tracked faces.

    For V1: if exactly one face is classified as SPEAKING, return it.
    If no face is speaking, return None.
    If multiple faces are speaking, return None (ambiguous — no feedback).

    Args:
        tracked_faces: List of tracked faces with speaking state.

    Returns:
        The active speaker TrackedFace, or None if ambiguous/no speaker.
    """
    speaking_faces = [
        f for f in tracked_faces if f.speaking_state.is_speaking
    ]

    if len(speaking_faces) == 1:
        return speaking_faces[0]
    elif len(speaking_faces) == 0:
        return None
    else:
        # Multiple speakers detected — ambiguous, do not issue feedback
        return None