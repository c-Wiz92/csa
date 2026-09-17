"""Speaker motion estimation module.

Estimates whether the active speaker is moving LEFT, RIGHT, or STATIONARY
based on bearing measurements over time.

Architecture:
    SpeakerMotionEstimator (ABC) ← BearingMotionEstimator

The motion estimator receives SpeakerBearing measurements and computes
angular velocity with temporal smoothing. It is independent of how the
active speaker was selected, supporting future multi-speaker pipelines.

Pipeline position:
    Bearing Estimation → Motion Estimation → [Future: Temporal Filtering] → [Future: Feedback]
"""

import statistics
from collections import deque
from abc import ABC, abstractmethod
from typing import Optional, Dict

from src.data_structures import SpeakerBearing, SpeakerMotion, MovementDirection


class SpeakerMotionEstimator(ABC):
    """Abstract base class for speaker motion estimation.

    Implementations analyze a sequence of bearing measurements to
    determine the speaker's movement direction.
    """

    @abstractmethod
    def update(self, bearing: SpeakerBearing) -> SpeakerMotion:
        """Process a new bearing measurement and return motion state.

        Args:
            bearing: The current bearing measurement.

        Returns:
            SpeakerMotion with angular velocity and movement direction.
        """
        pass

    @abstractmethod
    def reset(self, track_id: int):
        """Reset history for a specific track.

        Args:
            track_id: The track ID to reset.
        """
        pass


class BearingMotionEstimator(SpeakerMotionEstimator):
    """Estimates speaker motion from bearing measurements.

    Computes angular velocity as:
        velocity = (current_angle - previous_angle) / delta_time

    Applies temporal smoothing (median filter) to reduce noise.
    Classifies movement using a configurable stationary threshold.

    Per-track state is maintained independently, supporting future
    multi-speaker pipelines.

    Config keys:
        history_length: Number of angular velocity samples for smoothing.
        stationary_threshold_degrees_per_second: Velocity below this → STATIONARY.
    """

    def __init__(self, config: dict):
        self.history_length: int = config.get("history_length", 5)
        self.stationary_threshold: float = config.get(
            "stationary_threshold_degrees_per_second", 5.0
        )

        # Per-track state
        self._prev_bearing: Dict[int, SpeakerBearing] = {}
        self._velocity_history: Dict[int, deque] = {}

    def update(self, bearing: SpeakerBearing) -> SpeakerMotion:
        """Process a new bearing measurement.

        Computes angular velocity from the difference between current
        and previous bearing, applies temporal smoothing, and classifies
        movement direction.

        Args:
            bearing: The current bearing measurement.

        Returns:
            SpeakerMotion with angular velocity and movement direction.
        """
        track_id = bearing.track_id

        # Initialize history for new tracks
        if track_id not in self._velocity_history:
            self._velocity_history[track_id] = deque(maxlen=self.history_length)

        # First observation for this track → UNKNOWN
        if track_id not in self._prev_bearing:
            self._prev_bearing[track_id] = bearing
            return SpeakerMotion(
                track_id=track_id,
                current_angle=bearing.angle_degrees,
                previous_angle=bearing.angle_degrees,
                angular_velocity=0.0,
                movement=MovementDirection.UNKNOWN,
                timestamp=bearing.timestamp,
                frame_number=bearing.frame_number,
            )

        # Get previous bearing
        prev = self._prev_bearing[track_id]

        # Calculate delta time
        delta_time = bearing.timestamp - prev.timestamp

        # Calculate raw angular velocity
        if delta_time <= 0:
            # Invalid delta time → return UNKNOWN
            return SpeakerMotion(
                track_id=track_id,
                current_angle=bearing.angle_degrees,
                previous_angle=prev.angle_degrees,
                angular_velocity=0.0,
                movement=MovementDirection.UNKNOWN,
                timestamp=bearing.timestamp,
                frame_number=bearing.frame_number,
            )

        raw_velocity = (bearing.angle_degrees - prev.angle_degrees) / delta_time

        # Add to velocity history
        self._velocity_history[track_id].append(raw_velocity)

        # Smooth using median (robust to outliers)
        smoothed_velocity = self._smooth_velocity(track_id)

        # Classify movement
        movement = self._classify_movement(smoothed_velocity)

        # Update previous bearing
        self._prev_bearing[track_id] = bearing

        return SpeakerMotion(
            track_id=track_id,
            current_angle=bearing.angle_degrees,
            previous_angle=prev.angle_degrees,
            angular_velocity=round(smoothed_velocity, 2),
            movement=movement,
            timestamp=bearing.timestamp,
            frame_number=bearing.frame_number,
        )

    def _smooth_velocity(self, track_id: int) -> float:
        """Apply median smoothing to angular velocity history.

        Median is more robust to occasional noisy measurements
        than a simple moving average.

        Args:
            track_id: The track ID to smooth.

        Returns:
            Smoothed angular velocity.
        """
        history = self._velocity_history[track_id]
        if not history:
            return 0.0
        return statistics.median(history)

    def _classify_movement(self, smoothed_velocity: float) -> MovementDirection:
        """Classify movement direction from smoothed angular velocity.

        Args:
            smoothed_velocity: Median-smoothed angular velocity in deg/s.

        Returns:
            MovementDirection enum value.
        """
        if smoothed_velocity > self.stationary_threshold:
            return MovementDirection.RIGHT
        elif smoothed_velocity < -self.stationary_threshold:
            return MovementDirection.LEFT
        else:
            return MovementDirection.STATIONARY

    def reset(self, track_id: int):
        """Reset history for a specific track.

        Args:
            track_id: The track ID to reset.
        """
        self._prev_bearing.pop(track_id, None)
        self._velocity_history.pop(track_id, None)