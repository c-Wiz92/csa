"""Unit tests for speaker motion estimation."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_structures import SpeakerBearing, MovementDirection
from src.speaker_motion import BearingMotionEstimator


def make_bearing(track_id, angle, timestamp, frame_number=0, center_x=320):
    """Helper to create a SpeakerBearing."""
    return SpeakerBearing(
        track_id=track_id,
        angle_degrees=angle,
        center_x=center_x,
        normalized_x=0.0,
        frame_number=frame_number,
        timestamp=timestamp,
    )


class TestBearingMotionEstimator:
    """Tests for BearingMotionEstimator."""

    def test_first_observation_unknown(self):
        """First observation for a track should return UNKNOWN."""
        config = {"history_length": 5, "stationary_threshold_degrees_per_second": 5.0}
        estimator = BearingMotionEstimator(config)

        bearing = make_bearing(1, 10.0, time.time())
        motion = estimator.update(bearing)

        assert motion.movement == MovementDirection.UNKNOWN
        assert motion.angular_velocity == 0.0
        assert motion.track_id == 1

    def test_increasing_angle_right(self):
        """Increasing angle should classify as RIGHT."""
        config = {"history_length": 5, "stationary_threshold_degrees_per_second": 5.0}
        estimator = BearingMotionEstimator(config)

        t0 = time.time()
        # First observation
        bearing1 = make_bearing(1, 10.0, t0, frame_number=0)
        estimator.update(bearing1)

        # Second observation: angle increased by 5° in 0.1s = 50°/s
        bearing2 = make_bearing(1, 15.0, t0 + 0.1, frame_number=1)
        motion = estimator.update(bearing2)

        assert motion.movement == MovementDirection.RIGHT
        assert motion.angular_velocity > 0

    def test_decreasing_angle_left(self):
        """Decreasing angle should classify as LEFT."""
        config = {"history_length": 5, "stationary_threshold_degrees_per_second": 5.0}
        estimator = BearingMotionEstimator(config)

        t0 = time.time()
        bearing1 = make_bearing(1, 15.0, t0, frame_number=0)
        estimator.update(bearing1)

        # Angle decreased by 5° in 0.1s = -50°/s
        bearing2 = make_bearing(1, 10.0, t0 + 0.1, frame_number=1)
        motion = estimator.update(bearing2)

        assert motion.movement == MovementDirection.LEFT
        assert motion.angular_velocity < 0

    def test_constant_angle_stationary(self):
        """Nearly constant angle should classify as STATIONARY."""
        config = {"history_length": 5, "stationary_threshold_degrees_per_second": 5.0}
        estimator = BearingMotionEstimator(config)

        t0 = time.time()
        bearing1 = make_bearing(1, 10.0, t0, frame_number=0)
        estimator.update(bearing1)

        # Angle barely changed: 0.1° in 0.1s = 1°/s (below threshold)
        bearing2 = make_bearing(1, 10.1, t0 + 0.1, frame_number=1)
        motion = estimator.update(bearing2)

        assert motion.movement == MovementDirection.STATIONARY

    def test_angular_velocity_magnitude(self):
        """Angular velocity should be correctly calculated."""
        config = {"history_length": 5, "stationary_threshold_degrees_per_second": 5.0}
        estimator = BearingMotionEstimator(config)

        t0 = time.time()
        bearing1 = make_bearing(1, 10.0, t0, frame_number=0)
        estimator.update(bearing1)

        # 10° change in 0.2s = 50°/s
        bearing2 = make_bearing(1, 20.0, t0 + 0.2, frame_number=1)
        motion = estimator.update(bearing2)

        assert abs(motion.angular_velocity - 50.0) < 0.1

    def test_zero_delta_time_handling(self):
        """Zero delta time should return UNKNOWN safely."""
        config = {"history_length": 5, "stationary_threshold_degrees_per_second": 5.0}
        estimator = BearingMotionEstimator(config)

        t0 = time.time()
        bearing1 = make_bearing(1, 10.0, t0, frame_number=0)
        estimator.update(bearing1)

        # Same timestamp (zero delta)
        bearing2 = make_bearing(1, 15.0, t0, frame_number=1)
        motion = estimator.update(bearing2)

        assert motion.movement == MovementDirection.UNKNOWN
        assert motion.angular_velocity == 0.0

    def test_separate_track_histories(self):
        """Different tracks should maintain separate histories."""
        config = {"history_length": 5, "stationary_threshold_degrees_per_second": 5.0}
        estimator = BearingMotionEstimator(config)

        t0 = time.time()

        # Track 1: first observation
        bearing_t1_1 = make_bearing(1, 10.0, t0, frame_number=0)
        motion_t1_1 = estimator.update(bearing_t1_1)

        # Track 2: first observation
        bearing_t2_1 = make_bearing(2, -5.0, t0, frame_number=0)
        motion_t2_1 = estimator.update(bearing_t2_1)

        # Both should be UNKNOWN (first observations)
        assert motion_t1_1.movement == MovementDirection.UNKNOWN
        assert motion_t2_1.movement == MovementDirection.UNKNOWN

        # Track 1: moving right
        bearing_t1_2 = make_bearing(1, 20.0, t0 + 0.1, frame_number=1)
        motion_t1_2 = estimator.update(bearing_t1_2)

        # Track 2: still unknown (only one observation)
        # But we need to check track 2 doesn't leak from track 1
        bearing_t2_2 = make_bearing(2, -5.0, t0 + 0.1, frame_number=1)
        motion_t2_2 = estimator.update(bearing_t2_2)

        assert motion_t1_2.movement == MovementDirection.RIGHT
        assert motion_t2_2.movement == MovementDirection.STATIONARY  # barely moved

    def test_no_history_leakage_between_tracks(self):
        """Track 2 should not be affected by Track 1's history."""
        config = {"history_length": 5, "stationary_threshold_degrees_per_second": 5.0}
        estimator = BearingMotionEstimator(config)

        t0 = time.time()

        # Track 1: establish history with large movement
        estimator.update(make_bearing(1, 0.0, t0, frame_number=0))
        estimator.update(make_bearing(1, 30.0, t0 + 0.1, frame_number=1))

        # Track 2: first observation should be UNKNOWN regardless of track 1
        motion_t2 = estimator.update(make_bearing(2, 0.0, t0 + 0.1, frame_number=0))
        assert motion_t2.movement == MovementDirection.UNKNOWN

    def test_noisy_measurements_smoothed(self):
        """Noisy measurements should be smoothed by median filter."""
        config = {"history_length": 5, "stationary_threshold_degrees_per_second": 5.0}
        estimator = BearingMotionEstimator(config)

        t0 = time.time()
        # Establish baseline
        estimator.update(make_bearing(1, 10.0, t0, frame_number=0))

        # Add several measurements with noise but overall right trend
        angles = [12.0, 11.0, 14.0, 13.0, 16.0]  # noisy but increasing
        for i, angle in enumerate(angles):
            motion = estimator.update(
                make_bearing(1, angle, t0 + 0.1 * (i + 1), frame_number=i + 1)
            )

        # After enough samples, should detect RIGHT despite noise
        assert motion.movement == MovementDirection.RIGHT

    def test_history_bounded(self):
        """History should not exceed history_length."""
        config = {"history_length": 3, "stationary_threshold_degrees_per_second": 5.0}
        estimator = BearingMotionEstimator(config)

        t0 = time.time()
        # Add more observations than history_length
        for i in range(10):
            bearing = make_bearing(
                1, 10.0 + i * 0.01, t0 + 0.1 * i, frame_number=i
            )
            estimator.update(bearing)

        # History should be bounded (internal check)
        assert len(estimator._velocity_history[1]) <= 3

    def test_configurable_stationary_threshold(self):
        """Different thresholds should produce different classifications."""
        config_low = {"history_length": 5, "stationary_threshold_degrees_per_second": 1.0}
        config_high = {"history_length": 5, "stationary_threshold_degrees_per_second": 50.0}

        estimator_low = BearingMotionEstimator(config_low)
        estimator_high = BearingMotionEstimator(config_high)

        t0 = time.time()

        # Both start with same baseline
        estimator_low.update(make_bearing(1, 10.0, t0, frame_number=0))
        estimator_high.update(make_bearing(1, 10.0, t0, frame_number=0))

        # 3° change in 0.1s = 30°/s
        motion_low = estimator_low.update(make_bearing(1, 13.0, t0 + 0.1, frame_number=1))
        motion_high = estimator_high.update(make_bearing(1, 13.0, t0 + 0.1, frame_number=1))

        # Low threshold: 30°/s > 1°/s → RIGHT
        assert motion_low.movement == MovementDirection.RIGHT
        # High threshold: 30°/s < 50°/s → STATIONARY
        assert motion_high.movement == MovementDirection.STATIONARY

    def test_track_reset(self):
        """Reset should clear track history."""
        config = {"history_length": 5, "stationary_threshold_degrees_per_second": 5.0}
        estimator = BearingMotionEstimator(config)

        t0 = time.time()
        # Build history
        estimator.update(make_bearing(1, 10.0, t0, frame_number=0))
        estimator.update(make_bearing(1, 20.0, t0 + 0.1, frame_number=1))

        # Reset track 1
        estimator.reset(1)

        # Next observation should be UNKNOWN (no previous)
        motion = estimator.update(make_bearing(1, 30.0, t0 + 0.2, frame_number=2))
        assert motion.movement == MovementDirection.UNKNOWN

    def test_track_reappears_safely(self):
        """Track that disappears and reappears should be handled safely."""
        config = {"history_length": 5, "stationary_threshold_degrees_per_second": 5.0}
        estimator = BearingMotionEstimator(config)

        t0 = time.time()

        # Track 1 appears
        motion1 = estimator.update(make_bearing(1, 10.0, t0, frame_number=0))
        assert motion1.movement == MovementDirection.UNKNOWN

        # Track 1 continues
        motion2 = estimator.update(make_bearing(1, 15.0, t0 + 0.1, frame_number=1))
        assert motion2.movement == MovementDirection.RIGHT

        # Track 1 disappears (reset)
        estimator.reset(1)

        # Track 1 reappears
        motion3 = estimator.update(make_bearing(1, 20.0, t0 + 0.5, frame_number=2))
        assert motion3.movement == MovementDirection.UNKNOWN

    def test_speaker_stops_moving(self):
        """Speaker that was moving then stops should transition to STATIONARY."""
        config = {"history_length": 5, "stationary_threshold_degrees_per_second": 5.0}
        estimator = BearingMotionEstimator(config)

        t0 = time.time()

        # Start
        estimator.update(make_bearing(1, 10.0, t0, frame_number=0))

        # Moving right
        estimator.update(make_bearing(1, 20.0, t0 + 0.1, frame_number=1))
        motion = estimator.update(make_bearing(1, 30.0, t0 + 0.2, frame_number=2))
        assert motion.movement == MovementDirection.RIGHT

        # Now stationary (small changes) — need enough samples to flush median history
        for i in range(5):
            motion = estimator.update(
                make_bearing(1, 30.0 + (i + 1) * 0.1, t0 + 0.3 + (i + 1) * 0.1, frame_number=3 + i)
            )
        assert motion.movement == MovementDirection.STATIONARY


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])