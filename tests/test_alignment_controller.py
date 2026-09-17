"""Unit tests for the alignment controller."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_structures import (
    SpeakerBearing,
    SpeakerMotion,
    MovementDirection,
    AlignmentState,
)
from src.alignment_controller import BearingAlignmentController


def make_bearing(track_id, angle, frame_number=0):
    """Helper to create a SpeakerBearing."""
    return SpeakerBearing(
        track_id=track_id,
        angle_degrees=angle,
        center_x=320.0,
        normalized_x=0.0,
        frame_number=frame_number,
        timestamp=time.time() + frame_number * 0.033,
    )


def make_motion(angular_velocity):
    """Helper to create a SpeakerMotion."""
    return SpeakerMotion(
        track_id=1,
        current_angle=0.0,
        previous_angle=0.0,
        angular_velocity=angular_velocity,
        movement=MovementDirection.STATIONARY,
        timestamp=time.time(),
        frame_number=0,
    )


class TestBearingAlignmentController:
    """Tests for BearingAlignmentController."""

    def get_config(self, **overrides):
        """Get default config with optional overrides."""
        config = {
            "aligned_enter_threshold_degrees": 5.0,
            "aligned_exit_threshold_degrees": 8.0,
            "bearing_history_length": 5,
            "turn_confirm_frames": 3,
            "aligned_confirm_frames": 3,
        }
        config.update(overrides)
        return config

    def test_no_target(self):
        """No bearing should produce NO_TARGET."""
        controller = BearingAlignmentController(self.get_config())
        result = controller.update(None, None)
        assert result.state == AlignmentState.NO_TARGET
        assert result.command == "NO ACTIVE SPEAKER"

    def test_first_observation_unknown(self):
        """First observation should return UNKNOWN."""
        controller = BearingAlignmentController(self.get_config())
        bearing = make_bearing(1, 0.0, frame_number=0)
        result = controller.update(bearing, None)
        assert result.state == AlignmentState.UNKNOWN

    def test_target_centered_aligned(self):
        """Target directly centered should eventually become ALIGNED."""
        controller = BearingAlignmentController(self.get_config())
        # Send several centered readings
        for i in range(5):
            bearing = make_bearing(1, 0.0, frame_number=i)
            result = controller.update(bearing, None)
        assert result.state == AlignmentState.ALIGNED

    def test_target_left_turn_left(self):
        """Target clearly left should produce TURN_LEFT."""
        controller = BearingAlignmentController(self.get_config())
        # Send several left readings (-20 degrees)
        for i in range(5):
            bearing = make_bearing(1, -20.0, frame_number=i)
            result = controller.update(bearing, None)
        assert result.state == AlignmentState.TURN_LEFT

    def test_target_right_turn_right(self):
        """Target clearly right should produce TURN_RIGHT."""
        controller = BearingAlignmentController(self.get_config())
        for i in range(5):
            bearing = make_bearing(1, 20.0, frame_number=i)
            result = controller.update(bearing, None)
        assert result.state == AlignmentState.TURN_RIGHT

    def test_hysteresis_remain_aligned(self):
        """Should remain ALIGNED when bearing is between enter and exit thresholds."""
        controller = BearingAlignmentController(self.get_config())
        # First establish ALIGNED state
        for i in range(5):
            bearing = make_bearing(1, 0.0, frame_number=i)
            controller.update(bearing, None)
        # Now send bearings between enter (5) and exit (8) thresholds
        for i in range(5, 10):
            bearing = make_bearing(1, 6.0, frame_number=i)
            result = controller.update(bearing, None)
        assert result.state == AlignmentState.ALIGNED

    def test_hysteresis_leave_aligned(self):
        """Should leave ALIGNED when bearing exceeds exit threshold."""
        controller = BearingAlignmentController(self.get_config())
        # Establish ALIGNED
        for i in range(5):
            bearing = make_bearing(1, 0.0, frame_number=i)
            controller.update(bearing, None)
        # Now exceed exit threshold (8 degrees)
        for i in range(5, 10):
            bearing = make_bearing(1, 10.0, frame_number=i)
            result = controller.update(bearing, None)
        assert result.state == AlignmentState.TURN_RIGHT

    def test_hysteresis_enter_aligned(self):
        """Should enter ALIGNED when bearing drops to enter threshold."""
        controller = BearingAlignmentController(self.get_config())
        # Establish TURN_RIGHT
        for i in range(5):
            bearing = make_bearing(1, 20.0, frame_number=i)
            controller.update(bearing, None)
        # Now drop to within enter threshold
        for i in range(5, 10):
            bearing = make_bearing(1, 3.0, frame_number=i)
            result = controller.update(bearing, None)
        assert result.state == AlignmentState.ALIGNED

    def test_hysteresis_left_side(self):
        """Hysteresis should work symmetrically on the left side."""
        controller = BearingAlignmentController(self.get_config())
        # Establish ALIGNED
        for i in range(5):
            bearing = make_bearing(1, 0.0, frame_number=i)
            controller.update(bearing, None)
        # Exceed exit threshold on left
        for i in range(5, 10):
            bearing = make_bearing(1, -10.0, frame_number=i)
            result = controller.update(bearing, None)
        assert result.state == AlignmentState.TURN_LEFT

    def test_debounce_prevents_flicker(self):
        """Debouncing should prevent rapid state changes."""
        controller = BearingAlignmentController(self.get_config())
        # Establish ALIGNED
        for i in range(5):
            bearing = make_bearing(1, 0.0, frame_number=i)
            controller.update(bearing, None)
        # Send one frame above exit threshold — should NOT change yet
        bearing = make_bearing(1, 10.0, frame_number=5)
        result = controller.update(bearing, None)
        assert result.state == AlignmentState.ALIGNED

    def test_confirmation_required(self):
        """State change should require N consecutive frames."""
        controller = BearingAlignmentController(self.get_config(
            bearing_history_length=3,
            turn_confirm_frames=3,
            aligned_confirm_frames=3,
        ))
        # Establish ALIGNED
        for i in range(5):
            bearing = make_bearing(1, 0.0, frame_number=i)
            controller.update(bearing, None)
        # Send frames above exit threshold
        # With history_length=3: frame 5→median=0, frame 6→median=10 (desired TURN_RIGHT)
        # Need 3 confirmations: frame 6 (counter=1), frame 7 (counter=2), frame 8 (counter=3)
        for i in range(5, 9):
            bearing = make_bearing(1, 10.0, frame_number=i)
            result = controller.update(bearing, None)
        assert result.state == AlignmentState.TURN_RIGHT

    def test_noisy_bearing_smoothed(self):
        """Noisy bearing measurements should be smoothed by median."""
        controller = BearingAlignmentController(self.get_config(
            bearing_history_length=5,
            turn_confirm_frames=5,
        ))
        # Send noisy but overall centered readings
        angles = [0.5, -0.3, 0.8, -0.2, 0.4]
        for i, angle in enumerate(angles):
            bearing = make_bearing(1, angle, frame_number=i)
            result = controller.update(bearing, None)
        # Median should be close to 0 → ALIGNED
        assert result.state == AlignmentState.ALIGNED

    def test_track_change_resets_state(self):
        """Changing track ID should reset alignment state."""
        controller = BearingAlignmentController(self.get_config())
        # Establish TURN_RIGHT for track 1
        for i in range(5):
            bearing = make_bearing(1, 20.0, frame_number=i)
            controller.update(bearing, None)
        # Now track 2 appears with centered bearing
        result = controller.update(make_bearing(2, 0.0, frame_number=0), None)
        assert result.state == AlignmentState.UNKNOWN

    def test_no_history_leakage_between_tracks(self):
        """Track 2 should not inherit Track 1's alignment state."""
        controller = BearingAlignmentController(self.get_config())
        # Track 1: establish TURN_RIGHT
        for i in range(5):
            bearing = make_bearing(1, 20.0, frame_number=i)
            controller.update(bearing, None)
        # Track 2: first observation should be UNKNOWN
        result = controller.update(make_bearing(2, 0.0, frame_number=0), None)
        assert result.state == AlignmentState.UNKNOWN

    def test_speaker_disappears(self):
        """Speaker disappearing should produce NO_TARGET."""
        controller = BearingAlignmentController(self.get_config())
        # Establish some state
        for i in range(5):
            bearing = make_bearing(1, 20.0, frame_number=i)
            controller.update(bearing, None)
        # Speaker disappears
        result = controller.update(None, None)
        assert result.state == AlignmentState.NO_TARGET

    def test_speaker_reappears(self):
        """Speaker reappearing should start from UNKNOWN."""
        controller = BearingAlignmentController(self.get_config())
        # Speaker present
        for i in range(5):
            bearing = make_bearing(1, 20.0, frame_number=i)
            controller.update(bearing, None)
        # Speaker disappears
        controller.update(None, None)
        # Speaker reappears
        result = controller.update(make_bearing(1, 0.0, frame_number=10), None)
        assert result.state == AlignmentState.UNKNOWN

    def test_configurable_thresholds(self):
        """Different thresholds should produce different results."""
        # With enter=15, exit=20, bearing=12 should be ALIGNED (12 <= 15)
        controller = BearingAlignmentController(self.get_config(
            aligned_enter_threshold_degrees=15.0,
            aligned_exit_threshold_degrees=20.0,
            bearing_history_length=3,
            turn_confirm_frames=3,
            aligned_confirm_frames=3,
        ))
        for i in range(5):
            bearing = make_bearing(1, 12.0, frame_number=i)
            result = controller.update(bearing, None)
        assert result.state == AlignmentState.ALIGNED

    def test_configurable_confirmation_frames(self):
        """More confirmation frames should delay state change."""
        controller = BearingAlignmentController(self.get_config(
            turn_confirm_frames=5,
            bearing_history_length=3,
            aligned_confirm_frames=3,
        ))
        # Establish ALIGNED
        for i in range(5):
            bearing = make_bearing(1, 0.0, frame_number=i)
            controller.update(bearing, None)
        # With history_length=3: median becomes 10 at frame 6
        # Need 5 confirmations from frame 6: frames 6,7,8,9,10
        for i in range(5, 11):
            bearing = make_bearing(1, 10.0, frame_number=i)
            result = controller.update(bearing, None)
        assert result.state == AlignmentState.TURN_RIGHT

    def test_bearing_median_filter(self):
        """Median filter should smooth bearing history."""
        controller = BearingAlignmentController(self.get_config(
            bearing_history_length=5,
            turn_confirm_frames=5,
        ))
        # Send readings with one outlier
        angles = [0.0, 0.0, 50.0, 0.0, 0.0]  # outlier at index 2
        for i, angle in enumerate(angles):
            bearing = make_bearing(1, angle, frame_number=i)
            result = controller.update(bearing, None)
        # Median of [0, 0, 50, 0, 0] = 0 → ALIGNED
        assert result.state == AlignmentState.ALIGNED

    def test_command_string(self):
        """Command string should match state."""
        controller = BearingAlignmentController(self.get_config())
        # No target
        result = controller.update(None, None)
        assert "NO ACTIVE SPEAKER" in result.command

    def test_result_contains_motion_info(self):
        """Result should include angular velocity from motion data."""
        controller = BearingAlignmentController(self.get_config())
        bearing = make_bearing(1, 10.0, frame_number=0)
        motion = make_motion(15.0)
        result = controller.update(bearing, motion)
        assert result.angular_velocity == 15.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])