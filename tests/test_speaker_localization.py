"""Unit tests for speaker localization and bearing estimation."""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_structures import BoundingBox, TrackedFace, SpeakingState
from src.speaker_localization import CameraBearingEstimator, select_active_speaker


def make_tracked_face(track_id, x1, x2, is_speaking=False):
    """Helper to create a TrackedFace with a bounding box."""
    return TrackedFace(
        track_id=track_id,
        bbox=BoundingBox(x1=x1, y1=100, x2=x2, y2=300),
        detection_confidence=0.9,
        speaking_state=SpeakingState(is_speaking=is_speaking, confidence=0.8),
    )


class TestCameraBearingEstimator:
    """Tests for CameraBearingEstimator."""

    def test_speaker_at_center(self):
        """Speaker exactly at image center should give ~0 degrees."""
        config = {"focal_length_pixels": 500, "horizontal_fov_degrees": 60}
        estimator = CameraBearingEstimator(config)

        # Frame width 640, face center at 320
        face = make_tracked_face(1, 270, 370)  # center_x = 320
        bearing = estimator.compute_bearing(face, frame_width=640)

        assert abs(bearing.angle_degrees) < 0.1, f"Expected ~0, got {bearing.angle_degrees}"
        assert bearing.track_id == 1
        assert bearing.center_x == 320.0

    def test_speaker_left_of_center(self):
        """Speaker left of center should give negative angle."""
        config = {"focal_length_pixels": 500, "horizontal_fov_degrees": 60}
        estimator = CameraBearingEstimator(config)

        # Frame width 640, face center at 160 (left quarter)
        face = make_tracked_face(1, 110, 210)  # center_x = 160
        bearing = estimator.compute_bearing(face, frame_width=640)

        assert bearing.angle_degrees < 0, f"Expected negative, got {bearing.angle_degrees}"
        assert bearing.direction == "LEFT"

    def test_speaker_right_of_center(self):
        """Speaker right of center should give positive angle."""
        config = {"focal_length_pixels": 500, "horizontal_fov_degrees": 60}
        estimator = CameraBearingEstimator(config)

        # Frame width 640, face center at 480 (right quarter)
        face = make_tracked_face(1, 430, 530)  # center_x = 480
        bearing = estimator.compute_bearing(face, frame_width=640)

        assert bearing.angle_degrees > 0, f"Expected positive, got {bearing.angle_degrees}"
        assert bearing.direction == "RIGHT"

    def test_symmetric_positions(self):
        """Left and right positions should give approximately symmetric angles."""
        config = {"focal_length_pixels": 500, "horizontal_fov_degrees": 60}
        estimator = CameraBearingEstimator(config)

        # Left: center_x = 160, Right: center_x = 480 (symmetric around 320)
        face_left = make_tracked_face(1, 110, 210)
        face_right = make_tracked_face(2, 430, 530)

        bearing_left = estimator.compute_bearing(face_left, frame_width=640)
        bearing_right = estimator.compute_bearing(face_right, frame_width=640)

        # Angles should be approximately symmetric
        assert abs(bearing_left.angle_degrees + bearing_right.angle_degrees) < 0.1, (
            f"Expected symmetric, got {bearing_left.angle_degrees} and {bearing_right.angle_degrees}"
        )

    def test_normalized_x_range(self):
        """Normalized x should be in [-1, 1]."""
        config = {"focal_length_pixels": 500, "horizontal_fov_degrees": 60}
        estimator = CameraBearingEstimator(config)

        # Far left
        face_left = make_tracked_face(1, 0, 100)
        bearing_left = estimator.compute_bearing(face_left, frame_width=640)
        assert -1.0 <= bearing_left.normalized_x <= 1.0

        # Far right
        face_right = make_tracked_face(2, 540, 640)
        bearing_right = estimator.compute_bearing(face_right, frame_width=640)
        assert -1.0 <= bearing_right.normalized_x <= 1.0

    def test_fov_fallback(self):
        """When focal_length_pixels is 0, should use FOV to estimate."""
        config = {"focal_length_pixels": 0, "horizontal_fov_degrees": 60}
        estimator = CameraBearingEstimator(config)

        face = make_tracked_face(1, 270, 370)  # center
        bearing = estimator.compute_bearing(face, frame_width=640)

        assert abs(bearing.angle_degrees) < 0.1

    def test_angle_magnitude_increases_with_displacement(self):
        """Larger displacement from center should give larger angle."""
        config = {"focal_length_pixels": 500, "horizontal_fov_degrees": 60}
        estimator = CameraBearingEstimator(config)

        face_center = make_tracked_face(1, 270, 370)  # center_x = 320
        face_mid = make_tracked_face(2, 170, 270)     # center_x = 220
        face_far = make_tracked_face(3, 70, 170)      # center_x = 120

        bearing_center = estimator.compute_bearing(face_center, frame_width=640)
        bearing_mid = estimator.compute_bearing(face_mid, frame_width=640)
        bearing_far = estimator.compute_bearing(face_far, frame_width=640)

        assert abs(bearing_far.angle_degrees) > abs(bearing_mid.angle_degrees)
        assert abs(bearing_mid.angle_degrees) > abs(bearing_center.angle_degrees)

    def test_known_angle(self):
        """Test with a known angle calculation."""
        config = {"focal_length_pixels": 500, "horizontal_fov_degrees": 60}
        estimator = CameraBearingEstimator(config)

        # displacement = 100 pixels, focal_length = 500
        # angle = atan(100/500) = atan(0.2) ≈ 11.31 degrees
        face = make_tracked_face(1, 270, 370)  # center_x = 320
        # Move to center_x = 420 (displacement = 100)
        face = make_tracked_face(1, 370, 470)  # center_x = 420
        bearing = estimator.compute_bearing(face, frame_width=640)

        expected_angle = math.degrees(math.atan(100 / 500))
        assert abs(bearing.angle_degrees - expected_angle) < 0.1, (
            f"Expected {expected_angle}, got {bearing.angle_degrees}"
        )


class TestSelectActiveSpeaker:
    """Tests for select_active_speaker function."""

    def test_single_speaker(self):
        """Exactly one speaking face should be selected."""
        faces = [
            make_tracked_face(1, 100, 200, is_speaking=False),
            make_tracked_face(2, 300, 400, is_speaking=True),
            make_tracked_face(3, 500, 600, is_speaking=False),
        ]
        result = select_active_speaker(faces)
        assert result is not None
        assert result.track_id == 2

    def test_no_speaker(self):
        """No speaking faces should return None."""
        faces = [
            make_tracked_face(1, 100, 200, is_speaking=False),
            make_tracked_face(2, 300, 400, is_speaking=False),
        ]
        result = select_active_speaker(faces)
        assert result is None

    def test_multiple_speakers_ambiguous(self):
        """Multiple speaking faces should return None (ambiguous)."""
        faces = [
            make_tracked_face(1, 100, 200, is_speaking=True),
            make_tracked_face(2, 300, 400, is_speaking=True),
        ]
        result = select_active_speaker(faces)
        assert result is None

    def test_empty_list(self):
        """Empty list should return None."""
        result = select_active_speaker([])
        assert result is None

    def test_single_speaking_face(self):
        """Single face that is speaking should be selected."""
        faces = [make_tracked_face(1, 100, 200, is_speaking=True)]
        result = select_active_speaker(faces)
        assert result is not None
        assert result.track_id == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])