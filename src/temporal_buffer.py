"""Temporal buffer for visual speaking detection.

Maintains a fixed-length history of mouth observations per tracked face.
Each track_id gets its own independent buffer. Buffers are automatically
created for new tracks and removed when tracks disappear.

This module is independent of the speaking-detection algorithm — it simply
stores temporal data. Future detectors (MAR, LSTM, Transformer, etc.) can
query the buffer via get_history(track_id).
"""

import time
from collections import deque
from typing import Dict, List, Optional

from src.data_structures import TrackedFace, MouthObservation, FaceLandmarks


class TemporalBuffer:
    """Manages per-track mouth observation histories.

    Config keys:
        max_length: Maximum number of observations stored per track
        stale_frames: Remove track after this many frames without landmarks
    """

    def __init__(self, config: dict):
        self.max_length: int = config.get("max_length", 30)
        self.stale_frames: int = config.get("stale_frames", 15)

        # track_id → deque of MouthObservation
        self._buffers: Dict[int, deque] = {}
        # track_id → consecutive frames without landmarks
        self._missing_count: Dict[int, int] = {}
        # track_id → last frame number seen
        self._last_seen: Dict[int, int] = {}

    def update(self, tracked_faces: List[TrackedFace], frame_number: int) -> None:
        """Update buffers with current frame's tracked faces.

        For each tracked face:
        - If landmarks exist, create a MouthObservation and append
        - If landmarks are missing, increment missing counter
        - Reset missing counter when landmarks reappear

        Removes stale tracks that haven't been seen recently.

        Args:
            tracked_faces: List of tracked faces from the pipeline.
            frame_number: Current frame number.
        """
        active_ids = set()

        for face in tracked_faces:
            track_id = face.track_id
            active_ids.add(track_id)

            # Initialize buffer for new tracks
            if track_id not in self._buffers:
                self._buffers[track_id] = deque(maxlen=self.max_length)
                self._missing_count[track_id] = 0

            self._last_seen[track_id] = frame_number

            if face.landmarks is not None:
                # Reset missing counter
                self._missing_count[track_id] = 0

                # Create observation from landmarks
                observation = self._create_observation(
                    face.landmarks, frame_number
                )
                self._buffers[track_id].append(observation)
            else:
                # Increment missing counter
                self._missing_count[track_id] = (
                    self._missing_count.get(track_id, 0) + 1
                )

        # Remove stale tracks
        self._cleanup_stale(active_ids, frame_number)

    def _create_observation(
        self, landmarks: FaceLandmarks, frame_number: int
    ) -> MouthObservation:
        """Create a MouthObservation from facial landmarks.

        Extracts mouth-specific measurements and stores both the key
        mouth landmarks and the full landmark set.

        Args:
            landmarks: FaceLandmarks object with all 478 landmarks.
            frame_number: Current frame number.

        Returns:
            MouthObservation with mouth measurements.
        """
        lm = landmarks

        # Extract key mouth landmarks in pixel coordinates
        # Landmarks are normalized [0, 1], we store them as-is
        mouth_landmarks = {
            lm.MOUTH_LEFT: (lm.mouth_left[0], lm.mouth_left[1]),
            lm.MOUTH_RIGHT: (lm.mouth_right[0], lm.mouth_right[1]),
            lm.UPPER_LIP_TOP: (lm.upper_lip[0], lm.upper_lip[1]),
            lm.LOWER_LIP_BOTTOM: (lm.lower_lip[0], lm.lower_lip[1]),
            lm.UPPER_LIP_INNER: (
                lm.landmarks[lm.UPPER_LIP_INNER][0],
                lm.landmarks[lm.UPPER_LIP_INNER][1],
            ),
            lm.LOWER_LIP_INNER: (
                lm.landmarks[lm.LOWER_LIP_INNER][0],
                lm.landmarks[lm.LOWER_LIP_INNER][1],
            ),
        }

        # Calculate mouth width (distance between corners)
        dx = lm.mouth_right[0] - lm.mouth_left[0]
        dy = lm.mouth_right[1] - lm.mouth_left[1]
        mouth_width = (dx ** 2 + dy ** 2) ** 0.5

        # Calculate mouth height (distance between upper and lower lip)
        dx_h = lm.lower_lip[0] - lm.upper_lip[0]
        dy_h = lm.lower_lip[1] - lm.upper_lip[1]
        mouth_height = (dx_h ** 2 + dy_h ** 2) ** 0.5

        # MAR (Mouth Aspect Ratio)
        mar = mouth_height / mouth_width if mouth_width > 0 else 0.0

        return MouthObservation(
            frame_number=frame_number,
            timestamp=time.time(),
            mouth_landmarks=mouth_landmarks,
            mouth_width=mouth_width,
            mouth_height=mouth_height,
            mar=mar,
            all_landmarks=lm.landmarks,
        )

    def _cleanup_stale(
        self, active_ids: set, frame_number: int
    ) -> None:
        """Remove buffers for tracks that are no longer active.

        A track is removed if:
        - It's not in the current active set AND hasn't been seen for
          stale_frames consecutive frames

        Args:
            active_ids: Set of track IDs present in current frame.
            frame_number: Current frame number.
        """
        stale_ids = []
        for track_id in self._buffers:
            if track_id not in active_ids:
                last_frame = self._last_seen.get(track_id, 0)
                if frame_number - last_frame > self.stale_frames:
                    stale_ids.append(track_id)

        for track_id in stale_ids:
            del self._buffers[track_id]
            del self._missing_count[track_id]
            del self._last_seen[track_id]

    def get_history(self, track_id: int) -> List[MouthObservation]:
        """Get the temporal history for a specific track.

        Args:
            track_id: The track ID to query.

        Returns:
            List of MouthObservation objects, oldest first.
            Empty list if track not found.
        """
        if track_id in self._buffers:
            return list(self._buffers[track_id])
        return []

    def get_latest(self, track_id: int) -> Optional[MouthObservation]:
        """Get the most recent observation for a track.

        Args:
            track_id: The track ID to query.

        Returns:
            Most recent MouthObservation, or None if track not found.
        """
        if track_id in self._buffers and self._buffers[track_id]:
            return self._buffers[track_id][-1]
        return None

    def get_buffer_length(self, track_id: int) -> int:
        """Get the current number of observations for a track.

        Args:
            track_id: The track ID to query.

        Returns:
            Number of observations in the buffer (0 if not found).
        """
        if track_id in self._buffers:
            return len(self._buffers[track_id])
        return 0

    def get_active_tracks(self) -> List[int]:
        """Get all track IDs with active buffers.

        Returns:
            List of active track IDs.
        """
        return list(self._buffers.keys())

    def get_all_histories(self) -> Dict[int, List[MouthObservation]]:
        """Get all buffer histories.

        Returns:
            Dict mapping track_id → list of MouthObservation.
        """
        return {tid: list(buf) for tid, buf in self._buffers.items()}

    def reset(self) -> None:
        """Clear all buffers and state."""
        self._buffers.clear()
        self._missing_count.clear()
        self._last_seen.clear()