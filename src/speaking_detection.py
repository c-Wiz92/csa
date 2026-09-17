"""Visual speaking detection module.

Provides an abstract interface and a MAR-based implementation.
The MAR detector analyzes mouth aspect ratio over time to detect
speaking activity. This is a V1 baseline — it measures mouth geometry
changes, not actual speech.

Known limitations:
- False positives: yawning, chewing, laughing, smiling, exaggerated expressions
- False negatives: quiet/subtle speech, profile views, occlusion, poor lighting
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Optional, Deque

import numpy as np

from src.data_structures import SpeakingState, TrackedFace
from src.temporal_buffer import TemporalBuffer


class SpeakingDetector(ABC):
    """Abstract base class for visual speaking detectors.

    Implementations analyze face crops to determine if the person
    is speaking, using only visual information (no audio).
    """

    @abstractmethod
    def detect(
        self, face_crop: np.ndarray, tracked_face: TrackedFace
    ) -> SpeakingState:
        """Determine if the person in the face crop is speaking.

        Args:
            face_crop: Cropped image of the face region (BGR).
            tracked_face: The tracked face metadata (for accessing track_id, history, etc.).

        Returns:
            SpeakingState with boolean and confidence.
        """
        pass

    @abstractmethod
    def reset(self, track_id: int):
        """Reset internal state for a specific track (e.g., track lost).

        Args:
            track_id: The track ID to reset state for.
        """
        pass


class MARSpeakingDetector(SpeakingDetector):
    """MAR-based speaking detector with temporal analysis.

    Analyzes Mouth Aspect Ratio (MAR) over a temporal window to detect
    speaking activity. Uses hysteresis and debouncing to avoid flickering.

    State machine:
        UNKNOWN → SILENT: Default when insufficient data
        SILENT → SPEAKING: Requires sustained high MAR variation + transitions
        SPEAKING → SILENT: Requires sustained low MAR variation

    Config keys:
        mar_speaking_threshold: MAR value above which mouth is "open"
        mar_baseline_frames: Number of frames to establish baseline
        variation_threshold: Minimum MAR variation to consider "active"
        min_transitions: Minimum mouth open/close transitions for speaking
        speak_confirm_frames: Consecutive active frames to confirm SPEAKING
        silent_confirm_frames: Consecutive inactive frames to confirm SILENT
        analysis_window: Number of recent frames to analyze
    """

    def __init__(self, config: dict, temporal_buffer: TemporalBuffer):
        self._buffer = temporal_buffer

        # Thresholds
        self.mar_speaking_threshold: float = config.get("mar_speaking_threshold", 0.3)
        self.mar_baseline_frames: int = config.get("mar_baseline_frames", 10)
        self.variation_threshold: float = config.get("variation_threshold", 0.05)
        self.min_transitions: int = config.get("min_transitions", 2)
        self.speak_confirm_frames: int = config.get("speak_confirm_frames", 5)
        self.silent_confirm_frames: int = config.get("silent_confirm_frames", 8)
        self.analysis_window: int = config.get("analysis_window", 15)

        # Per-track state
        self._track_states: Dict[int, str] = {}  # track_id → "SPEAKING"/"SILENT"/"UNKNOWN"
        self._track_counters: Dict[str, Dict[int, int]] = {
            "speak_confirm": {},
            "silent_confirm": {},
        }
        self._track_baselines: Dict[int, Deque[float]] = {}  # track_id → recent MAR values

    def detect(
        self, face_crop: np.ndarray, tracked_face: TrackedFace
    ) -> SpeakingState:
        """Determine if the person is speaking using MAR temporal analysis.

        Args:
            face_crop: Cropped image of the face region (BGR).
            tracked_face: The tracked face metadata.

        Returns:
            SpeakingState with is_speaking boolean and confidence.
        """
        track_id = tracked_face.track_id

        # Initialize state for new tracks
        if track_id not in self._track_states:
            self._track_states[track_id] = "UNKNOWN"
            self._track_counters["speak_confirm"][track_id] = 0
            self._track_counters["silent_confirm"][track_id] = 0
            self._track_baselines[track_id] = deque(maxlen=self.mar_baseline_frames)

        # Get MAR history from temporal buffer
        history = self._buffer.get_history(track_id)

        if not history or len(history) < 3:
            # Not enough data yet
            self._track_states[track_id] = "UNKNOWN"
            return SpeakingState(is_speaking=False, confidence=0.0)

        # Analyze recent MAR values
        recent = history[-self.analysis_window:]
        mar_values = [obs.mar for obs in recent]

        # Compute statistics
        current_mar = mar_values[-1]
        mean_mar = np.mean(mar_values)
        std_mar = np.std(mar_values)
        mar_range = max(mar_values) - min(mar_values)

        # Update baseline
        self._track_baselines[track_id].append(current_mar)
        baseline_mar = np.mean(list(self._track_baselines[track_id]))

        # Count transitions (mouth opening/closing events)
        transitions = self._count_transitions(mar_values)

        # Determine if mouth is actively moving
        is_active = self._is_mouth_active(
            mar_values, mean_mar, std_mar, transitions
        )

        # State machine with hysteresis
        current_state = self._track_states[track_id]

        if current_state in ("UNKNOWN", "SILENT"):
            if is_active:
                self._track_counters["speak_confirm"][track_id] += 1
                self._track_counters["silent_confirm"][track_id] = 0

                if self._track_counters["speak_confirm"][track_id] >= self.speak_confirm_frames:
                    self._track_states[track_id] = "SPEAKING"
            else:
                self._track_counters["speak_confirm"][track_id] = 0
                self._track_counters["silent_confirm"][track_id] += 1

                if self._track_counters["silent_confirm"][track_id] >= self.silent_confirm_frames:
                    self._track_states[track_id] = "SILENT"

        elif current_state == "SPEAKING":
            if not is_active:
                self._track_counters["silent_confirm"][track_id] += 1
                self._track_counters["speak_confirm"][track_id] = 0

                if self._track_counters["silent_confirm"][track_id] >= self.silent_confirm_frames:
                    self._track_states[track_id] = "SILENT"
            else:
                self._track_counters["silent_confirm"][track_id] = 0
                self._track_counters["speak_confirm"][track_id] += 1

        # Build result
        state = self._track_states[track_id]
        is_speaking = state == "SPEAKING"

        # Confidence based on how clearly the pattern matches
        if is_speaking:
            # Higher confidence with more transitions and higher variation
            confidence = min(1.0, 0.5 + std_mar * 2 + transitions * 0.05)
        else:
            # Confidence in silence
            confidence = 0.0 if state == "UNKNOWN" else min(1.0, 0.5 + (self.variation_threshold - std_mar) * 5)

        # Store debug info on the face for visualization
        tracked_face.metadata["mar_debug"] = {
            "mar": round(current_mar, 4),
            "var": round(std_mar, 4),
            "mean": round(mean_mar, 4),
            "transitions": transitions,
            "state": state,
            "buffer_len": len(history),
        }

        return SpeakingState(is_speaking=is_speaking, confidence=round(confidence, 2))

    def _count_transitions(self, mar_values: list) -> int:
        """Count mouth opening/closing transitions.

        A transition occurs when MAR crosses the speaking threshold
        in either direction.

        Args:
            mar_values: List of MAR values over time.

        Returns:
            Number of threshold crossings.
        """
        if len(mar_values) < 2:
            return 0

        transitions = 0
        above = mar_values[0] > self.mar_speaking_threshold

        for i in range(1, len(mar_values)):
            currently_above = mar_values[i] > self.mar_speaking_threshold
            if currently_above != above:
                transitions += 1
                above = currently_above

        return transitions

    def _is_mouth_active(
        self,
        mar_values: list,
        mean_mar: float,
        std_mar: float,
        transitions: int,
    ) -> bool:
        """Determine if mouth movement is consistent with speaking.

        Args:
            mar_values: Recent MAR values.
            mean_mar: Mean MAR over the window.
            std_mar: Standard deviation of MAR.
            transitions: Number of open/close transitions.

        Returns:
            True if mouth activity suggests speaking.
        """
        # Check 1: MAR variation must be significant
        if std_mar < self.variation_threshold:
            return False

        # Check 2: Must have enough transitions (opening/closing)
        if transitions < self.min_transitions:
            return False

        # Check 3: At least some frames must show open mouth
        open_frames = sum(1 for m in mar_values if m > self.mar_speaking_threshold)
        if open_frames < 1:
            return False

        return True

    def reset(self, track_id: int):
        """Reset internal state for a specific track.

        Args:
            track_id: The track ID to reset state for.
        """
        self._track_states.pop(track_id, None)
        self._track_counters["speak_confirm"].pop(track_id, None)
        self._track_counters["silent_confirm"].pop(track_id, None)
        self._track_baselines.pop(track_id, None)


class PlaceholderSpeakingDetector(SpeakingDetector):
    """Placeholder that always returns silent.

    Used for scaffolding. Replace with MAR-based or temporal
    speaking detector when ready.
    """

    def __init__(self, config: dict):
        self._device = config.get("device", "cpu")

    def detect(
        self, face_crop: np.ndarray, tracked_face: TrackedFace
    ) -> SpeakingState:
        """Always returns silent. Replace with real implementation."""
        return SpeakingState(is_speaking=False, confidence=0.0)

    def reset(self, track_id: int):
        pass