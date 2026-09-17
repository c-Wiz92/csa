"""Active speaker alignment controller.

Determines what the user should do based on the active speaker's bearing.

Architecture:
    AlignmentController (ABC) ← BearingAlignmentController

The controller receives bearing and motion information and produces
a semantic alignment command (TURN_LEFT, TURN_RIGHT, ALIGNED, NO_TARGET).

It does NOT directly call TTS, audio APIs, GPIO, or vibration motors.
A future FeedbackManager will translate commands into physical feedback.

Pipeline position:
    ... → Motion Estimation → Alignment Controller → [Future: Feedback Manager]
"""

import statistics
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, Dict

from src.data_structures import (
    SpeakerBearing,
    SpeakerMotion,
    AlignmentState,
    AlignmentResult,
)


class AlignmentController(ABC):
    """Abstract base class for alignment controllers.

    Implementations analyze bearing/motion data to determine
    the user's alignment state relative to the active speaker.
    """

    @abstractmethod
    def update(
        self,
        bearing: Optional[SpeakerBearing],
        motion: Optional[SpeakerMotion],
    ) -> AlignmentResult:
        """Process new bearing/motion data and return alignment decision.

        Args:
            bearing: Current bearing measurement, or None if no active speaker.
            motion: Current motion estimate, or None if unavailable.

        Returns:
            AlignmentResult with state and command.
        """
        pass

    @abstractmethod
    def reset(self, track_id: int):
        """Reset alignment state for a specific track.

        Args:
            track_id: The track ID to reset.
        """
        pass


class BearingAlignmentController(AlignmentController):
    """Alignment controller using bearing with hysteresis and debouncing.

    Pipeline:
        raw bearing
        → bearing median filter (noise reduction)
        → hysteresis (separate enter/exit thresholds)
        → state confirmation/debounce (temporal stability)
        → alignment state

    Hysteresis:
        - When TURNING: enter ALIGNED only when |bearing| <= enter_threshold
        - When ALIGNED: remain ALIGNED while |bearing| < exit_threshold
        - This prevents flickering around the alignment boundary

    Debouncing:
        - A state transition must persist for N consecutive frames
        - turn_confirm_frames: frames needed to confirm a turn command
        - aligned_confirm_frames: frames needed to confirm alignment

    Per-track state:
        - Each track_id has independent alignment history
        - Track changes reset alignment state safely

    Config keys:
        aligned_enter_threshold_degrees: Enter ALIGNED when |bearing| <= this
        aligned_exit_threshold_degrees: Leave ALIGNED when |bearing| >= this
        bearing_history_length: Median filter window for bearing smoothing
        turn_confirm_frames: Consecutive frames to confirm turn
        aligned_confirm_frames: Consecutive frames to confirm aligned
    """

    def __init__(self, config: dict):
        self.enter_threshold: float = config.get(
            "aligned_enter_threshold_degrees", 5.0
        )
        self.exit_threshold: float = config.get(
            "aligned_exit_threshold_degrees", 8.0
        )
        self.bearing_history_length: int = config.get(
            "bearing_history_length", 5
        )
        self.turn_confirm_frames: int = config.get(
            "turn_confirm_frames", 3
        )
        self.aligned_confirm_frames: int = config.get(
            "aligned_confirm_frames", 3
        )

        # Per-track state
        self._bearing_history: Dict[int, deque] = {}
        self._track_states: Dict[int, AlignmentState] = {}
        self._confirm_counters: Dict[int, int] = {}
        self._pending_states: Dict[int, AlignmentState] = {}

    def update(
        self,
        bearing: Optional[SpeakerBearing],
        motion: Optional[SpeakerMotion],
    ) -> AlignmentResult:
        """Process new bearing/motion and return alignment decision.

        Args:
            bearing: Current bearing measurement, or None.
            motion: Current motion estimate, or None.

        Returns:
            AlignmentResult with state and command.
        """
        # No active speaker — reset all track states
        if bearing is None:
            self._track_states.clear()
            self._confirm_counters.clear()
            self._pending_states.clear()
            return AlignmentResult(
                state=AlignmentState.NO_TARGET,
                track_id=-1,
                bearing_degrees=0.0,
                command="NO ACTIVE SPEAKER",
                timestamp=time.time(),
                frame_number=0,
            )

        track_id = bearing.track_id

        # Initialize per-track state
        if track_id not in self._bearing_history:
            self._bearing_history[track_id] = deque(
                maxlen=self.bearing_history_length
            )
            self._track_states[track_id] = AlignmentState.UNKNOWN
            self._confirm_counters[track_id] = 0
            self._pending_states[track_id] = AlignmentState.UNKNOWN

        # Check for track change — reset if track_id changed from previous
        # (This is handled naturally since we key by track_id)

        # Add bearing to history and compute smoothed bearing
        self._bearing_history[track_id].append(bearing.angle_degrees)
        smoothed_bearing = self._smooth_bearing(track_id)

        # Apply hysteresis to determine raw desired state
        desired_state = self._apply_hysteresis(
            track_id, smoothed_bearing
        )

        # Apply debouncing / temporal confirmation
        confirmed_state = self._apply_debounce(track_id, desired_state)

        # Update track state
        self._track_states[track_id] = confirmed_state

        # Build command string
        command = self._state_to_command(confirmed_state)

        # Get angular velocity if motion available
        angular_velocity = 0.0
        if motion is not None:
            angular_velocity = motion.angular_velocity

        # Confidence based on how far from threshold
        confidence = self._compute_confidence(confirmed_state, smoothed_bearing)

        return AlignmentResult(
            state=confirmed_state,
            track_id=track_id,
            bearing_degrees=round(smoothed_bearing, 2),
            command=command,
            timestamp=bearing.timestamp,
            frame_number=bearing.frame_number,
            angular_velocity=angular_velocity,
            confidence=round(confidence, 2),
        )

    def _smooth_bearing(self, track_id: int) -> float:
        """Apply median filter to bearing history.

        Args:
            track_id: The track ID.

        Returns:
            Median-smoothed bearing in degrees.
        """
        history = self._bearing_history[track_id]
        if not history:
            return 0.0
        return statistics.median(history)

    def _apply_hysteresis(
        self, track_id: int, smoothed_bearing: float
    ) -> AlignmentState:
        """Apply hysteresis to determine desired alignment state.

        Uses separate enter/exit thresholds to prevent flickering.

        Args:
            track_id: The track ID.
            smoothed_bearing: Median-smoothed bearing.

        Returns:
            Desired alignment state.
        """
        current_state = self._track_states.get(track_id, AlignmentState.UNKNOWN)
        abs_bearing = abs(smoothed_bearing)

        if current_state == AlignmentState.ALIGNED:
            # Currently aligned — only leave if bearing exceeds exit threshold
            if abs_bearing >= self.exit_threshold:
                if smoothed_bearing > 0:
                    return AlignmentState.TURN_RIGHT
                else:
                    return AlignmentState.TURN_LEFT
            return AlignmentState.ALIGNED

        elif current_state in (
            AlignmentState.TURN_LEFT,
            AlignmentState.TURN_RIGHT,
        ):
            # Currently turning — enter aligned only within enter threshold
            if abs_bearing <= self.enter_threshold:
                return AlignmentState.ALIGNED
            return current_state

        else:
            # UNKNOWN state — classify directly
            if abs_bearing <= self.enter_threshold:
                return AlignmentState.ALIGNED
            elif smoothed_bearing > 0:
                return AlignmentState.TURN_RIGHT
            else:
                return AlignmentState.TURN_LEFT

    def _apply_debounce(
        self, track_id: int, desired_state: AlignmentState
    ) -> AlignmentState:
        """Apply temporal debouncing to state transitions.

        A state transition must persist for N consecutive frames
        before becoming active.

        Args:
            track_id: The track ID.
            desired_state: The desired state from hysteresis.

        Returns:
            Confirmed alignment state.
        """
        current_state = self._track_states.get(track_id, AlignmentState.UNKNOWN)

        if desired_state == current_state:
            # No change — reset counter and return current
            self._confirm_counters[track_id] = 0
            self._pending_states[track_id] = desired_state
            return current_state

        # State change desired
        if self._pending_states.get(track_id) != desired_state:
            # New pending state — start counting
            self._pending_states[track_id] = desired_state
            self._confirm_counters[track_id] = 1
            return current_state

        # Same pending state — increment counter
        self._confirm_counters[track_id] += 1

        # Check if we've reached confirmation threshold
        required = self._get_confirm_threshold(desired_state)
        if self._confirm_counters[track_id] >= required:
            # Confirmed — transition
            return desired_state

        # Not yet confirmed — stay at current
        return current_state

    def _get_confirm_threshold(self, state: AlignmentState) -> int:
        """Get the confirmation frame count for a state.

        Args:
            state: The desired state.

        Returns:
            Number of consecutive frames required.
        """
        if state == AlignmentState.ALIGNED:
            return self.aligned_confirm_frames
        else:
            return self.turn_confirm_frames

    def _state_to_command(self, state: AlignmentState) -> str:
        """Convert alignment state to human-readable command.

        Args:
            state: The alignment state.

        Returns:
            Command string.
        """
        commands = {
            AlignmentState.ALIGNED: "ALIGNED — HOLD POSITION",
            AlignmentState.TURN_LEFT: "TURN LEFT",
            AlignmentState.TURN_RIGHT: "TURN RIGHT",
            AlignmentState.NO_TARGET: "NO ACTIVE SPEAKER",
            AlignmentState.UNKNOWN: "UNKNOWN",
        }
        return commands.get(state, "UNKNOWN")

    def _compute_confidence(
        self, state: AlignmentState, smoothed_bearing: float
    ) -> float:
        """Compute confidence in the alignment decision.

        Higher confidence when bearing is far from thresholds.

        Args:
            state: Current alignment state.
            smoothed_bearing: Smoothed bearing.

        Returns:
            Confidence value between 0 and 1.
        """
        if state == AlignmentState.ALIGNED:
            # Higher confidence when closer to center
            return max(0.0, 1.0 - abs(smoothed_bearing) / self.exit_threshold)
        elif state in (AlignmentState.TURN_LEFT, AlignmentState.TURN_RIGHT):
            # Higher confidence when further from center
            return min(1.0, abs(smoothed_bearing) / self.exit_threshold)
        return 0.0

    def reset(self, track_id: int):
        """Reset alignment state for a specific track.

        Args:
            track_id: The track ID to reset.
        """
        self._bearing_history.pop(track_id, None)
        self._track_states.pop(track_id, None)
        self._confirm_counters.pop(track_id, None)
        self._pending_states.pop(track_id, None)