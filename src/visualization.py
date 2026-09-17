"""Visualization and debug display module.

Renders detection, tracking, and speaking information on frames
for development and debugging purposes.
"""

from typing import List, Optional

import cv2
import numpy as np

from src.data_structures import TrackedFace, SpeakerBearing, SpeakerMotion, AlignmentResult


class Visualizer:
    """Renders pipeline results on frames for debugging."""

    def __init__(self, config: dict):
        self.window_name: str = config.get("window_name", "Assistive Vision - Debug")
        self.show_boxes: bool = config.get("show_bounding_boxes", True)
        self.show_ids: bool = config.get("show_track_ids", True)
        self.show_speaking: bool = config.get("show_speaking_status", True)
        self.show_confidence: bool = config.get("show_confidence", True)
        self.speaking_color: tuple = tuple(config.get("speaking_color", [0, 255, 0]))
        self.silent_color: tuple = tuple(config.get("silent_color", [0, 0, 255]))
        self.text_scale: float = config.get("text_scale", 0.6)
        self.text_thickness: int = config.get("text_thickness", 2)

    def draw(
        self,
        frame: np.ndarray,
        tracked_faces: List[TrackedFace],
        fps: float = 0.0,
        detector_device: str = "cpu",
        detection_count: int = 0,
        active_speaker_bearing: Optional[SpeakerBearing] = None,
        active_speaker_motion: Optional[SpeakerMotion] = None,
        active_speaker_alignment: Optional[AlignmentResult] = None,
    ) -> np.ndarray:
        """Draw all tracked face information on the frame.

        Args:
            frame: BGR image to draw on.
            tracked_faces: List of tracked faces to visualize.
            fps: Current FPS to display.
            detector_device: Device used for inference (e.g., "cuda", "cpu").
            detection_count: Number of raw detections before tracking.
            active_speaker_bearing: Bearing of the active speaker, if any.
            active_speaker_motion: Motion of the active speaker, if any.
            active_speaker_alignment: Alignment decision, if any.

        Returns:
            Annotated frame (modified in-place, also returned).
        """
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        for face in tracked_faces:
            color = (
                self.speaking_color
                if face.speaking_state.is_speaking
                else self.silent_color
            )

            # Bounding box
            if self.show_boxes:
                cv2.rectangle(
                    overlay,
                    (face.bbox.x1, face.bbox.y1),
                    (face.bbox.x2, face.bbox.y2),
                    color,
                    2,
                )

            # Build label lines
            labels = []
            if self.show_ids:
                labels.append(f"ID {face.track_id}")

            # Speaking status with MAR info
            if self.show_speaking:
                status = "SPEAKING" if face.speaking_state.is_speaking else "SILENT"
                labels.append(status)

            # MAR and variation from speaking detector
            mar_debug = face.metadata.get("mar_debug", {})
            if mar_debug:
                labels.append(f"MAR:{mar_debug.get('mar', 0):.2f}")
                labels.append(f"VAR:{mar_debug.get('var', 0):.2f}")
            else:
                labels.append("MAR:0.00")
                labels.append("VAR:0.00")

            if self.show_confidence:
                labels.append(f"CONF:{face.speaking_state.confidence:.2f}")

            # Draw labels above the bounding box
            y_offset = face.bbox.y1 - 10
            for label in reversed(labels):
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
                )
                # Background rectangle for readability
                cv2.rectangle(
                    overlay,
                    (face.bbox.x1, y_offset - text_h - 4),
                    (face.bbox.x1 + text_w + 4, y_offset + 4),
                    color,
                    -1,
                )
                cv2.putText(
                    overlay,
                    label,
                    (face.bbox.x1 + 2, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale,
                    (255, 255, 255),
                    self.text_thickness,
                )
                y_offset -= text_h + 8

            # Draw facial landmarks
            if face.landmarks is not None:
                h, w = overlay.shape[:2]
                lm = face.landmarks

                # Draw all landmarks as small dots
                for i, (lx, ly, lz) in enumerate(lm.landmarks):
                    px = int(lx * w)
                    py = int(ly * h)
                    # Mouth landmarks in green, others in cyan
                    if i in (
                        lm.MOUTH_LEFT, lm.MOUTH_RIGHT,
                        lm.UPPER_LIP_TOP, lm.LOWER_LIP_BOTTOM,
                        lm.UPPER_LIP_INNER, lm.LOWER_LIP_INNER,
                    ):
                        cv2.circle(overlay, (px, py), 3, (0, 255, 0), -1)
                    else:
                        cv2.circle(overlay, (px, py), 1, (255, 255, 0), -1)

                # Draw mouth region connections (simplified)
                mouth_indices = [
                    lm.MOUTH_LEFT, lm.UPPER_LIP_TOP,
                    lm.MOUTH_RIGHT, lm.LOWER_LIP_BOTTOM,
                ]
                for i in range(len(mouth_indices)):
                    x1, y1, _ = lm.landmarks[mouth_indices[i]]
                    x2, y2, _ = lm.landmarks[mouth_indices[(i + 1) % len(mouth_indices)]]
                    pt1 = (int(x1 * w), int(y1 * h))
                    pt2 = (int(x2 * w), int(y2 * h))
                    cv2.line(overlay, pt1, pt2, (0, 255, 255), 1)

        # FPS counter
        if fps > 0:
            fps_label = f"FPS: {fps:.1f}"
            cv2.putText(
                overlay,
                fps_label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # Face count
        count_label = f"Tracked: {len(tracked_faces)} | Detections: {detection_count}"
        cv2.putText(
            overlay,
            count_label,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Detector device info
        device_label = f"Detector: {detector_device.upper()}"
        cv2.putText(
            overlay,
            device_label,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Draw center line
        center_x = w // 2
        cv2.line(
            overlay,
            (center_x, 0),
            (center_x, h),
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )

        # Draw active speaker bearing HUD
        if active_speaker_bearing is not None:
            # Draw line from center to speaker
            speaker_cx = int(active_speaker_bearing.center_x)
            cv2.line(
                overlay,
                (center_x, h // 2),
                (speaker_cx, h // 2),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            # Arrow head
            arrow_dir = 1 if speaker_cx > center_x else -1
            cv2.arrowedLine(
                overlay,
                (center_x, h // 2),
                (speaker_cx, h // 2),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
                tipLength=0.3,
            )

            # HUD text
            hud_x = 10
            hud_y = 120
            hud_lines = [
                f"ACTIVE SPEAKER: ID {active_speaker_bearing.track_id}",
                f"BEARING: {active_speaker_bearing.angle_degrees:+.1f} deg",
                f"DIRECTION: {active_speaker_bearing.direction}",
            ]
            # Add motion info if available
            if active_speaker_motion is not None:
                hud_lines.append(f"MOVEMENT: {active_speaker_motion.movement.value}")
                hud_lines.append(f"ANGULAR VEL: {active_speaker_motion.angular_velocity:+.1f} deg/s")
            if active_speaker_alignment is not None:
                hud_lines.append(f"ALIGNMENT: {active_speaker_alignment.command}")
            for line in hud_lines:
                cv2.putText(
                    overlay,
                    line,
                    (hud_x, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                hud_y += 25
        else:
            cv2.putText(
                overlay,
                "ACTIVE SPEAKER: NONE",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 100),
                2,
            )

        return overlay

    def show(self, frame: np.ndarray):
        """Display the frame in a window."""
        cv2.imshow(self.window_name, frame)

    def should_exit(self) -> bool:
        """Check if the user pressed 'q' to quit."""
        key = cv2.waitKey(1) & 0xFF
        return key == ord("q")

    def cleanup(self):
        """Destroy the display window."""
        cv2.destroyAllWindows()