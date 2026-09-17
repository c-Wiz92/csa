"""Pipeline orchestrator.

Coordinates video capture, face detection, face tracking,
speaking detection, and visualization into a single real-time loop.
"""

import time
from typing import Optional, List

import numpy as np

from src.config import load_config, get_device
from src.data_structures import (
    FrameResult,
    TrackedFace,
    Detection,
    SpeakingState,
    BoundingBox,
)
from src.video_capture import VideoCapture
from src.face_detection import FaceDetector, YOLOv8FaceDetector, PlaceholderFaceDetector
from src.face_tracking import FaceTracker, BotSORTFaceTracker, PlaceholderFaceTracker
from src.face_landmarks import FaceLandmarker, MediaPipeFaceLandmarker, PlaceholderFaceLandmarker
from src.temporal_buffer import TemporalBuffer
from src.speaking_detection import SpeakingDetector, MARSpeakingDetector, PlaceholderSpeakingDetector
from src.speaker_localization import SpeakerLocalizer, CameraBearingEstimator, select_active_speaker
from src.speaker_motion import SpeakerMotionEstimator, BearingMotionEstimator
from src.alignment_controller import AlignmentController, BearingAlignmentController
from src.visualization import Visualizer


class Pipeline:
    """Main processing pipeline.

    Wires together all modules and runs the real-time loop.
    Each component can be swapped by passing a different implementation.
    """

    def __init__(
        self,
        config: dict,
        face_detector: Optional[FaceDetector] = None,
        face_tracker: Optional[FaceTracker] = None,
        speaking_detector: Optional[SpeakingDetector] = None,
    ):
        self.config = config

        # Camera
        self.camera_config = config.get("camera", {})
        self.capture = VideoCapture(self.camera_config)

        # Face detection
        det_config = config.get("face_detection", {})
        if face_detector is not None:
            self.detector = face_detector
        elif det_config.get("enabled", True):
            self.detector = YOLOv8FaceDetector(det_config)
        else:
            self.detector = PlaceholderFaceDetector(det_config)

        # Face tracking
        track_config = config.get("face_tracking", {})
        if face_tracker is not None:
            self.tracker = face_tracker
        elif track_config.get("enabled", True):
            self.tracker = BotSORTFaceTracker(track_config)
        else:
            self.tracker = PlaceholderFaceTracker(track_config)

        # Facial landmarks
        landmark_config = config.get("face_landmarks", {})
        if landmark_config.get("enabled", True):
            self.landmarker = MediaPipeFaceLandmarker(landmark_config)
        else:
            self.landmarker = PlaceholderFaceLandmarker(landmark_config)

        # Temporal buffer (for speaking detection)
        temporal_config = config.get("temporal_buffer", {})
        self.temporal_buffer = TemporalBuffer(temporal_config)

        # Speaking detection
        speak_config = config.get("speaking_detection", {})
        if speaking_detector is not None:
            self.speaking_detector = speaking_detector
        elif speak_config.get("enabled", True):
            self.speaking_detector = MARSpeakingDetector(speak_config, self.temporal_buffer)
        else:
            self.speaking_detector = PlaceholderSpeakingDetector(speak_config)

        # Speaker localization (bearing estimation)
        loc_config = config.get("speaker_localization", {})
        if loc_config.get("enabled", True):
            self.localizer = CameraBearingEstimator(loc_config)
        else:
            self.localizer = None

        # Speaker motion estimation
        motion_config = config.get("speaker_motion", {})
        if motion_config.get("enabled", True):
            self.motion_estimator = BearingMotionEstimator(motion_config)
        else:
            self.motion_estimator = None

        # Alignment controller
        alignment_config = config.get("alignment", {})
        if alignment_config.get("enabled", True):
            self.alignment_controller = BearingAlignmentController(alignment_config)
        else:
            self.alignment_controller = None

        # Visualization
        disp_config = config.get("display", {})
        self.visualizer = Visualizer(disp_config)

        # State
        self._running = False
        self._fps = 0.0
        self._frame_times: list = []

    def _extract_face_crop(
        self, frame: np.ndarray, bbox: BoundingBox
    ) -> Optional[np.ndarray]:
        """Safely extract a face crop from the frame."""
        h, w = frame.shape[:2]
        x1 = max(0, bbox.x1)
        y1 = max(0, bbox.y1)
        x2 = min(w, bbox.x2)
        y2 = min(h, bbox.y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _update_fps(self, elapsed: float):
        """Update FPS calculation with exponential moving average."""
        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            if self._fps == 0.0:
                self._fps = instant_fps
            else:
                self._fps = 0.9 * self._fps + 0.1 * instant_fps

    def _process_frame(self, frame: np.ndarray, frame_number: int) -> FrameResult:
        """Run the full pipeline on a single frame."""
        start_time = time.perf_counter()

        # 1. Face detection
        detections: List[Detection] = self.detector.detect(frame)

        # 2. Face tracking
        tracked_faces: List[TrackedFace] = self.tracker.update(detections, frame)

        # 3. Facial landmarks (per tracked face)
        self.landmarker.detect_landmarks(frame, tracked_faces)

        # 4. Update temporal buffer with current landmarks
        self.temporal_buffer.update(tracked_faces, frame_number)

        # 5. Speaking detection (per tracked face)
        for face in tracked_faces:
            face.face_crop = self._extract_face_crop(frame, face.bbox)
            if face.face_crop is not None:
                face.speaking_state = self.speaking_detector.detect(
                    face.face_crop, face
                )

        # 6. Active speaker bearing estimation
        active_speaker_bearing = None
        if self.localizer is not None:
            active_speaker = select_active_speaker(tracked_faces)
            if active_speaker is not None:
                frame_width = frame.shape[1]
                active_speaker_bearing = self.localizer.compute_bearing(
                    active_speaker, frame_width, frame_number
                )

        # 7. Speaker motion estimation
        active_speaker_motion = None
        if self.motion_estimator is not None and active_speaker_bearing is not None:
            active_speaker_motion = self.motion_estimator.update(active_speaker_bearing)

        # 8. Alignment controller
        active_speaker_alignment = None
        if self.alignment_controller is not None:
            active_speaker_alignment = self.alignment_controller.update(
                active_speaker_bearing, active_speaker_motion
            )

        elapsed = time.perf_counter() - start_time
        self._update_fps(elapsed)

        return FrameResult(
            frame=frame,
            frame_number=frame_number,
            timestamp=time.time(),
            tracked_faces=tracked_faces,
            detection_count=len(detections),
            processing_time_ms=elapsed * 1000,
            active_speaker_bearing=active_speaker_bearing,
            active_speaker_motion=active_speaker_motion,
            active_speaker_alignment=active_speaker_alignment,
        )

    def run(self):
        """Run the main pipeline loop.

        Captures frames, processes them, and displays results.
        Press 'q' to quit.
        """
        if not self.capture.start():
            print("ERROR: Could not open camera.")
            return

        actual_w, actual_h = self.capture.actual_resolution
        print(f"Camera opened: {actual_w}x{actual_h}")
        print(f"Face detector: {self.detector.__class__.__name__} on {self.detector.device}")
        print(f"Face tracker: {self.tracker.__class__.__name__}")
        print(f"Face landmarker: {self.landmarker.__class__.__name__}")
        print(f"Temporal buffer: max_length={self.temporal_buffer.max_length}")
        print(f"Speaking detector: {self.speaking_detector.__class__.__name__}")
        if self.localizer is not None:
            print(f"Speaker localizer: {self.localizer.__class__.__name__}")
        else:
            print("Speaker localizer: disabled")
        if self.motion_estimator is not None:
            print(f"Motion estimator: {self.motion_estimator.__class__.__name__}")
        else:
            print("Motion estimator: disabled")
        if self.alignment_controller is not None:
            print(f"Alignment controller: {self.alignment_controller.__class__.__name__}")
        else:
            print("Alignment controller: disabled")
        print("Press 'q' to quit.\n")

        self._running = True

        try:
            while self._running:
                ret, frame = self.capture.read()
                if not ret or frame is None:
                    print("WARNING: Failed to read frame.")
                    continue

                # Process
                result = self._process_frame(frame, self.capture.frame_count)

                # Visualize
                annotated = self.visualizer.draw(
                    result.frame,
                    result.tracked_faces,
                    self._fps,
                    detector_device=self.detector.device,
                    detection_count=result.detection_count,
                    active_speaker_bearing=result.active_speaker_bearing,
                    active_speaker_motion=result.active_speaker_motion,
                    active_speaker_alignment=result.active_speaker_alignment,
                )
                self.visualizer.show(annotated)

                # Check for quit
                if self.visualizer.should_exit():
                    self._running = False

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.stop()

    def stop(self):
        """Stop the pipeline and release resources."""
        self._running = False
        self.capture.release()
        self.visualizer.cleanup()
        print("Pipeline stopped.")