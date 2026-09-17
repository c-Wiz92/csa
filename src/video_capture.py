"""Video capture module.

Provides a clean interface for reading frames from a webcam.
Wraps OpenCV VideoCapture with context manager support and
consistent error handling.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np


class VideoCapture:
    """Wrapper around OpenCV VideoCapture for webcam input.

    Usage:
        config = {"device_id": 0, "width": 640, "height": 480, "fps": 30}
        with VideoCapture(config) as cap:
            ret, frame = cap.read()
    """

    def __init__(self, config: dict):
        self.device_id: int = config.get("device_id", 0)
        self.width: int = config.get("width", 640)
        self.height: int = config.get("height", 480)
        self.fps: int = config.get("fps", 30)
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: int = 0

    def start(self) -> bool:
        """Initialize and open the camera.

        Returns:
            True if camera opened successfully.
        """
        self._cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            # Try without DSHOW backend as fallback
            self._cap = cv2.VideoCapture(self.device_id)

        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera.

        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if ret:
            self._frame_count += 1
        return ret, frame

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def actual_resolution(self) -> Tuple[int, int]:
        """Return actual resolution the camera is delivering."""
        if self._cap is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (self.width, self.height)

    def release(self):
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False