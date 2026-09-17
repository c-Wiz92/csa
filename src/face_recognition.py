"""Face recognition module — EXTENSION POINT (NOT IMPLEMENTED).

This module is a placeholder for V2. It defines the interface that
a future face recognition implementation must follow.

When implemented, this module will:
- Take a face crop and tracked face metadata
- Return an identity label and confidence
- Operate independently of the detection/tracking/speaking pipeline

The pipeline is designed so this can be plugged in without modifying
any other module. The TrackedFace data structure already has
`identity` and `identity_confidence` fields ready to be populated.

DO NOT implement this in V1.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from src.data_structures import TrackedFace


class FaceRecognizer(ABC):
    """Abstract base class for face recognition.

    Future implementation will identify known individuals from face crops.
    This runs independently of the detection/tracking/speaking pipeline
    and populates the `identity` field on TrackedFace.
    """

    @abstractmethod
    def recognize(
        self, face_crop: np.ndarray, tracked_face: TrackedFace
    ) -> Tuple[Optional[str], Optional[float]]:
        """Identify a person from their face crop.

        Args:
            face_crop: Cropped image of the face region (BGR).
            tracked_face: The tracked face metadata.

        Returns:
            Tuple of (identity_label, confidence).
            identity_label is None if person is unknown.
        """
        pass

    @abstractmethod
    def register_identity(self, name: str, face_crops: list):
        """Register a new known identity.

        Args:
            name: Label for the person.
            face_crops: List of face crop images for enrollment.
        """
        pass
