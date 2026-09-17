"""Configuration loader.

Reads config.yaml and provides typed access to configuration values.
Falls back to defaults if the config file is missing or incomplete.
"""

import os
from typing import Any, Dict
from dataclasses import dataclass, field

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "device_id": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
    },
    "face_detection": {
        "enabled": True,
        "model_path": "models/yolov8n-face.pt",
        "confidence_threshold": 0.5,
        "device": "cuda",
        "img_size": 640,
    },
    "face_tracking": {
        "enabled": True,
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.6,
        "track_buffer": 30,
        "with_reid": False,
        "use_cmc": False,
        "frame_rate": 30,
    },
    "face_landmarks": {
        "enabled": True,
        "model_path": "models/face_landmarker.task",
        "num_faces": 5,
        "min_face_detection_confidence": 0.5,
        "min_face_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
    },
    "temporal_buffer": {
        "max_length": 30,
        "stale_frames": 15,
    },
    "speaking_detection": {
        "enabled": True,
        "mar_speaking_threshold": 0.3,
        "mar_baseline_frames": 10,
        "variation_threshold": 0.05,
        "min_transitions": 2,
        "speak_confirm_frames": 5,
        "silent_confirm_frames": 8,
        "analysis_window": 15,
    },
    "speaker_localization": {
        "enabled": True,
        "focal_length_pixels": 0,
        "horizontal_fov_degrees": 60.0,
        "center_deadband_degrees": 5.0,
    },
    "speaker_motion": {
        "enabled": True,
        "history_length": 5,
        "stationary_threshold_degrees_per_second": 5.0,
    },
    "alignment": {
        "enabled": True,
        "aligned_enter_threshold_degrees": 5.0,
        "aligned_exit_threshold_degrees": 8.0,
        "bearing_history_length": 5,
        "turn_confirm_frames": 3,
        "aligned_confirm_frames": 3,
    },
    "display": {
        "window_name": "Assistive Vision - Debug",
        "show_bounding_boxes": True,
        "show_track_ids": True,
        "show_speaking_status": True,
        "show_confidence": True,
        "speaking_color": [0, 255, 0],
        "silent_color": [0, 0, 255],
        "text_scale": 0.6,
        "text_thickness": 2,
    },
}


def _merge_defaults(defaults: Dict, override: Dict) -> Dict:
    """Recursively merge override into defaults."""
    result = defaults.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_defaults(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file, falling back to defaults.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Merged configuration dictionary.
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        return _merge_defaults(DEFAULT_CONFIG, user_config)
    return DEFAULT_CONFIG.copy()


def get_device(preferred: str = "cuda") -> str:
    """Return preferred device if available, else CPU fallback.

    Args:
        preferred: Preferred device string ("cuda" or "cpu").

    Returns:
        Device string safe to use with PyTorch.
    """
    if preferred == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    return "cpu"