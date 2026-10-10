"""Independent scene-awareness module using SmolVLM.

Runs in a background thread so the real-time speaker-localization
pipeline never waits for caption generation. Produces concise
descriptions of visible people / activity / surroundings for BVI users.

STRICT SEPARATION: this module never reads SpeakingState, never touches
MAR detection, speaker selection, bearing, motion, or alignment. Its
only consumer is the presentation layer (visualizer / future audio).
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SceneCaption:
    text: str
    timestamp: float = 0.0
    latency_ms: float = 0.0


PROMPT = (
    "Describe the visible people, their arrangement, and any obvious "
    "activity in one short sentence (under 20 words). "
    "Only state what is clearly visible. If unsure, say 'unclear'. "
    "Do not guess about speech."
)


class SmolVLMSceneDescriber:
    """Async SmolVLM captioner. Thread-safe. Never blocks the caller."""

    def __init__(self, config: dict):
        self.enabled: bool = config.get("enabled", True)
        self.model_id: str = config.get(
            "model_id", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        )
        self.period_s: float = config.get("period_seconds", 4.5)
        self.max_new_tokens: int = config.get("max_new_tokens", 60)
        self.image_size: int = config.get("image_size", 512)

        self._lock = threading.Lock()
        self._latest: Optional[SceneCaption] = None
        self._pending: Optional[np.ndarray] = None
        self._last_submit: float = 0.0
        self._busy: bool = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._model = None
        self._processor = None
        self._device = "cpu"
        self._load_error: Optional[str] = None

    def start(self):
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker, name="smolvlm-scene", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def submit(self, frame_bgr: np.ndarray):
        """Offer a frame for captioning. Drops it if busy or too soon."""
        if not self.enabled or self._load_error is not None:
            return
        now = time.time()
        if now - self._last_submit < self.period_s:
            return
        with self._lock:
            if self._busy or self._pending is not None:
                return
            try:
                import cv2

                h, w = frame_bgr.shape[:2]
                scale = self.image_size / max(h, w)
                if scale < 1.0:
                    small = cv2.resize(
                        frame_bgr,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    small = frame_bgr
                self._pending = small[:, :, ::-1].copy()
            except Exception:
                return
            self._last_submit = now

    def get_cached(self) -> Optional[SceneCaption]:
        with self._lock:
            return self._latest

    def caption_age_s(self) -> Optional[float]:
        cap = self.get_cached()
        if cap is None:
            return None
        return time.time() - cap.timestamp

    # -- background worker ----------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
            )

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            dtype = (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype=dtype
            )
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
                self._device = "cuda"
            self._model.eval()
            print(f"[scene] SmolVLM loaded on {self._device}: {self.model_id}")
            return True
        except Exception as e:
            self._load_error = str(e)
            print(f"[scene] WARNING: SmolVLM failed to load: {e}")
            print("[scene] Scene awareness disabled; localization unaffected.")
            return False

    def _worker(self):
        if not self._ensure_loaded():
            return
        import torch
        from PIL import Image

        while not self._stop_event.is_set():
            frame_rgb = None
            with self._lock:
                if self._pending is not None:
                    frame_rgb = self._pending
                    self._pending = None
                    self._busy = True
            if frame_rgb is None:
                self._stop_event.wait(0.2)
                continue
            try:
                image = Image.fromarray(frame_rgb)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ]
                inputs = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {
                    k: v.to(self._device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = self._model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens
                    )
                latency_ms = (time.perf_counter() - t0) * 1000
                text = self._processor.batch_decode(
                    out, skip_special_tokens=True
                )[0].strip()
                if "Assistant:" in text:
                    text = text.split("Assistant:")[-1].strip()
                with self._lock:
                    self._latest = SceneCaption(
                        text=text,
                        timestamp=time.time(),
                        latency_ms=round(latency_ms, 1),
                    )
                print(f"[scene] ({latency_ms:.0f}ms) {self._latest.text}")
            except Exception as e:
                print(f"[scene] WARNING: inference failed: {e}")
            finally:
                with self._lock:
                    self._busy = False
