"""Pluggable face-embedding backends.

Two backends, selected by `config.detector.embedder`:

  facenet  InceptionResnetV1 pretrained on VGGFace2 (facenet-pytorch). A real
           face-identity embedding: robust to lighting, background and
           clothing. This is the one you want.
  vision   Apple's VNGenerateImageFeaturePrintRequest applied to the face
           crop. Zero extra dependencies and ANE-accelerated, but it is a
           generic image descriptor, not a face descriptor -- it will happily
           confuse two people with similar hair in similar lighting.

  auto     facenet if importable, otherwise vision.

Both backends return an L2-normalized float32 vector, so distances live on a
comparable [0, 2] scale and `match_threshold` means roughly the same thing
either way (though the defaults still differ -- see DEFAULT_THRESHOLD).
"""

from __future__ import annotations

import numpy as np


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n).astype(np.float32)


class VisionFeaturePrintEmbedder:
    name = "vision-featureprint"
    # Generic descriptor -> distances between different people are small, so
    # the usable threshold is tight. Expect to tune this one.
    DEFAULT_THRESHOLD = 0.55

    def __init__(self):
        from vision_mac import image_feature_print

        self._fn = image_feature_print

    def embed(self, face_rgb: np.ndarray) -> np.ndarray | None:
        vec = self._fn(face_rgb)
        return None if vec is None else _l2_normalize(vec)


class FacenetEmbedder:
    name = "facenet-vggface2"
    DEFAULT_THRESHOLD = 0.9

    def __init__(self):
        import torch
        from facenet_pytorch import InceptionResnetV1

        self._torch = torch
        # MPS gives no real win at batch size 1 and has had flaky kernels in
        # this model; CPU on Apple Silicon runs a 160x160 forward pass in a
        # few milliseconds, which is plenty.
        self._device = torch.device("cpu")
        self._model = (
            InceptionResnetV1(pretrained="vggface2").eval().to(self._device)
        )

    def embed(self, face_rgb: np.ndarray) -> np.ndarray | None:
        import cv2

        if face_rgb.size == 0:
            return None
        chip = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_AREA)
        x = chip.astype(np.float32)
        x = (x - 127.5) / 128.0  # facenet-pytorch's fixed_image_standardization
        t = self._torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            vec = self._model(t)[0].cpu().numpy()
        return _l2_normalize(vec)


def load_embedder(preference: str = "auto"):
    """Return an embedder instance for `preference` ('auto'|'facenet'|'vision')."""
    preference = (preference or "auto").lower()

    if preference in ("auto", "facenet"):
        try:
            return FacenetEmbedder()
        except Exception as exc:
            if preference == "facenet":
                raise RuntimeError(
                    "facenet backend requested but unavailable: "
                    f"{exc}\nInstall it with: ./install.sh --with-facenet"
                ) from exc
            print(f"[embedder] facenet unavailable ({exc}); falling back to Vision.")

    return VisionFeaturePrintEmbedder()


def crop_face(rgb: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.25):
    """Square, margin-padded crop around a face box, clamped to the frame."""
    h, w = rgb.shape[:2]
    x, y, bw, bh = box
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(bw, bh) * (1.0 + margin)
    x0 = int(max(0, round(cx - side / 2)))
    y0 = int(max(0, round(cy - side / 2)))
    x1 = int(min(w, round(cx + side / 2)))
    y1 = int(min(h, round(cy + side / 2)))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return rgb[y0:y1, x0:x1]
