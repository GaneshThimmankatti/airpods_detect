"""Thin wrappers around Apple's Vision framework, via PyObjC.

Vision runs on the Neural Engine / GPU, so per-frame detection cost on an
Apple Silicon MacBook Air is a couple of milliseconds and barely touches the
battery. Everything in here takes and returns plain numpy / tuples so the rest
of the project never has to know about Objective-C.

Pixel coordinates are (x, y, w, h) with the origin at the *top-left*, matching
OpenCV. Vision itself uses normalized bottom-left coordinates; the conversion
happens in `_to_pixels`.
"""

from __future__ import annotations

import numpy as np
import Quartz
import Vision
from Foundation import NSData


def rgb_to_cgimage(rgb: np.ndarray):
    """Wrap a contiguous HxWx3 uint8 RGB array as a CGImage (no re-encoding)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    data = NSData.dataWithBytes_length_(rgb.tobytes(), rgb.nbytes)
    provider = Quartz.CGDataProviderCreateWithCFData(data)
    return Quartz.CGImageCreate(
        w,
        h,
        8,
        24,
        w * 3,
        Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGBitmapByteOrderDefault | Quartz.kCGImageAlphaNone,
        provider,
        None,
        False,
        Quartz.kCGRenderingIntentDefault,
    )


def _to_pixels(bbox, w: int, h: int) -> tuple[int, int, int, int]:
    """Vision's normalized bottom-left CGRect -> top-left pixel (x, y, w, h)."""
    (nx, ny), (nw, nh) = bbox
    x = nx * w
    y = (1.0 - ny - nh) * h
    return int(round(x)), int(round(y)), int(round(nw * w)), int(round(nh * h))


def _perform(cgimage, requests) -> bool:
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cgimage, {}
    )
    ok, err = handler.performRequests_error_(requests, None)
    if not ok and err is not None:
        raise RuntimeError(f"Vision request failed: {err}")
    return bool(ok)


class FrameAnalyzer:
    """Runs person + face detection over a frame in a single Vision pass."""

    def __init__(self, min_person_confidence: float = 0.5):
        self.min_person_confidence = min_person_confidence
        self._people_req = Vision.VNDetectHumanRectanglesRequest.alloc().init()
        # Revision 2+ can return full-body boxes rather than upper-body only.
        if hasattr(self._people_req, "setUpperBodyOnly_"):
            try:
                self._people_req.setUpperBodyOnly_(False)
            except Exception:
                pass
        self._faces_req = Vision.VNDetectFaceRectanglesRequest.alloc().init()

    def analyze(self, rgb: np.ndarray) -> tuple[list, list]:
        """Return (people, faces); each item is ((x, y, w, h), confidence)."""
        h, w = rgb.shape[:2]
        cgimage = rgb_to_cgimage(rgb)
        _perform(cgimage, [self._people_req, self._faces_req])

        people = []
        for obs in self._people_req.results() or []:
            conf = float(obs.confidence())
            if conf < self.min_person_confidence:
                continue
            people.append((_to_pixels(obs.boundingBox(), w, h), conf))

        faces = []
        for obs in self._faces_req.results() or []:
            faces.append((_to_pixels(obs.boundingBox(), w, h), float(obs.confidence())))

        return people, faces


def detect_faces(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Face boxes only -- used by the enrollment script on still photos."""
    h, w = rgb.shape[:2]
    req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    _perform(rgb_to_cgimage(rgb), [req])
    return [_to_pixels(o.boundingBox(), w, h) for o in (req.results() or [])]


def image_feature_print(rgb: np.ndarray) -> np.ndarray | None:
    """Apple's general-purpose visual descriptor for an image crop.

    This is *not* a face-identity embedding -- it describes the whole crop,
    including hair, lighting and background. It is the zero-dependency
    fallback; `facenet` is the accurate path. See README.
    """
    req = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
    _perform(rgb_to_cgimage(rgb), [req])
    results = req.results() or []
    if not results:
        return None
    obs = results[0]
    raw = bytes(obs.data())
    # VNElementType: 1 = float32, 2 = float64.
    dtype = np.float64 if int(obs.elementType()) == 2 else np.float32
    return np.frombuffer(raw, dtype=dtype).astype(np.float32).copy()
