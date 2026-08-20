"""Live webcam detector: person boxes, face recognition, and the trigger.

    ./run.sh detect              # normal run
    ./run.sh detect --selftest   # open the camera briefly, no recognition
    ./run.sh detect --no-window  # headless (LaunchAgent use)

Per frame:
  1. Vision finds people and faces in one pass.
  2. Each face is matched against the enrolled embeddings.
  3. A person box is labelled "Birthday girl" when a matching face sits inside
     it; otherwise it stays a plain "Person" box.
  4. Frames with at least one match increment a counter; any frame without one
     resets it to zero. At `consecutive_frames` in a row, the video plays and a
     cooldown starts.

Press q or Esc to quit the window.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, log, resolve  # noqa: E402
from embedder import crop_face, load_embedder  # noqa: E402
from vision_mac import FrameAnalyzer  # noqa: E402

MATCH_COLOR = (80, 220, 80)  # BGR
PLAIN_COLOR = (170, 170, 170)
ALERT_COLOR = (60, 200, 255)


def center_in(box, outer) -> bool:
    x, y, w, h = box
    ox, oy, ow, oh = outer
    cx, cy = x + w / 2.0, y + h / 2.0
    return ox <= cx <= ox + ow and oy <= cy <= oy + oh


def play_video(path: Path, fullscreen: bool) -> None:
    """Hand the file to QuickTime and start playback."""
    if not path.exists():
        log("trigger", f"video not found: {path} -- nothing to play")
        return
    log("trigger", f"playing {path}")
    try:
        subprocess.run(["open", "-a", "QuickTime Player", str(path)], check=True)
        script = 'tell application "QuickTime Player"\n activate\n'
        if fullscreen:
            script += " tell document 1 to present\n"
        script += " tell document 1 to play\nend tell"
        # QuickTime needs a beat to open the document before it can be driven.
        time.sleep(1.5)
        subprocess.run(["osascript", "-e", script], check=False)
    except Exception as exc:
        log("trigger", f"QuickTime failed ({exc}); falling back to `open`")
        subprocess.run(["open", str(path)], check=False)


def draw_box(frame, box, color, label: str | None) -> None:
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if not label:
        return
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(0, y - th - base - 4)
    cv2.rectangle(frame, (x, ty), (x + tw + 8, ty + th + base + 4), color, -1)
    cv2.putText(
        frame, label, (x + 4, ty + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA,
    )


class Recognizer:
    def __init__(self, cfg, embedder_pref: str):
        self.embedder = load_embedder(embedder_pref)
        path = resolve(cfg["enrollment"]["embeddings_path"])
        if not path.exists():
            raise SystemExit(
                f"no enrolled embeddings at {path}\n"
                "Run `./run.sh enroll` first (see README)."
            )
        data = np.load(path, allow_pickle=False)
        self.bank = data["embeddings"].astype(np.float32)
        self.label = str(data["label"])
        saved_backend = str(data["backend"])
        if saved_backend != self.embedder.name:
            raise SystemExit(
                f"embeddings were enrolled with '{saved_backend}' but the "
                f"detector loaded '{self.embedder.name}'. Embeddings are not "
                "comparable across backends -- re-run enrollment, or pin "
                "detector.embedder in config.json."
            )
        thr = cfg["detector"]["match_threshold"]
        self.threshold = (
            float(thr) if thr is not None else float(self.embedder.DEFAULT_THRESHOLD)
        )
        log("detect", f"backend {self.embedder.name}, {len(self.bank)} refs, "
                      f"threshold {self.threshold:.2f}")

    def distance(self, face_rgb) -> float | None:
        vec = self.embedder.embed(face_rgb)
        if vec is None:
            return None
        return float(np.min(np.linalg.norm(self.bank - vec, axis=1)))


def selftest(camera_index: int) -> int:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log("selftest", "could not open camera -- grant Camera permission "
                        "to your terminal in System Settings > Privacy & Security.")
        return 1
    ok, frame = cap.read()
    cap.release()
    if not ok:
        log("selftest", "camera opened but returned no frame.")
        return 1
    log("selftest", f"camera OK, frame {frame.shape[1]}x{frame.shape[0]}")
    return 0


def main() -> int:
    cfg = load_config()
    d = cfg["detector"]
    ap = argparse.ArgumentParser(description="Webcam birthday-girl detector.")
    ap.add_argument("--camera", type=int, default=d["camera_index"])
    ap.add_argument("--consec", type=int, default=d["consecutive_frames"])
    ap.add_argument("--cooldown", type=float, default=d["cooldown_seconds"])
    ap.add_argument("--embedder", default=d["embedder"])
    ap.add_argument("--video", default=cfg["video"]["path"])
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preview", action="store_true",
                    help="show person/face boxes only, no recognition -- "
                         "works before anyone is enrolled")
    ap.add_argument("--seconds", type=float, default=None,
                    help="auto-quit after this many seconds (for scripted runs)")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.camera)

    recognizer = None if args.preview else Recognizer(cfg, args.embedder)
    analyzer = FrameAnalyzer(min_person_confidence=d["min_person_confidence"])
    video_path = resolve(args.video)
    show = d["show_window"] and not args.no_window
    mirror = d["mirror"]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        log("detect", "could not open camera -- see README on Camera permission.")
        return 1

    streak = 0
    last_trigger = 0.0
    start = time.monotonic()
    window = "Birthday Surprise" + (" (preview)" if args.preview else "")
    if args.preview:
        log("detect", "preview mode: person/face boxes only, no recognition.")
    else:
        log("detect", f"running; need {args.consec} consecutive matching frames.")

    try:
        while True:
            if args.seconds is not None and (time.monotonic() - start) > args.seconds:
                log("detect", f"--seconds elapsed, stopping.")
                break
            ok, frame = cap.read()
            if not ok:
                log("detect", "camera read failed; stopping.")
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            people, faces = analyzer.analyze(rgb)

            if args.preview:
                for pbox, conf in people:
                    draw_box(frame, pbox, PLAIN_COLOR, f"Person {conf:.2f}")
                for fbox, conf in faces:
                    draw_box(frame, fbox, ALERT_COLOR, f"Face {conf:.2f}")
                if show:
                    cv2.putText(frame, f"people={len(people)} faces={len(faces)}",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                ALERT_COLOR, 2, cv2.LINE_AA)
                    cv2.imshow(window, frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                continue

            # Score every face once, then attach results to person boxes.
            scored = []
            for fbox, _conf in faces:
                chip = crop_face(rgb, fbox)
                dist = recognizer.distance(chip) if chip is not None else None
                scored.append((fbox, dist))

            matched_frame = False
            claimed = set()

            for pbox, conf in people:
                best = None
                for i, (fbox, dist) in enumerate(scored):
                    if i in claimed or dist is None:
                        continue
                    if center_in(fbox, pbox) and (best is None or dist < best[1]):
                        best = (i, dist)
                if best is None:
                    draw_box(frame, pbox, PLAIN_COLOR, f"Person {conf:.2f}")
                    continue
                idx, dist = best
                claimed.add(idx)
                if dist <= recognizer.threshold:
                    matched_frame = True
                    draw_box(frame, pbox, MATCH_COLOR,
                             f"{recognizer.label}  d={dist:.2f}")
                else:
                    draw_box(frame, pbox, PLAIN_COLOR, f"Person  d={dist:.2f}")

            # A face with no enclosing person box still counts -- Vision often
            # drops the body box when someone leans close to the camera, and
            # silently failing at exactly that moment would be the worst case.
            for i, (fbox, dist) in enumerate(scored):
                if i in claimed or dist is None:
                    continue
                if dist <= recognizer.threshold:
                    matched_frame = True
                    draw_box(frame, fbox, MATCH_COLOR,
                             f"{recognizer.label}  d={dist:.2f}")
                else:
                    draw_box(frame, fbox, PLAIN_COLOR, f"Face  d={dist:.2f}")

            streak = streak + 1 if matched_frame else 0

            now = time.monotonic()
            if streak >= args.consec and (now - last_trigger) > args.cooldown:
                last_trigger = now
                streak = 0
                play_video(video_path, cfg["video"]["fullscreen"])

            if show:
                hud = f"streak {streak}/{args.consec}"
                cool = args.cooldown - (time.monotonic() - last_trigger)
                if last_trigger and cool > 0:
                    hud += f"   cooldown {cool:.0f}s"
                cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, ALERT_COLOR, 2, cv2.LINE_AA)
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

    log("detect", "stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
