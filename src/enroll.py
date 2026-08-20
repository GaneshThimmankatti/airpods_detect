"""Enrollment: turn reference photos of one person into stored embeddings.

    ./run.sh enroll                     # uses config's photos_dir
    ./run.sh enroll --photos ~/pics     # or point it somewhere else

Drop 5-15 photos of the birthday girl into `enroll_photos/`. Vary the angle,
expression, hair and lighting -- the spread of the enrolled set is what decides
how forgiving recognition is at runtime. Each photo should contain exactly one
clearly visible face; photos with several faces are skipped with a warning
rather than guessed at.

The report printed at the end is your threshold-tuning tool: it shows how far
apart the enrolled photos are from each other. A sensible threshold sits above
that spread but below the distance to strangers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, log, resolve  # noqa: E402
from embedder import crop_face, load_embedder  # noqa: E402
from vision_mac import detect_faces  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tiff"}


def read_rgb(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    # Long side capped at 1600px: Vision gains nothing from more, and HEIC
    # shots off a phone are needlessly large.
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        s = 1600.0 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Enroll reference photos.")
    ap.add_argument("--photos", default=cfg["enrollment"]["photos_dir"])
    ap.add_argument("--out", default=cfg["enrollment"]["embeddings_path"])
    ap.add_argument("--embedder", default=cfg["detector"]["embedder"])
    ap.add_argument("--name", default="Birthday girl", help="label to display")
    args = ap.parse_args()

    photos_dir = resolve(args.photos)
    if not photos_dir.is_dir():
        log("enroll", f"no such directory: {photos_dir}")
        log("enroll", f"create it and add 5-15 photos, then re-run.")
        return 1

    paths = sorted(
        p for p in photos_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        log("enroll", f"no images found in {photos_dir}")
        return 1

    emb = load_embedder(args.embedder)
    log("enroll", f"backend: {emb.name}")

    vectors: list[np.ndarray] = []
    used: list[str] = []

    for path in paths:
        rgb = read_rgb(path)
        if rgb is None:
            log("enroll", f"skip {path.name}: unreadable (HEIC may need conversion)")
            continue

        faces = detect_faces(rgb)
        if not faces:
            log("enroll", f"skip {path.name}: no face detected")
            continue
        if len(faces) > 1:
            log("enroll", f"skip {path.name}: {len(faces)} faces, need exactly one")
            continue

        chip = crop_face(rgb, faces[0])
        if chip is None:
            log("enroll", f"skip {path.name}: face too small in frame")
            continue

        vec = emb.embed(chip)
        if vec is None:
            log("enroll", f"skip {path.name}: embedding failed")
            continue

        vectors.append(vec)
        used.append(path.name)
        log("enroll", f"ok   {path.name}")

    if len(vectors) < 3:
        log("enroll", f"only {len(vectors)} usable photo(s); need at least 3.")
        return 1

    matrix = np.stack(vectors).astype(np.float32)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        embeddings=matrix,
        backend=np.array(emb.name),
        label=np.array(args.name),
        sources=np.array(used),
    )

    # Pairwise spread across the enrolled set -- the tuning signal.
    dists = [
        float(np.linalg.norm(matrix[i] - matrix[j]))
        for i in range(len(matrix))
        for j in range(i + 1, len(matrix))
    ]
    log("enroll", f"saved {len(vectors)} embeddings -> {out_path}")
    print()
    print("  Spread within the enrolled photos (same person, so these are 'match' distances):")
    print(f"    min {min(dists):.3f}   median {float(np.median(dists)):.3f}   max {max(dists):.3f}")

    raw = float(np.percentile(dists, 90)) * 1.10
    # A tight spread usually means the photos are too alike (a burst, one
    # sitting) rather than that recognition is genuinely that precise. Trusting
    # it would produce a threshold so strict the person never matches live, so
    # floor the suggestion at a fraction of the backend default.
    floor = 0.6 * emb.DEFAULT_THRESHOLD
    suggested = max(raw, floor)
    print()
    print(f"  Suggested match_threshold: {suggested:.2f}")
    if raw < floor:
        print("    NOTE: your reference photos are very similar to each other.")
        print("    Add photos with different lighting, angles and expressions,")
        print("    then re-enroll -- otherwise live matching will be brittle.")
    print(f"    (backend default is {emb.DEFAULT_THRESHOLD})")
    print("    Set it in config.json under detector.match_threshold, or leave")
    print("    null to use the backend default. The detector window shows the")
    print("    live distance on each box -- watch it and adjust.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
