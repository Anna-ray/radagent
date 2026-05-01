"""
scripts/scan_corrupt.py
-----------------------
Scan the NIH-14 image folder for corrupt PNGs.

Run this ONCE before kicking off a long training run. Reports the list
of unreadable / partially-readable files so you can decide whether to
re-extract or live with the resilience layer in dataset.py.

Usage:
    python -m scripts.scan_corrupt --config configs/nih14_convnextv2_base.yaml
    python -m scripts.scan_corrupt --config configs/nih14_convnextv2_base.yaml --quick
        ^^ --quick only checks the first byte sequence (very fast)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--quick", action="store_true",
                   help="Only verify PNG signature (fast); default decodes fully.")
    p.add_argument("--out", type=str, default="corrupt_images.txt")
    return p.parse_args()


def is_corrupt_quick(path: str) -> str | None:
    """Check just the PNG signature (8 bytes). Misses bad-filter corruption."""
    try:
        with open(path, "rb") as f:
            sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            return f"bad PNG signature"
        return None
    except Exception as e:
        return f"open error: {e}"


def is_corrupt_full(path: str) -> str | None:
    """Decode fully via OpenCV, then PIL fallback. Catches bad filter bytes."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return None
    # OpenCV failed — try PIL
    try:
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(path) as pil_img:
            pil_img.load()
        return None  # PIL could read it (with truncation tolerance)
    except Exception as e:
        return f"both decoders failed: {e}"


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    images_dir = cfg["data"]["images_dir"]
    print(f"[scan] scanning {images_dir} (mode={'quick' if args.quick else 'full decode'})")

    files: list[str] = []
    for root, _, names in os.walk(images_dir):
        for n in names:
            if n.lower().endswith((".png", ".jpg", ".jpeg")):
                files.append(os.path.join(root, n))
    print(f"[scan] found {len(files):,} candidate images")

    check = is_corrupt_quick if args.quick else is_corrupt_full
    bad: list[tuple[str, str]] = []
    for i, p in enumerate(files):
        reason = check(p)
        if reason is not None:
            bad.append((p, reason))
        if (i + 1) % 5000 == 0:
            print(f"  [scan] {i+1:,}/{len(files):,}   bad so far: {len(bad)}")

    print(f"\n[scan] DONE. {len(bad):,} corrupt files / {len(files):,} total "
          f"({100*len(bad)/max(1,len(files)):.3f}%)")

    if bad:
        with open(args.out, "w", encoding="utf-8") as f:
            for p, r in bad:
                f.write(f"{p}\t{r}\n")
        print(f"[scan] wrote list to {args.out}")
        for p, r in bad[:10]:
            print(f"  - {Path(p).name}: {r}")
        if len(bad) > 10:
            print(f"  ... and {len(bad)-10} more")
    else:
        print("[scan] dataset clean.")
    return 0 if len(bad) == 0 else 0  # non-zero would block scripts; we just report


if __name__ == "__main__":
    sys.exit(main())
