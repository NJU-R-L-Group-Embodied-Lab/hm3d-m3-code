"""Self-test for :mod:`benchmark.iin_verify` — offline, no Habitat, no network.

Builds deterministic procedural "scene" images (textured shapes so SIFT finds
stable keypoints), then checks the verifier separates:

* a **same-instance** pair (the image under a mild perspective warp + photometric
  jitter) — should verify as the same instance with many geometric inliers;
* a **different-instance** pair (a different procedural layout) — should NOT
  verify, with far fewer inliers.

This validates the verification logic itself; integration on real Habitat-rendered
goal images is a separate step (P2 render -> verify).
"""
from __future__ import annotations

import cv2
import numpy as np

from benchmark.iin_verify import InstanceVerifier


def make_scene(seed: int, h: int = 480, w: int = 640) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # gradient background — direction/scale vary per seed so unrelated scenes
    # share no identical low-texture content
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy, base = rng.uniform(40, 140), rng.uniform(30, 110), rng.uniform(10, 60)
    bg = ((xx / w) * cx + (yy / h) * cy + base).astype(np.uint8)
    img = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    def col():
        return tuple(int(c) for c in rng.integers(40, 255, size=3))

    for _ in range(14):  # filled shapes -> corners/blobs for SIFT
        kind = int(rng.integers(0, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if kind == 0:
            x2, y2 = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.rectangle(img, (x, y), (x2, y2), col(), -1)
        elif kind == 1:
            cv2.circle(img, (x, y), int(rng.integers(15, 60)), col(), -1)
        else:
            cv2.ellipse(img, (x, y), (int(rng.integers(20, 70)), int(rng.integers(20, 70))),
                        int(rng.integers(0, 180)), 0, 360, col(), -1)
    for _ in range(10):  # lines -> edges
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, col(), int(rng.integers(1, 4)))
    # non-repeating speckle texture -> many distinct blob keypoints. Unlike a
    # checkerboard, random dots are NOT periodic, so unrelated scenes produce no
    # geometrically-consistent cross-matches (periodic texture is the classic
    # SIFT false-match trap; real object instances aren't periodic either).
    for _ in range(90):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (x, y), int(rng.integers(2, 5)), col(), -1)
    return img


def warp_same(img: np.ndarray, seed: int) -> np.ndarray:
    """Mild perspective warp + photometric jitter = a different view of the SAME scene."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    m = 0.08
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-m, m, src.shape) * np.float32([w, h])
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    out = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REFLECT)
    # brightness/contrast jitter + mild blur + noise
    alpha, beta = 0.85 + 0.3 * rng.random(), rng.uniform(-15, 15)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    noise = rng.normal(0, 4, out.shape).astype(np.int16)
    return np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main() -> None:
    cv2.setRNGSeed(0)  # make RANSAC reproducible
    print("== iin_verify self-test (SIFT backend, offline) ==")
    verifier = InstanceVerifier(backend="sift")

    scene = make_scene(1)
    same_view = warp_same(scene, 2)
    other = make_scene(999)

    v_same = verifier.verify(same_view, scene)
    v_diff = verifier.verify(other, scene)

    print(f"  same-instance : inliers={v_same.n_inliers:3d} matches={v_same.n_matches:3d} "
          f"ratio={v_same.inlier_ratio:.2f} score={v_same.score:.2f} is_same={v_same.is_same}")
    print(f"  diff-instance : inliers={v_diff.n_inliers:3d} matches={v_diff.n_matches:3d} "
          f"ratio={v_diff.inlier_ratio:.2f} score={v_diff.score:.2f} is_same={v_diff.is_same}")

    assert v_same.is_same, "same-instance pair should verify as same"
    assert not v_diff.is_same, "different-instance pair should NOT verify as same"
    assert v_same.n_inliers > v_diff.n_inliers, "same pair must have more geometric inliers"
    assert v_same.score > v_diff.score, "same pair must score higher"
    print("ALL IIN-VERIFY SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
