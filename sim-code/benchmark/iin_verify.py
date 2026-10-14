"""Instance verification for IIN goals — "is the thing I see THE goal instance?"

This is the perfect-match / goal-verification step that UniGoal performs with
LightGlue keypoint matching (RGB observation <-> goal image). Our LaVira VA only
produces a category bounding box; it has no notion of *instance identity*, which
IIN requires. This module fills that gap and is deliberately:

* **standalone** — pure ``numpy`` + ``opencv``, no Habitat, no torch required for
  the default backend, so it is unit-testable anywhere;
* **pluggable** — the default ``sift`` backend (SIFT + Lowe ratio + RANSAC
  homography inliers) runs fully offline today; a ``lightglue`` backend activates
  automatically when ``lightglue`` (and its weights) are importable.

Both backends return the same verdict dict, so the VA integration is
backend-agnostic. The geometric-inlier signal is the same idea LightGlue uses; we
fall back to the classical detector because the container is offline and neither
``lightglue`` nor its pretrained weights can be fetched (see README).

CLI::

    python -m benchmark.iin_verify --query obs.png --goal goal.png [--backend sift]
    python -m benchmark.iin_verify --selftest
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Optional, Union

import cv2
import numpy as np

ImageLike = Union[str, np.ndarray]


@dataclass
class Verdict:
    n_kp_query: int
    n_kp_goal: int
    n_matches: int          # mutual NN matches surviving the ratio test
    n_inliers: int          # geometrically consistent (RANSAC homography) matches
    inlier_ratio: float     # n_inliers / max(1, n_matches)
    score: float            # backend confidence in [0, 1]
    is_same: bool
    backend: str


def _load_gray(img: ImageLike) -> np.ndarray:
    if isinstance(img, str):
        arr = cv2.imread(img, cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"cannot read image: {img}")
    else:
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # callers pass RGB (habitat)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr


class InstanceVerifier:
    """Keypoint-matching instance verifier with a pluggable backend.

    Decision: ``is_same`` iff geometric inliers >= ``min_inliers`` and
    inlier_ratio >= ``min_inlier_ratio``. Defaults are conservative; tune per the
    HM3D-IIN success criterion (1m + STOP) during P2 evaluation.
    """

    def __init__(self, backend: str = "sift", min_inliers: int = 18,
                 min_inlier_ratio: float = 0.18, ratio_test: float = 0.75):
        self.backend = backend
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.ratio_test = ratio_test
        self._lg = None
        if backend == "sift":
            self._sift = cv2.SIFT_create()
            self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif backend == "lightglue":
            self._init_lightglue()
        else:
            raise ValueError(f"unknown backend: {backend}")

    # ------------------------------------------------------------------ backends
    def _init_lightglue(self):
        try:
            import torch  # noqa: F401
            from lightglue import LightGlue, SuperPoint  # type: ignore
            from lightglue.utils import rbd  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "lightglue backend unavailable (offline / not installed). "
                "Install lightglue + weights, or use backend='sift'."
            ) from e
        import torch
        self._torch = torch
        self._rbd = rbd
        self._extractor = SuperPoint(max_num_keypoints=2048).eval()
        self._lg = LightGlue(features="superpoint").eval()

    # ------------------------------------------------------------------ verify
    def verify(self, query: ImageLike, goal: ImageLike) -> Verdict:
        if self.backend == "sift":
            return self._verify_sift(query, goal)
        return self._verify_lightglue(query, goal)

    def _ransac_inliers(self, pts_q: np.ndarray, pts_g: np.ndarray) -> int:
        if len(pts_q) < 4:
            return len(pts_q)
        _, mask = cv2.findHomography(pts_q, pts_g, cv2.RANSAC, 5.0)
        return int(mask.sum()) if mask is not None else 0

    def _verify_sift(self, query: ImageLike, goal: ImageLike) -> Verdict:
        gq, gg = _load_gray(query), _load_gray(goal)
        kq, dq = self._sift.detectAndCompute(gq, None)
        kg, dg = self._sift.detectAndCompute(gg, None)
        if dq is None or dg is None or len(kq) < 2 or len(kg) < 2:
            return Verdict(len(kq or []), len(kg or []), 0, 0, 0.0, 0.0, False, "sift")

        knn = self._matcher.knnMatch(dq, dg, k=2)
        good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < self.ratio_test * n.distance]
        if len(good) >= 4:
            pts_q = np.float32([kq[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts_g = np.float32([kg[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            inliers = self._ransac_inliers(pts_q, pts_g)
        else:
            inliers = len(good)
        n_matches = len(good)
        ratio = inliers / max(1, n_matches)
        score = self._score(inliers, ratio)
        is_same = inliers >= self.min_inliers and ratio >= self.min_inlier_ratio
        return Verdict(len(kq), len(kg), n_matches, inliers, round(ratio, 4),
                       round(score, 4), bool(is_same), "sift")

    def _verify_lightglue(self, query: ImageLike, goal: ImageLike) -> Verdict:
        torch, rbd = self._torch, self._rbd

        def to_tensor(img):
            g = _load_gray(img).astype(np.float32) / 255.0
            return torch.from_numpy(g)[None, None]

        with torch.no_grad():
            fq = self._extractor.extract(to_tensor(query))
            fg = self._extractor.extract(to_tensor(goal))
            matches01 = self._lg({"image0": fq, "image1": fg})
            fq, fg, m = [rbd(x) for x in (fq, fg, matches01)]
        pairs = m["matches"]
        n_matches = int(pairs.shape[0])
        kq = fq["keypoints"].cpu().numpy()
        kg = fg["keypoints"].cpu().numpy()
        if n_matches >= 4:
            idx = pairs.cpu().numpy()
            pts_q = kq[idx[:, 0]].reshape(-1, 1, 2).astype(np.float32)
            pts_g = kg[idx[:, 1]].reshape(-1, 1, 2).astype(np.float32)
            inliers = self._ransac_inliers(pts_q, pts_g)
        else:
            inliers = n_matches
        ratio = inliers / max(1, n_matches)
        score = self._score(inliers, ratio)
        is_same = inliers >= self.min_inliers and ratio >= self.min_inlier_ratio
        return Verdict(len(kq), len(kg), n_matches, inliers, round(ratio, 4),
                       round(score, 4), bool(is_same), "lightglue")

    def _score(self, inliers: int, ratio: float) -> float:
        # squashed inlier count, gently weighted by geometric consistency
        sat = 1.0 - np.exp(-inliers / max(1, self.min_inliers))
        return float(np.clip(0.5 * sat + 0.5 * min(ratio / max(1e-6, self.min_inlier_ratio), 1.0), 0.0, 1.0))


def verify(query: ImageLike, goal: ImageLike, backend: str = "sift", **kw) -> Verdict:
    return InstanceVerifier(backend=backend, **kw).verify(query, goal)


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description="IIN instance verification")
    p.add_argument("--query")
    p.add_argument("--goal")
    p.add_argument("--backend", default="sift", choices=["sift", "lightglue"])
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)
    if args.selftest:
        from tests.selftests.iin_verify_selftest import main as st
        st()
        return
    if not (args.query and args.goal):
        p.error("--query and --goal required (or --selftest)")
    import json
    print(json.dumps(asdict(verify(args.query, args.goal, backend=args.backend)), indent=2))


if __name__ == "__main__":
    main()
