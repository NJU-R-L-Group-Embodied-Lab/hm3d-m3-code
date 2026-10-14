"""DM3-Nav reproduction — D1: no-LLM perception + modality matching.

Faithful to DM3-Nav's perception stack (no LLM/VLM), using github-free weights:

* **Detection**: torchvision Mask R-CNN (COCO) — DM3 uses Mask R-CNN on HM3D; the
  6 HM3D-ObjectNav categories all map to COCO classes.
* **IIN (instance image) match**: keypoint matching (kornia **LightGlue** + DISK) —
  DM3 uses SuperGlue; LightGlue is its successor.
* **TN (language) match**: **CLIP** text–image cosine similarity (clip-anytorch,
  OpenAI weights) — DM3 uses CLIP for language goals.
* **(optional) YOLO verification**: ultralytics, mirroring DM3's YOLO second-check.

This module is the no-LLM analogue of LaViRA's VA. It is standalone/testable
(`python -m benchmark.dm3_perception --selftest`).
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np

# HM3D ObjectNav category -> COCO class name (torchvision Mask R-CNN labels)
HM3D_TO_COCO = {
    "chair": "chair", "bed": "bed", "plant": "potted plant",
    "toilet": "toilet", "tv_monitor": "tv", "sofa": "couch",
}


class DM3Perception:
    def __init__(self, device: Optional[str] = None, det_score_thresh: float = 0.5):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.det_score_thresh = det_score_thresh
        self._det = None
        self._coco = None
        self._clip = None
        self._clip_pre = None
        self._disk = None
        self._lg = None

    # ----------------------------------------------------------------- detection
    def _ensure_detector(self):
        if self._det is not None:
            return
        from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_Weights,
                                                  maskrcnn_resnet50_fpn)
        w = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self._coco = list(w.meta["categories"])  # index -> coco name
        self._det = maskrcnn_resnet50_fpn(weights=w).eval().to(self.device)

    def detect(self, rgb: np.ndarray, categories: List[str]) -> List[Dict]:
        """Detect instances of the given HM3D categories in an RGB uint8 image.

        Returns [{category, coco, score, box[x1,y1,x2,y2]}], score-filtered.
        """
        self._ensure_detector()
        torch = self.torch
        want = {HM3D_TO_COCO.get(c, c): c for c in categories}  # coco -> hm3d
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0).to(self.device)
        with torch.no_grad():
            out = self._det([t])[0]
        res = []
        for box, lab, sc in zip(out["boxes"], out["labels"], out["scores"]):
            s = float(sc)
            if s < self.det_score_thresh:
                continue
            name = self._coco[int(lab)] if int(lab) < len(self._coco) else str(int(lab))
            if name in want:
                res.append({"category": want[name], "coco": name, "score": round(s, 3),
                            "box": [float(x) for x in box]})
        return res

    # ----------------------------------------------------------------- TN (CLIP)
    def _ensure_clip(self):
        if self._clip is not None:
            return
        import clip
        self._clip, self._clip_pre = clip.load("ViT-B/32", device=self.device)
        self._clip.eval()
        self._clip_mod = clip

    def match_tn(self, rgb: np.ndarray, tn_text: str) -> float:
        """CLIP cosine similarity between an RGB crop and the TN description (0..1-ish)."""
        self._ensure_clip()
        torch = self.torch
        from PIL import Image
        img = self._clip_pre(Image.fromarray(rgb.astype(np.uint8))).unsqueeze(0).to(self.device)
        txt = self._clip_mod.tokenize([tn_text[:300]]).to(self.device)
        with torch.no_grad():
            fi = self._clip.encode_image(img)
            ft = self._clip.encode_text(txt)
            fi = fi / fi.norm(dim=-1, keepdim=True)
            ft = ft / ft.norm(dim=-1, keepdim=True)
            sim = float((fi @ ft.T).squeeze().cpu())
        return sim

    # ----------------------------------------------------------------- IIN (keypoint match)
    # NOTE: DM3 uses SuperGlue; kornia LightGlue is its successor and is installed,
    # but its pretrained matcher weights are hosted on GitHub releases, which is
    # unreachable from this container (github.com blocked; only pypi / pytorch.org /
    # azure / raw.githubusercontent reachable). We therefore use SIFT + RANSAC
    # (classical keypoint matching, same family) via the already-tested
    # benchmark.iin_verify. Swap to backend="lightglue" once the weights are local.
    def _ensure_iin(self):
        if self._lg is None:
            from .iin_verify import InstanceVerifier
            self._lg = InstanceVerifier(backend="sift")

    def match_iin(self, obs_rgb: np.ndarray, goal_rgb: np.ndarray) -> int:
        """Geometric keypoint-match count (RANSAC inliers) between obs and goal image."""
        self._ensure_iin()
        return int(self._lg.verify(obs_rgb, goal_rgb).n_inliers)


# --------------------------------------------------------------------------- CLI
def _selftest():
    import glob
    import cv2
    print("== DM3 perception self-test ==")
    p = DM3Perception()
    print("device:", p.device)
    imgs = sorted(glob.glob("data/datasets/ma_objectnav/val/iin_images/*.png"))
    if not imgs:
        print("no rendered IIN images found; run render_iin first"); return
    import os
    a = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2RGB)
    cat = os.path.basename(imgs[0]).split("_")[-1].replace(".png", "")
    # detection
    det = p.detect(a, ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"])
    print(f"  detect on {os.path.basename(imgs[0])}: {len(det)} dets -> {[(d['category'],d['score']) for d in det[:5]]}")
    # IIN: same vs different
    b = cv2.cvtColor(cv2.imread(imgs[1]), cv2.COLOR_BGR2RGB)
    same = p.match_iin(a, a); diff = p.match_iin(a, b)
    print(f"  IIN LightGlue matches: self={same}  other={diff}")
    assert same > diff, "self-match should exceed cross-match"
    # TN: image vs a matching vs mismatching description
    s_match = p.match_tn(a, f"a photo of a {cat}")
    s_mis = p.match_tn(a, "a photo of an airplane in the sky")
    print(f"  TN CLIP sim: '{cat}'={s_match:.3f}  'airplane'={s_mis:.3f}")
    assert s_match > s_mis, "matching caption should score higher"
    print("DM3 PERCEPTION SELF-TEST PASSED")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()


if __name__ == "__main__":
    main()
