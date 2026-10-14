"""VLFM top-down value map (Yokoyama et al., 2024).

Official VLFM builds a value map by scoring each observation with BLIP-2
ITC cosine against ``Seems like there is a {object} ahead.`` and splatting
that score onto a forward cone of ground cells, then steering to the
frontier adjacent to the highest accumulated value.

Spatial machinery matches the released ``vlfm.mapping.value_map`` cone
splat (confidence-weighted running mean). The value head is BLIP-2
(``BLIP2ValueMap``), with the BLIP-2 scorer in this module.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# --------------------------------------------------------------------------
# BLIP-2 ITM/ITC scorer
#
# Official VLFM value head: BLIP-2 image-text cosine (ITC).
#
# VLFM (Yokoyama et al., 2024) scores each RGB with LAVIS
# ``blip2_image_text_matching`` / ``pretrain`` and ``match_head="itc"``
# (see ``vlfm.vlm.blip2itm.BLIP2ITM.cosine``). The paper calls the model
# ITM; the released code uses the ITC cosine, not the ITM softmax.
#
# This module uses HuggingFace ``Blip2ForImageTextRetrieval``
# (``Salesforce/blip2-itm-vit-g``). Default ``match_head="itc"`` is
# ``use_image_text_matching_head=False`` (contrastive cosine).
# ``match_head="itm"`` is the CA-Nav-style matching head: softmax over
# the two-class ITM logits, using the "match" probability.
#
# Prompt matches VLFMConfig:
#
#     Seems like there is a {category} ahead.
# --------------------------------------------------------------------------
_DEFAULT_MODEL_ID = "Salesforce/blip2-itm-vit-g"
_DEFAULT_LOCAL = "data/models/blip2-itm-vit-g"  # relative to sim-code/; override with VLFM_BLIP2_DIR
_PROMPT = "Seems like there is a {category} ahead."
_VALID_HEADS = ("itc", "itm")

_SHARED = {}


def official_prompt(category: str) -> str:
    return _PROMPT.format(category=str(category).replace("|", "/"))


def resolve_model_path() -> str:
    local = os.environ.get("VLFM_BLIP2_DIR", _DEFAULT_LOCAL)
    if os.path.isfile(os.path.join(local, "config.json")):
        return local
    return os.environ.get("VLFM_BLIP2_ID", _DEFAULT_MODEL_ID)


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


class BLIP2ITM:
    """Process-wide BLIP-2 scorer (one copy per device)."""

    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import Blip2ForImageTextRetrieval, Blip2Processor

        self.device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
        self._torch = torch
        path = resolve_model_path()
        local_only = os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))
        print(f"[vlfm-blip2] loading {path} local_only={local_only} device={self.device}", flush=True)
        self.processor = Blip2Processor.from_pretrained(path, local_files_only=local_only)
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            path, local_files_only=local_only, torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self._prompts: List[str] = []

    def set_targets(self, categories: Union[str, Sequence[str]]) -> None:
        cats = [categories] if isinstance(categories, str) else list(categories)
        self._prompts = [official_prompt(c) for c in cats if str(c).strip()]

    def score_pair(self, rgb: np.ndarray, txt: str, match_head: str = "itc") -> float:
        """Score one RGB–text pair. ``itc`` = cosine; ``itm`` = P(match)."""
        if rgb is None or not txt:
            return 0.0
        head = str(match_head or "itc").lower()
        if head not in _VALID_HEADS:
            raise ValueError("match_head must be itc or itm, got %r" % match_head)
        torch = self._torch
        from PIL import Image

        img = Image.fromarray(np.asarray(rgb)[:, :, :3].astype(np.uint8))
        inputs = self.processor(images=img, text=txt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        use_itm = head == "itm"
        with torch.inference_mode():
            out = self.model(**inputs, use_image_text_matching_head=use_itm)
        logits = getattr(out, "logits_per_image", None)
        if logits is None:
            logits = getattr(out, "logits", None)
        if logits is None:
            logits = getattr(out, "itm_score", None)
        if logits is None:
            raise RuntimeError("BLIP-2 %s forward returned no logits" % head)
        logits = logits.float()
        if use_itm:
            # HF Blip2ForImageTextRetrieval puts the 2-class ITM scores in
            # logits_per_image with shape (B, 2); index 1 is "match".
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            if logits.shape[-1] != 2:
                raise RuntimeError("BLIP-2 ITM logits expected (*, 2), got %s" % (tuple(logits.shape),))
            val = float(torch.softmax(logits, dim=-1)[0, 1].item())
            return _clamp01(val)
        return _clamp01(float(logits.reshape(-1)[0].item()))

    def cosine(self, rgb: np.ndarray, txt: str) -> float:
        return self.score_pair(rgb, txt, match_head="itc")

    def itm(self, rgb: np.ndarray, txt: str) -> float:
        return self.score_pair(rgb, txt, match_head="itm")

    def score_image(self, rgb: np.ndarray, match_head: str = "itc") -> float:
        if rgb is None or not self._prompts:
            return 0.0
        return max(self.score_pair(rgb, p, match_head=match_head) for p in self._prompts)


def shared_blip2(device: str = "cuda") -> BLIP2ITM:
    key = str(device)
    obj = _SHARED.get(key)
    if obj is None:
        obj = BLIP2ITM(device=device)
        _SHARED[key] = obj
    return obj


class ConeValueMap:
    """Confidence-weighted forward-cone splat; subclasses implement score_image."""

    def __init__(self, cell_size: float = 0.5, cone_range: float = 3.0,
                 cone_halfangle_deg: float = 39.0, device: str = "cuda"):
        self.cell = cell_size
        self.range = cone_range
        self.half = math.radians(cone_halfangle_deg)
        self.device = device
        self._sum: Dict[Tuple[int, int], float] = {}
        self._wt: Dict[Tuple[int, int], float] = {}

    def set_targets(self, categories):
        raise NotImplementedError

    def set_target(self, category: str):
        self.set_targets([category])

    def score_image(self, rgb: np.ndarray) -> float:
        raise NotImplementedError

    def _key(self, x: float, z: float) -> Tuple[int, int]:
        return (int(math.floor(x / self.cell)), int(math.floor(z / self.cell)))

    def update(self, value: float, pos_xz: Tuple[float, float], yaw_rad: float,
               confidence: float = 1.0):
        """Splat ``value`` onto a forward cone of cells (confidence-weighted mean)."""
        px, pz = pos_xz
        r = self.range
        steps = max(1, int(r / self.cell))
        for di in range(-steps, steps + 1):
            for dj in range(-steps, steps + 1):
                dx, dz = di * self.cell, dj * self.cell
                dist = math.hypot(dx, dz)
                if dist > r or dist < 1e-6:
                    continue
                ang = math.atan2(dz, dx) - yaw_rad
                ang = (ang + math.pi) % (2 * math.pi) - math.pi
                if abs(ang) > self.half:
                    continue
                w = confidence * (1.0 - dist / r)
                k = self._key(px + dx, pz + dz)
                self._sum[k] = self._sum.get(k, 0.0) + w * value
                self._wt[k] = self._wt.get(k, 0.0) + w

    def value_at(self, x: float, z: float, radius: float = 1.5) -> float:
        """Max confidence-weighted-mean value among cells within ``radius``.

        Returns -1.0 when the area is unobserved (caller treats as 'explore').
        """
        steps = max(1, int(radius / self.cell))
        cx, cz = self._key(x, z)
        best = -1.0
        for di in range(-steps, steps + 1):
            for dj in range(-steps, steps + 1):
                k = (cx + di, cz + dj)
                w = self._wt.get(k, 0.0)
                if w > 0:
                    best = max(best, self._sum[k] / w)
        return best

    @property
    def observed_cells(self) -> int:
        return len(self._wt)


class BLIP2ValueMap(ConeValueMap):
    """BLIP-2 value head + cone splat.

    ``match_head="itc"`` is the official VLFM cosine. ``match_head="itm"``
    is the two-class matching probability (CA-Nav CVM head).
    """

    def __init__(self, cell_size: float = 0.5, cone_range: float = 3.0,
                 cone_halfangle_deg: float = 39.0, device: str = "cuda",
                 match_head: str = "itc"):
        super().__init__(cell_size, cone_range, cone_halfangle_deg, device)
        self.match_head = str(match_head or "itc").lower()
        self._itm = None

    def _ensure(self):
        if self._itm is None:
            self._itm = shared_blip2(self.device)

    def set_targets(self, categories):
        self._ensure()
        self._itm.set_targets(categories)

    def score_image(self, rgb: np.ndarray) -> float:
        if rgb is None:
            return 0.0
        self._ensure()
        return float(self._itm.score_image(rgb, match_head=self.match_head))
