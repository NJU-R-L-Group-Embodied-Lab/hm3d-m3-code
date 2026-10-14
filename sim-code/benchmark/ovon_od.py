"""Open-vocabulary detector gates for FOUND / STOP.

ON / TN stay official: OWL 0.4 → ON accept / TN CLIP ≥ 0.24, then PointNav.
IIN keeps SuperGlue ≥ 24.5. A far pass only walks in; the walk target is a
point in front of the box, SuperGlue sees an expanded OWL crop, and the
stop gate is inside APPROACH_RADIUS. After a pass, later close-frame misses
do not reject: the best score sticks. STOP uses the same 0.9 m radius as
ON/TN (hold's extra 1.0 m IIN stop is off). Stale unlock is the same as
ON/TN (far + no progress). A close miss is rejected only when SuperGlue
never passed.
"""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from benchmark.owl_shm import RgbShmSlot, default_shm_dir


# --------------------------------------------------------------------------
# Camera / depth geometry
#
# Shared RGB-D geometry for the OD found gate.
#
# Unproject a box from the current view.
# --------------------------------------------------------------------------
def habitat_forward_xz(yaw_rad: float) -> np.ndarray:
    """Habitat yaw0 looks down -Z; +X is right."""
    return np.array([math.sin(yaw_rad), -math.cos(yaw_rad)], dtype=np.float32)


def habitat_depth_to_meters(
    depth: np.ndarray,
    dmin: float = 0.1,
    dmax: float = 5.0,
) -> np.ndarray:
    """Habitat ``normalize_depth=True``: 0..1 -> [dmin, dmax] meters."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    return float(dmin) + d * (float(dmax) - float(dmin))


def depth_at_box(
    depth_m: np.ndarray,
    xyxy: Sequence[float],
    dmax: float = 5.0,
    inner: float = 0.25,
) -> Optional[float]:
    """Median of valid meters in the inner box. None if empty / at far plane."""
    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1 = int(max(0, math.floor(x1 + inner * bw)))
    y1 = int(max(0, math.floor(y1 + inner * bh)))
    x2 = int(min(w, math.ceil(x2 - inner * bw)))
    y2 = int(min(h, math.ceil(y2 - inner * bh)))
    if x2 <= x1 or y2 <= y1:
        cx = int(np.clip((xyxy[0] + xyxy[2]) * 0.5, 0, w - 1))
        cy = int(np.clip((xyxy[1] + xyxy[3]) * 0.5, 0, h - 1))
        val = float(depth_m[cy, cx])
        if 0.08 < val < 0.98 * float(dmax):
            return val
        return None
    patch = depth_m[y1:y2, x1:x2]
    valid = patch[(patch > 0.08) & (patch < 0.98 * float(dmax))]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def pixel_to_xz(
    col: float,
    row: float,
    depth_m: float,
    pos_xz: Tuple[float, float],
    yaw_rad: float,
    width: int,
    height: int,
    hfov_deg: float = 79.0,
) -> Optional[Tuple[float, float]]:
    """Unproject a pixel along Habitat yaw0 = -Z, X right. ``row`` unused (floor xz)."""
    del row
    if depth_m is None or not math.isfinite(depth_m):
        return None
    d = float(depth_m)
    if d <= 0.08 or d > 8.0:
        return None
    half = math.tan(math.radians(float(hfov_deg)) / 2.0)
    nx = (float(col) / max(int(width) - 1, 1) - 0.5) * 2.0 * half
    fwd = habitat_forward_xz(yaw_rad)
    right = np.array([-float(fwd[1]), float(fwd[0])], dtype=np.float32)
    return (
        float(pos_xz[0]) + d * float(fwd[0]) + nx * d * float(right[0]),
        float(pos_xz[1]) + d * float(fwd[1]) + nx * d * float(right[1]),
    )


def rho_theta_vlfm(agent_xz: Tuple[float, float], yaw: float, goal_xz: Tuple[float, float]):
    """VLFM polar goal. theta>0 is turn left."""
    curr = np.array([float(agent_xz[0]), float(agent_xz[1])], dtype=np.float32)
    goal = np.array([float(goal_xz[0]), float(goal_xz[1])], dtype=np.float32)
    c, s = math.cos(-float(yaw)), math.sin(-float(yaw))
    local = goal - curr
    rot = np.array([[c, -s], [s, c]], dtype=np.float32) @ local
    rho = float(np.linalg.norm(rot))
    theta = float(math.atan2(rot[1], rot[0]))
    return rho, theta

OWL_THRESHOLD = 0.4
SUPERGLUE_THRESH = 24.5
LANG_THRESH = 0.24
STOP_RADIUS = 0.9
APPROACH_RADIUS = 2.0
APPROACH_STANDOFF = 1.0
SG_CROP_SCALE = 2.0
REJECT_RADIUS = 0.6


def should_run_owl(step: int, every: int, force: bool = False) -> bool:
    if force:
        return True
    every = max(1, int(every))
    return int(step) % every == 0


def _as_uint8_rgb(rgb: np.ndarray) -> Optional[np.ndarray]:
    if rgb is None or np.asarray(rgb).ndim != 3:
        return None
    img = np.ascontiguousarray(np.asarray(rgb)[..., :3])
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _encode_jpg(rgb: np.ndarray) -> Optional[str]:
    import base64

    img = _as_uint8_rgb(rgb)
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def expand_box(
    box: Sequence[float],
    width: int,
    height: int,
    scale: float = SG_CROP_SCALE,
) -> List[float]:
    """Grow an OWL box about its center, clamped to the frame."""
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(1.0, (x2 - x1) * float(scale))
    bh = max(1.0, (y2 - y1) * float(scale))
    nx1 = max(0.0, cx - 0.5 * bw)
    ny1 = max(0.0, cy - 0.5 * bh)
    nx2 = min(float(width), cx + 0.5 * bw)
    ny2 = min(float(height), cy + 0.5 * bh)
    if nx2 <= nx1 or ny2 <= ny1:
        return [x1, y1, x2, y2]
    return [nx1, ny1, nx2, ny2]


def approach_depth(
    depth_m: float,
    standoff: float = APPROACH_STANDOFF,
    min_d: float = 0.8,
) -> float:
    """Walk to a point in front of the box, not onto the depth hit."""
    return max(float(min_d), float(depth_m) - float(standoff))


def _crop_box(rgb: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    h, w = np.asarray(rgb).shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray(rgb)[y1:y2, x1:x2]


def goal_spec(goal: dict, iin_path: Optional[str] = None) -> Dict[str, Any]:
    mod = str(goal.get("modality", "ON")).upper()
    cat = str(goal.get("object_category") or "object").replace("_", " ").strip()
    text = str(goal.get("tn_text") or goal.get("lang_desc") or cat).strip()
    return {
        "modality": mod,
        "category": cat,
        "owl_query": cat,
        "goal_text": text,
        "goal_image": iin_path or "",
    }


def gate_ok(modality: str, match_score: Optional[float]) -> bool:
    """ON needs no second score; IIN / TN use the official thresholds."""
    mod = str(modality).upper()
    if mod == "ON":
        return True
    if match_score is None:
        return False
    if mod == "IIN":
        return float(match_score) >= SUPERGLUE_THRESH
    if mod == "TN":
        return float(match_score) >= LANG_THRESH
    return False


def iin_sticky_ok(best_match: float) -> bool:
    """True after any SuperGlue pass; later frames may drop below 24.5."""
    return gate_ok("IIN", best_match)


def sticky_stop_radius(best_match: float) -> float:
    """Same STOP radius for every modality. ``best_match`` kept for callers."""
    return STOP_RADIUS


def face_box_turn(look_theta: float, eps: float = 0.15) -> str:
    """Turn toward the box. ``look_theta>0`` is left, matching VLFM."""
    if float(look_theta) >= float(eps):
        return "left"
    if float(look_theta) <= -float(eps):
        return "right"
    return "left"


class OwlClient:
    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        shm_dir: Optional[str] = None,
        shm_name: Optional[str] = None,
    ):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.shm_dir = (shm_dir if shm_dir is not None else default_shm_dir()).strip()
        self.shm_name = shm_name or ("owl_%d" % os.getpid())
        self._slots: Dict[str, RgbShmSlot] = {}
        self.n_shm = 0
        self.n_jpeg = 0
        self.n_detect = 0
        self.n_box = 0
        self.n_err = 0

    def _slot(self, suffix: str = "") -> Optional[RgbShmSlot]:
        if not self.shm_dir:
            return None
        name = self.shm_name + suffix
        slot = self._slots.get(name)
        if slot is None:
            try:
                os.makedirs(self.shm_dir, exist_ok=True)
                slot = RgbShmSlot(self.shm_dir, name)
                self._slots[name] = slot
            except OSError:
                return None
        return slot

    def _put_rgb(self, rgb: np.ndarray, suffix: str = "") -> Optional[dict]:
        slot = self._slot(suffix)
        if slot is None:
            return None
        img = _as_uint8_rgb(rgb)
        if img is None:
            return None
        try:
            meta = slot.write(img)
        except OSError:
            return None
        # Sidecar uses its own OWL_SHM_DIR (host path). Do not send the
        # Docker bind-mount path — it is not valid on the host.
        return {"shm_name": meta["shm_name"], "seq": meta["seq"]}

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _image_fields(self, rgb: np.ndarray, suffix: str = "") -> dict:
        fields: Dict[str, Any] = {}
        meta = self._put_rgb(rgb, suffix)
        if meta is not None:
            self.n_shm += 1
            fields.update(meta)
        img = _encode_jpg(rgb)
        if img is not None:
            self.n_jpeg += 1
            fields["image_b64"] = img
        return fields

    def detect(self, rgb: np.ndarray, text: str, threshold: float = OWL_THRESHOLD):
        if not text:
            return []
        fields = self._image_fields(rgb, "")
        if not fields:
            return []
        try:
            out = self._post(
                "/detect",
                dict(fields, text=text, threshold=float(threshold)),
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            self.n_err = getattr(self, "n_err", 0) + 1
            if self.n_err <= 5:
                print(f"[owl] detect failed ({self.n_err}): {exc!r} keys={list(fields)}")
            return []
        boxes = out.get("boxes") or []
        scores = out.get("scores") or []
        hits = list(zip(boxes, scores))
        if hits:
            self.n_box = getattr(self, "n_box", 0) + 1
        self.n_detect = getattr(self, "n_detect", 0) + 1
        return hits

    def match_image(self, rgb: np.ndarray, goal_rgb: np.ndarray) -> float:
        frame = self._image_fields(rgb, "")
        goal = self._image_fields(goal_rgb, "_g")
        if not frame or not goal:
            return 0.0
        payload = dict(frame)
        if "image_b64" in goal:
            payload["goal_image_b64"] = goal["image_b64"]
        else:
            payload["goal_shm_name"] = goal.get("shm_name")
            payload["goal_seq"] = goal.get("seq")
        try:
            out = self._post("/match_image", payload)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return 0.0
        return float(out.get("score") or 0.0)

    def score_lang(self, rgb: np.ndarray, text: str) -> float:
        if not text:
            return 0.0
        fields = self._image_fields(rgb, "")
        if not fields:
            return 0.0
        try:
            out = self._post("/score_lang", dict(fields, text=text))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return 0.0
        return float(out.get("score") or 0.0)


def _near_xz(
    a: Optional[Tuple[float, float]],
    b: Optional[Tuple[float, float]],
    radius: float = REJECT_RADIUS,
) -> bool:
    if a is None or b is None:
        return False
    dx = float(a[0]) - float(b[0])
    dz = float(a[1]) - float(b[1])
    return (dx * dx + dz * dz) <= float(radius) * float(radius)


class OvonODSkill:
    """Per-agent detection-head state for one assigned focus spec."""

    def __init__(self, owl_url: Optional[str] = None, shm_name: Optional[str] = None):
        url = owl_url or os.environ.get("OWL_URL", "")
        self.client = OwlClient(url, shm_name=shm_name) if url else None
        self.detected = False
        self.approaching = False
        self.goal_xz: Optional[Tuple[float, float]] = None
        self.box_xz: Optional[Tuple[float, float]] = None
        self.best_match = 0.0
        self.n_approach = 0
        self.n_reject = 0
        self.rejected_xz: List[Tuple[float, float]] = []
        self._goal_rgb: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.detected = False
        self.approaching = False
        self.goal_xz = None
        self.box_xz = None
        self.best_match = 0.0
        self.n_approach = 0
        self.n_reject = 0
        self.rejected_xz = []
        self._goal_rgb = None

    def _dump_best_match(
        self,
        crop: np.ndarray,
        goal_img: np.ndarray,
        score: float,
        spec: Dict[str, Any],
        box: Sequence[float],
        depth_m: Optional[float],
        frame: Optional[np.ndarray] = None,
    ) -> None:
        dump = (os.environ.get("IIN_SG_DUMP") or "").strip()
        if not dump:
            return
        os.makedirs(dump, exist_ok=True)
        stem = os.path.splitext(os.path.basename(spec.get("goal_image") or "iin"))[0]
        stem = stem.replace(" ", "_")
        crop_u8 = _as_uint8_rgb(crop)
        ref_u8 = _as_uint8_rgb(goal_img)
        if crop_u8 is not None:
            cv2.imwrite(os.path.join(dump, stem + "_query.png"), cv2.cvtColor(crop_u8, cv2.COLOR_RGB2BGR))
        if ref_u8 is not None:
            cv2.imwrite(os.path.join(dump, stem + "_ref.png"), cv2.cvtColor(ref_u8, cv2.COLOR_RGB2BGR))
        if frame is not None:
            vis = _as_uint8_rgb(frame)
            if vis is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(dump, stem + "_frame.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        with open(os.path.join(dump, stem + "_best.json"), "w") as f:
            json.dump(
                {
                    "score": float(score),
                    "box": [float(v) for v in box],
                    "depth": None if depth_m is None else float(depth_m),
                    "goal_image": spec.get("goal_image") or "",
                    "category": spec.get("category") or "",
                },
                f,
                indent=2,
            )

    def _goal_image(self, path: str) -> Optional[np.ndarray]:
        if self._goal_rgb is not None:
            return self._goal_rgb
        if path and os.path.isfile(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                self._goal_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._goal_rgb

    def _rejected(self, xz: Tuple[float, float]) -> bool:
        return any(_near_xz(xz, prev) for prev in self.rejected_xz)

    def _reject(self, xz: Tuple[float, float]) -> None:
        if self._rejected(xz):
            return
        self.rejected_xz.append((float(xz[0]), float(xz[1])))
        self.n_reject += 1
        if _near_xz(self.goal_xz, xz) and not self.detected:
            self.goal_xz = None
            self.box_xz = None
            self.approaching = False

    def _set_approach(
        self,
        xz: Tuple[float, float],
        box_xz: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        if not self.approaching or not _near_xz(self.goal_xz, xz):
            self.n_approach += 1
        self.approaching = True
        self.goal_xz = (float(xz[0]), float(xz[1]))
        if box_xz is not None:
            self.box_xz = (float(box_xz[0]), float(box_xz[1]))
        return self.goal_xz

    def update(
        self,
        rgb: np.ndarray,
        depth,
        spec: Dict[str, Any],
        pos_xz: Tuple[float, float],
        yaw: float,
        hfov_deg: float = 79.0,
        dmin: float = 0.1,
        dmax: float = 5.0,
    ) -> Optional[Tuple[float, float]]:
        if self.detected and self.goal_xz is not None:
            return self.goal_xz
        if self.client is None or rgb is None or depth is None:
            return self.goal_xz
        query = spec.get("owl_query") or spec.get("category") or ""
        hits = self.client.detect(rgb, query, OWL_THRESHOLD)
        if not hits:
            return self.goal_xz
        hits = sorted(hits, key=lambda x: x[1], reverse=True)
        mod = str(spec.get("modality") or "ON").upper()
        depth_m = habitat_depth_to_meters(depth, dmin, dmax)
        h, w = depth_m.shape[:2]
        candidate: Optional[Tuple[float, float]] = None
        candidate_box: Optional[Tuple[float, float]] = None
        for box, _score in hits:
            match = None
            crop = _crop_box(rgb, box)
            if mod == "IIN":
                goal_img = self._goal_image(spec.get("goal_image") or "")
                if crop is None or goal_img is None:
                    continue
                rh, rw = np.asarray(rgb).shape[:2]
                sg_crop = _crop_box(rgb, expand_box(box, rw, rh))
                if sg_crop is None:
                    sg_crop = crop
                match = self.client.match_image(sg_crop, goal_img)
                if match > self.best_match:
                    self.best_match = match
                    self._dump_best_match(
                        sg_crop, goal_img, match, spec, box,
                        depth_at_box(depth_m, box, dmax=dmax), frame=rgb,
                    )
            elif mod == "TN":
                desc = spec.get("goal_text") or spec.get("goal_description") or ""
                if crop is None or not desc:
                    continue
                match = self.client.score_lang(crop, desc)
                self.best_match = max(self.best_match, match)
            d = depth_at_box(depth_m, box, dmax=dmax)
            if d is None:
                continue
            cx = 0.5 * (float(box[0]) + float(box[2]))
            cy = 0.5 * (float(box[1]) + float(box[3]))
            xz = pixel_to_xz(cx, cy, d, pos_xz, yaw, w, h, hfov_deg=hfov_deg)
            if xz is None:
                continue
            if mod == "IIN" and self._rejected(xz):
                continue
            if mod == "IIN":
                if float(d) > APPROACH_RADIUS:
                    if candidate is None:
                        axz = pixel_to_xz(
                            cx, cy, approach_depth(d), pos_xz, yaw, w, h,
                            hfov_deg=hfov_deg,
                        )
                        candidate = axz if axz is not None else xz
                        candidate_box = xz
                    continue
                if gate_ok(mod, match) or iin_sticky_ok(self.best_match):
                    self.detected = True
                    self.approaching = False
                    self.goal_xz = xz
                    self.box_xz = xz
                    return xz
                self._reject(xz)
                continue
            if gate_ok(mod, match):
                self.detected = True
                self.approaching = False
                self.goal_xz = xz
                return xz
        if mod == "IIN" and candidate is not None:
            return self._set_approach(candidate, box_xz=candidate_box)
        return self.goal_xz

    def update_many(
        self,
        rgb: np.ndarray,
        depth,
        specs: Sequence[Tuple[int, Dict[str, Any]]],
        pos_xz: Tuple[float, float],
        yaw: float,
        hfov_deg: float = 79.0,
        dmin: float = 0.1,
        dmax: float = 5.0,
    ) -> Optional[int]:
        """Watch several specs. First committed hit wins; returns that goal index."""
        if self.detected and self.goal_xz is not None:
            return None
        kept = (
            self.detected, self.approaching, self.goal_xz, self.box_xz,
            self.best_match, list(self.rejected_xz),
        )
        for j, spec in specs:
            self._goal_rgb = None
            self.detected = False
            self.approaching = False
            self.goal_xz = None
            self.box_xz = None
            self.best_match = 0.0
            xz = self.update(rgb, depth, spec, pos_xz, yaw, hfov_deg, dmin, dmax)
            if self.detected and xz is not None:
                return int(j)
        (
            self.detected, self.approaching, self.goal_xz, self.box_xz,
            self.best_match, self.rejected_xz,
        ) = kept
        return None
