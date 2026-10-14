"""Benchmark generation constants: difficulty axes, buckets, strata, defaults.

These define the *controlled difficulty* layer. Every
generated episode is labelled along these axes and binned into a ``stratum`` so
results can be sliced ("when does cooperation / allocation help?").

Thresholds are deliberately centralised here so a single edit re-bins the whole
dataset deterministically.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

# --- HM3D-v2 ObjectNav categories (the 6 goal classes shipped with the dataset).
HM3D_CATEGORIES = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]

# --- Goal modalities (UniGoal-style heterogeneous goals).
#   ON  = object-category name        (always available)
#   IIN = instance image goal         (image rendered from a viewpoint -> P2)
#   TN  = text / language description  (free-text description         -> P2)
MODALITIES = ["ON", "IIN", "TN"]

# --- Difficulty axis: number of targets per episode (the unordered multi-object set).
DEFAULT_M_VALUES = [3, 4, 5]

# --- Spatial spread of the goal set: mean pairwise geodesic distance (metres).
#   Drives "does division of labour pay off" (clustered vs scattered targets).
SPREAD_BUCKETS = [("near", 0.0, 4.0), ("mid", 4.0, 8.0), ("far", 8.0, float("inf"))]

# --- Start spacing: realised minimum pairwise geodesic distance between the K
#   agent start points (metres). Drives "are starts co-located / dispersed".
START_SPACING_BUCKETS = [
    ("coloc", 0.0, 1.0),
    ("near", 1.0, 5.0),
    ("far", 5.0, float("inf")),
]

# --- Modality mix bucket from the per-goal modality tags.
#   all_on  : every goal is ON (easy)
#   mixed   : a blend of ON and IIN/TN
#   all_alt : every goal is IIN or TN (hard — no category-name shortcut)
def modality_bucket(tags) -> str:
    s = set(tags)
    if s == {"ON"}:
        return "all_on"
    if "ON" not in s:
        return "all_alt"
    return "mixed"


# --- "allocation-critical" flag.
#   True when a purely geometric (nearest-agent, load-blind) allocation yields a
#   team makespan this much worse than the oracle min-max routing. These are the
#   episodes where *semantic* allocation should beat geometric heuristics.
ALLOCATION_CRITICAL_GAP = 0.25  # relative makespan gap (geo - oracle)/oracle

# --- Start-point sampling.
START_MIN_ISLAND_RADIUS = 2.0    # reject starts on tiny disconnected islands
START_SAMPLE_ATTEMPTS = 4000     # rejection-sampling budget per episode's starts

# --- IIN goal-image camera spec (aligned to the eval-time RGB sensor in
#   habitat_extensions/config/objectnav_v2.yaml so a P2-rendered goal image
#   matches the distribution the agent actually observes). HM3D-IIN renders its
#   goal images from a stored camera pose with a calibrated pinhole camera; we
#   record the same so P2 can render from ``iin_viewpoint`` without guessing.
IIN_CAMERA = {"hfov": 79, "height_m": 0.88, "width_px": 640, "height_px": 480}
# Stretch robot RGB camera (OVON eval setting). Use at P2 render so SuperGlue sees the same
# intrinsics as the live crop (Locobot IIN_CAMERA is the frozen episode spec).
IIN_CAMERA_STRETCH = {"hfov": 42, "height_m": 1.31, "width_px": 360, "height_px": 640}
# Stretch embodiment, square issuer: same HFoV / camera height as above, 512x512.
# Used by --camera-profile stretch512. Does not replace IIN_CAMERA_STRETCH.
IIN_CAMERA_STRETCH_512 = {"hfov": 42, "height_m": 1.31, "width_px": 512, "height_px": 512}
# Official InstanceImageNav goal camera (Krantz et al. / Habitat Challenge).
# Independent of the eval robot: 512x512, HFoV in [60, 120]. height_m is unused
# because the render camera is placed in world coordinates and looks at the object.
IIN_CAMERA_OFFICIAL = {"hfov": 90, "height_m": 0.0, "width_px": 512, "height_px": 512}
IIN_OFFICIAL_HFOV_BINS = (60, 75, 90, 105, 120)
IIN_OFFICIAL_RADII_M = (0.5, 1.0, 1.5, 2.0)
IIN_OFFICIAL_HEIGHTS_M = (0.8, 1.15, 1.5)  # legacy discrete; official now samples U(0.8, 1.5)
IIN_OFFICIAL_HEIGHT_RANGE_M = (0.8, 1.5)
IIN_OFFICIAL_HFOV_RANGE_DEG = (60.0, 120.0)
IIN_OFFICIAL_LOOKAT_JITTER_DEG = 5.0
IIN_OFFICIAL_THETA_STEP_DEG = 10.0
IIN_OFFICIAL_MIN_OBJECT_IN_FRAME = 0.7  # Krantz c_o
IIN_OFFICIAL_MIN_FRAME_COVERAGE = 0.04  # fallback if OSA unavailable
IIN_OFFICIAL_CO_MIN = 0.7
IIN_OFFICIAL_CF_SLOPE = 0.0232
IIN_OFFICIAL_CF_INTERCEPT = 0.02
# GOAT LanguageNav issuer (Stretch embodiment, 512x512). Distinct from IIN.
TN_CAMERA_GOAT = {"hfov": 42, "height_m": 1.31, "width_px": 512, "height_px": 512}
TN_GOAT_RADII_M = (1.0, 1.5, 2.0)
TN_GOAT_THETA_STEP_DEG = 10.0
TN_GOAT_NEARBY_MAX_DEPTH_M = 4.5
TN_GOAT_NEARBY_MIN_COVERAGE = 0.005

# --- Goal viewpoint selection: keep the K best (highest-IoU) *navigable*
#   viewpoints per goal as its reachable target set; the best one is the
#   representative point used for goal-goal geodesics.
MAX_VIEWPOINTS_PER_GOAL = 8

# --- Principled budget: step/length budget scales with
#   the team-optimal route. ``budget_geo = alpha * oracle_makespan`` (metres),
#   convertible to a step budget by the runner using its step length.
BUDGET_ALPHA = 3.0

# --- Oracle makespan: brute-force min-max assignment is K**M; above this cap we
#   fall back to the balanced-greedy makespan and flag the value approximate.
ORACLE_BRUTE_FORCE_CAP = 100_000

# --- Scene-scale bucket from navmesh navigable area (m^2).
SCENE_SCALE_BUCKETS = [("small", 0.0, 40.0), ("medium", 40.0, 100.0), ("large", 100.0, float("inf"))]

# --- Cross-floor difficulty axis.
#   HM3D-v2 ObjectNav is ~3% must-cross-floor; HM3D-OVON ~38% (open-vocab rare
#   instances). We inherit OVON's vocab so the axis comes for free, and in the
#   multi-agent setting "which agent is on the goal's floor" becomes a
#   coordination signal. A goal is "cross-floor" w.r.t. an agent when the
#   vertical gap exceeds one floor height.
FLOOR_HEIGHT_M = 1.5             # min vertical gap (m) that counts as a floor change
FLOOR_SPAN_BUCKETS = [("flat", 0.0, FLOOR_HEIGHT_M),
                      ("split", FLOOR_HEIGHT_M, 2 * FLOOR_HEIGHT_M),
                      ("multi", 2 * FLOOR_HEIGHT_M, float("inf"))]

# --- Same-class distractor protocol (MultiON 2.0). For IIN/TN goals the agent
#   must reach the *specific* instance; other navigable instances of the same
#   category are distractors. We record them per goal so the success check and
#   false-positive metric can use them.
MAX_DISTRACTORS_PER_GOAL = 6


def _bucket(value: float, buckets) -> str:
    for name, lo, hi in buckets:
        if lo <= value < hi:
            return name
    return buckets[-1][0]


# --------------------------------------------------------------------------
# Difficulty labelling and stratum keys
#
# Difficulty labelling + stratum keys (controlled difficulty).
#
# Each episode is tagged along controlled axes and binned into a coarse ``stratum``
# used for balanced sampling. ``allocation_critical`` is kept as a *separate* flag
# (it is derived from routing and correlates with the other axes) so it does not
# fragment the strata, but the stratifier can still balance on it.
# --------------------------------------------------------------------------
def goal_spread(goal_goal: Sequence[Sequence[float]]) -> float:
    """Mean pairwise geodesic distance among goal reach-points (metres)."""
    gg = np.asarray(goal_goal, dtype=np.float64)
    M = gg.shape[0]
    if M < 2:
        return 0.0
    iu = np.triu_indices(M, k=1)
    vals = gg[iu]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else 0.0


def start_spacing(start_start: Sequence[Sequence[float]]) -> float:
    """Minimum pairwise geodesic distance among agent starts (metres)."""
    ss = np.asarray(start_start, dtype=np.float64)
    K = ss.shape[0]
    if K < 2:
        return 0.0
    iu = np.triu_indices(K, k=1)
    vals = ss[iu]
    vals = vals[np.isfinite(vals)]
    return float(vals.min()) if vals.size else 0.0


def cross_floor_stats(
    start_y: Optional[Sequence[float]],
    goal_y: Optional[Sequence[float]],
    start_goal: Optional[Sequence[Sequence[float]]] = None,
) -> dict:
    """Vertical (cross-floor) difficulty from start/goal heights.

    ``start_y`` (K agent start heights) and ``goal_y`` (M goal heights) are the
    world-frame y coordinates. ``start_goal`` (K x M geodesic matrix) lets us ask
    whether the *geodesically nearest* agent to each goal is on another floor —
    the multi-agent coordination signal ("send whoever's already upstairs").
    """
    H = FLOOR_HEIGHT_M
    if not start_y or not goal_y:
        return {
            "floor_span_m": 0.0, "floor_span_bucket": "flat",
            "cross_floor": False, "n_cross_floor_goals": 0,
            "nearest_agent_cross_floor": False,
        }
    sy = np.asarray(start_y, dtype=np.float64)
    gy = np.asarray(goal_y, dtype=np.float64)
    ally = np.concatenate([sy, gy])
    span = float(ally.max() - ally.min())
    # per-goal: gap to the *closest-in-height* agent (min over agents of |Δy|);
    # cross-floor when even the best-aligned agent is a floor away.
    gaps = np.abs(gy[:, None] - sy[None, :])  # M x K
    min_gap = gaps.min(axis=1)                # M
    n_cross = int((min_gap >= H).sum())
    cross_floor = bool(n_cross > 0)
    # nearest-by-distance agent on another floor (allocation-relevant)
    nearest_cross = False
    if start_goal is not None:
        sg = np.asarray(start_goal, dtype=np.float64)  # K x M
        for j in range(gy.shape[0]):
            col = sg[:, j]
            if np.all(~np.isfinite(col)):
                continue
            a = int(np.nanargmin(np.where(np.isfinite(col), col, np.inf)))
            if abs(float(gy[j] - sy[a])) >= H:
                nearest_cross = True
                break
    return {
        "floor_span_m": round(span, 3),
        "floor_span_bucket": _bucket(span, FLOOR_SPAN_BUCKETS),
        "cross_floor": cross_floor,
        "n_cross_floor_goals": n_cross,
        "nearest_agent_cross_floor": bool(nearest_cross),
    }


def assemble(
    *,
    num_agents: int,
    modality_tags: List[str],
    goal_goal: Sequence[Sequence[float]],
    start_start: Sequence[Sequence[float]],
    navigable_area: float,
    allocation_critical: bool,
    start_y: Optional[Sequence[float]] = None,
    goal_y: Optional[Sequence[float]] = None,
    start_goal: Optional[Sequence[Sequence[float]]] = None,
) -> dict:
    M = len(modality_tags)
    spread = goal_spread(goal_goal)
    spacing = start_spacing(start_start)
    spread_b = _bucket(spread, SPREAD_BUCKETS)
    spacing_b = _bucket(spacing, START_SPACING_BUCKETS)
    modality_b = modality_bucket(modality_tags)
    scale_b = _bucket(navigable_area, SCENE_SCALE_BUCKETS)
    floor = cross_floor_stats(start_y, goal_y, start_goal)

    from collections import Counter

    mix = dict(Counter(modality_tags))
    out = {
        "num_agents": int(num_agents),
        "M": int(M),
        "spatial_spread_geo": round(spread, 3),
        "spread_bucket": spread_b,
        "start_spacing_geo": round(spacing, 3),
        "start_spacing_bucket": spacing_b,
        "modality_mix": {k: int(v) for k, v in mix.items()},
        "modality_bucket": modality_b,
        "scene_navmesh_area": round(float(navigable_area), 2),
        "scene_scale_bucket": scale_b,
        "allocation_critical": bool(allocation_critical),
    }
    out.update(floor)
    return out


def heights_same_floor(ys, h: Optional[float] = None) -> bool:
    """True iff all heights lie in one floor band (default ``FLOOR_HEIGHT_M``)."""
    h = FLOOR_HEIGHT_M if h is None else float(h)
    vals = [float(y) for y in (ys or []) if y is not None]
    if len(vals) < 2:
        return True
    return (max(vals) - min(vals)) < h


def stratum_key(diff: dict) -> str:
    """Coarse balancing bucket. Excludes allocation_critical by design."""
    return "K{num_agents}-M{M}-spread_{spread_bucket}-start_{start_spacing_bucket}-{modality_bucket}".format(**diff)
