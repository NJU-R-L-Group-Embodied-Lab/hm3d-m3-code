"""Panorama view layout for LaViRA MA decisions.

Default is the LaViRA 12×30° spin, keeping every 3rd frame (4 cardinals).
Stretch-42 FOV runs use 8×45° and keep all 8 frames.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

CARDINALS = {
    0: ("forward", "FORWARD"),
    90: ("left", "LEFT"),
    180: ("behind", "BEHIND"),
    270: ("right", "RIGHT"),
}

DIRECTION_TO_ANGLE = {
    "forward": 0,
    "left": 90,
    "behind": 180,
    "right": 270,
}


def spin_plan(pano_n: int = 4, pano_stride_deg: float = 90.0) -> Tuple[int, float, int]:
    """Return (n_turns, turn_deg, keep_every)."""
    n = int(pano_n)
    stride = float(pano_stride_deg)
    if n == 4 and abs(stride - 90.0) < 1e-6:
        return 12, 30.0, 3
    if n <= 0 or abs(n * stride - 360.0) > 1.0:
        return 12, 30.0, 3
    return n, stride, 1


def align_spin_frames(frames: Sequence[dict]) -> List[dict]:
    if not frames:
        return []
    return [frames[-1]] + list(frames[:-1])


def keep_stride(frames: Sequence[dict], keep_every: int) -> List[dict]:
    return list(frames[:: max(1, int(keep_every))])


def frame_idx_for_angle(angle_deg: float, n_frames: int) -> int:
    n = max(1, int(n_frames))
    stride = 360.0 / float(n)
    return int(round(float(angle_deg) / stride)) % n


def direction_to_frame_idx(direction: str, n_frames: int) -> int:
    key = str(direction or "forward").lower()
    return frame_idx_for_angle(DIRECTION_TO_ANGLE.get(key, 0), n_frames)


def view_specs(n_frames: int, *, prefix: str = "Current") -> List[Dict]:
    n = max(1, int(n_frames))
    stride = 360.0 / float(n)
    out: List[Dict] = []
    for i in range(n):
        angle = int(round(i * stride)) % 360
        name, alias = CARDINALS.get(angle, (f"view_{i}", ""))
        if alias:
            label = f"{prefix} view {i} ({angle}°, {alias}):"
        else:
            label = f"{prefix} view {i} ({angle}°):"
        out.append({"index": i, "angle": angle, "name": name, "label": label})
    return out


def view_index_legend(n_frames: int) -> str:
    bits = []
    for spec in view_specs(n_frames):
        alias = CARDINALS.get(spec["angle"], ("", ""))[1]
        extra = f" {alias}" if alias else ""
        bits.append(f"{spec['index']}={spec['angle']}°{extra}".rstrip())
    return ", ".join(bits)


def parse_focus_view(value, n_frames: int) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < int(n_frames):
        return idx
    return None


def _depth_center_mean(depth_image) -> Optional[float]:
    if depth_image is None:
        return None
    import numpy as np

    d_img = depth_image[:, :, 0] if len(getattr(depth_image, "shape", ())) == 3 else depth_image
    try:
        h, w = d_img.shape[:2]
    except Exception:
        return None
    center = d_img[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    valid = center > 0.01
    if not valid.any():
        return None
    return float(center[valid].mean())


def blocked_frame_indices(panorama_images: Sequence[dict], threshold: float = 0.15) -> set:
    """Frame indices whose center depth looks too close to walk into."""
    blocked = set()
    for i, frame in enumerate(panorama_images or []):
        mean_d = _depth_center_mean(frame.get("depth") if isinstance(frame, dict) else None)
        if mean_d is not None and mean_d < float(threshold):
            blocked.add(int(i))
    return blocked


def action_line(spec: Dict) -> str:
    alias = CARDINALS.get(int(spec["angle"]), ("", ""))[1]
    extra = f", {alias}" if alias else ""
    return (
        f"   - navigate to view {spec['index']} ({spec['angle']}°{extra}) "
        "- face this view and go forward"
    )


def navigate_action_choices(
    n_frames: int,
    blocked_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[str], str]:
    """Prompt bullets and JSON action enum for an N-view panorama."""
    blocked = {int(i) for i in (blocked_indices or [])}
    items: List[str] = []
    tokens: List[str] = []
    for spec in view_specs(n_frames):
        if int(spec["index"]) in blocked:
            continue
        items.append(action_line(spec))
        tokens.append("navigate to view %d" % int(spec["index"]))
    if not items:
        spec0 = view_specs(n_frames)[0]
        items.append(action_line(spec0) + " (Emergency)")
        tokens.append("navigate to view 0")
    return items, "|".join(tokens)


def parse_navigate_action(action: str, n_frames: int) -> Optional[int]:
    """Map an LA action string to a panorama frame index."""
    import re

    text = str(action or "").lower().strip()
    if not text or text.startswith("backtrack"):
        return None
    n = max(1, int(n_frames))
    m = re.search(r"view\s*[_\-\s]?\s*(\d+)", text)
    if m:
        idx = int(m.group(1))
        return idx if 0 <= idx < n else None
    m = re.search(r"(\d+)\s*(?:°|deg)", text)
    if m:
        return frame_idx_for_angle(int(m.group(1)), n)
    if re.search(r"\bforward\b", text):
        return direction_to_frame_idx("forward", n)
    if re.search(r"\bbehind\b", text):
        return direction_to_frame_idx("behind", n)
    if re.search(r"\bleft\b", text):
        return direction_to_frame_idx("left", n)
    if re.search(r"\bright\b", text):
        return direction_to_frame_idx("right", n)
    return None


def nearest_cardinal_name(frame_idx: int, n_frames: int) -> str:
    spec = None
    for item in view_specs(n_frames):
        if int(item["index"]) == int(frame_idx):
            spec = item
            break
    if spec is None:
        return "forward"
    name = str(spec.get("name") or "")
    if name in DIRECTION_TO_ANGLE:
        return name
    angle = float(spec["angle"])
    best = "forward"
    best_d = 360.0
    for deg, key in ((0, "forward"), (90, "left"), (180, "behind"), (270, "right")):
        d = min((angle - deg) % 360.0, (deg - angle) % 360.0)
        if d < best_d:
            best_d = d
            best = key
    return best


def views_phrase(n_frames: int, stride_deg: float = 90.0) -> str:
    if int(n_frames) == 8 and abs(float(stride_deg) - 45.0) < 1e-6:
        return "current 8-view 45° panorama"
    if int(n_frames) == 4:
        return "current 4-directional views"
    return "current %d-view panorama" % int(n_frames)


def doublecheck_phrase(n_frames: int, stride_deg: float = 90.0) -> str:
    if int(n_frames) == 8 and abs(float(stride_deg) - 45.0) < 1e-6:
        return "An 8-view 45° panorama double-check will verify that claimed target."
    return "A four-direction panorama double-check will verify that claimed target."


def _action_space(agent):
    cfg = getattr(agent, "agent_config", None)
    return getattr(cfg, "action_space", None) if cfg is not None else None


def _is_turn_key(key) -> bool:
    if "turn" in str(key).lower():
        return True
    try:
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        return int(key) in (
            int(HabitatSimActions.TURN_LEFT),
            int(HabitatSimActions.TURN_RIGHT),
        )
    except Exception:
        return False


def snapshot_turn_amounts(sim) -> List[Tuple[object, str, float]]:
    snaps: List[Tuple[object, str, float]] = []
    n = 1
    hc = getattr(sim, "habitat_config", None)
    if hc is not None:
        n = int(getattr(hc, "NUM_AGENTS", 1) or 1)
    get_agent = getattr(sim, "get_agent", None)
    if get_agent is None:
        return snaps
    for i in range(n):
        try:
            agent = get_agent(i)
        except Exception:
            continue
        space = _action_space(agent)
        if not space:
            continue
        for key in list(space.keys()):
            spec = space[key]
            act = getattr(spec, "actuation", None)
            if act is None or not hasattr(act, "amount"):
                continue
            if _is_turn_key(key) or _is_turn_key(getattr(spec, "name", "")):
                snaps.append((act, "amount", float(act.amount)))
    return snaps


def apply_turn_amounts(snaps: Sequence[Tuple[object, str, float]], deg: float) -> None:
    for act, attr, _old in snaps:
        setattr(act, attr, float(deg))


def restore_turn_amounts(snaps: Sequence[Tuple[object, str, float]]) -> None:
    for act, attr, old in snaps:
        setattr(act, attr, float(old))


@contextmanager
def temporary_turn_deg(sim, deg: float):
    """Patch habitat-sim turn actuation for the panorama spin, then restore."""
    snaps = snapshot_turn_amounts(sim)
    changed = bool(snaps) and abs(float(deg) - 30.0) > 1e-6
    if changed:
        apply_turn_amounts(snaps, deg)
    try:
        yield bool(snaps) or abs(float(deg) - 30.0) < 1e-6
    finally:
        if changed:
            restore_turn_amounts(snaps)


def yaw_left_rotation(rotation, deg: float):
    """Return habitat [x, y, z, w] after a left yaw of ``deg``."""
    import quaternion as qt
    import numpy as np

    if hasattr(rotation, "w"):
        q = rotation
    else:
        x, y, z, w = [float(v) for v in rotation]
        q = qt.quaternion(w, x, y, z)
    delta = qt.from_rotation_vector(np.array([0.0, math.radians(float(deg)), 0.0]))
    new_q = delta * q
    return [float(new_q.x), float(new_q.y), float(new_q.z), float(new_q.w)]


def rotate_agent_left(sim, agent_id: int, deg: float) -> None:
    state = sim.get_agent_state(agent_id)
    pos = state.position
    rot = yaw_left_rotation(state.rotation, deg)
    sim.set_agent_state(
        [float(pos[0]), float(pos[1]), float(pos[2])],
        rot,
        agent_id,
    )
