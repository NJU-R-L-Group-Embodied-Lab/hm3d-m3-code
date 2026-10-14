"""
Per-agent LA + VA decision helper for the multi-agent runner.

Wraps lavira's `VLMReasoningAgent` so the multi-agent loop can call:

    decision = ma_decision.decide(
        sim, agent_id, panorama, agent_state, vlm_agent,
        instruction, announcement_text, …
    )

and get back either:
  * NewTarget(world_xz, stop_approach=...)  → walk to the VA waypoint;
    if stop_approach, double-check after map-Euclidean arrival
  * Backtrack(waypoint_id)  → caller pops waypoint history
  * EpisodeStop is only produced by confirm_arrival_stop, not this same turn

Everything else (rotation panorama collection, FMM nav, sim stepping) lives
in the runner.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from vlnce_baselines.utils.depth_utils import get_world_xz_from_pixel


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeStop:
    reason: str = "double_check_stop"
    verification: Optional[Dict] = None
    goal_id: Optional[str] = None


@dataclass
class NewTarget:
    world_xz_m: Tuple[float, float]              # habitat (x, z) for sim ops
    direction: str  # 'forward' | 'left' | 'behind' | 'right'
    mapping_cell_rc: Optional[Tuple[int, int]] = None   # (row, col) in env's full_map
    description: str = ""
    bbox: Optional[Dict] = None
    bbox_frame_idx: int = 0
    progress_analysis: str = ""   # LA's "what I just saw / how it's going"
    reasoning_action: str = ""    # LA's "why this direction"
    stop_approach: bool = False   # LA stop=true: walk here, then double-check
    claim_goal_id: Optional[str] = None


@dataclass
class Backtrack:
    waypoint_id: int


Decision = "EpisodeStop | NewTarget | Backtrack"


@dataclass
class LaVaRaw:
    """LA + VA HTTP result. No Habitat sim access; projection happens later."""

    error: Optional[str] = None
    action: str = "NAVIGATE"
    stop_proposed: bool = False
    reported_claim_goal_id: str = ""
    direction: str = "forward"
    frame_idx: int = 0
    la_decision: Dict = field(default_factory=dict)
    bbox: Optional[Dict] = None
    progress_analysis: str = ""
    reasoning_action: str = ""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
DIRECTION_TO_FRAME_IDX = {
    "forward": 0,
    "left": 1,
    "behind": 2,
    "right": 3,
}


def yaw_from_quat(rotation) -> float:
    if hasattr(rotation, "w"):
        w, x, y, z = rotation.w, rotation.x, rotation.y, rotation.z
    else:
        w, x, y, z = rotation
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    return float(math.atan2(siny_cosp, cosy_cosp))


def build_camera_intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)


def habitat_full_pose(sim, agent_id: int) -> np.ndarray:
    """Return lavira-MAPPING-frame pose `[x_map, y_map, heading_deg]` for
    `get_world_xz_from_pixel` (which is written in mapping frame, not habitat).

    Mapping frame is defined by `habitat_extensions.pose_utils.get_sim_location`:
        x_map = -habitat_z
        y_map = -habitat_x
        heading positive = CCW from +x_map (= habitat -z = "north")
    `get_world_xz_from_pixel` internally calls `np.deg2rad(heading_rad)`, so
    despite the var name, it actually expects DEGREES."""
    # Reuse lavira's authoritative formula (`get_sim_location`) so the angle
    # convention matches `mapping.full_pose` exactly.
    import quaternion
    state = sim.get_agent_state(agent_id)
    x = -float(state.position[2])
    y = -float(state.position[0])
    axis = quaternion.as_euler_angles(state.rotation)[0]
    if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
        o = quaternion.as_euler_angles(state.rotation)[1]
    else:
        o = 2 * np.pi - quaternion.as_euler_angles(state.rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return np.array([x, y, math.degrees(float(o))], dtype=np.float32)


def preprocess_depth_meters(depth_norm: np.ndarray, min_d: float, max_d: float) -> np.ndarray:
    """Habitat normalized depth [0,1] → meters."""
    d = depth_norm.squeeze().astype(np.float32)
    return min_d + d * (max_d - min_d)


def normalized_bbox_depth_meters(
    panorama_frames: List[Dict],
    verification: Dict,
    min_depth_m: float,
    max_depth_m: float,
) -> Optional[float]:
    """Measure inset-median depth for a verified Qwen 0..999 current-view bbox."""

    try:
        view_index = int(verification.get("view_index"))
        bbox = [float(v) for v in verification.get("bbox_2d", [])]
    except (TypeError, ValueError):
        return None
    if not (0 <= view_index < len(panorama_frames)) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    if not (0.0 <= x1 < x2 <= 999.0 and 0.0 <= y1 < y2 <= 999.0):
        return None
    depth = np.asarray(panorama_frames[view_index].get("depth"), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.size == 0:
        return None
    height, width = depth.shape
    xlo = max(0, min(width - 1, int(math.floor(x1 * width / 999.0))))
    xhi = max(0, min(width, int(math.ceil(x2 * width / 999.0))))
    ylo = max(0, min(height - 1, int(math.floor(y1 * height / 999.0))))
    yhi = max(0, min(height, int(math.ceil(y2 * height / 999.0))))
    if xhi - xlo < 2 or yhi - ylo < 2:
        return None
    xpad = max(1, (xhi - xlo) // 6)
    ypad = max(1, (yhi - ylo) // 6)
    crop = depth[ylo + ypad:yhi - ypad, xlo + xpad:xhi - xpad]
    valid = crop[np.isfinite(crop) & (crop > 0.0) & (crop < 0.995)]
    if valid.size < 16:
        return None
    raw = float(np.median(valid))
    if raw <= 1.0:
        raw = float(min_depth_m) + raw * (float(max_depth_m) - float(min_depth_m))
    return raw


# --------------------------------------------------------------------------- #
# VLM-only LA + VA (no Habitat sim)
# --------------------------------------------------------------------------- #
def run_la_va(
    *,
    vlm_agent,
    instruction: str,
    visited_targets: List[Dict],
    panorama_frames: List[Dict],
    announcement_text: str = "",
    episode_id: Optional[str] = None,
    current_step: int = 0,
    history_images: Optional[List[Dict]] = None,
    target_reference_images: Optional[List[Dict]] = None,
    team_topdown_map=None,
    team_topdown_legend: Optional[str] = None,
    width: int = 640,
    height: int = 480,
    agent_id: int = 0,
    **_ignored: Any,
) -> LaVaRaw:
    """Call LA then VA. Safe to run on a worker thread (no sim)."""

    feedback = announcement_text or ""
    try:
        decision = vlm_agent.navigate_or_backtrack(
            instruction=instruction,
            visited_targets=visited_targets,
            feedback=feedback,
            episode_id=episode_id,
            step=current_step,
            history_images=history_images,
            target_reference_images=target_reference_images,
            team_topdown_map=team_topdown_map,
            team_topdown_legend=team_topdown_legend,
        )
    except Exception as e:
        print(f"[ma-decide][warn] navigate_or_backtrack failed agent_{agent_id}: {e!r}")
        return LaVaRaw(error=f"la:{e!r}")

    action = str(decision.get("action", "NAVIGATE")).upper()
    stop_proposed = action == "STOP" or bool(decision.get("stop_signal", False))
    reported_goal_id = str(decision.get("claim_goal_id") or "").strip()
    progress_analysis = str(decision.get("progress_analysis", "")).strip()
    reasoning_action = str(decision.get("reasoning_action", "")).strip()
    if action == "BACKTRACK":
        return LaVaRaw(
            action="BACKTRACK",
            stop_proposed=stop_proposed,
            reported_claim_goal_id=reported_goal_id,
            la_decision=decision,
            progress_analysis=progress_analysis,
            reasoning_action=reasoning_action,
        )

    from benchmark.pano_views import (
        direction_to_frame_idx,
        nearest_cardinal_name,
        parse_focus_view,
        parse_navigate_action,
    )

    n_frames = len(panorama_frames)
    raw_action = ""
    if isinstance(decision.get("action"), str):
        raw_action = decision.get("action")
    la_action = ""
    if isinstance(decision.get("la_action"), str):
        la_action = decision.get("la_action")
    frame_idx = parse_navigate_action(raw_action or la_action, n_frames)
    if frame_idx is None:
        frame_idx = parse_focus_view(
            decision.get("focus_view", decision.get("view_index")),
            n_frames,
        )
    direction = str(decision.get("direction", "forward")).lower()
    if frame_idx is None:
        if direction not in DIRECTION_TO_FRAME_IDX and direction not in (
            "forward", "left", "behind", "right",
        ):
            direction = "forward"
        frame_idx = direction_to_frame_idx(direction, n_frames)
    if frame_idx >= n_frames:
        frame_idx = 0
    direction = nearest_cardinal_name(frame_idx, n_frames)
    frame = panorama_frames[frame_idx]
    try:
        bbox = vlm_agent.query_llm(
            instruction=instruction,
            visited_targets=visited_targets,
            rgb_image=frame["rgb"],
            depth_image=frame["depth"],
            width=width,
            height=height,
            current_step=current_step,
            progress_analysis=progress_analysis,
            planned_action=("STOP" if stop_proposed else str(decision.get("action") or "NAVIGATE")),
            episode_id=episode_id,
        )
    except Exception as e:
        print(f"[ma-decide][warn] VA query_llm failed agent_{agent_id}: {e!r}")
        return LaVaRaw(
            error=f"va:{e!r}",
            action=action,
            stop_proposed=stop_proposed,
            reported_claim_goal_id=reported_goal_id,
            direction=direction,
            frame_idx=frame_idx,
            la_decision=decision,
            progress_analysis=progress_analysis,
            reasoning_action=reasoning_action,
        )
    return LaVaRaw(
        action=action,
        stop_proposed=stop_proposed,
        reported_claim_goal_id=reported_goal_id,
        direction=direction,
        frame_idx=frame_idx,
        la_decision=decision,
        bbox=bbox,
        progress_analysis=progress_analysis,
        reasoning_action=reasoning_action,
    )


DCS_REJECT_FEEDBACK = (
    "System: Your previous decision to STOP was rejected "
    "by the double-check mechanism because the target was "
    "not clearly visible or not close enough. Please "
    "navigate closer to the target or verify your position."
)


@dataclass
class AgentApiResult:
    """One agent's API pipeline: optional arrival DCS, then LA+VA."""

    dcs_should_stop: Optional[bool] = None
    dcs_verification: Optional[Dict] = None
    raw: Optional[LaVaRaw] = None


def run_agent_api_pipeline(
    *,
    need_dcs: bool = False,
    claim_goal_id: Optional[str] = None,
    stop_event_sink: Optional[Callable[[Dict], None]] = None,
    **la_va_kwargs: Any,
) -> AgentApiResult:
    """Same-agent order is DCS → LA → VA. Safe on a worker thread (no sim)."""

    instruction = str(la_va_kwargs.get("instruction") or "")
    dcs_should_stop = None
    dcs_verification = None
    if need_dcs:
        should_stop, verification = confirm_arrival_stop(
            vlm_agent=la_va_kwargs.get("vlm_agent"),
            instruction=instruction,
            panorama_frames=la_va_kwargs.get("panorama_frames") or [],
            visited_targets=la_va_kwargs.get("visited_targets") or [],
            episode_id=la_va_kwargs.get("episode_id"),
            step=la_va_kwargs.get("current_step"),
            history_images=la_va_kwargs.get("history_images"),
            target_reference_images=la_va_kwargs.get("target_reference_images"),
            claim_goal_id=claim_goal_id,
            stop_event_sink=stop_event_sink,
            agent_id=int(la_va_kwargs.get("agent_id") or 0),
        )
        dcs_should_stop = bool(should_stop)
        dcs_verification = dict(verification or {})
        if not should_stop:
            instruction = (instruction + "\n" + DCS_REJECT_FEEDBACK).strip()
            la_va_kwargs = dict(la_va_kwargs)
            la_va_kwargs["instruction"] = instruction
    raw = run_la_va(**la_va_kwargs)
    return AgentApiResult(
        dcs_should_stop=dcs_should_stop,
        dcs_verification=dcs_verification,
        raw=raw,
    )


def run_agent_api_parallel(
    jobs: List[Dict],
    max_workers: Optional[int] = None,
) -> List[AgentApiResult]:
    """Concurrent per-agent pipelines. Each job keeps DCS→LA→VA on one worker."""

    n = len(jobs)
    if n == 0:
        return []
    workers = n if max_workers is None or int(max_workers) <= 0 else int(max_workers)
    workers = max(1, min(n, workers))
    dcs_n = sum(1 for job in jobs if job.get("need_dcs"))
    if workers == 1 or n == 1:
        return [run_agent_api_pipeline(**job) for job in jobs]
    print(f"[ma-vlm] parallel n={n} workers={workers} dcs={dcs_n}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_agent_api_pipeline, **job) for job in jobs]
        return [fut.result() for fut in futures]


def run_la_va_parallel(
    jobs: List[Dict],
    max_workers: Optional[int] = None,
) -> List[LaVaRaw]:
    """Run independent LA+VA HTTP jobs. ``max_workers<=1`` or one job stays serial."""

    n = len(jobs)
    if n == 0:
        return []
    workers = n if max_workers is None or int(max_workers) <= 0 else int(max_workers)
    workers = max(1, min(n, workers))
    if workers == 1 or n == 1:
        return [run_la_va(**job) for job in jobs]
    print(f"[ma-vlm] parallel n={n} workers={workers}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_la_va, **job) for job in jobs]
        return [fut.result() for fut in futures]


def project_la_va_to_decision(
    *,
    raw: LaVaRaw,
    sim=None,
    agent_id: int,
    panorama_frames: List[Dict],
    width: int,
    height: int,
    hfov_deg: float,
    depth_min: float,
    depth_max: float,
    mapping_pose_override: Optional[np.ndarray] = None,
    initial_habitat_yaw_rad: float = 0.0,
    agent_habitat_override: Optional[np.ndarray] = None,
    active_goal_id: Optional[str] = None,
) -> "EpisodeStop | NewTarget | Backtrack | None":
    """Bbox → world. Must run on the Habitat thread."""

    if raw.action == "BACKTRACK":
        wp = (raw.la_decision or {}).get("waypoint", 0)
        if isinstance(wp, int):
            return Backtrack(wp)
        return None
    if raw.error or raw.bbox is None:
        return None

    direction = raw.direction if raw.direction in DIRECTION_TO_FRAME_IDX else "forward"
    frame_idx = int(raw.frame_idx)
    if frame_idx >= len(panorama_frames):
        frame_idx = 0
    frame = panorama_frames[frame_idx]
    bbox = raw.bbox
    x1 = int(bbox.get("x1", 0))
    x2 = int(bbox.get("x2", width))
    y2 = int(bbox.get("y2", height))
    pixel_u = int((x1 + x2) / 2)
    pixel_v = min(y2, height - 1)
    depth_m = preprocess_depth_meters(frame["depth"], depth_min, depth_max)
    if mapping_pose_override is not None:
        pose = np.asarray(mapping_pose_override, dtype=np.float32).copy()
    else:
        pose = habitat_full_pose(sim, agent_id)
    direction_offset_deg = {
        "forward": 0.0,
        "left": 90.0,
        "behind": 180.0,
        "right": -90.0,
    }[direction]
    try:
        frame_angle = float(frame.get("angle", direction_offset_deg))
    except (TypeError, ValueError):
        frame_angle = direction_offset_deg
    pose_adj = pose.copy()
    pose_adj[2] = pose[2] + frame_angle
    K = build_camera_intrinsics(width, height, hfov_deg)
    try:
        wxz = get_world_xz_from_pixel(
            pixel_coords=(pixel_u, pixel_v),
            depth_image=depth_m,
            full_pose=pose_adj,
            camera_intrinsics=K,
        )
    except Exception as e:
        print(f"[ma-decide][warn] bbox→world failed agent_{agent_id}: {e!r}")
        return None
    if wxz is None or not np.all(np.isfinite(wxz)):
        return None

    target_x_map = float(wxz[0])
    target_y_map = float(wxz[1])
    res_cm = 5.0
    target_row = int(target_y_map * 100.0 / res_cm)
    target_col = int(target_x_map * 100.0 / res_cm)
    if agent_habitat_override is not None:
        agent_habitat = np.asarray(agent_habitat_override, dtype=np.float32)
    else:
        agent_state = sim.get_agent_state(agent_id)
        agent_habitat = np.asarray(agent_state.position, dtype=np.float32)
    dx_map = target_x_map - float(pose[0])
    dy_map = target_y_map - float(pose[1])
    s0 = math.sin(initial_habitat_yaw_rad)
    c0 = math.cos(initial_habitat_yaw_rad)
    d_world_x = -s0 * dx_map - c0 * dy_map
    d_world_z = -c0 * dx_map + s0 * dy_map
    habitat_x = float(agent_habitat[0]) + d_world_x
    habitat_z = float(agent_habitat[2]) + d_world_z
    return NewTarget(
        world_xz_m=(habitat_x, habitat_z),
        direction=direction,
        mapping_cell_rc=(target_row, target_col),
        description=bbox.get("target", ""),
        bbox=bbox,
        bbox_frame_idx=frame_idx,
        progress_analysis=raw.progress_analysis,
        reasoning_action=raw.reasoning_action,
        stop_approach=bool(raw.stop_proposed),
        claim_goal_id=str(active_goal_id or "") or None,
    )


# --------------------------------------------------------------------------- #
# Main decision entrypoint
# --------------------------------------------------------------------------- #
def decide(
    *,
    sim,
    agent_id: int,
    panorama_frames: List[Dict],
    visited_targets: List[Dict],
    vlm_agent,
    instruction: str,
    width: int,
    height: int,
    hfov_deg: float,
    depth_min: float,
    depth_max: float,
    current_step: int,
    announcement_text: str = "",
    episode_id: Optional[str] = None,
    history_images: Optional[List[Dict]] = None,
    target_reference_images: Optional[List[Dict]] = None,
    active_goal_id: Optional[str] = None,
    stop_max_depth_m: float = 1.0,
    stop_event_sink: Optional[Callable[[Dict], None]] = None,
    mapping_pose_override: Optional[np.ndarray] = None,
    initial_habitat_yaw_rad: float = 0.0,
    team_topdown_map=None,
    team_topdown_legend: Optional[str] = None,
) -> "EpisodeStop | NewTarget | Backtrack | None":
    """Run LA decision, STOP panorama verification, then VA on `agent_id`.

    Returns one of the result types above (or None if the decision was a
    BACKTRACK we can't act on yet).
    """
    def emit_stop_event(event_type: str, **extra) -> None:
        if stop_event_sink is None:
            return
        event = {
            "event_type": str(event_type),
            "agent_id": int(agent_id),
            "step": int(current_step),
            "goal_id": str(active_goal_id or ""),
        }
        try:
            position = np.asarray(sim.get_agent_state(agent_id).position, dtype=np.float32)
            event["attempt_position"] = [float(x) for x in position]
        except Exception:
            event["attempt_position"] = None
        event.update(extra)
        try:
            stop_event_sink(event)
        except Exception as exc:
            print(f"[ma-decide][warn] stop event sink failed: {exc!r}")

    raw = run_la_va(
        vlm_agent=vlm_agent,
        instruction=instruction,
        visited_targets=visited_targets,
        panorama_frames=panorama_frames,
        announcement_text=announcement_text,
        episode_id=episode_id,
        current_step=current_step,
        history_images=history_images,
        target_reference_images=target_reference_images,
        team_topdown_map=team_topdown_map,
        team_topdown_legend=team_topdown_legend,
        width=width,
        height=height,
        agent_id=agent_id,
    )
    if raw.stop_proposed:
        if not active_goal_id and raw.reported_claim_goal_id:
            active_goal_id = raw.reported_claim_goal_id
        emit_stop_event("la_stop_proposed")
    return project_la_va_to_decision(
        raw=raw,
        sim=sim,
        agent_id=agent_id,
        panorama_frames=panorama_frames,
        width=width,
        height=height,
        hfov_deg=hfov_deg,
        depth_min=depth_min,
        depth_max=depth_max,
        mapping_pose_override=mapping_pose_override,
        initial_habitat_yaw_rad=initial_habitat_yaw_rad,
        active_goal_id=active_goal_id,
    )


def confirm_arrival_stop(
    *,
    vlm_agent,
    instruction: str,
    panorama_frames: List[Dict],
    visited_targets: List[Dict],
    episode_id: Optional[str] = None,
    step: Optional[int] = None,
    history_images: Optional[List[Dict]] = None,
    target_reference_images: Optional[List[Dict]] = None,
    claim_goal_id: Optional[str] = None,
    stop_event_sink: Optional[Callable[[Dict], None]] = None,
    agent_id: int = 0,
) -> Tuple[bool, Dict]:
    """LaViRA post-arrival double-check: visible and ~1 m. No depth gate."""

    def emit(event_type: str, **extra) -> None:
        if stop_event_sink is None:
            return
        event = {
            "event_type": str(event_type),
            "agent_id": int(agent_id),
            "step": int(step or 0),
            "goal_id": str(claim_goal_id or ""),
        }
        event.update(extra)
        try:
            stop_event_sink(event)
        except Exception as exc:
            print(f"[ma-decide][warn] arrival stop event failed: {exc!r}")

    stop_references = target_reference_images
    if claim_goal_id and target_reference_images:
        stop_references = [
            item for item in target_reference_images
            if str(item.get("goal_id") or "") == str(claim_goal_id)
        ]
    stop_instruction = instruction
    if claim_goal_id:
        stop_instruction += (
            f"\nClaim under verification: {claim_goal_id}. "
            "Verify only this target using the current views."
        )
    try:
        should_stop, verification = vlm_agent.double_check_stop(
            stop_instruction,
            panorama_frames,
            visited_targets,
            episode_id=episode_id,
            step=step,
            history_images=history_images,
            target_reference_images=stop_references,
            claim_goal_id=claim_goal_id,
        )
    except Exception as exc:
        print(f"[ma-decide][warn] arrival double_check failed agent_{agent_id}: {exc!r}")
        emit("double_check_error", error=type(exc).__name__)
        return False, {}
    verification = dict(verification or {})
    if should_stop:
        emit("double_check_confirmed", verification=dict(verification))
        return True, verification
    emit("double_check_rejected", verification=dict(verification))
    return False, verification
