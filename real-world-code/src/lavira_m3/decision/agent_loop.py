"""Platform-agnostic LA → VA → waypoint → arrival DCS / FOUND loop."""

from __future__ import annotations

import copy
import logging
import math
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ..contracts import NavResult, RobotRuntime
from . import config as _cfg
from .board_format import parse_goal_index
from .config import (
    REAR_LA_DIRS,
    STEP_MAX_M,
    STEP_MIN_M,
    VA_MAX_DIM,
    LLMConfig,
    RobotConfig,
    board_all_found,
    claim_max_m_for,
    history_root,
    la_action_dirs,
    load_goals,
    short_hop_m,
    step_max_m,
    t0_joint_enabled,
    t0_leader_name,
    t0_static,
    t0_wait_s,
    view_mask_min_depth_m,
)
from .interrupt import requested
from .llm_router import LLMRouter
from .map_geometry import fused_map_aabb, in_world_xy
from .model_messages import (
    build_la_content,
    build_va_content,
    encode_image,
    image_part,
    parse_json_object,
)
from .nav_round import ROUND_S, short_hop_candidates
from .navigation_geometry import (
    already_within_waypoint,
    heading_step,
    heading_yaw,
    inside_body_envelope,
    ray_step,
    reach_stop_message,
)
from .navigation_history import (
    copy_view_for_archive,
    history_public,
    history_rgb_period_s,
    save_front_image,
    save_history_manifest,
    waypoint_has_rgb,
)
from .navigation_results import (
    json_public,
    navigation_succeeded,
    parse_map_xy,
    planner_path_from_result,
    trace_polyline,
)
from .observations import MissingViewsError, require_all_views
from .path_geometry import in_map_progress_point, path_is_bad_loop
from .prompts_real import (
    AFTER_BACKTRACK_FEEDBACK,
    AFTER_FOUND_SWITCH_FEEDBACK,
    BACKTRACK_STUCK_FEEDBACK,
    NAV_BLOCKED_FEEDBACK,
    SCB_FAIL_HIDE_CYCLES,
    SHORT_HOP_VA_IOU_MAX,
    backtrack_prompt_block,
    dcs_reject_feedback,
    history_text,
    image_goal_scenes_block,
    la_backtrack_replan_prompt,
    la_prompt,
    parse_backtrack_id,
    parse_direction,
    scb_walk_failed_feedback,
    short_hop_la_feedback,
    short_hop_va_feedback,
    stop_check_prompt,
    va_prompt,
)
from .t0_joint import (
    collect_poses,
    fetch_scene,
    rasterize_map_2d,
    run_t0_la,
)
from .team_client import TeamClient
from .vision_geometry import (
    CAM_YAW,
    bbox_ground_px,
    bbox_inset_median_depth,
    bbox_iou,
    blocked_prompt_note,
    fallback_va_bbox,
    legal_la_directions,
    normalize_va_target,
    usable_va_bbox,
    view_center_mean_depth,
    view_mask_blocked,
)
from .visual_projection import project_visual_target

HTTP_REACH_M = float(getattr(_cfg, "HTTP_REACH_M", 0.75))
ROS1_REACH_M = float(getattr(_cfg, "ROS1_REACH_M", 0.75))
ROS2_REACH_M = float(getattr(_cfg, "ROS2_REACH_M", 0.75))

TRACE_IMAGE_DIM = 480
_UNPREPARED_TARGET = object()


log = logging.getLogger("lavira_m3.agent")


class T0BarrierError(RuntimeError):
    """T0 joint observations incomplete or failed; refuse independent LA/nav."""


def wait_board_goals(board: TeamClient, poll_s: float = 1.0) -> List[Dict]:
    """Block until the webpage submits goals. No LA/VA/nav before started=true."""
    log.info("waiting for webpage submit on board %s", getattr(board, "hub", ""))
    while True:
        if requested():
            raise KeyboardInterrupt
        hello = board.t0_hello()
        if hello.get("ok") is False:
            log.warning("T0 hello failed: %s", hello.get("error"))
        snap = board.snapshot()
        goals = list(snap.get("goals") or [])
        started = bool(snap.get("started"))
        if snap.get("ok") is not False and started and goals:
            log.info("loaded %d goals after webpage submit", len(goals))
            return goals
        if snap.get("ok") is False:
            log.warning("board snapshot failed: %s", snap.get("error"))
        time.sleep(max(0.2, float(poll_s)))


class LaviraM3Agent:
    def __init__(
        self,
        robot: RobotConfig,
        runtime: RobotRuntime,
        goals: List[Dict],
        llm: Optional[LLMRouter] = None,
        from_board: bool = False,
    ):
        self.robot = robot
        self.runtime = runtime
        self._ra_started = None
        self._ra_time_ms = None
        self.goals = goals
        self.from_board = from_board
        self.llm = llm or LLMRouter(LLMConfig())
        self.board = TeamClient(robot.hub, robot.name)
        self._trace_queue = queue.Queue(maxsize=128)
        self._trace_worker = None
        self._trace_stop = threading.Event()
        self.history: List[Dict] = []
        self.history_session = time.strftime("%Y%m%d_%H%M%S")
        self._bind_llm_logger()
        self._hist_lock = threading.Lock()
        self._nav_leg_start: Optional[int] = None
        self.step = 0
        self.stop_feedback = ""
        self.nav_feedback = ""
        self.board_ok = False
        self.last_va_depth = None
        self.last_va_depth_estimated = False
        self.last_va_search_step = False
        self.last_va_heading_step = False
        self.last_va_requested_xy = None
        self.t0_joint = t0_joint_enabled()
        self.t0_consumed = False
        self._last_pano: Optional[List[Dict[str, Any]]] = None
        self._origin_views: Optional[List[Dict[str, Any]]] = None
        self._last_la: Optional[Dict[str, Any]] = None
        self._last_va: Optional[Dict[str, Any]] = None
        self._last_va_view: Optional[Dict[str, Any]] = None
        self._short_hop = False
        self._short_travel_streak = 0
        self._forbid_la_dirs: List[str] = []
        self._scb_fail_hide: Dict[int, int] = {}

    def _bind_llm_logger(self) -> None:
        binder = getattr(self.llm, "bind_call_logger", None)
        if not callable(binder):
            return
        from .llm_call_log import LlmCallLogger

        folder = os.path.join(history_root(), self.robot.name, self.history_session)
        logger = LlmCallLogger(self.robot.name, folder, self.history_session)
        logger.advertise()
        binder(logger)
        self._llm_call_logger = logger

    def _note_llm(self, kind: str) -> None:
        note = getattr(self.llm, "note_record", None)
        if callable(note):
            note(kind, self.step)

    def setup(self) -> None:
        advertise = getattr(getattr(self, "_llm_call_logger", None), "advertise", None)
        if callable(advertise):
            advertise()
        if self.from_board:
            self.goals = wait_board_goals(self.board)
            self.board_ok = bool(self.goals)
            self.board.t0_hello()
            return
        snap = self.board.init_goals(self.goals)
        self.board_ok = snap.get("ok") is not False
        if not self.board_ok:
            log.error("board init failed: %s", snap.get("error"))
        self.board.t0_hello()

    def _image_goal_scenes_text(self) -> str:
        return image_goal_scenes_block(self._live_goals())

    def _claim_max_m(self) -> float:
        z = getattr(self.robot, "found_max_depth_m", None)
        if z is not None:
            return float(z)
        return claim_max_m_for(self.robot.name)

    def _instruction(self) -> str:
        found = []
        try:
            snap = self.board.snapshot()
            found = list(snap.get("found_by") or [])
            goals = list(snap.get("goals") or self.goals or [])
        except Exception:
            goals = list(self.goals or [])
        parts = []
        for i, g in enumerate(goals):
            idx = int(g.get("goal_index", i))
            if idx < len(found) and found[idx] not in ("", None, -1, "none"):
                continue
            gid = f"goal_{idx}"
            line = f"{gid} [{g.get('modality', 'ON')}]: {g.get('object_category')}"
            if g.get("tn_text"):
                line += f"; {g['tn_text']}"
            elif str(g.get("modality") or "").upper() == "IIN":
                if g.get("reference_image"):
                    line += "; match the attached reference image"
                else:
                    line += "; reference image unavailable"
            parts.append(line)
        if not parts:
            return "All board goals are found. Do not keep searching found objects."
        return "Find remaining unresolved targets:\n" + "\n".join(parts)

    def _la(self, frames: List[Dict]) -> Dict:
        blocks = self.board.prompt_blocks()
        self.board_ok = bool(blocks.get("available"))
        if not self.board_ok:
            log.warning("board unavailable: %s", blocks.get("error"))
        min_z = view_mask_min_depth_m()
        blocked, depths = view_mask_blocked(frames, self.robot.directions, min_z)
        legal = legal_la_directions(self.robot.directions, blocked, exclude_rear=True)
        if self._forbid_la_dirs:
            filtered = [d for d in legal if d not in self._forbid_la_dirs]
            if filtered:
                legal = filtered
        note = blocked_prompt_note(blocked, depths, min_z)
        if blocked:
            log.info("LA view mask robot=%s blocked=%s legal=%s", self.robot.name, blocked, legal)
        extras = [self.stop_feedback, self.nav_feedback]
        if self._short_hop:
            extras.append(short_hop_la_feedback(self._last_la, self._last_va))
        if self._forbid_la_dirs:
            extras.append(
                "System: Prefer a different legal direction than %s."
                % ", ".join(self._forbid_la_dirs)
            )
        bt_items = self._available_backtrack_items()
        self._tick_scb_fail_hides()
        extra_actions: List[str] = []
        backtrack_note = ""
        if bt_items:
            extra_actions = ["backtrack to %s" % i for i, _xy, _dist in bt_items]
            backtrack_note = backtrack_prompt_block(bt_items, step_max_m())
            if self._short_travel_streak >= 2 or not legal:
                extras.append(BACKTRACK_STUCK_FEEDBACK)
        prompt = la_prompt(
            f"agent_{self.robot.name}",
            legal,
            blocks.get("board") or "",
            blocks.get("progress") or "",
            extra_feedback="\n".join(x for x in extras if str(x).strip()),
            hardware_dirs=self.robot.directions,
            blocked_note=note,
            current_step=self.step,
            instruction=self._instruction(),
            claim_max_m=self._claim_max_m(),
            image_goal_scenes=self._image_goal_scenes_text(),
            t0_sector=str(blocks.get("t0_sector") or ""),
            teammate_live=str(blocks.get("teammate_live") or ""),
            extra_actions=extra_actions,
            backtrack_note=backtrack_note,
        )
        self._remember_origin_views(frames)
        content = build_la_content(
            prompt, self._origin_views or [], self.history, frames, self._reference_parts()
        )
        self._note_llm("la")
        raw = self.llm.generate_la([{"role": "user", "content": content}])
        data = parse_json_object(raw)
        data["_raw"] = raw
        self.stop_feedback = ""
        self.nav_feedback = ""
        return data

    def _goal_image(self, goal: Dict):
        spec = str(goal.get("reference_image") or "")
        if not spec:
            return None
        try:
            return self.board.fetch_image(spec)
        except Exception:
            log.exception("load reference image failed: %s", spec)
            return None

    def _live_goals(self) -> List[Dict]:
        try:
            snap = self.board.snapshot()
            goals = list(snap.get("goals") or self.goals or [])
        except Exception:
            goals = list(self.goals or [])
        return goals

    def _reference_parts(
        self,
        claim_goal_id: Optional[str] = None,
        max_dim: Optional[int] = None,
    ) -> List[Dict]:
        goals = self._live_goals()
        claim_idx = parse_goal_index(claim_goal_id, goals) if claim_goal_id else None
        ordered: List[Dict] = []
        if claim_idx is not None:
            ordered.append(goals[claim_idx])
        for g in goals:
            if claim_idx is not None and int(g.get("goal_index", -999)) == int(claim_idx):
                continue
            ordered.append(g)
        parts: List[Dict] = []
        attached = 0
        for g in ordered:
            if not g.get("reference_image"):
                continue
            img = self._goal_image(g)
            if img is None:
                continue
            gid = g.get("goal_index")
            label = "goal_%s reference image (identity only)" % gid
            if claim_idx is not None and int(gid if gid is not None else -999) == int(claim_idx):
                label = "Claimed goal_%s reference image (identity only)" % gid
            parts.append({"type": "text", "text": label})
            parts.append(image_part(img, max_dim) if max_dim else image_part(img))
            attached += 1
        if parts:
            parts.insert(0, {"type": "text", "text": "Target reference images (identity only):"})
            log.info("attached %s IIN reference image(s) claim=%s", attached, claim_goal_id)
        elif any(str(g.get("modality") or "").upper() == "IIN" for g in goals):
            log.warning("IIN goal(s) on the board but no reference image loaded for VA/DCS")
        return parts

    def _t0_payload(self, frames: List[Dict]) -> List[Dict]:
        out = []
        for fr in frames:
            rec = {"name": fr.get("name") or "unknown"}
            if fr.get("rgb") is not None:
                rec["jpeg_b64"] = encode_image(fr["rgb"])
            z = view_center_mean_depth(fr.get("depth"))
            if z is not None:
                rec["center_depth_m"] = round(float(z), 3)
            out.append(rec)
        return out

    def _t0_references(self, goals: List[Dict]):
        refs = []
        for g in goals or self._live_goals():
            img = self._goal_image(g)
            if img is not None:
                refs.append((f"goal_{g.get('goal_index', 0)}", img))
        return refs

    def _run_t0_leader(self) -> None:
        locked = self.board.t0_lock()
        if locked.get("ok") is False or not locked.get("acquired"):
            return
        bundle = self.board.t0_bundle()
        if bundle.get("ok") is False:
            self.board.t0_fail(str(bundle.get("error") or "T0 bundle failed"))
            return
        scene = fetch_scene(self.robot.nav_url)
        observes = dict(bundle.get("observes") or {})
        map_image = rasterize_map_2d(scene, collect_poses(observes, scene))
        goals = list(bundle.get("goals") or self.goals)
        robots = [
            n
            for n in (bundle.get("online") or bundle.get("robots") or bundle.get("expected") or [])
            if n
        ]
        try:
            assignments, team, raw = run_t0_la(
                self.llm,
                observes,
                str(bundle.get("board") or ""),
                map_image,
                self._t0_references(goals),
                robots=robots or None,
            )
        except Exception as exc:
            log.exception("T0 joint LA call failed")
            self.board.t0_fail(str(exc))
            return
        if not assignments:
            self.board.t0_fail("T0 joint LA returned no valid assignments")
            return
        posted = self.board.t0_assign(assignments, team, raw)
        if posted.get("ok") is False:
            self.board.t0_fail(str(posted.get("error") or "T0 assign failed"))

    def _t0_or_wait(self, frames: List[Dict]) -> Dict:
        payload = self._t0_payload(frames)
        blank = [str(fr.get("name") or "unknown") for fr in payload if not fr.get("jpeg_b64")]
        if blank:
            raise T0BarrierError(
                "%s T0 observe missing images %s; refusing navigation" % (self.robot.name, blank)
            )
        posted = self.board.t0_observe(
            self.runtime.pose(),
            list(self.robot.directions),
            payload,
        )
        if posted.get("ok") is False:
            raise T0BarrierError(
                "%s T0 observe rejected: %s; refusing navigation"
                % (self.robot.name, posted.get("error") or "unknown")
            )
        deadline = time.time() + t0_wait_s()
        ready_seen = False
        while True:
            if requested():
                raise KeyboardInterrupt
            status = self.board.t0_status()
            la = status.get("la")
            if status.get("assigned"):
                if isinstance(la, dict) and la.get("action"):
                    rec = dict(la)
                    if status.get("la_time_ms") is not None:
                        rec["_t0_la_time_ms"] = status.get("la_time_ms")
                    self._emit(
                        "T0 assignment action=%s la_time_ms=%s",
                        rec.get("action"),
                        rec.get("_t0_la_time_ms"),
                    )
                    return rec
                have = list((status.get("assignments") or {}).keys())
                log.warning(
                    "T0 already assigned without %s (assignments=%s); falling back to independent LA",
                    self.robot.name,
                    have,
                )
                return self._la(frames)
            if status.get("failed"):
                raise T0BarrierError(
                    "T0 joint failed: %s; refusing navigation" % status.get("failed")
                )
            if status.get("ready"):
                ready_seen = True
                leader = str(status.get("leader") or t0_leader_name())
                if self.robot.name == leader and not status.get("assigned"):
                    self._run_t0_leader()
                    time.sleep(0.5)
                    continue
            elif not ready_seen and time.time() >= deadline:
                missing = list(status.get("missing") or [])
                offline = list(status.get("offline") or [])
                raise T0BarrierError(
                    "T0 incomplete after %.0fs; missing observations from %s (offline=%s); refusing navigation"
                    % (t0_wait_s(), missing, offline)
                )
            time.sleep(1.0)

    def _va(self, view: Dict, direction: str, progress: str) -> Dict:
        rgb = view["rgb"]
        height, width = rgb.shape[:2]
        extra = short_hop_va_feedback(self._last_va, direction) if self._short_hop else ""
        blocks = {}
        try:
            blocks = self.board.prompt_blocks() or {}
        except Exception:
            blocks = {}
        prompt = va_prompt(
            self._instruction(),
            direction,
            progress,
            current_step=self.step,
            width=width,
            height=height,
            image_goal_scenes=self._image_goal_scenes_text(),
            extra_feedback=extra,
            t0_sector=str(blocks.get("t0_sector") or ""),
            teammate_live=str(blocks.get("teammate_live") or ""),
        )
        content = build_va_content(prompt, view, self._reference_parts(max_dim=VA_MAX_DIM))
        self._note_llm("va")
        raw = self.llm.generate_va([{"role": "user", "content": content}])
        data = parse_json_object(raw)
        data["_raw"] = raw
        return data

    def _project(
        self, view: Dict, bbox: List[float], direction: str, *, force_search=False
    ) -> Optional[Tuple[float, float]]:
        projection = project_visual_target(
            self.robot, self.runtime, view, bbox, direction, force_search=force_search
        )
        self.last_va_depth = projection.depth
        self.last_va_depth_estimated = projection.depth_estimated
        self.last_va_search_step = projection.search_step
        self.last_va_map_point = projection.map_point
        self.last_va_requested_xy = projection.requested_xy
        return projection.waypoint

    def _claim_goal(self, claim: str) -> Optional[Dict]:
        goals = self._live_goals()
        idx = parse_goal_index(claim, goals)
        if idx is None:
            return None
        return goals[idx]

    def _dcs(self, frames: List[Dict], claim: str) -> Tuple[bool, Dict]:
        goal = self._claim_goal(claim) or {}
        prompt = stop_check_prompt(
            self._instruction(),
            claim,
            claim_max_m=self._claim_max_m(),
            goal=goal,
        )
        content = [{"type": "text", "text": prompt}]
        modality = str(goal.get("modality") or "").upper()
        if modality == "IIN":
            ref = self._goal_image(goal)
            if ref is not None:
                content.append(
                    {
                        "type": "text",
                        "text": "Reference image for the claimed target (identity only)",
                    }
                )
                content.append(image_part(ref))
        for fr in frames:
            if fr.get("rgb") is None:
                continue
            content.append({"type": "text", "text": f"Current {fr['name']}"})
            content.append(image_part(fr["rgb"]))
        raw = self.llm.generate_la([{"role": "user", "content": content}], max_tokens=256)
        data = parse_json_object(raw)
        data["_raw"] = raw
        ok = str(data.get("decision", "")).upper() == "FOUND"
        log.info(
            "DCS claim=%s ok=%s %s", claim, ok, {k: data.get(k) for k in ("reasoning", "decision")}
        )
        return ok, data

    def _dcs_measured_depth(self, frames: List[Dict], dcs: Dict) -> Optional[float]:
        """Closest valid inset-median depth (m) of the claimed box, else last measured VA depth."""
        bbox = dcs.get("bbox_2d")
        zs: List[float] = []
        if usable_va_bbox(bbox):
            for fr in frames:
                rgb = fr.get("rgb")
                if rgb is None or fr.get("depth") is None:
                    continue
                h, w = rgb.shape[:2]
                if not usable_va_bbox(bbox, w, h):
                    continue
                z = bbox_inset_median_depth(fr.get("depth"), list(bbox), w, h)
                if z is not None:
                    zs.append(float(z))
        if zs:
            return min(zs)
        z = getattr(self, "last_va_depth", None)
        if z is not None and not getattr(self, "last_va_depth_estimated", False):
            return float(z)
        return None

    def _select_claim_view(
        self,
        frames: List[Dict],
        direction: str = "",
        dcs: Optional[Dict] = None,
    ) -> Tuple[Optional[Dict], str]:
        named: Dict[str, Dict] = {}
        for fr in frames or []:
            name = str(fr.get("name") or "").strip()
            if name and fr.get("rgb") is not None:
                named[name] = fr
        if not named:
            return None, ""
        dcs = dcs or {}
        hinted = str(dcs.get("camera") or dcs.get("view") or "").strip().lower().replace(" ", "_")
        if hinted in named:
            return named[hinted], hinted
        direction = str(direction or "").strip()
        if direction in named:
            return named[direction], direction
        bbox = dcs.get("bbox_2d")
        best: Optional[Tuple[float, Dict, str]] = None
        if usable_va_bbox(bbox):
            for name, fr in named.items():
                rgb = fr.get("rgb")
                try:
                    h, w = rgb.shape[:2]
                except Exception:
                    continue
                if not usable_va_bbox(bbox, w, h):
                    continue
                z = None
                if fr.get("depth") is not None:
                    z = bbox_inset_median_depth(fr.get("depth"), list(bbox), w, h)
                score = 0.0 if z is None else 1.0 / (1.0 + float(z))
                if best is None or score > best[0]:
                    best = (score, fr, name)
            if best is not None:
                return best[1], best[2]
        if "front" in named:
            return named["front"], "front"
        name, fr = next(iter(named.items()))
        return fr, name

    def _publish_found_view(
        self,
        frames: List[Dict],
        goal_id: str,
        dcs: Dict,
        direction: str = "",
    ) -> None:
        view, camera = self._select_claim_view(frames, direction, dcs)
        if view is None:
            return
        rgb = view.get("rgb")
        image_b64 = ""
        image_size = None
        try:
            import numpy as np

            arr = rgb if isinstance(rgb, np.ndarray) else np.asarray(rgb)
            if getattr(arr, "ndim", 0) >= 2 and arr.size > 4:
                image_b64 = encode_image(arr, TRACE_IMAGE_DIM)
                image_size = [int(arr.shape[1]), int(arr.shape[0])]
        except Exception:
            log.exception("encode FOUND %s view failed", goal_id)
        bbox = dcs.get("bbox_2d")
        if not usable_va_bbox(bbox):
            bbox = None
        va_pub = {
            "target": str(goal_id),
            "found": True,
            "camera": camera,
            "decision": dcs.get("decision"),
            "reasoning": dcs.get("reasoning"),
        }
        if bbox:
            va_pub["bbox_2d"] = list(bbox)
        pixel = None
        if image_size and bbox:
            pixel = list(bbox_ground_px(list(bbox), image_size[0], image_size[1]))
        if not hasattr(self.board, "post_trace"):
            return
        try:
            self._trace(
                va=va_pub,
                va_bbox=list(bbox) if bbox else None,
                va_pixel=pixel,
                va_image=image_b64,
                image_size=image_size,
                direction=camera,
                found_goal_id=goal_id,
            )
        except Exception:
            log.exception("FOUND view trace failed")

    def _goal_still_open(self, goal_id: str) -> bool:
        snap = self.board.snapshot()
        goals = snap.get("goals") or self.goals or []
        found = snap.get("found_by") or []
        idx = parse_goal_index(goal_id, goals)
        if idx is None:
            return False
        if idx >= len(found):
            return True
        return found[idx] in ("", None, -1, "none")

    def _handle_claim(self, goal_id: str, frames: List[Dict], direction: str = "") -> bool:
        """DCS-only FOUND on the current panorama. True when every goal is found."""
        if t0_static():
            return False
        if not self._goal_still_open(goal_id):
            log.info("claim %s skipped; already found or unknown", goal_id)
            return board_all_found(self.board.snapshot())
        dcs_ok, dcs = self._dcs(frames, goal_id)
        depth = self._dcs_measured_depth(frames, dcs)
        log.info(
            "FOUND dcs-only claim=%s z=%s dcs_ok=%s",
            goal_id,
            "none" if depth is None else "%.2f" % float(depth),
            dcs_ok,
        )
        if not dcs_ok:
            log.info(
                "claim %s rejected dcs=%s depth=%s",
                goal_id,
                {k: dcs.get(k) for k in ("reasoning", "decision")},
                depth,
            )
            self.stop_feedback = dcs_reject_feedback(self._claim_max_m())
            return False
        self._publish_found_view(frames, goal_id, dcs, direction)
        xy = self._pose_xy()
        posted = self.board.post_found(
            goal_id,
            {
                "source": "dcs",
                "depth_m": depth,
                "dcs": {
                    "decision": dcs.get("decision"),
                    "reasoning": dcs.get("reasoning"),
                },
            },
            xy=list(xy) if xy else None,
        )
        if posted.get("ok") is False:
            log.error("FOUND %s but board write failed: %s", goal_id, posted.get("error"))
            return False
        log.info("FOUND %s via DCS", goal_id)
        snap = posted if posted.get("found_by") is not None else self.board.snapshot()
        return board_all_found(snap)

    def _emit(self, msg: str, *args) -> None:
        text = msg % args if args else msg
        log.info("%s", text)
        try:
            import rospy

            rospy.loginfo("[lavira] " + text)
        except Exception:
            pass

    def _plan_ready(self, path, xy=None) -> None:
        """Freeze RA at the first returned trajectory, before any motion wait."""
        if len(path or []) < 2 or self._ra_started is None or self._ra_time_ms is not None:
            return
        self._ra_time_ms = round((time.perf_counter() - self._ra_started) * 1000.0, 1)
        try:
            self._trace(ra_time_ms=self._ra_time_ms, va_path=path, waypoint=xy)
        except Exception:
            log.exception("planning-time trace failed")

    def _trace(self, **kwargs) -> None:
        payload = {"step": self.step}
        payload.update(kwargs)
        if self.robot.navigation_backend == "http":
            payload["trace_created_ns"] = time.time_ns()
            if self._trace_worker is None:
                self._trace_worker = threading.Thread(
                    target=self._send_traces, name="board-trace", daemon=True
                )
                self._trace_worker.start()
            try:
                self._trace_queue.put_nowait(copy.deepcopy(payload))
            except queue.Full:
                log.warning(
                    "board trace queue full; dropping telemetry step=%s (motion is not blocked)",
                    self.step,
                )
            return
        posted = self.board.post_trace(payload)
        if posted.get("ok") is False:
            log.warning("trace post failed: %s", posted.get("error"))

    def _send_traces(self) -> None:
        while not self._trace_stop.is_set():
            payload = self._trace_queue.get()
            try:
                if payload is None:
                    return
                posted = self.board.post_trace(payload)
                if posted.get("ok") is False:
                    log.warning(
                        "trace post failed step=%s: %s", payload["step"], posted.get("error")
                    )
            except Exception:
                log.exception("background trace post failed")
            finally:
                self._trace_queue.task_done()

    def _publish_va_path(self, path, xy=None) -> None:
        """Post the planner polyline as soon as preview exists, before blocking walk."""
        cleaned = trace_polyline(path)
        if not cleaned or not hasattr(self.board, "post_trace"):
            return
        waypoint = None
        if xy is not None and len(xy) >= 2:
            try:
                waypoint = [float(xy[0]), float(xy[1])]
            except (TypeError, ValueError):
                waypoint = None
        try:
            self._trace(va_path=cleaned, waypoint=waypoint)
        except Exception:
            log.exception("early va_path trace failed")

    def _approach_points(
        self, xy: Tuple[float, float], pose: Optional[Tuple[float, float, float]]
    ) -> List[Tuple[float, float]]:
        """Original goal first, then points walking back toward the robot (nearest-to-goal first)."""
        points = [xy]
        if pose is None:
            return points
        x0, y0, _yaw = pose
        dx, dy = xy[0] - x0, xy[1] - y0
        dist = math.hypot(dx, dy)
        if dist <= 0.70:
            return points
        t = dist - 0.15
        while t >= 0.70:
            s = t / dist
            points.append((x0 + dx * s, y0 + dy * s))
            t -= 0.15
        out: List[Tuple[float, float]] = []
        seen = set()
        for pt in points:
            key = (round(pt[0], 2), round(pt[1], 2))
            if key in seen:
                continue
            seen.add(key)
            out.append(pt)
        return out

    def _sector_approach_points(
        self, xy: Tuple[float, float], pose: Optional[Tuple[float, float, float]]
    ) -> List[Tuple[float, float]]:
        """Generate a small left/right sector after the straight-line fallback fails."""
        if pose is None:
            return []
        x0, y0, _yaw = pose
        dx, dy = xy[0] - x0, xy[1] - y0
        dist = math.hypot(dx, dy)
        if dist <= 0.70:
            return []
        base = math.atan2(dy, dx)
        radii = [max(0.70, dist - delta) for delta in (0.30, 1.00, 2.00, 3.00)]
        points: List[Tuple[float, float]] = []
        for angle_deg in (15.0, -15.0, 30.0, -30.0, 45.0, -45.0):
            angle = base + math.radians(angle_deg)
            ca, sa = math.cos(angle), math.sin(angle)
            for radius in radii:
                points.append((x0 + radius * ca, y0 + radius * sa))
        out: List[Tuple[float, float]] = []
        seen = set()
        for pt in points:
            key = (round(pt[0], 2), round(pt[1], 2))
            if key in seen:
                continue
            seen.add(key)
            out.append(pt)
        return out

    def _cap_step(
        self, xy: Tuple[float, float], pose: Optional[Tuple[float, float, float]]
    ) -> Tuple[float, float]:
        """Do not pre-clip walk length. Push too-close / in-body VA along heading."""
        if pose is None:
            return xy
        target = (float(xy[0]), float(xy[1]))
        have_real_depth = self.last_va_depth is not None and not self.last_va_depth_estimated
        if have_real_depth:
            self.last_va_heading_step = False
            return target
        inside = inside_body_envelope(
            target[0],
            target[1],
            pose[0],
            pose[1],
            pose[2],
            str(self.robot.name),
        )
        dx, dy = target[0] - pose[0], target[1] - pose[1]
        dist = math.hypot(dx, dy)
        if inside or dist < float(STEP_MIN_M):
            if inside and self.robot.navigation_backend == "http":
                out = ray_step(pose[0], pose[1], target[0], target[1], STEP_MIN_M, pose[2])
            else:
                out = heading_step(pose[0], pose[1], pose[2], STEP_MIN_M)
            self.last_va_heading_step = True
            self._emit(
                "heading step %.2f m (inside=%s dist=%.2f) -> (%.2f, %.2f)",
                STEP_MIN_M,
                inside,
                dist,
                out[0],
                out[1],
            )
            return out
        self.last_va_heading_step = False
        return target

    def _bound_goal(
        self, xy: Tuple[float, float], pose: Optional[Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Discard a VA that sits outside the fixed 100 m canvas; caller re-runs LA+VA."""
        del pose
        if in_world_xy(xy[0], xy[1]):
            return xy
        self._emit("discard VA (%.2f, %.2f) outside 100 m map; new LA+VA", xy[0], xy[1])
        return None

    def _preview_path(
        self,
        x: float,
        y: float,
        yaw: float,
        pose: Optional[Tuple[float, float, float]],
    ) -> List[Tuple[float, float]]:
        nav = getattr(self.runtime, "nav", None)
        if nav is None:
            return []
        if hasattr(nav, "preview_plan"):
            try:
                return list(nav.preview_plan(x, y, yaw, pose) or [])
            except Exception:
                log.exception("preview_plan failed")
                return []
        navigate = getattr(nav, "navigate_to", None)
        if navigate is None:
            return []
        try:
            result = navigate(x, y, yaw, execute=False)
        except TypeError:
            return []
        except Exception:
            log.exception("Aloha preview failed")
            return []
        return list(planner_path_from_result(result) or [])

    def _trim_loop_path(
        self,
        xy: Tuple[float, float],
        pose: Optional[Tuple[float, float, float]],
    ) -> Tuple[float, float]:
        if pose is None:
            return xy
        yaw = heading_yaw(xy[0], xy[1], *pose)
        path = self._preview_path(xy[0], xy[1], yaw, pose)
        if not path:
            return xy
        scene = {}
        try:
            scene = fetch_scene(self.robot.nav_url)
        except Exception:
            scene = {}
        aabb = fused_map_aabb(scene)
        if not path_is_bad_loop(path, pose[0], pose[1], xy[0], xy[1], aabb):
            return xy
        step = in_map_progress_point(path, pose[0], pose[1], aabb, STEP_MIN_M, STEP_MAX_M)
        if step is None:
            step = heading_step(pose[0], pose[1], pose[2], STEP_MIN_M)
            self.last_va_heading_step = True
            self._emit(
                "reject out-of-map/loop path; heading fallback (%.2f, %.2f)",
                step[0],
                step[1],
            )
            return step
        self._emit("reject out-of-map/loop path; in-map step (%.2f, %.2f)", step[0], step[1])
        return self._cap_step(step, pose)

    def _goal_ok(self, xy: Tuple[float, float], pose: Optional[Tuple[float, float, float]]) -> bool:
        if pose is None:
            return True
        if math.hypot(xy[0] - pose[0], xy[1] - pose[1]) < float(STEP_MIN_M) - 1e-3:
            return False
        return not inside_body_envelope(
            xy[0], xy[1], pose[0], pose[1], pose[2], str(self.robot.name)
        )

    def _prepare_go_target(
        self, view: Dict, bbox: List[float], direction: str, *, force_search=False
    ):
        rgb = view.get("rgb")
        if rgb is not None:
            h, w = rgb.shape[:2]
            if not usable_va_bbox(bbox, w, h):
                force_search = True
                bbox = fallback_va_bbox(w, h)
        xy = self._project(view, bbox, direction, force_search=force_search)
        pose = self.runtime.pose()
        self.last_va_selection_pose = pose
        self.last_va_heading_step = False
        if xy is None:
            return None
        xy = self._bound_goal(self._cap_step(xy, pose), pose)
        if xy is not None:
            xy = self._bound_goal(self._trim_loop_path(xy, pose), pose)
        if xy is not None:
            self.last_va_requested_xy = (float(xy[0]), float(xy[1]))
        return xy

    def _go(
        self, view: Dict, bbox: List[float], direction: str, *, prepared_xy=_UNPREPARED_TARGET
    ) -> Tuple[Optional[Tuple[float, float]], NavResult]:
        nav = getattr(self.runtime, "nav", None)
        previous = getattr(nav, "round_deadline", None)
        self._nav_round_deadline = time.monotonic() + ROUND_S
        if nav is not None:
            nav.round_deadline = self._nav_round_deadline
        try:
            return self._go_with_budget(view, bbox, direction, prepared_xy=prepared_xy)
        except BaseException:
            if nav is not None:
                nav.cancel()
            raise
        finally:
            if nav is not None:
                nav.round_deadline = previous

    def _go_with_budget(
        self, view: Dict, bbox: List[float], direction: str, *, prepared_xy=_UNPREPARED_TARGET
    ) -> Tuple[Optional[Tuple[float, float]], NavResult]:
        if self._nav_leg_start is None:
            self._nav_leg_start = len(self.history)
        self._ra_started = time.perf_counter()
        self._ra_time_ms = None
        if getattr(self.runtime, "nav", None) is not None:
            self.runtime.nav.on_plan = self._plan_ready
            if self.robot.navigation_backend == "http":
                self.runtime.nav.on_execution = lambda nav: self._trace(nav=nav)
        xy = (
            self._prepare_go_target(view, bbox, direction)
            if prepared_xy is _UNPREPARED_TARGET
            else prepared_xy
        )
        if xy is None:
            result = NavResult(False, "no_waypoint", "no validated VA target; new LA+VA")
            self._emit("navigate %s", result.as_dict())
            return None, result
        pose = self.runtime.pose()
        try:
            self._trace(waypoint=[float(xy[0]), float(xy[1])])
        except Exception:
            log.exception("walk-target waypoint trace failed")
        reach = self._walk_reach_m()
        if already_within_waypoint(pose, xy[0], xy[1], reach_m=reach):
            self._emit("filtered walk target within %.2f m; skip navigation", reach)
            return xy, NavResult(True, "succeeded", reach_stop_message(pose, xy[0], xy[1], reach))
        result = NavResult(False, "no_waypoint", "no candidate")
        used = xy
        best_xy, best = None, None
        static = t0_static()
        candidates = [xy]
        if self.last_va_search_step:
            self._emit(
                "search_step_no_depth navigate (%.2f, %.2f); not a target approach", xy[0], xy[1]
            )
        for candidate_index, cand in enumerate(candidates):
            if time.monotonic() >= self._nav_round_deadline - 1.0:
                result = NavResult(
                    False, "timed_out", "30 s navigation budget exhausted; new LA+VA"
                )
                break
            cand = self._bound_goal(cand, pose)
            if cand is None:
                continue
            yaw = heading_yaw(cand[0], cand[1], *(pose or (cand[0], cand[1], 0.0)))
            preview = self._preview_path(cand[0], cand[1], yaw, pose)
            if preview:
                self._publish_va_path(preview, cand)
            requested = cand
            if self.robot.navigation_backend == "http" and self.last_va_search_step:
                result = self._navigate_with_history(
                    cand[0],
                    cand[1],
                    yaw,
                    exploration=True,
                    requested_goal=requested,
                )
            else:
                result = self._navigate_with_history(
                    cand[0], cand[1], yaw, requested_goal=requested
                )
            self._emit(
                "navigate try (%.2f, %.2f) -> %s",
                cand[0],
                cand[1],
                result.as_dict() if hasattr(result, "as_dict") else result,
            )
            if self.robot.navigation_backend == "http" and getattr(result, "target", None):
                cand = (float(result.target[0]), float(result.target[1]))
            planned = planner_path_from_result(result)
            if self.robot.navigation_backend == "http" and planned:
                self._publish_va_path(planned, cand)
            if planned and best is None:
                best_xy, best = (
                    cand,
                    result
                    if isinstance(result, NavResult)
                    else NavResult(bool(result), "unknown", ""),
                )
                if static:
                    return cand, best
            if navigation_succeeded(result):
                if candidate_index > 0 and isinstance(result, NavResult):
                    result.extra["fallback_from"] = list(xy)
                    result.message = "short safe hop completed; new LA+VA: " + result.message
                return cand, result if isinstance(result, NavResult) else NavResult(
                    True, "unknown", ""
                )
            if static or getattr(result, "state", "") not in (
                "planning_failed",
                "planning_timeout",
                "stalled",
                "blocked",
            ):
                break
            used = cand
            # Do not spend the rest of this round retrying after useful motion.
            now_pose = self.runtime.pose()
            if (
                pose is not None
                and now_pose is not None
                and math.hypot(now_pose[0] - pose[0], now_pose[1] - pose[1]) >= 0.30
            ):
                break
            nav = getattr(self.runtime, "nav", None)
            if (
                getattr(result, "execution_id", None) is not None
                and getattr(result, "state", "") != "stalled"
            ):
                cancelled = nav.cancel() if nav is not None else None
                if cancelled is not None and not cancelled.ok:
                    result = NavResult(
                        False, "cancel_failed", "cannot replace active task: " + cancelled.message
                    )
                    break
            if candidate_index == 0:
                timeout = min(2.0, self._nav_round_deadline - 1.0 - time.monotonic())
                if timeout <= 0:
                    break
                if self.robot.navigation_backend == "ros1":
                    hops = nav.short_hop_candidates(now_pose, xy)
                    hops = [
                        hop for hop in hops if math.hypot(hop[0] - cand[0], hop[1] - cand[1]) > 0.05
                    ]
                else:
                    scene = fetch_scene(self.robot.nav_url, timeout=timeout)
                    hops = short_hop_candidates(now_pose, xy, scene, str(self.robot.name))
                candidates.extend(hops)
                self._emit(
                    "original target %s; %d safe 1.5 m fallback candidates, %.1f s left",
                    getattr(result, "state", "failed"),
                    len(hops),
                    max(0.0, self._nav_round_deadline - time.monotonic()),
                )
                pose = now_pose
        if static and best is not None:
            return best_xy, best
        if self.robot.navigation_backend == "http":
            return None, result if isinstance(result, NavResult) else NavResult(
                bool(result), "unknown", ""
            )
        return used, result if isinstance(result, NavResult) else NavResult(
            bool(result), "unknown", ""
        )

    def _save_front_disk(self, rec: Dict[str, Any]) -> None:
        if save_front_image(rec, self.robot.name, self.history_session):
            self._save_history_manifest()

    def _save_history_manifest(self) -> None:
        save_history_manifest(self.history, self.robot.name, self.history_session)

    def _post_history(self, rec: Dict[str, Any], include_image: bool = True) -> None:
        payload = history_public(rec)
        if include_image and rec.get("rgb") is not None:
            try:
                payload["front_image"] = encode_image(rec["rgb"], TRACE_IMAGE_DIM)
            except Exception:
                log.exception("encode front history failed waypoint=%s", rec.get("waypoint_id"))
        posted = None
        if hasattr(self.board, "post_history"):
            posted = self.board.post_history(payload)
        elif hasattr(self.board, "post_trace"):
            posted = {"ok": True}
        if isinstance(posted, dict) and posted.get("ok") is False:
            log.warning("history post failed: %s", posted.get("error"))

    def _grab_front_rgb(self):
        """Latest body-front RGB only. Never a stopped-waypoint panorama sweep."""
        cap = getattr(self.runtime, "capture_direction", None)
        if cap is None:
            return None
        try:
            view = cap("front")
        except Exception:
            log.exception("front RGB grab failed")
            return None
        if not isinstance(view, dict):
            return None
        return view.get("rgb")

    def _append_nav_history_frame(self, kind: str) -> Optional[Dict[str, Any]]:
        rgb = self._grab_front_rgb()
        pose = None
        try:
            pose = self.runtime.pose()
        except Exception:
            pose = None
        if rgb is None:
            log.warning("%s missing front RGB during %s", self.robot.name, kind)
            return None
        with self._hist_lock:
            rec = {
                "waypoint_id": len(self.history),
                "step": self.step,
                "rgb": rgb,
                "pose": pose,
                "action": "",
                "target": "",
                "xy": None,
                "nav_ok": None,
                "kind": kind,
                "t": time.time(),
            }
            self.history.append(rec)
        self._save_front_disk(rec)
        self._post_history(rec, include_image=True)
        return rec

    def _discard_history_from(self, mark: int) -> None:
        """Drop frames grabbed during a motion that never actually moved."""
        with self._hist_lock:
            dropped = self.history[mark:]
            del self.history[mark:]
        for rec in dropped:
            path = rec.get("disk_path")
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        if dropped:
            self._save_history_manifest()

    def _run_motion_with_history(self, motion_fn, start_kind: str, hz_kind: str, drop_states=()):
        """Sample front RGB at motion start, then every period, until motion returns."""
        if self._nav_leg_start is None:
            self._nav_leg_start = len(self.history)
        hist_mark = len(self.history)
        stop = threading.Event()
        self._append_nav_history_frame(start_kind)

        def _loop() -> None:
            period = history_rgb_period_s()
            while not stop.wait(period):
                self._append_nav_history_frame(hz_kind)

        worker = threading.Thread(target=_loop, name="hist-rgb", daemon=True)
        worker.start()
        result = None
        try:
            result = motion_fn()
            return result
        finally:
            stop.set()
            worker.join(timeout=max(2.0, history_rgb_period_s() + 1.0))
            state = getattr(result, "state", "") if result is not None else ""
            if state in drop_states:
                self._discard_history_from(hist_mark)

    def _walk_reach_m(self) -> float:
        name = self.robot.navigation_backend
        if name == "ros1":
            return self.robot.arrival_tolerance
        if name == "ros2":
            return self.robot.arrival_tolerance
        return self.robot.arrival_tolerance

    def _stationary_nav(self, x: float, y: float) -> bool:
        if t0_static():
            return True
        try:
            pose = self.runtime.pose()
        except Exception:
            pose = None
        name = self.robot.navigation_backend
        if name == "ros1":
            reach = ROS1_REACH_M
        elif name == "ros2":
            reach = ROS2_REACH_M
        else:
            reach = None
        return already_within_waypoint(pose, float(x), float(y), reach_m=reach)

    def _face_with_history(self, direction: str) -> NavResult:
        """Record RGB while the chassis yaws. Skip when the body will stay still."""
        name = self.robot.navigation_backend
        if name in ("ros1", "ros2") or (name == "http" and str(direction) == "front"):
            return self.runtime.face_direction(direction)
        return self._run_motion_with_history(
            lambda: self.runtime.face_direction(direction),
            "turn_start",
            "turn_hz",
            drop_states=("skipped", "previewed"),
        )

    def _call_navigate(
        self, x: float, y: float, yaw: Optional[float] = None, **kwargs
    ) -> NavResult:
        nav = getattr(self.runtime, "nav", None)
        standalone = nav is not None and getattr(nav, "round_deadline", None) is None
        if standalone:
            nav.round_deadline = time.monotonic() + ROUND_S
        try:
            try:
                return self.runtime.navigate_to(x, y, yaw, **kwargs)
            except TypeError:
                kwargs.pop("requested_goal", None)
                return self.runtime.navigate_to(x, y, yaw, **kwargs)
        finally:
            if standalone:
                nav.round_deadline = None

    def _navigate_with_history(
        self, x: float, y: float, yaw: Optional[float] = None, **kwargs
    ) -> NavResult:
        """Record RGB while walking. Skip when already at the goal or preview-only."""
        if self._stationary_nav(x, y):
            return self._call_navigate(x, y, yaw, **kwargs)
        return self._run_motion_with_history(
            lambda: self._call_navigate(x, y, yaw, **kwargs),
            "nav_start",
            "nav_hz",
            drop_states=("skipped", "previewed"),
        )

    def _record(self, **kwargs) -> None:
        kwargs.pop("rgb", None)
        start = self._nav_leg_start
        self._nav_leg_start = None
        if start is None:
            return
        with self._hist_lock:
            recs = list(self.history[start:])
        if not recs:
            return
        for rec in recs:
            if rec.get("is_waypoint"):
                continue
            rec.update(kwargs)
            if rec.get("nav_ok") and rec.get("xy"):
                rec["reached_xy"] = rec["xy"]
            self._post_history(rec, include_image=False)
        self._save_history_manifest()

    def _reached_records(self) -> List[Dict]:
        return [h for h in self.history if h.get("is_waypoint") and h.get("reached_xy")]

    def _tick_scb_fail_hides(self) -> None:
        for idx in list(self._scb_fail_hide):
            self._scb_fail_hide[idx] -= 1
            if self._scb_fail_hide[idx] <= 0:
                self._scb_fail_hide.pop(idx, None)

    def _hide_failed_backtrack(self, idx: int, state: str = "") -> None:
        self._scb_fail_hide[int(idx)] = int(SCB_FAIL_HIDE_CYCLES)
        self.nav_feedback = scb_walk_failed_feedback(idx, state)
        self._emit(
            "SCB walk failed id=%s state=%s; hide %s cycles",
            idx,
            state or "failed",
            SCB_FAIL_HIDE_CYCLES,
        )

    def _land_xy(self, requested: Tuple[float, float], pose=None) -> Tuple[float, float]:
        if pose is None:
            try:
                pose = self.runtime.pose()
            except Exception:
                pose = None
        if pose is not None and len(pose) >= 2:
            return (float(pose[0]), float(pose[1]))
        return (float(requested[0]), float(requested[1]))

    def _available_backtrack_items(self) -> List[Tuple[int, Tuple[float, float], float]]:
        """Archived-RGB waypoint IDs (0-based) within STEP_MAX_M of the current pose."""
        recs = self._reached_records()
        if not recs:
            return []
        pose = None
        try:
            pose = self.runtime.pose()
        except Exception:
            pose = None
        max_m = step_max_m()
        reach = self._walk_reach_m()
        items: List[Tuple[int, Tuple[float, float], float]] = []
        for i, rec in enumerate(recs):
            if self._scb_fail_hide.get(i, 0) > 0:
                continue
            if not waypoint_has_rgb(rec):
                continue
            raw = rec.get("reached_xy")
            if not raw or len(raw) < 2:
                continue
            xy = (float(raw[0]), float(raw[1]))
            if pose is None:
                dist = 0.0
            else:
                dist = math.hypot(xy[0] - float(pose[0]), xy[1] - float(pose[1]))
                if dist > max_m:
                    continue
                if dist <= reach:
                    continue
            items.append((i, xy, dist))
        return items

    def _mark_leave_direction(self, direction: str) -> None:
        recs = self._reached_records()
        if recs and str(direction).strip() and str(direction) != "backtrack":
            recs[-1]["leave_direction"] = str(direction)

    def _archive_arrival_waypoint(
        self,
        xy: Tuple[float, float],
        arriving_direction: str = "",
        mark_leave: bool = True,
    ) -> None:
        """Snapshot RGB panorama at the arrival pose. SCB listing requires this archive."""
        if t0_static():
            return
        if mark_leave and arriving_direction:
            self._mark_leave_direction(arriving_direction)
        pose = None
        try:
            pose = self.runtime.pose()
        except Exception:
            pose = None
        requested = (float(xy[0]), float(xy[1]))
        land = self._land_xy(requested, pose)
        try:
            frames = self.runtime.capture_panorama()
        except Exception:
            log.exception("arrival panorama archive failed")
            frames = []
        pano = [copy_view_for_archive(fr) for fr in frames or [] if fr]
        rec = {
            "waypoint_id": len(self.history),
            "step": self.step,
            "pose": pose,
            "reached_xy": land,
            "requested_xy": requested,
            "xy": land,
            "yaw": None if pose is None or len(pose) < 3 else float(pose[2]),
            "pano": pano,
            "is_waypoint": True,
            "leave_direction": "",
            "kind": "waypoint",
            "t": time.time(),
            "nav_ok": True,
        }
        if not waypoint_has_rgb(rec):
            self._emit("arrival waypoint missing RGB; not listable for backtrack")
            return
        with self._hist_lock:
            self.history.append(rec)
        self._save_history_manifest()
        self._emit(
            "archived waypoint rgb views=%s land=(%.2f, %.2f) requested=(%.2f, %.2f)",
            len(pano),
            land[0],
            land[1],
            requested[0],
            requested[1],
        )

    def _waypoint_target_yaw(self, rec: Dict[str, Any], direction: str) -> float:
        pose = rec.get("pose")
        base = 0.0
        if isinstance(pose, (list, tuple)) and len(pose) >= 3:
            base = float(pose[2])
        elif rec.get("yaw") is not None:
            base = float(rec["yaw"])
        return base + float(CAM_YAW.get(str(direction), 0.0))

    def _failed_path_note(self, target: Dict[str, Any]) -> str:
        try:
            mark = self.history.index(target)
        except ValueError:
            return "None"
        bits = []
        for rec in self.history[mark + 1 :]:
            action = str(rec.get("action") or rec.get("leave_direction") or "").strip()
            if action:
                bits.append(action)
        return "; ".join(bits[:12]) if bits else "None"

    def _clear_va_trace(self, la: Optional[Dict] = None) -> None:
        try:
            self._trace(clear_va=True, direction="backtrack", la=json_public(la or {}))
        except Exception:
            log.exception("clear VA trace failed")

    def _replan_at_backtrack(self, target: Dict[str, Any]) -> Dict[str, Any]:
        pano = list(target.get("pano") or [])
        min_z = view_mask_min_depth_m()
        blocked, _depths = view_mask_blocked(pano, self.robot.directions, min_z)
        legal = legal_la_directions(self.robot.directions, blocked, exclude_rear=True)
        if not legal:
            legal = [d for d in la_action_dirs(self.robot.directions)]
        prev = str(target.get("leave_direction") or "unknown")
        prompt = la_backtrack_replan_prompt(
            f"agent_{self.robot.name}",
            legal,
            instruction=self._instruction(),
            previous_action=prev,
            failed_path_note=self._failed_path_note(target),
            hardware_dirs=self.robot.directions,
        )
        content: List[Dict] = [{"type": "text", "text": prompt}]
        content.append({"type": "text", "text": "Navigation History (up to this waypoint):"})
        try:
            mark = self.history.index(target)
        except ValueError:
            mark = len(self.history)
        content.append({"type": "text", "text": history_text(self.history[: mark + 1])})
        content.append({"type": "text", "text": "Previous Trajectory (Path taken from here):"})
        failed = [r for r in self.history[mark + 1 :] if r.get("rgb") is not None][-8:]
        if not failed:
            content.append({"type": "text", "text": self._failed_path_note(target)})
        for rec in failed:
            content.append(
                {"type": "text", "text": "Failed-path front (%s)" % rec.get("kind", "nav")}
            )
            content.append(image_part(rec["rgb"]))
        content.append(
            {"type": "text", "text": "Current 4-directional views at this waypoint (archived RGB):"}
        )
        for fr in pano:
            name = str(fr.get("name") or "?")
            if fr.get("rgb") is None:
                content.append({"type": "text", "text": f"{name}: NO IMAGE"})
                continue
            content.append({"type": "text", "text": f"View {name}"})
            content.append(image_part(fr["rgb"]))
        self._note_llm("la")
        raw = self.llm.generate_la([{"role": "user", "content": content}])
        data = parse_json_object(raw)
        data["_raw"] = raw
        action = str(data.get("action") or "navigate to front")
        direction = parse_direction(action, legal or self.robot.directions)
        if direction == "backtrack":
            direction = "front"
        direction = self._clamp_async_la_direction(direction)
        data["action"] = "navigate to %s" % direction
        data["direction"] = direction
        data["stop"] = bool(data.get("stop") is True or str(data.get("stop")).lower() == "true")
        self._emit("SCB replan dir=%s stop=%s prev=%s", direction, data["stop"], prev)
        return data

    def _open_goal_ids(self) -> List[str]:
        try:
            snap = self.board.snapshot() if hasattr(self.board, "snapshot") else {}
            goals = list(snap.get("goals") or self.goals or [])
            found = list(snap.get("found_by") or [])
        except Exception:
            goals = list(self.goals or [])
            found = []
        ids: List[str] = []
        for i, g in enumerate(goals):
            try:
                idx = int(g.get("goal_index", i))
            except (TypeError, ValueError):
                idx = i
            if idx < len(found) and found[idx] not in ("", None, -1, "none"):
                continue
            ids.append("goal_%s" % idx)
        return ids

    def _dcs_open_goals_now(self, frames: List[Dict], direction: str = "front") -> bool:
        """Live-front DCS for unresolved goals. True when the board is complete."""
        for gid in self._open_goal_ids():
            if self._handle_claim(gid, frames, direction=direction):
                return True
        return board_all_found(self.board.snapshot() if hasattr(self.board, "snapshot") else {})

    def _second_chance_backtrack(self, action: str) -> Dict[str, Any]:
        idx = parse_backtrack_id(action)
        recs = self._reached_records()
        empty = {"continue_va": False, "stop": False, "action": action}
        if idx < 0 or idx >= len(recs):
            log.warning(
                "backtrack id %s not in archived waypoints (n=%s); not walking", idx, len(recs)
            )
            return empty
        target = recs[idx]
        if not waypoint_has_rgb(target):
            log.warning("backtrack id %s has no archived RGB; not walking", idx)
            return empty
        replan = self._replan_at_backtrack(target)
        direction = str(replan.get("direction") or "front")
        xy = (float(target["reached_xy"][0]), float(target["reached_xy"][1]))
        yaw = self._waypoint_target_yaw(target, direction)
        target["leave_direction"] = direction
        self._clear_va_trace(replan)
        start_pose = None
        try:
            start_pose = self.runtime.pose()
        except Exception:
            start_pose = None
        self._nav_leg_start = None
        result = self._navigate_with_history(xy[0], xy[1], yaw)
        ok = navigation_succeeded(result)
        self._record(action=action, target="backtrack", xy=xy, yaw=yaw, nav_ok=ok)
        self._mark_travel(start_pose, result)
        if not ok:
            self._hide_failed_backtrack(idx, getattr(result, "state", "") or "failed")
            return empty
        self._archive_arrival_waypoint(xy, mark_leave=False)
        self.nav_feedback = AFTER_BACKTRACK_FEEDBACK
        self._short_travel_streak = 0
        return {
            "continue_va": True,
            "stop": bool(replan.get("stop")),
            "action": str(replan.get("action") or ("navigate to %s" % direction)),
            "direction": direction,
            "replan": replan,
        }

    def _backtrack(self, action: str) -> bool:
        """Compat wrapper around second-chance backtrack."""
        self._second_chance_backtrack(action)
        return True

    def _run_scb_then_front_va(
        self,
        action: str,
        la: Dict,
        pa: str,
        static: bool,
    ) -> Optional[Tuple[List[Dict], str, Dict, Dict, Dict]]:
        """Walk SCB target pose, then prepare live-front VA. None ends the cycle."""
        if t0_static():
            self._emit("static T0: skip backtrack motion")
            return None
        scb = self._second_chance_backtrack(action)
        if not scb.get("continue_va"):
            return None
        self._clear_short_hop()
        view = self._acquire_va_view("front", [], static)
        if view is None or view.get("rgb") is None:
            raise MissingViewsError(
                "%s missing VA view front after SCB; refusing VA" % self.robot.name
            )
        live = [view]
        if scb.get("stop") and self._dcs_open_goals_now(live, "front"):
            self._clear_short_hop()
            return None
        va = self._va(view, "front", pa)
        la = dict(la)
        la["action"] = scb.get("action") or "navigate to front"
        la["claim_goal_id"] = ""
        la["stop"] = bool(scb.get("stop"))
        return live, "front", view, va, la

    def _clear_short_hop(self) -> None:
        self._short_hop = False
        self._short_travel_streak = 0

    def _remember_origin_views(self, frames: List[Dict[str, Any]]) -> None:
        """Keep the first complete session panorama. Never overwrite."""
        if self._origin_views:
            return
        by_name = {str((fr or {}).get("name") or ""): fr for fr in frames or []}
        pose = None
        try:
            pose = self.runtime.pose()
        except Exception:
            pose = None
        out: List[Dict[str, Any]] = []
        for name in self.robot.directions:
            fr = by_name.get(str(name)) or {}
            out.append({"name": str(name), "rgb": fr.get("rgb"), "pose": pose})
        if any(rec.get("rgb") is None for rec in out):
            return
        self._origin_views = out
        self._emit(
            "remember initial panorama %d views names=%s",
            len(out),
            ",".join(rec["name"] for rec in out),
        )

    def _cycle_frames(self) -> List[Dict[str, Any]]:
        reuse = bool(
            self._short_hop and self._last_pano and not (self.t0_joint and not self.t0_consumed)
        )
        if reuse:
            try:
                require_all_views(self.robot.name, self._last_pano, self.robot.directions)
                self._emit("short hop: reuse last panorama; skip capture_panorama")
                return list(self._last_pano)
            except MissingViewsError:
                self._emit("short hop: last panorama incomplete; recapture")
                self._clear_short_hop()
        frames = self.runtime.capture_panorama()
        require_all_views(self.robot.name, frames, self.robot.directions)
        self._last_pano = frames
        self._remember_origin_views(frames)
        return frames

    def _pose_xy(self) -> Optional[Tuple[float, float]]:
        try:
            pose = self.runtime.pose()
        except Exception:
            pose = None
        return parse_map_xy(pose)

    def _strip_repeat_claim(self, la: Dict) -> None:
        """Drop a claim only when that goal is already found. Keep approach claims."""
        claim = str(la.get("claim_goal_id") or "").strip()
        if not claim:
            return
        try:
            still_open = self._goal_still_open(claim)
        except Exception:
            return
        if still_open:
            return
        la["claim_goal_id"] = ""
        self._emit("drop claim %s; already found or unknown", claim)

    def _va_repeats_last(self, va: Dict, direction: str) -> bool:
        if not self._short_hop:
            return False
        prev = self._last_va or {}
        prev_dir = str(
            prev.get("direction") or (self._last_la or {}).get("direction") or ""
        ).strip()
        if not prev_dir or str(direction) != prev_dir:
            return False
        if bbox_iou(va.get("bbox_2d"), prev.get("bbox_2d")) >= SHORT_HOP_VA_IOU_MAX:
            return True
        prev_t = normalize_va_target(prev.get("target"))
        new_t = normalize_va_target(va.get("target"))
        if prev_t and new_t and prev_t == new_t and prev_t != "unknown target":
            return True
        return False

    def _clamp_async_la_direction(self, direction: str) -> str:
        if str(direction) not in REAR_LA_DIRS:
            return direction
        alt = self._pick_other_direction(str(direction))
        self._emit("async LA: rear %s not allowed; using %s", direction, alt)
        return alt

    def _pick_other_direction(self, forbidden: str) -> str:
        min_z = view_mask_min_depth_m()
        frames = self._last_pano or []
        blocked, _depths = view_mask_blocked(frames, self.robot.directions, min_z)
        legal = [
            d
            for d in legal_la_directions(self.robot.directions, blocked, exclude_rear=True)
            if d != forbidden
        ]
        if legal:
            return legal[0]
        others = [d for d in la_action_dirs(self.robot.directions) if d != forbidden]
        return others[0] if others else "front"

    def _acquire_va_view(self, direction: str, frames: List[Dict], static: bool) -> Optional[Dict]:
        last_dir = str((self._last_va or {}).get("direction") or "")
        reuse_same = (
            self._short_hop
            and bool(direction)
            and direction == last_dir
            and isinstance(self._last_va_view, dict)
            and self._last_va_view.get("rgb") is not None
        )
        if reuse_same:
            self._emit("short hop: reuse last %s VA view; skip face", direction)
            return self._last_va_view
        if static and self.robot.navigation_backend != "http":
            by_name = {str((fr or {}).get("name") or ""): fr for fr in frames}
            view = by_name.get(direction)
            self._emit("static T0: VA from onboard %s, no chassis rotate", direction)
            return view
        faced = self._face_with_history(direction)
        if not navigation_succeeded(faced) and getattr(faced, "state", "") not in ("skipped",):
            self._emit(
                "face_direction %s failed (%s); continue with current heading",
                direction,
                getattr(faced, "message", faced),
            )
        else:
            self._emit("faced %s for VA front camera (%s)", direction, getattr(faced, "state", ""))
        return self.runtime.capture_direction(direction)

    def _run_va_with_short_hop_guard(
        self,
        view: Dict,
        direction: str,
        pa: str,
        frames: List[Dict],
        la: Dict,
    ) -> Tuple[Dict, str, Dict, Dict]:
        va = self._va(view, direction, pa)
        if not self._short_hop or not self._va_repeats_last(va, direction):
            return la, direction, view, va
        self._emit("short hop: VA reselected last object; retry VA once")
        va = self._va(view, direction, pa)
        if not self._va_repeats_last(va, direction):
            return la, direction, view, va
        forbidden = direction
        self._emit("short hop: VA still reselected; force another direction than %s", forbidden)
        self._forbid_la_dirs = [forbidden]
        try:
            la = self._la(frames)
        finally:
            self._forbid_la_dirs = []
        self._strip_repeat_claim(la)
        pa = str(la.get("progress_analysis") or pa)
        if pa:
            posted = self.board.post_progress(pa, self.step)
            if posted.get("ok") is False:
                log.warning("progress post failed: %s", posted.get("error"))
        direction = parse_direction(str(la.get("action") or ""), self.robot.directions)
        if direction == "backtrack":
            self._emit("short hop: keep backtrack; skip VA")
            return la, direction, view, va
        if not direction or direction == forbidden or direction in REAR_LA_DIRS:
            direction = self._pick_other_direction(forbidden)
            la["action"] = "navigate to %s" % direction
            self._emit("short hop: override LA direction to %s", direction)
        view = self._acquire_va_view(direction, frames, t0_static())
        if view is None or view.get("rgb") is None:
            raise MissingViewsError(
                "%s missing VA view %s; refusing VA" % (self.robot.name, direction)
            )
        va = self._va(view, direction, pa)
        return la, direction, view, va

    def _remember_decisions(self, la: Dict, va: Dict, direction: str, view: Optional[Dict]) -> None:
        self._last_la = {
            "action": la.get("action"),
            "claim_goal_id": la.get("claim_goal_id"),
            "direction": direction,
        }
        bbox = va.get("bbox_2d")
        self._last_va = {
            "direction": direction,
            "target": va.get("target"),
            "bbox_2d": list(bbox) if isinstance(bbox, (list, tuple)) else None,
        }
        if view is not None:
            self._last_va_view = view

    def _mark_travel(
        self, start_pose: Optional[Tuple[float, float, float]], result: NavResult
    ) -> None:
        end_pose = None
        try:
            end_pose = self.runtime.pose()
        except Exception:
            end_pose = None
        state = str(getattr(result, "state", "") or "")
        msg = str(getattr(result, "message", "") or "")
        if (
            state in ("skipped", "previewed")
            or "visual target within" in msg
            or "already at" in msg.lower()
        ):
            self._short_hop = True
            self._short_travel_streak += 1
            self._emit("short hop next cycle: nav %s", state or msg)
            return
        if start_pose is None or end_pose is None:
            self._short_hop = False
            self._short_travel_streak = 0
            return
        dist = math.hypot(
            float(end_pose[0]) - float(start_pose[0]),
            float(end_pose[1]) - float(start_pose[1]),
        )
        self._short_hop = dist < short_hop_m()
        if self._short_hop:
            self._short_travel_streak += 1
        else:
            self._short_travel_streak = 0
        self._emit(
            "travel %.2f m (threshold %.2f); short_hop_next=%s streak=%s",
            dist,
            short_hop_m(),
            self._short_hop,
            self._short_travel_streak,
        )

    def cycle(self) -> bool:
        """One decision cycle. Returns False to stop the episode."""
        if requested():
            return False
        self.step += 1
        self._nav_leg_start = None
        self._ra_started = None
        self._ra_time_ms = None
        self._trace(ra_time_ms=None)
        frames = self._cycle_frames()

        la_started = time.perf_counter()
        from_t0 = False
        if self.t0_joint and not self.t0_consumed:
            la = self._t0_or_wait(frames)
            self.t0_consumed = True
            from_t0 = True
            if not la or not la.get("action"):
                raise T0BarrierError("T0 produced no assignment; refusing navigation")
        else:
            la = self._la(frames)
        if la.get("_t0_la_time_ms") is not None:
            try:
                la_time_ms = round(float(la.get("_t0_la_time_ms")), 1)
            except (TypeError, ValueError):
                la_time_ms = round((time.perf_counter() - la_started) * 1000.0, 1)
        else:
            la_time_ms = round((time.perf_counter() - la_started) * 1000.0, 1)
        pa = str(la.get("progress_analysis") or "")
        if pa:
            posted = self.board.post_progress(pa, self.step)
            if posted.get("ok") is False:
                log.warning("progress post failed: %s", posted.get("error"))
        self._strip_repeat_claim(la)
        action = str(la.get("action") or "")
        stop = bool(la.get("stop") is True or str(la.get("stop")).lower() == "true")
        claim = str(la.get("claim_goal_id") or "").strip()
        direction = parse_direction(action, self.robot.directions)
        if not from_t0:
            clamped = self._clamp_async_la_direction(direction)
            if clamped != direction:
                direction = clamped
                la["action"] = "navigate to %s" % direction
                action = la["action"]
        self._emit("LA stop=%s claim=%s dir=%s action=%s", stop, claim, direction, action)
        self._trace(
            la=json_public(la), la_raw=la.get("_raw"), la_time_ms=la_time_ms, direction=direction
        )
        static = t0_static()
        if claim and not static:
            if self._handle_claim(claim, frames, direction=direction):
                self._clear_short_hop()
                return False
            if not self._goal_still_open(claim):
                self._clear_short_hop()
                self.stop_feedback = AFTER_FOUND_SWITCH_FEEDBACK
                self._emit("FOUND %s; skip VA/nav this cycle and search remaining goals", claim)
                return True

        va_started = time.perf_counter()
        if direction == "backtrack":
            prepared = self._run_scb_then_front_va(action, la, pa, static)
            if prepared is None:
                return self.step < self.robot.max_cycles
            frames, direction, view, va, la = prepared
            action = str(la.get("action") or action)
        else:
            view = self._acquire_va_view(direction, frames, static)
            if view is None or view.get("rgb") is None:
                raise MissingViewsError(
                    "%s missing VA view %s; refusing VA" % (self.robot.name, direction)
                )
            la, direction, view, va = self._run_va_with_short_hop_guard(
                view, direction, pa, frames, la
            )
            action = str(la.get("action") or action)
            if direction == "backtrack":
                prepared = self._run_scb_then_front_va(action, la, pa, static)
                if prepared is None:
                    return self.step < self.robot.max_cycles
                frames, direction, view, va, la = prepared
                action = str(la.get("action") or action)
        claim = str(la.get("claim_goal_id") or "").strip()
        va_time_ms = round((time.perf_counter() - va_started) * 1000.0, 1)

        bbox = va.get("bbox_2d")
        force_search = False
        rgb = view.get("rgb")
        size = rgb.shape[:2] if rgb is not None else (0, 0)
        if not usable_va_bbox(bbox, size[1], size[0]):
            if isinstance(bbox, (list, tuple)):
                self._emit(
                    "VA bbox discarded as empty/degenerate: %s target=%s", bbox, va.get("target")
                )
            if rgb is not None:
                h, w = rgb.shape[:2]
                bbox = fallback_va_bbox(w, h)
                va["bbox_2d"] = list(bbox)
                va["bbox_fallback"] = True
                force_search = True
                self._emit(
                    "VA missing/tiny bbox; 1.5 m heading search (center-quarter display %s)",
                    bbox,
                )
            else:
                bbox = None
                force_search = True
        va["target"] = normalize_va_target(va.get("target"))
        self._emit("VA target=%s bbox=%s", va.get("target"), bbox)
        rgb = view.get("rgb")
        pixel = None
        va_map = None
        if rgb is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            h, w = rgb.shape[:2]
            pixel = list(bbox_ground_px(list(bbox), w, h))
        image_b64 = ""
        image_size = None
        if rgb is not None:
            try:
                image_b64 = encode_image(rgb, TRACE_IMAGE_DIM)
                image_size = [int(rgb.shape[1]), int(rgb.shape[0])]
            except Exception:
                log.exception("encode VA image for dashboard failed")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            va_map = self._prepare_go_target(view, list(bbox), direction, force_search=force_search)
        depth_now = None
        if rgb is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            h, w = rgb.shape[:2]
            depth_now = bbox_inset_median_depth(view.get("depth"), list(bbox), w, h)
        va_pub = json_public(va)
        va_pub["target_state"] = "candidate" if va_map is not None else "blocked"
        depth_m = getattr(self, "last_va_depth", None)
        if depth_m is None and not getattr(self, "last_va_search_step", False):
            depth_m = depth_now
        if depth_m is not None:
            va_pub["depth_m"] = round(float(depth_m), 3)
        if getattr(self, "last_va_depth_estimated", False):
            va_pub["depth_estimated"] = True
        if getattr(self, "last_va_search_step", False):
            va_pub["search_step_no_depth"] = True
        # Push VA image/bbox as soon as the model returns; do not wait for RA motion.
        self._trace(
            la=json_public(la),
            la_raw=la.get("_raw"),
            la_time_ms=la_time_ms,
            va=va_pub,
            va_time_ms=va_time_ms,
            va_raw=va.get("_raw"),
            va_bbox=list(bbox) if isinstance(bbox, (list, tuple)) else None,
            va_pixel=pixel,
            va_depth=None if depth_m is None else round(float(depth_m), 3),
            va_map=list(va_map) if va_map else None,
            va_image=image_b64,
            image_size=image_size,
            direction=direction,
        )
        xy = None
        result = NavResult(False, "no_bbox", "VA did not return bbox_2d")
        start_pose = None
        try:
            start_pose = self.runtime.pose()
        except Exception:
            start_pose = None
        arrived_xy = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            xy, result = self._go(view, list(bbox), direction, prepared_xy=va_map)
            if not navigation_succeeded(result):
                self.nav_feedback = NAV_BLOCKED_FEEDBACK
            elif xy is not None:
                arrived_xy = xy
        else:
            self._emit("VA missing bbox and no view size; cannot apply center-quarter fallback")
        self._mark_travel(start_pose, result)
        self._remember_decisions(la, va, direction, view)
        self._record(
            action=action,
            target=va.get("target") or claim,
            xy=xy,
            yaw=None
            if xy is None
            else heading_yaw(xy[0], xy[1], *(self.runtime.pose() or (0, 0, 0))),
            direction=direction,
            nav_ok=navigation_succeeded(result),
            stop=stop,
        )
        if arrived_xy is not None:
            self._archive_arrival_waypoint(arrived_xy, arriving_direction=str(direction))
        nav = result.as_dict() if hasattr(result, "as_dict") else {"ok": bool(result)}
        va_path = trace_polyline(result) or trace_polyline(nav)
        if not va_path and hasattr(self.runtime, "nav"):
            va_path = trace_polyline(getattr(self.runtime.nav, "last_preview", {}))
        ra_time_ms = self._ra_time_ms
        self._trace(
            la=json_public(la),
            la_raw=la.get("_raw"),
            la_time_ms=la_time_ms,
            va=va_pub,
            va_time_ms=va_time_ms,
            ra_time_ms=ra_time_ms,
            va_raw=va.get("_raw"),
            va_bbox=list(bbox) if isinstance(bbox, (list, tuple)) else None,
            va_pixel=pixel,
            va_depth=None if depth_m is None else round(float(depth_m), 3),
            va_map=list(va_map) if va_map else None,
            va_path=va_path,
            va_image=image_b64,
            image_size=image_size,
            direction=direction,
            nav=nav,
            waypoint=list(xy) if xy else None,
        )
        if static:
            self._emit("static T0: first cycle done, not executing further RA")
            return False
        return self.step < self.robot.max_cycles

    def run(self) -> None:
        self.setup()
        try:
            while not requested() and self.cycle():
                if requested():
                    break
                snap = self.board.snapshot()
                if board_all_found(snap):
                    log.info("all goals found")
                    break
                if snap.get("ok") is False:
                    log.warning("board snapshot failed: %s", snap.get("error"))
        except MissingViewsError:
            log.exception("required views missing; stopping agent")
            raise
        except T0BarrierError:
            log.exception("T0 barrier failed; stopping agent before navigation")
            raise
        except KeyboardInterrupt:
            log.info("agent interrupted")
        finally:
            try:
                self.runtime.stop()
            except Exception:
                log.exception("runtime.stop failed")
            # Stop motion first. Telemetry must never delay emergency shutdown.
            if self._trace_worker is not None:
                try:
                    self._trace_queue.put_nowait(None)
                except queue.Full:
                    log.warning("board trace queue still full at shutdown")
                self._trace_worker.join(timeout=2.0)
                self._trace_stop.set()
                if self._trace_worker.is_alive():
                    log.warning("board trace flush incomplete at shutdown")


def run_from_args(robot: RobotConfig, runtime: RobotRuntime, goals_path: str) -> None:
    agent = LaviraM3Agent(robot, runtime, load_goals(goals_path))
    agent.run()
