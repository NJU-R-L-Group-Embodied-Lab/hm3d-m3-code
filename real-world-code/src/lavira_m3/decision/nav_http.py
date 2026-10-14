"""Aloha map2d :8770 navigation client. Agent never publishes /cmd_vel."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from ..contracts import NavResult
from .config import EXPLORATION_REACH_M, HTTP_REACH_M
from .map_geometry import in_world_xy
from .nav_round import (
    PLAN_S,
    ROUND_S,
    STALL_S,
    MotionProgress,
    TravelBudget,
    remaining,
    travel_budget_hit,
)
from .navigation_geometry import (
    COMPLETION_BLOCKED,
    COMPLETION_INTERMEDIATE,
    COMPLETION_ORIGINAL,
    IN_PLACE_XY_M,
    classify_nav_completion,
    nav_completion_fields,
    nav_completion_message,
    pose_xy,
    start_goal_dist,
)
from .path_geometry import thin_planner_path

log = logging.getLogger("lavira_m3.nav")

ACTIVE = ("planning", "navigating", "aligning")
SUCCESS = ("succeeded", "progressed")
FAILURE = ("failed", "unreachable", "timed_out", "cancelled", "stopped", "blocked")


def _requested_xy(requested_goal, x: float, y: float) -> Tuple[float, float]:
    if isinstance(requested_goal, (list, tuple)) and len(requested_goal) >= 2:
        try:
            rx, ry = float(requested_goal[0]), float(requested_goal[1])
        except (TypeError, ValueError):
            return float(x), float(y)
        if math.isfinite(rx) and math.isfinite(ry):
            return rx, ry
    return float(x), float(y)


def apply_goal_completion(kind, pose, requested_xy, accepted_xy, reach_m, **kwargs):
    if kind not in (COMPLETION_ORIGINAL, COMPLETION_INTERMEDIATE, COMPLETION_BLOCKED):
        return None
    extra = dict(kwargs.pop("extra", None) or {})
    extra.update(nav_completion_fields(kind, requested_xy, accepted_xy))
    state = {
        COMPLETION_ORIGINAL: "succeeded",
        COMPLETION_INTERMEDIATE: "progressed",
        COMPLETION_BLOCKED: "blocked",
    }[kind]
    return NavResult(
        kind != COMPLETION_BLOCKED,
        state,
        nav_completion_message(kind, pose, requested_xy, accepted_xy, reach_m),
        extra=extra,
        **kwargs,
    )


def apply_travel_budget(travel, pose, requested_xy, accepted_xy, reach_m, **kwargs):
    """Cancel-equivalent: walked TRAVEL_BUDGET_M of real XY, treat as intermediate."""
    extra = dict(kwargs.pop("extra", None) or {})
    extra["travel_m"] = round(float(getattr(travel, "travel_m", 0.0)), 3)
    extra["travel_budget_m"] = float(getattr(travel, "budget_m", 0.0))
    extra["travel_source"] = str(getattr(travel, "source", "odom_ekf") or "odom_ekf")
    stop = getattr(travel, "last", None) or pose_xy(pose) or accepted_xy
    finished = apply_goal_completion(
        COMPLETION_INTERMEDIATE,
        pose,
        requested_xy,
        stop,
        reach_m,
        extra=extra,
        **kwargs,
    )
    if finished is None:
        return None
    dist = (
        start_goal_dist(stop, requested_xy[0], requested_xy[1])
        if requested_xy is not None
        else None
    )
    remain = "?" if dist is None else ("%.2f" % dist)
    finished.message = "travel budget %.2f m used %.2f m; original still %s m away" % (
        extra["travel_budget_m"],
        extra["travel_m"],
        remain,
    )
    return finished


def finish_travel_budget(client, travel, pose, requested_xy, accepted_xy, reach_m, **kwargs):
    cancelled = client.cancel()
    if not cancelled.ok:
        return NavResult(
            False,
            "cancel_failed",
            cancelled.message,
            execution_id=kwargs.get("execution_id"),
            preview_id=kwargs.get("preview_id"),
            target=kwargs.get("target"),
            extra=kwargs.get("extra"),
        )
    finished = apply_travel_budget(travel, pose, requested_xy, accepted_xy, reach_m, **kwargs)
    if finished is not None:
        log.info("%s", finished.message)
    return finished


def _opener():
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    return build_opener(ProxyHandler({}))


def post_robot_plan(
    robot: str, x: float, y: float, yaw: float, path: Any, frame: str = "map"
) -> None:
    """Upload a planner polyline to map2d :8770. Never invent a start–goal chord."""
    if not isinstance(path, list) or len(path) < 3:
        return
    path = thin_planner_path(path)
    if len(path) < 2:
        return
    hub = (
        os.environ.get("MAP2D_NAV_URL")
        or os.environ.get("MAP2D_URL")
        or os.environ.get("FLEET_HUB")
        or "http://127.0.0.1:8770"
    )
    body = json.dumps(
        {
            "goal": [float(x), float(y)],
            "yaw": float(yaw),
            "path": path,
            "frame": frame or "map",
        }
    ).encode()
    req = Request(
        hub.rstrip("/") + "/v1/plan/%s" % robot,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        _opener().open(req, timeout=3).read()
    except Exception as exc:
        log.warning("POST /v1/plan/%s failed: %s", robot, exc)


class Map2dNavClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        poll_s: float = 0.4,
        total_s: float = ROUND_S,
        http_timeout_s: float = 8.0,
        progress_stall_s: float = STALL_S,
    ):
        self.base = (base_url or os.environ.get("MAP2D_NAV_URL") or "http://127.0.0.1:8770").rstrip(
            "/"
        )
        self.poll_s = poll_s
        self.total_s = total_s
        self.http_timeout_s = http_timeout_s
        self.progress_stall_s = progress_stall_s
        self._op = _opener()
        self.last_execution_id: Optional[int] = None
        self.last_preview: Dict[str, Any] = {}
        self.travel_pose_fn = None

    def request(
        self, method: str, path: str, body: Optional[Dict] = None, timeout: Optional[float] = None
    ):
        timeout = (
            min(timeout or self.http_timeout_s, 1.0)
            if path == "/v1/cancel"
            else remaining(self, timeout or self.http_timeout_s)
        )
        if timeout <= 0:
            return 0, {"error": "navigation round deadline exceeded"}
        data = None if body is None else json.dumps(body).encode()
        req = Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Cache-Control": "no-store"},
        )
        try:
            with self._op.open(req, timeout=timeout) as resp:
                raw = resp.read() or b"{}"
                payload = json.loads(raw.decode() or "{}")
                return int(getattr(resp, "status", 200) or 200), payload
        except HTTPError as exc:
            raw = exc.read() or b"{}"
            try:
                payload = json.loads(raw.decode() or "{}")
            except json.JSONDecodeError:
                payload = {"error": raw.decode(errors="replace")}
            payload.setdefault("error", str(exc))
            return int(exc.code), payload
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            return 0, {"error": str(exc)}

    def scene(self) -> Dict[str, Any]:
        code, data = self.request("GET", "/v1/scene")
        if code != 200:
            return {
                "ok": False,
                "error": data.get("error") or f"scene http {code}",
                "navigation": {},
            }
        data["ok"] = True
        return data

    def navigation(self) -> Dict[str, Any]:
        return dict((self.scene().get("navigation") or {}))

    def _bound_goal(self, x: float, y: float, scene: Optional[Dict[str, Any]] = None):
        del scene
        if in_world_xy(x, y):
            return float(x), float(y)
        log.info("discard goal (%.2f, %.2f) outside 100 m map", x, y)
        return None

    def nearest_traversable(self, x: float, y: float, max_m: float = 1.0):
        code, data = self.request(
            "GET",
            "/v1/nearest-traversable?x=%.6f&y=%.6f&max_m=%.3f" % (float(x), float(y), float(max_m)),
        )
        point = data.get("point") if isinstance(data, dict) else None
        if code != 200 or not isinstance(point, list) or len(point) < 2:
            return None
        return float(point[0]), float(point[1])

    def _ready_error(self, nav: Dict[str, Any]) -> Optional[str]:
        if not nav:
            return "navigation snapshot empty"
        if nav.get("estop_locked"):
            return "estop_locked"
        message = str(nav.get("message") or "")
        if message in (
            "odometry",
            "raw_localization",
            "imu",
            "wheel",
            "heartbeat",
            "shared_map",
            "peer_state",
        ) or message.startswith("stale_"):
            return "navigation input stale: " + message
        pose = nav.get("pose") or {}
        if pose and pose.get("live") is False:
            return "pose not live"
        motion = nav.get("motion") or {}
        if motion and motion.get("live") is False:
            return "motion not live"
        state = str(nav.get("state") or "")
        if state == "unavailable":
            return nav.get("message") or "navigation unavailable"
        return None

    def cancel(self) -> NavResult:
        code, data = self.request("POST", "/v1/cancel", {})
        nav = data.get("navigation") or {}
        state = str(nav.get("state") or data.get("state") or "cancelled")
        message = str(nav.get("message") or data.get("error") or data.get("message") or "")
        ok = code == 200 and bool(data.get("ok", code == 200))
        return NavResult(
            ok=ok, state=state, message=message, execution_id=nav.get("id", self.last_execution_id)
        )

    def lock_estop(self) -> NavResult:
        """Latch Aloha software estop so PathTracker publishes zero cmd. Does not touch GO1/GO2."""
        latch = os.environ.get("LAVIRA_ESTOP_FILE", "/tmp/lavira_estop")
        try:
            with open(latch, "a"):
                pass
        except OSError as exc:
            log.warning("touch %s failed: %s", latch, exc)
        code, data = self.request("POST", "/v1/estop", {"locked": True})
        locked = bool(data.get("locked") or os.path.exists(latch))
        ok = code == 200 and bool(data.get("ok", code == 200)) and locked
        return NavResult(
            ok=ok,
            state="stopped" if locked else str(data.get("state") or "failed"),
            message=str(
                data.get("message")
                or data.get("error")
                or ("estop locked" if locked else "estop lock failed")
            ),
        )

    def halt_navigation(self) -> NavResult:
        """Cancel the active HTTP goal and lock Aloha estop. Safe to call twice."""
        cancel = self.cancel()
        estop = self.lock_estop()
        ok = bool(cancel.ok or estop.ok)
        return NavResult(
            ok=ok,
            state="stopped" if estop.ok else cancel.state,
            message="cancel=%s; estop=%s" % (cancel.message, estop.message),
            execution_id=cancel.execution_id,
        )

    def navigate_to(
        self,
        x: float,
        y: float,
        yaw: float,
        execute: Optional[bool] = None,
        rotation_only: bool = False,
        exploration: bool = False,
        requested_goal: Optional[Tuple[float, float]] = None,
    ) -> NavResult:
        reach_m = EXPLORATION_REACH_M if exploration else HTTP_REACH_M
        target = (float(x), float(y), float(yaw))
        origin = _requested_xy(requested_goal, target[0], target[1])
        if not all(math.isfinite(v) for v in target):
            return NavResult(False, "failed", "target must be finite", target=target)

        scene = self.scene()
        nav = dict(scene.get("navigation") or {})
        if scene.get("ok"):
            bounded = self._bound_goal(x, y, scene)
            if bounded is None:
                return NavResult(False, "no_waypoint", "goal outside world map", target=target)
            x, y = bounded
            target = (float(x), float(y), float(yaw))
            if requested_goal is None:
                origin = (target[0], target[1])
        if not scene.get("ok"):
            return NavResult(
                False, "unavailable", scene.get("error") or "scene read failed", target=target
            )
        ready = self._ready_error(nav)
        if ready:
            return NavResult(
                False,
                str(nav.get("state") or "unavailable"),
                ready,
                target=target,
                frame=nav.get("frame"),
            )
        if str(nav.get("state") or "") in ACTIVE:
            return NavResult(
                False,
                "busy",
                "another navigation task is active; not cancelling it",
                execution_id=nav.get("id"),
                target=target,
                frame=nav.get("frame"),
            )

        from .config import t0_static

        if execute is None:
            execute = not t0_static()
        pose0 = nav.get("pose")
        dist_a = start_goal_dist(pose0, origin[0], origin[1])
        if (
            execute
            and not rotation_only
            and dist_a is not None
            and dist_a > IN_PLACE_XY_M
            and dist_a <= reach_m
        ):
            finished = apply_goal_completion(
                COMPLETION_ORIGINAL,
                pose0,
                origin,
                origin,
                reach_m,
                target=target,
                frame=nav.get("frame"),
            )
            if finished is not None:
                return finished
        body = {"x": target[0], "y": target[1], "yaw": target[2]}
        if rotation_only:
            body["rotation_only"] = True
        if exploration:
            body["exploration"] = True
        code, preview = self.request(
            "POST", "/v1/preview", body, timeout=max(self.http_timeout_s, 12.0)
        )
        # The robot-side planner reports a stale preview with an error containing
        # "过期" ("expired"); this is a protocol string, so it is matched verbatim.
        if code == 409 and "过期" in str(preview.get("error") or preview.get("message") or ""):
            code, preview = self.request(
                "POST", "/v1/preview", body, timeout=max(self.http_timeout_s, 12.0)
            )
        if isinstance(preview, dict) and preview.get("path"):
            self.last_preview = preview
        if code != 200:
            return NavResult(
                False,
                "preview_failed",
                str(preview.get("error") or preview.get("message") or f"preview http {code}"),
                target=target,
                frame=preview.get("frame") or nav.get("frame"),
                extra=preview,
            )
        if preview.get("can_submit", preview.get("executable")) is not True or not preview.get(
            "id"
        ):
            return NavResult(
                False,
                "not_executable",
                str(preview.get("warning") or preview.get("error") or "preview not executable"),
                preview_id=preview.get("id"),
                target=target,
                frame=preview.get("frame"),
                extra=preview,
            )

        plan_ready_at = time.monotonic()
        timing = {}
        preview["timing"] = timing
        self.last_preview = preview
        on_plan = getattr(self, "on_plan", None)
        if on_plan is not None and not rotation_only and preview.get("executable") is True:
            on_plan(preview.get("path") or [], preview.get("goal") or list(target[:2]))

        if not execute:
            return NavResult(
                True,
                "previewed",
                "static T0: preview only, not executing",
                preview_id=str(preview.get("id") or ""),
                target=target,
                frame=preview.get("frame") or nav.get("frame"),
                extra=preview,
            )

        pg = preview.get("goal") or [target[0], target[1]]
        wait_x, wait_y = float(pg[0]), float(pg[1])
        wait_yaw = float(preview.get("yaw") if preview.get("yaw") is not None else target[2])
        wait = (wait_x, wait_y, wait_yaw)

        start_xy = pose_xy(nav.get("pose"))
        kind = (
            None
            if rotation_only
            else classify_nav_completion(
                start_xy,
                start_xy,
                origin,
                (wait_x, wait_y),
                reach_m,
            )
        )
        dist_req = start_goal_dist(nav.get("pose"), origin[0], origin[1])
        in_place = dist_req is not None and dist_req <= IN_PLACE_XY_M
        if kind == COMPLETION_ORIGINAL and in_place:
            kind = None
        finished = apply_goal_completion(
            kind,
            nav.get("pose"),
            origin,
            (wait_x, wait_y),
            reach_m,
            preview_id=str(preview.get("id") or ""),
            target=wait,
            frame=preview.get("frame") or nav.get("frame"),
            extra=dict(preview),
        )
        if finished is not None:
            return finished

        preview_id = str(preview["id"])
        submit = {
            "x": wait_x,
            "y": wait_y,
            "yaw": wait_yaw,
            "confirmed": True,
            "preview_id": preview_id,
        }
        timing["plan_to_submit_ms"] = round((time.monotonic() - plan_ready_at) * 1000.0, 1)
        planning_deadline = time.monotonic() + PLAN_S
        code, accepted = self.request(
            "POST", "/v1/goal", submit, timeout=min(self.http_timeout_s, PLAN_S)
        )
        timing["plan_to_ack_ms"] = round((time.monotonic() - plan_ready_at) * 1000.0, 1)
        if code == 0:
            # Timeout: request may have been accepted. Disambiguate; never resubmit.
            later = self.navigation()
            later_goal = later.get("goal") or []
            if (
                later.get("id") not in (None, 0, nav.get("id"))
                and isinstance(later_goal, (list, tuple))
                and len(later_goal) >= 2
                and math.hypot(float(later_goal[0]) - wait_x, float(later_goal[1]) - wait_y) < 0.05
            ):
                accepted = {"ok": True, "id": later.get("id")}
                code = 200
            else:
                self.cancel()
                return NavResult(
                    False,
                    "submit_uncertain",
                    "goal POST timed out and scene did not confirm our target; cancelled",
                    preview_id=preview_id,
                    target=wait,
                )
        if code != 200 or not accepted.get("ok"):
            return NavResult(
                False,
                "submit_failed",
                str(accepted.get("error") or accepted.get("message") or f"goal http {code}"),
                preview_id=preview_id,
                target=wait,
                extra=accepted,
            )

        execution_id = accepted.get("id")
        timing.update(accepted.get("timing") or {})
        log.info("Aloha dispatch timing id=%s %s", execution_id, timing)
        self._report_execution(accepted, timing)
        self.last_execution_id = execution_id
        deadline = time.monotonic() + remaining(self, self.total_s)
        progress = None
        travel = None if rotation_only else TravelBudget()
        last_log = 0.0
        while time.monotonic() < deadline:
            from lavira_m3.decision.interrupt import requested

            if requested():
                self.cancel()
                return NavResult(
                    False,
                    "cancelled",
                    "interrupted",
                    execution_id=execution_id,
                    preview_id=preview_id,
                    target=wait,
                    extra=self.last_preview,
                )
            current = self.navigation()
            cid = current.get("id")
            if cid != execution_id:
                # Never interpret a different task's pose/path/failure as our result.
                if cid is None:
                    cancelled = self.cancel()
                    return NavResult(
                        False,
                        "safety_stop" if cancelled.ok else "cancel_failed",
                        "current task feedback unavailable",
                        execution_id=execution_id,
                    )
                return NavResult(
                    False,
                    "interrupted",
                    "navigation id changed; control lost",
                    execution_id=execution_id,
                )
            ready = self._ready_error(current)
            if ready:
                cancelled = self.cancel()
                return NavResult(
                    False,
                    "safety_stop" if cancelled.ok else "cancel_failed",
                    ready,
                    execution_id=execution_id,
                )
            server_timing = current.get("timing") or {}
            if server_timing.get("execution_id") == execution_id:
                if any(timing.get(k) != v for k, v in server_timing.items()):
                    timing.update(server_timing)
                    self._report_execution(current, timing)
            pose = current.get("pose")
            accepted_now = current.get("accepted_goal") or [wait_x, wait_y]
            kind = (
                None
                if rotation_only
                else classify_nav_completion(
                    start_xy,
                    pose_xy(pose),
                    origin,
                    accepted_now,
                    reach_m,
                )
            )
            if kind == COMPLETION_ORIGINAL and in_place:
                kind = None
            finished = apply_goal_completion(
                kind,
                pose,
                origin,
                accepted_now,
                reach_m,
                execution_id=execution_id,
                preview_id=preview_id,
                target=wait,
                frame=current.get("frame"),
                extra=dict(self.last_preview or {}),
            )
            if finished is not None:
                cancelled = self.cancel()
                if not cancelled.ok:
                    return NavResult(
                        False, "cancel_failed", cancelled.message, execution_id=execution_id
                    )
                log.info("Aloha %s", finished.message)
                return finished
            if travel_budget_hit(self, travel, pose, rotation_only=rotation_only):
                return finish_travel_budget(
                    self,
                    travel,
                    pose,
                    origin,
                    accepted_now,
                    reach_m,
                    execution_id=execution_id,
                    preview_id=preview_id,
                    target=wait,
                    frame=current.get("frame"),
                    extra=dict(self.last_preview or {}),
                )
            state = str(current.get("state") or "")
            message = str(current.get("message") or "")
            if state in SUCCESS:
                extra = dict(self.last_preview or {})
                extra.update(nav_completion_fields(kind, origin, accepted_now))
                return NavResult(
                    True,
                    state,
                    message or "arrived",
                    execution_id=execution_id,
                    preview_id=preview_id,
                    target=wait,
                    frame=current.get("frame"),
                    extra=extra,
                )
            if state in FAILURE:
                if state == "unreachable" and not current.get("path"):
                    state = "planning_timeout"
                elif state == "failed":
                    state = "safety_stop"  # Server currently folds gate faults into failed.
                return NavResult(
                    False,
                    state,
                    message or state,
                    execution_id=execution_id,
                    preview_id=preview_id,
                    target=wait,
                    frame=current.get("frame"),
                    extra=self.last_preview,
                )
            now = time.monotonic()
            motion = current.get("motion") or {}
            if progress is None and current.get("path"):
                progress = MotionProgress(pose)
            if progress is None and time.monotonic() >= planning_deadline:
                cancelled = self.cancel()
                return NavResult(
                    False,
                    "planning_timeout" if cancelled.ok else "cancel_failed",
                    "no execution trajectory within 6 s; own task cancelled",
                    execution_id=execution_id,
                    preview_id=preview_id,
                    target=wait,
                    frame=current.get("frame"),
                    extra=self.last_preview,
                )
            if (
                not rotation_only
                and progress is not None
                and progress.stalled(pose, self.progress_stall_s)
            ):
                cancelled = self.cancel()
                return NavResult(
                    False,
                    "stalled" if cancelled.ok else "cancel_failed",
                    "no progress for %.1f s (%s); cancelled"
                    % (self.progress_stall_s, state or "idle"),
                    execution_id=execution_id,
                    preview_id=preview_id,
                    target=wait,
                    frame=current.get("frame"),
                    extra=self.last_preview,
                )
            if now - last_log >= 2.0:
                dist = start_goal_dist(pose, wait_x, wait_y)
                log.info(
                    "Aloha wait state=%s dist=%s v=%.2f id=%s",
                    state or "-",
                    ("%.2f" % dist) if dist is not None else "?",
                    float((motion or {}).get("v") or 0.0),
                    execution_id,
                )
                last_log = now
            time.sleep(min(self.poll_s, max(0.0, deadline - time.monotonic())))
        cancelled = self.cancel()
        return NavResult(
            False,
            "timed_out" if cancelled.ok else "cancel_failed",
            "local wait exceeded; " + cancelled.message,
            execution_id=execution_id,
            preview_id=preview_id,
            target=wait,
            extra=self.last_preview,
        )

    def _report_execution(self, current, timing):
        callback = getattr(self, "on_execution", None)
        if callback is None:
            return
        try:
            callback(
                {
                    "state": current.get("state", "navigating"),
                    "message": current.get("message", ""),
                    "execution_id": current.get("id"),
                    "preview_id": self.last_preview.get("id"),
                    "path": self.last_preview.get("path"),
                    "timing": dict(timing),
                }
            )
        except Exception:
            log.exception("execution timing report failed")
