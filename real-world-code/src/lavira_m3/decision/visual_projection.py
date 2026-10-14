"""Project a visual target without mutating the agent workflow."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..contracts import RobotRuntime
from .config import (
    STEP_MAX_M,
    STEP_MIN_M,
    VA_PROJECT_MAX_M,
    RobotConfig,
    t0_static,
    va_preview_depth_m,
)
from .navigation_geometry import (
    body_to_map,
    heading_step,
)
from .vision_geometry import (
    CAM_YAW,
    bbox_ground_px,
    body_step_along_pixel,
    clip_va_project_depth,
    pixel_to_body,
    va_observe_depth,
)

log = logging.getLogger(__name__)


@dataclass
class VisualProjection:
    waypoint: Optional[Tuple[float, float]] = None
    depth: Optional[float] = None
    depth_estimated: bool = False
    search_step: bool = False
    map_point: Optional[Tuple[float, float]] = None
    requested_xy: Optional[Tuple[float, float]] = None


def project_visual_target(
    robot: RobotConfig,
    runtime: RobotRuntime,
    view: Dict,
    bbox: List[float],
    direction: str,
    *,
    force_search=False,
) -> VisualProjection:
    projection = VisualProjection()
    rgb = view.get("rgb")
    projection.search_step = False
    projection.map_point = None
    projection.requested_xy = None
    if rgb is None:
        projection.depth = None
        projection.depth_estimated = False
        projection.waypoint = None
        return projection
    h, w = rgb.shape[:2]
    u, v = bbox_ground_px(bbox, w, h)
    z, source = va_observe_depth(view.get("depth"), bbox, w, h, u, v)
    estimated = False
    search_step = z is None or force_search
    if force_search:
        z = None
        source = "heading_search"
    pose = runtime.pose()
    if pose is None:
        projection.depth = None
        projection.depth_estimated = False
        projection.waypoint = None
        return projection
    if force_search:
        xy = heading_step(pose[0], pose[1], pose[2], STEP_MIN_M)
        projection.map_point = (float(xy[0]), float(xy[1]))
        projection.requested_xy = (float(xy[0]), float(xy[1]))
        projection.depth = None
        projection.depth_estimated = False
        projection.search_step = True
        projection.waypoint = projection.map_point
        return projection
    fx = float(view.get("fx") or 0)
    fy = float(view.get("fy") or fx)
    cx = float(view.get("cx") or 0)
    cy = float(view.get("cy") or 0)
    if fx <= 1.0:
        if not search_step:
            log.warning("missing camera intrinsics for measured depth; refusing projection")
            projection.depth = None
            projection.depth_estimated = False
            projection.waypoint = None
            return projection
        log.warning(
            "missing camera intrinsics; no-depth VA uses the current body view ray for %.2f m",
            STEP_MIN_M,
        )
    elif z is None and t0_static() and robot.navigation_backend != "http":
        z = va_preview_depth_m()
        estimated = True
        source = "static_preview"
        log.info("static T0: preview depth %.2f m for %s (no valid depth)", z, direction)
        search_step = False
    elif z is None:
        search_step = True
        log.info(
            "search_step_no_depth dir=%s px=(%s,%s); walk %.2f m along ray, not FOUND",
            direction,
            u,
            v,
            STEP_MIN_M,
        )
    rx, ry, ryaw = pose
    cam_yaw = 0.0 if robot.panorama == "surround7" else CAM_YAW.get(direction, 0.0)
    if not search_step:
        clipped_z, clipped = clip_va_project_depth(z)
        if clipped:
            log.info(
                "clip VA project depth %.2f m -> %.2f m before projection",
                float(z),
                VA_PROJECT_MAX_M,
            )
            source = (str(source) + "+clip") if source else "clip"
        z = clipped_z
    if search_step:
        if fx <= 1.0:
            xb = STEP_MIN_M * math.cos(cam_yaw)
            yb = STEP_MIN_M * math.sin(cam_yaw)
        else:
            xb, yb = body_step_along_pixel(u, v, fx, fy, cx, cy, cam_yaw, STEP_MIN_M)
    elif robot.navigation_backend == "http":
        xb, yb = body_step_along_pixel(u, v, fx, fy, cx, cy, cam_yaw, float(z))
    else:
        xb, yb = pixel_to_body(u, v, float(z), fx, fy, cx, cy, cam_yaw)
    xy = body_to_map(xb, yb, rx, ry, ryaw)
    # Keep the visual projection for the dashboard even when the point is
    # not executable. Navigation still requires a traversable snapped point.
    projection.map_point = (float(xy[0]), float(xy[1]))
    projection.requested_xy = (float(xy[0]), float(xy[1]))
    projection.depth = None if search_step else float(z)
    projection.depth_estimated = bool(estimated)
    projection.search_step = bool(search_step)
    if robot.navigation_backend == "ros1":
        # Select on EGO's acknowledged grid after geometric preparation.
        projection.waypoint = projection.map_point
        return projection
    nav_client = getattr(runtime, "nav", None)
    snap = getattr(nav_client, "nearest_traversable", None)
    free_xy = snap(xy[0], xy[1], max_m=1.0) if snap else None
    if free_xy is None:
        log.warning(
            "VA projection has no free/soft map cell within %.2f m: (%.2f, %.2f)",
            STEP_MAX_M,
            xy[0],
            xy[1],
        )
        # Keep the visual projection for the dashboard. Executability is
        # still decided by preview/move_base downstream; do not erase the
        # VA point merely because map snapping found no nearby cell.
        projection.depth = None if search_step else float(z)
        projection.depth_estimated = bool(estimated)
        projection.search_step = bool(search_step)
        projection.waypoint = (float(xy[0]), float(xy[1]))
        return projection
    if math.hypot(free_xy[0] - xy[0], free_xy[1] - xy[1]) > 0.05:
        log.info(
            "snap VA projection (%s) (%.2f, %.2f) -> nearest free/soft cell (%.2f, %.2f)",
            "no_depth" if search_step else "depth",
            xy[0],
            xy[1],
            free_xy[0],
            free_xy[1],
        )
    xy = (float(free_xy[0]), float(free_xy[1]))
    projection.depth = None if search_step else float(z)
    projection.depth_estimated = bool(estimated)
    projection.search_step = bool(search_step)
    log.info(
        "project robot=%s dir=%s px=(%s,%s) depth=%s source=%s search=%s pose=(%.2f,%.2f,%.2f) xy=(%.2f,%.2f)",
        robot.name,
        direction,
        u,
        v,
        "none" if search_step else "%.2f" % float(z),
        source or ("search_step" if search_step else "none"),
        search_step,
        rx,
        ry,
        ryaw,
        xy[0],
        xy[1],
    )
    projection.waypoint = xy
    return projection
