"""
Multi-agent step visualizer.

Mirrors the full layout produced by single-agent LaViRA's `_create_combined_image`
(visualization.py) — header bar with panel labels, RGB view + VLM map side
by side, and a tall info panel below with step/episode/action/destination/
instruction/TODO list — and extends it for N agents:

  - The RGB column is per-agent forward-view tiles stacked vertically, each
    bordered + labelled in the agent's color.
  - The VLM map is an incrementally-built shared occupancy grid
    (depth → ternary white/gray/black ternary cells, lavira's palette).
    Each agent contributes obs every sim step; agents' positions, paths and
    waypoint markers are overlaid in distinct colors.
  - The info panel keeps lavira's exact text-block structure (Step / Episode
    / ACTION line, Destination, Instruction with LA action, TODO list) and
    adds a per-agent LA/decision row below the instruction.

Anything lavira's visualizer can render single-agent is preserved here;
multi-agent-only additions (per-agent colors, per-agent action rows, shared
incremental map ingestion API) are layered on top.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ----------------------------- palette (BGR) -------------------------------- #
# Same palette as lavira's create_vlm_map_from_state.
COLOR_BG_MAP    = (255, 255, 255)   # unexplored
COLOR_EXPLORED  = (220, 220, 220)   # explored free space
COLOR_OBSTACLE  = (0, 0, 0)         # obstacle
COLOR_GOAL      = (0, 255, 0)       # unfound goal X
COLOR_GOAL_FOUND = (190, 230, 190)  # same X, paler once FOUND
COLOR_WAYPOINT  = (255, 0, 0)       # waypoint marker (blue in BGR)
# Fixed K=2..4 identity. Same BGR for RGB borders and shared OCC
# (triangle / trail / waypoint X). Avoid green (FOUND F#) and black (walls).
AGENT_COLORS_BGR: List[Tuple[int, int, int]] = [
    (0, 0, 255),       # agent_0 red
    (255, 0, 0),       # agent_1 blue
    (0, 140, 255),     # agent_2 orange
    (255, 0, 255),     # agent_3 magenta
]
AGENT_COLOR_NAMES = ("RED", "BLUE", "ORANGE", "MAGENTA")


def agent_bgr(agent_id: int) -> Tuple[int, int, int]:
    return AGENT_COLORS_BGR[int(agent_id) % len(AGENT_COLORS_BGR)]


def snapshot_world_xyz(pos) -> np.ndarray:
    """Copy Habitat Vector3 / numpy views into a standalone xyz array.

    Habitat may reuse the same Vector3 buffer; ``np.asarray(pos).copy()`` can
    stay aliased, so every trail sample collapses to the live pose and the
    overlay becomes a start→current chord.
    """
    if pos is None:
        raise ValueError("pos is None")
    if hasattr(pos, "x") and hasattr(pos, "z"):
        return np.array(
            [float(pos.x), float(getattr(pos, "y", 0.0)), float(pos.z)],
            dtype=np.float32,
        )
    vals = [float(x) for x in np.asarray(pos).reshape(-1)]
    if len(vals) >= 3:
        return np.array(vals[:3], dtype=np.float32)
    if len(vals) == 2:
        return np.array([vals[0], 0.0, vals[1]], dtype=np.float32)
    raise ValueError("pos has fewer than 2 elements")


def draw_nesw_compass(img: np.ndarray, *, margin: int = 18, arm: int = 22) -> None:
    """Draw a small N/S/E/W rose. Image-up is North, right is East."""
    if img is None or img.size == 0:
        return
    h, w = img.shape[:2]
    arm = max(12, min(int(arm), w // 8, h // 8))
    ox = w - margin - arm - 12
    oy = margin + arm + 10
    if ox < arm + 8 or oy < arm + 8:
        return
    ink = (40, 40, 40)
    north = (0, 0, 180)
    bg_tl = (ox - arm - 14, oy - arm - 16)
    bg_br = (ox + arm + 16, oy + arm + 18)
    overlay = img.copy()
    cv2.rectangle(overlay, bg_tl, bg_br, (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    cv2.line(img, (ox, oy - arm), (ox, oy + arm), ink, 1, cv2.LINE_AA)
    cv2.line(img, (ox - arm, oy), (ox + arm, oy), ink, 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "N", (ox - 6, oy - arm - 3), font, 0.42, north, 1, cv2.LINE_AA)
    cv2.putText(img, "S", (ox - 6, oy + arm + 13), font, 0.42, ink, 1, cv2.LINE_AA)
    cv2.putText(img, "W", (ox - arm - 13, oy + 4), font, 0.42, ink, 1, cv2.LINE_AA)
    cv2.putText(img, "E", (ox + arm + 2, oy + 4), font, 0.42, ink, 1, cv2.LINE_AA)


def trail_point_xz(p) -> Optional[Tuple[float, float]]:
    """Normalize a stored trail sample to world (x, z)."""
    if p is None:
        return None
    if hasattr(p, "x") and hasattr(p, "z"):
        return (float(p.x), float(p.z))
    vals = [float(x) for x in np.asarray(p).reshape(-1)]
    if len(vals) >= 3:
        return (vals[0], vals[2])
    if len(vals) == 2:
        return (vals[0], vals[1])
    return None


def draw_dashed_polyline(
    img: np.ndarray,
    pts: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int],
    *,
    thickness: int = 2,
    dash_px: float = 10.0,
    gap_px: float = 7.0,
) -> None:
    """Draw a dashed polyline that follows every sample (no skipped vertices)."""
    if img is None or len(pts) < 2:
        return
    dash_px = max(2.0, float(dash_px))
    gap_px = max(2.0, float(gap_px))
    thickness = max(1, int(thickness))
    drawing_dash = True
    leftover = dash_px
    for i in range(len(pts) - 1):
        x0, y0 = float(pts[i][0]), float(pts[i][1])
        x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            continue
        ux, uy = dx / seg_len, dy / seg_len
        walked = 0.0
        cx, cy = x0, y0
        while walked < seg_len - 1e-6:
            take = min(leftover, seg_len - walked)
            nx = cx + ux * take
            ny = cy + uy * take
            if drawing_dash:
                cv2.line(
                    img,
                    (int(round(cx)), int(round(cy))),
                    (int(round(nx)), int(round(ny))),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            cx, cy = nx, ny
            walked += take
            leftover -= take
            if leftover <= 1e-6:
                drawing_dash = not drawing_dash
                leftover = dash_px if drawing_dash else gap_px


def polyline_in_bounds_segments(
    path: Sequence,
    world_to_px_fn,
) -> List[List[Tuple[int, int]]]:
    """Split a world-space trail at out-of-bounds samples instead of skipping them.

    Skipping OOB points used to join distant in-bounds samples with a fake
    shortcut, which looked like a broken / non-continuous trail.
    """
    segs: List[List[Tuple[int, int]]] = []
    cur: List[Tuple[int, int]] = []
    for p in path:
        xz = trail_point_xz(p)
        if xz is None:
            if len(cur) >= 2:
                segs.append(cur)
            cur = []
            continue
        pt = world_to_px_fn(xz)
        if pt is None:
            if len(cur) >= 2:
                segs.append(cur)
            cur = []
            continue
        pix = (int(pt[0]), int(pt[1]))
        if cur and cur[-1] == pix:
            continue
        cur.append(pix)
    if len(cur) >= 2:
        segs.append(cur)
    return segs


def draw_found_marker(
    img: np.ndarray,
    center: Tuple[int, int],
    number: int,
    *,
    color: Tuple[int, int, int] = (0, 180, 0),
    radius: int = 12,
    label: Optional[str] = None,
) -> None:
    """Filled F# disk + outer ring: used only after a target is FOUND."""
    x, y = int(center[0]), int(center[1])
    r = max(8, int(radius))
    cv2.circle(img, (x, y), r, color, -1)
    cv2.circle(img, (x, y), r, (0, 0, 0), 2)
    cv2.circle(img, (x, y), r + 3, color, 2)
    text = f"F{int(number)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45 if r <= 14 else 0.55
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    cv2.putText(
        img,
        text,
        (x - tw // 2, y + th // 2),
        font,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if label:
        name = str(label).strip()[:18]
        if name:
            cv2.putText(
                img,
                name,
                (x + r + 6, y + 4),
                font,
                max(0.4, scale),
                color,
                1,
                cv2.LINE_AA,
            )


# ----------------------------- occupancy grid ------------------------------- #
UNKNOWN = 0
EXPLORED = 1
OBSTACLE = 2


@dataclass
class OccGridConfig:
    size_m: float = 30.0
    resolution_m: float = 0.05
    obstacle_height_min: float = 0.20
    obstacle_height_max: float = 1.80
    depth_min: float = 0.5
    depth_max: float = 5.0


class IncrementalOccupancy:
    """Shared 2D occupancy grid built incrementally from N agents' depth obs.

    Coordinates: rows index world +z (south), cols index world +x (east);
    grid is centered at ``origin_xz`` so col/row = (world − origin ± size/2) / res.
    """

    def __init__(self, cfg: OccGridConfig, hfov_deg: float, depth_h: int,
                 depth_w: int, sensor_height_m: float,
                 origin_xz: Tuple[float, float] = (0.0, 0.0)):
        self.cfg = cfg
        self.cells = int(round(cfg.size_m / cfg.resolution_m))
        self.grid = np.full((self.cells, self.cells), UNKNOWN, dtype=np.uint8)
        self.hfov = math.radians(hfov_deg)
        self.depth_h = depth_h
        self.depth_w = depth_w
        self.sensor_h = sensor_height_m
        self.origin_x = float(origin_xz[0])
        self.origin_z = float(origin_xz[1])
        self._rays = self._make_rays(depth_h, depth_w, self.hfov)

    @staticmethod
    def _make_rays(h: int, w: int, hfov: float) -> np.ndarray:
        fx = (w / 2.0) / math.tan(hfov / 2.0)
        fy = fx
        cx, cy = w / 2.0, h / 2.0
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x = (u - cx) / fx
        y = -(v - cy) / fy
        z = -np.ones_like(x)
        rays = np.stack([x, y, z], axis=-1).astype(np.float32)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        return rays

    def world_to_cell(self, x: float, z: float) -> Tuple[int, int]:
        half = self.cfg.size_m / 2.0
        col = int((x - self.origin_x + half) / self.cfg.resolution_m)
        row = int((z - self.origin_z + half) / self.cfg.resolution_m)
        return row, col

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.cells and 0 <= c < self.cells

    def integrate(self, depth_meters: np.ndarray, agent_pos: np.ndarray,
                  agent_yaw_rad: float, stride: int = 2) -> None:
        d = depth_meters.squeeze().astype(np.float32)
        if d.shape != (self.depth_h, self.depth_w):
            return
        d = d[::stride, ::stride]
        rays = self._rays[::stride, ::stride]
        valid = (d > self.cfg.depth_min) & (d < self.cfg.depth_max)
        pts_cam = rays * d[..., None]
        cy_, sy_ = math.cos(agent_yaw_rad), math.sin(agent_yaw_rad)
        # Habitat yaw is around +Y, yaw=0 faces −Z, +yaw is left toward +X.
        # That is Ry(−yaw), not the textbook Ry(+yaw).
        R = np.array([[cy_, 0.0, -sy_],
                      [0.0, 1.0, 0.0],
                      [sy_, 0.0, cy_]], dtype=np.float32)
        cam_world = agent_pos.astype(np.float32).copy()
        cam_world[1] += self.sensor_h
        pts_world = pts_cam @ R.T + cam_world[None, None, :]
        y_rel = pts_world[..., 1] - agent_pos[1]
        is_obstacle = valid & (y_rel > self.cfg.obstacle_height_min) & (y_rel < self.cfg.obstacle_height_max)
        is_floor = valid & (y_rel < self.cfg.obstacle_height_min) & (y_rel > -1.0)
        self._stamp(pts_world[is_floor], EXPLORED)
        self._stamp(pts_world[is_obstacle], OBSTACLE)

    def _stamp(self, pts: np.ndarray, label: int) -> None:
        if pts.size == 0:
            return
        half = self.cfg.size_m / 2.0
        res = self.cfg.resolution_m
        cols = ((pts[:, 0] - self.origin_x + half) / res).astype(np.int32)
        rows = ((pts[:, 2] - self.origin_z + half) / res).astype(np.int32)
        mask = (rows >= 0) & (rows < self.cells) & (cols >= 0) & (cols < self.cells)
        rows, cols = rows[mask], cols[mask]
        if label == OBSTACLE:
            self.grid[rows, cols] = OBSTACLE
        else:
            unset = self.grid[rows, cols] != OBSTACLE
            self.grid[rows[unset], cols[unset]] = EXPLORED

    def render_base(self, crop_pad_m: float = 1.2) -> Tuple[np.ndarray, Dict]:
        img = np.full((self.cells, self.cells, 3), COLOR_BG_MAP, dtype=np.uint8)
        img[self.grid == EXPLORED] = COLOR_EXPLORED
        img[self.grid == OBSTACLE] = COLOR_OBSTACLE
        nz = np.argwhere(self.grid != UNKNOWN)
        if nz.size == 0:
            return img, {"r0": 0, "c0": 0, "r1": self.cells, "c1": self.cells}
        pad = int(round(crop_pad_m / self.cfg.resolution_m))
        r0 = max(0, int(nz[:, 0].min()) - pad)
        c0 = max(0, int(nz[:, 1].min()) - pad)
        r1 = min(self.cells, int(nz[:, 0].max()) + pad + 1)
        c1 = min(self.cells, int(nz[:, 1].max()) + pad + 1)
        return img, {"r0": r0, "c0": c0, "r1": r1, "c1": c1}


def occupancy_frame_from_bounds(
    lower,
    upper,
    pad_m: float = 6.0,
    min_size_m: float = 40.0,
    max_size_m: float = 120.0,
) -> Tuple[Tuple[float, float], float]:
    """Square occupancy canvas covering a pathfinder / mesh XZ AABB."""
    min_x = float(lower[0]) - float(pad_m)
    max_x = float(upper[0]) + float(pad_m)
    min_z = float(lower[2]) - float(pad_m)
    max_z = float(upper[2]) + float(pad_m)
    origin = (0.5 * (min_x + max_x), 0.5 * (min_z + max_z))
    span = max(max_x - min_x, max_z - min_z, float(min_size_m))
    size_m = float(min(float(max_size_m), max(float(min_size_m), math.ceil(span))))
    return origin, size_m


def occupancy_frame_from_sim(
    sim,
    pad_m: float = 6.0,
    min_size_m: float = 40.0,
    max_size_m: float = 120.0,
) -> Tuple[Tuple[float, float], float]:
    lower, upper = sim.pathfinder.get_bounds()
    return occupancy_frame_from_bounds(
        lower, upper, pad_m=pad_m, min_size_m=min_size_m, max_size_m=max_size_m,
    )


def crop_map_ink(img: np.ndarray, pad_px: int = 16) -> np.ndarray:
    """Crop near-white margins so a large occupancy canvas still fills the panel."""
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ink = gray < 250
    nz = cv2.findNonZero(ink.astype(np.uint8))
    if nz is None:
        return img
    x, y, w, h = cv2.boundingRect(nz)
    pad = max(0, int(pad_px))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + h + pad)
    side = max(x1 - x0, y1 - y0, 1)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(img.shape[1], x0 + side)
    y1 = min(img.shape[0], y0 + side)
    cropped = img[y0:y1, x0:x1]
    return cropped if cropped.size else img


# --------------------------------------------------------------------------- #
# Text helpers (ported from lavira's `_create_combined_image`)
# --------------------------------------------------------------------------- #
ACTION_LABEL = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
    4: "LOOK_UP",
    5: "LOOK_DOWN",
}


def scale_bbox_xyxy(
    bbox: Dict,
    src_wh: Tuple[int, int],
    dst_wh: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Map a VA box from source RGB pixels onto a resized tile."""
    if not bbox:
        return None
    src_w, src_h = int(src_wh[0]), int(src_wh[1])
    dst_w, dst_h = int(dst_wh[0]), int(dst_wh[1])
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return None
    x1 = int(bbox.get("x", bbox.get("x1", 0)))
    y1 = int(bbox.get("y", bbox.get("y1", 0)))
    if "width" in bbox and "height" in bbox:
        x2 = x1 + int(bbox["width"])
        y2 = y1 + int(bbox["height"])
    else:
        x2 = int(bbox.get("x2", src_w))
        y2 = int(bbox.get("y2", src_h))
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def _wrap_text(text: str, max_width: int, font, font_scale, thickness) -> List[str]:
    if not text:
        return [text]
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        test = cur + (" " if cur else "") + w
        sz = cv2.getTextSize(test, font, font_scale, thickness)[0]
        if sz[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
                cur = w
            else:
                lines.append(w)
    if cur:
        lines.append(cur)
    return lines or [text]


# --------------------------------------------------------------------------- #
# MAVisualizer
# --------------------------------------------------------------------------- #
class MAVisualizer:
    """Per-step renderer for multi-agent ObjectNav (lavira layout + extensions)."""

    def __init__(
        self,
        save_dir: str,
        hfov_deg: float = 79.0,
        depth_h: int = 480,
        depth_w: int = 640,
        depth_min_m: float = 0.1,
        depth_max_m: float = 5.0,
        sensor_height_m: float = 0.88,
        map_size_m: float = 80.0,
        map_res_m: float = 0.05,
        rgb_tile_h: int = 220,
        info_panel_h: int = 500,
        header_h: int = 25,
        prefer_full_occ: bool = False,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.hfov = hfov_deg
        self.depth_h = depth_h
        self.depth_w = depth_w
        self.depth_min = depth_min_m
        self.depth_max = depth_max_m
        self.sensor_height = sensor_height_m
        self.cfg = OccGridConfig(
            size_m=map_size_m, resolution_m=map_res_m,
            depth_min=depth_min_m, depth_max=depth_max_m,
        )
        self.rgb_tile_h = rgb_tile_h
        self.info_panel_h = info_panel_h
        self.header_h = header_h
        self.prefer_full_occ = bool(prefer_full_occ)
        self._occ_origin_xz: Tuple[float, float] = (0.0, 0.0)
        self.occ: Optional[IncrementalOccupancy] = None

    # --------------------------- bbox frame ------------------------ #
    def _annotate_rgb_with_bbox(self, rgb: np.ndarray, bbox: Dict) -> np.ndarray:
        """Head-to-head port of lavira's `_save_rgb_with_bbox` annotation
        block: draws the bbox rectangle (green=NAVIGATE / red=STOP), then
        extends the image downward with a black footer showing
        `Navigate: <target>`."""
        annotated = rgb.copy()
        if annotated.dtype != np.uint8:
            annotated = (annotated * 255).astype(np.uint8)

        x = int(bbox.get("x", bbox.get("x1", 0)))
        y = int(bbox.get("y", bbox.get("y1", 0)))
        if "width" in bbox and "height" in bbox:
            w_b = int(bbox["width"]); h_b = int(bbox["height"])
        else:
            w_b = int(bbox.get("x2", annotated.shape[1])) - x
            h_b = int(bbox.get("y2", annotated.shape[0])) - y
        target_description = bbox.get("target", "target")
        action_decision = bbox.get("action", "NAVIGATE")
        box_color = (0, 255, 0) if action_decision == "NAVIGATE" else (0, 0, 255)
        cv2.rectangle(annotated, (x, y), (x + w_b, y + h_b), box_color, 4)

        # Footer with action + target description, same formula as lavira.
        action_text = "Navigate" if action_decision == "NAVIGATE" else action_decision.capitalize()
        footer_text = f"{action_text}: {target_description}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        h, w = annotated.shape[:2]
        max_text_w = w - 40
        (tw, th), _ = cv2.getTextSize(footer_text, font, font_scale, thickness)
        if tw > max_text_w:
            font_scale = font_scale * (max_text_w / tw)
            (tw, th), _ = cv2.getTextSize(footer_text, font, font_scale, thickness)
        footer_h = int(th * 2.5)
        new_img = np.zeros((h + footer_h, w, 3), dtype=np.uint8)
        new_img[:h, :, :] = annotated
        text_x = max(20, (w - tw) // 2)
        text_y = h + (footer_h + th) // 2 - int(th * 0.2)
        cv2.putText(new_img, footer_text, (text_x, text_y), font, font_scale,
                    (255, 255, 255), thickness, lineType=cv2.LINE_AA)
        return new_img

    def save_bbox_frame(
        self,
        *,
        step: int,
        episode_id: str,
        forward_rgbs: List[Optional[np.ndarray]],
        va_bbox_per_agent: List[Optional[Dict]],
    ) -> Optional[Path]:
        """Save `bbox_step{N:04d}.png` — head-to-head with lavira's per-step
        bbox dump, extended for N agents by stacking annotated tiles vertically."""
        n = len(forward_rgbs)
        if n == 0:
            return None
        tiles: List[np.ndarray] = []
        for i in range(n):
            rgb = forward_rgbs[i]
            bbox = va_bbox_per_agent[i] if va_bbox_per_agent and i < len(va_bbox_per_agent) else None
            if rgb is None:
                tile = np.zeros((self.depth_h, self.depth_w, 3), dtype=np.uint8)
            else:
                arr = rgb if rgb.dtype == np.uint8 else (rgb * 255).astype(np.uint8)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                if bbox:
                    try:
                        tile = self._annotate_rgb_with_bbox(bgr, bbox)
                    except Exception:
                        tile = bgr
                else:
                    tile = bgr
            # Prepend a thin agent-color stripe so multi-agent tiles are
            # visually distinguishable.
            color = AGENT_COLORS_BGR[i % len(AGENT_COLORS_BGR)]
            stripe_h = 22
            stripe = np.full((stripe_h, tile.shape[1], 3), color, dtype=np.uint8)
            cv2.putText(stripe, f"Agent {i}", (10, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(np.concatenate([stripe, tile], axis=0))
        # Equalize widths (in case different agents have different sizes)
        widths = [t.shape[1] for t in tiles]
        max_w = max(widths)
        norm_tiles = []
        for t in tiles:
            if t.shape[1] < max_w:
                pad = np.full((t.shape[0], max_w - t.shape[1], 3), 0, dtype=np.uint8)
                t = np.concatenate([t, pad], axis=1)
            norm_tiles.append(t)
        bbox_panel = np.concatenate(norm_tiles, axis=0)

        # Halve the panel size to keep the file small (network transit).
        bbox_panel = cv2.resize(
            bbox_panel,
            (max(1, bbox_panel.shape[1] // 2), max(1, bbox_panel.shape[0] // 2)),
            interpolation=cv2.INTER_AREA,
        )
        ep_dir = self.save_dir / str(episode_id)
        ep_dir.mkdir(parents=True, exist_ok=True)
        out = ep_dir / f"bbox_step{step:04d}.jpg"
        cv2.imwrite(str(out), bbox_panel, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return out

    # --------------------------- lifecycle ------------------------- #
    def begin_episode(
        self,
        origin_xz: Optional[Tuple[float, float]] = None,
        size_m: Optional[float] = None,
    ):
        if origin_xz is not None:
            self._occ_origin_xz = (float(origin_xz[0]), float(origin_xz[1]))
        if size_m is not None:
            size_m = float(size_m)
            if abs(size_m - float(self.cfg.size_m)) > 1e-6:
                self.cfg = OccGridConfig(
                    size_m=size_m,
                    resolution_m=self.cfg.resolution_m,
                    obstacle_height_min=self.cfg.obstacle_height_min,
                    obstacle_height_max=self.cfg.obstacle_height_max,
                    depth_min=self.cfg.depth_min,
                    depth_max=self.cfg.depth_max,
                )
        self.occ = IncrementalOccupancy(
            self.cfg, hfov_deg=self.hfov,
            depth_h=self.depth_h, depth_w=self.depth_w,
            sensor_height_m=self.sensor_height,
            origin_xz=self._occ_origin_xz,
        )

    def ingest(self, *, depth_norm: np.ndarray, agent_pos: np.ndarray, agent_yaw_rad: float) -> None:
        if self.occ is None:
            self.begin_episode()
        d = np.asarray(depth_norm, dtype=np.float32).squeeze()
        if d.ndim == 3:
            d = d[..., 0]
        if d.ndim != 2 or d.size == 0:
            return
        if d.shape != (self.depth_h, self.depth_w):
            d = cv2.resize(d, (self.depth_w, self.depth_h), interpolation=cv2.INTER_NEAREST)
        # Habitat depth is usually 0-1. If the buffer is already metres, keep it.
        if float(np.nanmax(d)) > 1.5:
            d_m = np.clip(d, self.depth_min, self.depth_max)
        else:
            d_m = self.depth_min + d * (self.depth_max - self.depth_min)
        self.occ.integrate(d_m, agent_pos.astype(np.float32), agent_yaw_rad, stride=2)

    # --------------------------- map column ------------------------ #
    def _build_map_panel(
        self,
        *,
        out_h: int,
        agent_positions: List[np.ndarray],
        agent_yaws: List[float],
        agent_colors: List[Tuple[int, int, int]],
        agent_paths: Optional[List[List[np.ndarray]]],
        waypoints_per_agent: Optional[List[List[np.ndarray]]],
        goals_world: Optional[List[np.ndarray]],
        per_agent_target_world: Optional[List[Optional[np.ndarray]]],
        found_goal_mask: Optional[List[bool]] = None,
        found_markers: Optional[List[Dict]] = None,
    ) -> np.ndarray:
        assert self.occ is not None
        base, meta = self.occ.render_base(crop_pad_m=1.8)
        if self.prefer_full_occ:
            # Keep the pathfinder-sized canvas so the whole house stays in view.
            r0, c0, r1, c1 = 0, 0, self.occ.cells, self.occ.cells
        else:
            r0, c0, r1, c1 = meta["r0"], meta["c0"], meta["r1"], meta["c1"]
            pad = int(round(2.4 / self.occ.cfg.resolution_m))
            extra_xz = []
            for pos in agent_positions or []:
                extra_xz.append((float(pos[0]), float(pos[2])))
            if agent_paths:
                for path in agent_paths:
                    for p in path or []:
                        xz = trail_point_xz(p)
                        if xz is not None:
                            extra_xz.append(xz)
            if per_agent_target_world:
                for t in per_agent_target_world:
                    if t is None:
                        continue
                    extra_xz.append((float(t[0]), float(t[2])))
            for xz in extra_xz:
                rr, cc = self.occ.world_to_cell(float(xz[0]), float(xz[1]))
                rr = min(max(rr, 0), self.occ.cells - 1)
                cc = min(max(cc, 0), self.occ.cells - 1)
                r0 = min(r0, max(0, rr - pad))
                c0 = min(c0, max(0, cc - pad))
                r1 = max(r1, min(self.occ.cells, rr + pad + 1))
                c1 = max(c1, min(self.occ.cells, cc + pad + 1))
            if r1 <= r0 or c1 <= c0:
                r0, c0, r1, c1 = 0, 0, self.occ.cells, self.occ.cells
        crop = base[r0:r1, c0:c1].copy()
        h_in, w_in = crop.shape[:2]
        scale = out_h / max(h_in, 1)
        out_w = max(1, int(w_in * scale))
        resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        draw_nesw_compass(resized)

        def _px_in_crop(wxz):
            rr, cc = self.occ.world_to_cell(float(wxz[0]), float(wxz[1]))
            if not (r0 <= rr < r1 and c0 <= cc < c1):
                return None
            return (int(round((cc - c0 + 0.5) * scale)),
                    int(round((rr - r0 + 0.5) * scale)))

        def _px_clamped(wxz):
            rr, cc = self.occ.world_to_cell(float(wxz[0]), float(wxz[1]))
            rr = min(max(rr, r0), r1 - 1)
            cc = min(max(cc, c0), c1 - 1)
            return (int(round((cc - c0 + 0.5) * scale)),
                    int(round((rr - r0 + 0.5) * scale)))

        # FOUND objects: F# disk + category at the verified goal pose.
        if found_markers:
            for mi, marker in enumerate(found_markers):
                wxz = marker.get("world_xz") or marker.get("xz")
                if wxz is None:
                    continue
                px = _px_in_crop((float(wxz[0]), float(wxz[1])))
                if px is None:
                    continue
                draw_found_marker(
                    resized, px, int(marker.get("number", mi + 1)),
                    color=COLOR_GOAL, radius=13,
                    label=marker.get("label"),
                )

        # Per-agent historical trails as dashed polylines (every sample).
        if agent_paths:
            for i, path in enumerate(agent_paths):
                if not path or len(path) < 2:
                    continue
                for seg in polyline_in_bounds_segments(path, _px_in_crop):
                    draw_dashed_polyline(
                        resized, seg, agent_colors[i % len(agent_colors)],
                        thickness=2, dash_px=8.0, gap_px=6.0,
                    )

        # Per-agent visited waypoints (blue X — like lavira)
        if waypoints_per_agent:
            for wps in waypoints_per_agent:
                for w in wps or []:
                    px = _px_in_crop((float(w[0]), float(w[2])))
                    if px is None:
                        continue
                    cv2.drawMarker(resized, px, COLOR_WAYPOINT,
                                   markerType=cv2.MARKER_TILTED_CROSS,
                                   markerSize=12, thickness=2)

        # Per-agent active sub-goal targets (small filled circle in agent color)
        if per_agent_target_world:
            for i, t in enumerate(per_agent_target_world):
                if t is None:
                    continue
                px = _px_in_crop((float(t[0]), float(t[2])))
                if px is None:
                    continue
                cv2.circle(resized, px, 6, agent_colors[i % len(agent_colors)], -1)

        # Agent arrows drawn after resize so all K markers stay readable
        # even when the occupancy crop is large.
        for i, (pos, yaw) in enumerate(zip(agent_positions, agent_yaws)):
            ax, ay = _px_clamped((float(pos[0]), float(pos[2])))
            color = agent_colors[i % len(agent_colors)]
            # Image: +col = +X (east), +row = +Z (south), so up = −Z = yaw 0.
            # Forward = (sin yaw, −cos yaw) in (x, z) → (sin, −cos) in (col, −row).
            length = 16
            p_tip = (int(ax + length * math.sin(yaw)),
                     int(ay - length * math.cos(yaw)))
            p_left = (int(ax + (length / 2) * math.sin(yaw + math.pi * 5 / 6)),
                      int(ay - (length / 2) * math.cos(yaw + math.pi * 5 / 6)))
            p_right = (int(ax + (length / 2) * math.sin(yaw - math.pi * 5 / 6)),
                       int(ay - (length / 2) * math.cos(yaw - math.pi * 5 / 6)))
            arrow = np.array([[p_tip, p_left, p_right]], dtype=np.int32)
            cv2.fillPoly(resized, arrow, color)
            cv2.circle(resized, (ax, ay), 5, color, -1)
            cv2.circle(resized, (ax, ay), 5, (255, 255, 255), 1)
            cv2.putText(resized, f"a{i}", (ax + 8, ay - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        return resized

    # --------------------------- RGB column ------------------------ #
    def _build_rgb_column(
        self,
        *,
        out_h: int,
        forward_rgbs: List[Optional[np.ndarray]],
        agent_colors: List[Tuple[int, int, int]],
        action_per_agent: Optional[List[Optional[int]]],
        last_direction_per_agent: Optional[List[Optional[str]]],
        va_bbox_per_agent: Optional[List[Optional[Dict]]] = None,
        found_labels_per_agent: Optional[List[List[str]]] = None,
    ) -> np.ndarray:
        n = len(forward_rgbs)
        cols = 2 if n >= 3 else 1
        rows = int(math.ceil(n / float(cols)))
        tile_h = max(int(out_h / max(rows, 1)), 100)
        tile_w = int(tile_h * (self.depth_w / max(self.depth_h, 1)))
        tiles: List[np.ndarray] = []
        for i in range(rows * cols):
            rgb = forward_rgbs[i] if i < n else None
            if rgb is None:
                tile = np.full((tile_h, tile_w, 3), 200 if i < n else 255, dtype=np.uint8)
            else:
                arr = rgb if rgb.dtype == np.uint8 else (rgb * 255).astype(np.uint8)
                src_h, src_w = arr.shape[:2]
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                tile = cv2.resize(bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                bbox = None
                if va_bbox_per_agent and i < len(va_bbox_per_agent):
                    bbox = va_bbox_per_agent[i]
                xyxy = scale_bbox_xyxy(bbox, (src_w, src_h), (tile_w, tile_h)) if bbox else None
                if xyxy is not None:
                    action_decision = str((bbox or {}).get("action") or "NAVIGATE")
                    box_color = (0, 255, 0) if action_decision == "NAVIGATE" else (0, 0, 255)
                    cv2.rectangle(tile, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)
                    label = str((bbox or {}).get("target") or "VA")
                    if label:
                        cv2.putText(
                            tile, label[:28], (max(6, xyxy[0]), max(36, xyxy[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA,
                        )
            if i < n:
                c = agent_colors[i % len(agent_colors)]
                cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), c, 4)
                name = AGENT_COLOR_NAMES[i % len(AGENT_COLOR_NAMES)]
                cv2.putText(tile, f"Agent {i} {name}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
                if action_per_agent and i < len(action_per_agent) and action_per_agent[i] is not None:
                    lbl = ACTION_LABEL.get(action_per_agent[i], str(action_per_agent[i]))
                    cv2.putText(tile, lbl, (tile_w - 170, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
                if last_direction_per_agent and i < len(last_direction_per_agent) and last_direction_per_agent[i]:
                    cv2.putText(tile, f"LA: {last_direction_per_agent[i]}", (8, tile_h - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
                if found_labels_per_agent and i < len(found_labels_per_agent):
                    y0 = 46
                    for lab in (found_labels_per_agent[i] or [])[:3]:
                        text = f"FOUND {lab}"[:28]
                        (tw, th), _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                        )
                        cv2.rectangle(
                            tile, (6, y0 - th - 4), (10 + tw, y0 + 4),
                            (0, 160, 0), -1,
                        )
                        cv2.putText(
                            tile, text, (8, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),
                            1, cv2.LINE_AA,
                        )
                        y0 += th + 10
            tiles.append(tile)
        row_imgs = []
        for r in range(rows):
            row_imgs.append(np.hstack(tiles[r * cols:(r + 1) * cols]))
        col = np.vstack(row_imgs)
        if col.shape[0] < out_h:
            pad = np.full((out_h - col.shape[0], col.shape[1], 3), 255, dtype=np.uint8)
            col = np.vstack([col, pad])
        return col[:out_h]

    # --------------------------- header bar ------------------------ #
    def _build_header(self, total_w: int, rgb_w: int) -> np.ndarray:
        hdr = np.full((self.header_h, total_w, 3), 240, dtype=np.uint8)
        for text, color, x_pos in [
            ("RGB View", (0, 100, 0), rgb_w // 2),                       # dark green
            ("VLM Navigation Map", (0, 0, 139), rgb_w + (total_w - rgb_w) // 2),  # dark red
        ]:
            sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(hdr, text, (max(0, x_pos - sz[0] // 2), 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return hdr

    # --------------------------- info panel ------------------------ #
    def _build_info_panel(
        self,
        *,
        width: int,
        step: int,
        episode_id: str,
        action_per_agent: List[Optional[int]],
        destination: str,
        instruction: str,
        la_direction_per_agent: List[Optional[str]],
        la_reasoning_per_agent: List[Optional[str]],
        todo_list: Optional[str],
        agent_colors: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        bg = np.ones((self.info_panel_h, width, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        line_h = 25
        y = 22
        max_w = width - 30

        # 1) Step / Episode / per-agent ACTION (red)
        action_strs = []
        for i, a in enumerate(action_per_agent):
            lbl = ACTION_LABEL.get(a, "?") if a is not None else "—"
            action_strs.append(f"A{i}:{lbl}")
        head = f"Step: {step} | Episode: {episode_id} | ACTIONS: {' '.join(action_strs)}"
        cv2.putText(bg, head, (15, y), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        y += line_h

        # 2) Destination (orange) — for ObjectNav this is the goal_name
        if destination:
            cv2.putText(bg, f"Destination: {destination}", (15, y), font,
                        font_scale, (0, 165, 255), thickness, cv2.LINE_AA)
            y += line_h

        y += 6  # spacer

        # 3) Instruction (blue) — wrapped
        if instruction:
            cv2.putText(bg, "Instruction:", (15, y), font, font_scale,
                        (255, 0, 0), thickness, cv2.LINE_AA)
            y += line_h
            for line in _wrap_text(instruction, max_w - 30, font, font_scale, thickness):
                cv2.putText(bg, line, (30, y), font, font_scale,
                            (255, 0, 0), thickness, cv2.LINE_AA)
                y += line_h

        # 4) Per-agent last LA direction line (colored)
        for i, d in enumerate(la_direction_per_agent):
            if d is None:
                continue
            color = agent_colors[i % len(agent_colors)]
            text = f"Agent {i} LA: navigate to {d}"
            cv2.putText(bg, text, (15, y), font, font_scale, color,
                        thickness, cv2.LINE_AA)
            y += line_h

        # 5) Per-agent reasoning (gray, wrapped, truncated)
        for i, reason in enumerate(la_reasoning_per_agent):
            if not reason:
                continue
            short = reason if len(reason) <= 180 else reason[:177] + "..."
            cv2.putText(bg, f"Agent {i} reasoning:", (15, y), font, 0.5,
                        agent_colors[i % len(agent_colors)], 1, cv2.LINE_AA)
            y += 18
            for line in _wrap_text(short, max_w - 30, font, 0.5, 1):
                if y < self.info_panel_h - 30:
                    cv2.putText(bg, line, (30, y), font, 0.5,
                                (60, 60, 60), 1, cv2.LINE_AA)
                    y += 18

        # 6) TODO list (dark orange header, gray-if-done items)
        if todo_list:
            y += 4
            if y < self.info_panel_h - 30:
                cv2.putText(bg, "TODO List:", (15, y), font, 0.5,
                            (255, 140, 0), 1, cv2.LINE_AA)
                y += line_h
            for raw in todo_list.split("\n"):
                line = raw.strip()
                if not line:
                    continue
                done = ("- [x]" in line) or ("- [X]" in line)
                color = (100, 100, 100) if done else (0, 0, 0)
                for w in _wrap_text(line, max_w - 60, font, 0.5, 1):
                    if y < self.info_panel_h - 12:
                        cv2.putText(bg, w, (30, y), font, 0.5,
                                    color, 1, cv2.LINE_AA)
                        y += 18
        return bg

    # --------------------------- main render ----------------------- #
    def render_step(
        self,
        *,
        step: int,
        agent_positions: List[np.ndarray],
        agent_yaws: List[float],
        forward_rgbs: List[Optional[np.ndarray]],
        goal_name: str,
        episode_id: str,
        instruction: Optional[str] = None,
        agent_paths: Optional[List[List[np.ndarray]]] = None,
        waypoints_per_agent: Optional[List[List[np.ndarray]]] = None,
        goals_world: Optional[List[np.ndarray]] = None,
        found_goal_mask: Optional[List[bool]] = None,
        found_markers: Optional[List[Dict]] = None,
        per_agent_target_world: Optional[List[Optional[np.ndarray]]] = None,
        action_per_agent: Optional[List[Optional[int]]] = None,
        la_direction_per_agent: Optional[List[Optional[str]]] = None,
        la_reasoning_per_agent: Optional[List[Optional[str]]] = None,
        todo_list: Optional[str] = None,
        lavira_map_bgr: Optional[np.ndarray] = None,
        per_agent_lavira_maps: Optional[List[Optional[np.ndarray]]] = None,
        va_bbox_per_agent: Optional[List[Optional[Dict]]] = None,
        found_labels_per_agent: Optional[List[List[str]]] = None,
    ) -> Path:
        if self.occ is None:
            self.begin_episode()
        n = len(agent_positions)
        agent_colors = AGENT_COLORS_BGR[:max(n, 1)]
        # K<=2: stack RGB on the left. K>=3: 2x2 RGB grid so one shared OCC
        # stays readable instead of a 4-row tower.
        rgb_rows = 2 if n >= 3 else max(n, 1)
        per_agent_h = self.rgb_tile_h
        body_h = per_agent_h * rgb_rows

        rgb_col = self._build_rgb_column(
            out_h=body_h,
            forward_rgbs=forward_rgbs,
            agent_colors=agent_colors,
            action_per_agent=None,
            last_direction_per_agent=la_direction_per_agent,
            va_bbox_per_agent=va_bbox_per_agent,
            found_labels_per_agent=found_labels_per_agent,
        )
        if per_agent_lavira_maps:
            # Match lavira's single-agent _create_combined_image: each VLM map
            # is resized so its HEIGHT matches the RGB tile height (~480px),
            # preserving aspect ratio. Total combined is ~1120x1005 per row.
            tile_h = per_agent_h
            tiles_m: List[np.ndarray] = []
            for i, mm in enumerate(per_agent_lavira_maps):
                if mm is None:
                    tile = np.full((tile_h, tile_h, 3), 255, dtype=np.uint8)
                else:
                    h_in, w_in = mm.shape[:2]
                    out_w = max(1, int(w_in * tile_h / max(h_in, 1)))
                    tile = cv2.resize(mm, (out_w, tile_h), interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1),
                              agent_colors[i % len(agent_colors)], 3)
                cv2.putText(tile, f"Agent {i}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            agent_colors[i % len(agent_colors)], 2, cv2.LINE_AA)
                tiles_m.append(tile)
            # Equalize widths so vertical stack works
            max_w = max(t.shape[1] for t in tiles_m)
            for k in range(len(tiles_m)):
                if tiles_m[k].shape[1] < max_w:
                    pad = np.full((tile_h, max_w - tiles_m[k].shape[1], 3), 255, dtype=np.uint8)
                    tiles_m[k] = np.concatenate([tiles_m[k], pad], axis=1)
            map_col = np.concatenate(tiles_m, axis=0)
            body_h = map_col.shape[0]
            rgb_col = self._build_rgb_column(
                out_h=body_h,
                forward_rgbs=forward_rgbs,
                agent_colors=agent_colors,
                action_per_agent=None,
                last_direction_per_agent=la_direction_per_agent,
                va_bbox_per_agent=va_bbox_per_agent,
                found_labels_per_agent=found_labels_per_agent,
            )
        elif lavira_map_bgr is not None:
            # SharedMappingPlanner / Semantic_Mapping merge. Crop white
            # margins so an 80 m canvas still fills the right panel.
            base = crop_map_ink(lavira_map_bgr.copy())
            h_in, w_in = base.shape[:2]
            map_col = cv2.resize(
                base,
                (max(1, int(w_in * body_h / max(h_in, 1))), body_h),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            map_col = self._build_map_panel(
                out_h=body_h,
                agent_positions=agent_positions,
                agent_yaws=agent_yaws,
                agent_colors=agent_colors,
                agent_paths=agent_paths,
                waypoints_per_agent=waypoints_per_agent,
                goals_world=goals_world,
                per_agent_target_world=per_agent_target_world,
                found_goal_mask=found_goal_mask,
                found_markers=found_markers,
            )

        gap = np.full((body_h, 14, 3), 255, dtype=np.uint8)
        body = np.concatenate([rgb_col, gap, map_col], axis=1)
        total_w = body.shape[1]
        # Slim top bar showing just the target object — no info panel.
        hdr = np.full((self.header_h, total_w, 3), 240, dtype=np.uint8)
        key = "  ".join(
            f"a{i}={AGENT_COLOR_NAMES[i]}" for i in range(min(n, len(AGENT_COLOR_NAMES)))
        )
        cv2.putText(hdr, f"Target: {goal_name}   |   {key}   |  N-up  dash=trail  X=waypoint  F#=found",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)
        frame = np.concatenate([hdr, body], axis=0)

        ep_dir = self.save_dir / str(episode_id)
        ep_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ep_dir / f"rgb_step{step:04d}.jpg"), rgb_col, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        cv2.imwrite(str(ep_dir / f"map_step{step:04d}.jpg"), map_col, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        out = ep_dir / f"combined_step{step:04d}.jpg"
        cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return out
