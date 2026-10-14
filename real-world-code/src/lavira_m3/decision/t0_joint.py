"""T0 joint language-action: one LA call assigns every robot's first action."""

from __future__ import annotations

import io
import json
import logging
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import ProxyHandler, Request, build_opener

from PIL import Image, ImageDraw, ImageFont

from lavira_m3.profiles import AGENTS, profile

from .config import (
    ROBOT_IDS,
    T0_MAP_DIM,
    la_action_dirs,
    t0_robots,
    view_mask_min_depth_m,
)
from .model_messages import image_part, parse_json_object
from .prompts_real import (
    IMAGE_GOAL_SECTION_LABEL,
    T0_MAP2D_CAPTION,
    normalize_image_goal_scenes,
    t0_joint_la_prompt,
    t0_view_caption,
)
from .vision_geometry import blocked_prompt_note, legal_la_directions, view_mask_blocked

log = logging.getLogger("lavira_m3.t0")

ROBOT_COLORS = {
    n: ((185, 240, 75), (82, 110, 255), (255, 121, 89))[i % 3] for i, n in enumerate(AGENTS)
}
LA_FIELDS = ("action", "claim_goal_id")
TEAM_FIELDS = ("panoramic_observation", "map_understanding", "team_reasoning", "image_goal_scenes")


def legal_dirs(robot: str) -> Tuple[str, ...]:
    raw = tuple(profile(robot)["directions"])
    return tuple(la_action_dirs(raw))


def pose_xyyaw(pose: Any) -> Optional[Tuple[float, float, float]]:
    if not pose:
        return None
    if isinstance(pose, (list, tuple)) and len(pose) >= 2:
        yaw = float(pose[2]) if len(pose) > 2 else 0.0
        return float(pose[0]), float(pose[1]), yaw
    if isinstance(pose, dict):
        if "x" in pose and "y" in pose:
            return float(pose["x"]), float(pose["y"]), float(pose.get("yaw") or 0.0)
        pos = pose.get("position") or pose.get("xy")
        if pos and len(pos) >= 2:
            return float(pos[0]), float(pos[1]), float(pose.get("yaw") or 0.0)
    return None


def jpeg_to_pil(b64: str) -> Optional[Image.Image]:
    import base64

    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        log.exception("decode T0 jpeg failed")
        return None


def action_is_legal(action: str, dirs: Sequence[str]) -> bool:
    text = " ".join((action or "").lower().replace("_", " ").replace("-", " ").split())
    if "backtrack" in text:
        return False
    for name in dirs:
        token = name.replace("_", " ")
        if text == "navigate to " + token or text.endswith("navigate to " + token):
            return True
    return False


def normalize_la(raw: Dict[str, Any], robot: str) -> Dict[str, Any]:
    dirs = legal_dirs(robot)
    action = str(raw.get("action") or "")
    if not action_is_legal(action, dirs):
        return {}
    region = " ".join(str(raw.get("region") or "").split())
    if len(region) > 300:
        region = region[:300]
    return {
        "action": action,
        "claim_goal_id": str(raw.get("claim_goal_id") or "").strip(),
        "region": region,
    }


def _robot_aliases(robots: Sequence[str]) -> Dict[str, str]:
    aliases = {}
    for name in robots:
        aliases[name] = name
        aliases[str(name).strip().lower()] = name
        aliases[f"agent_{ROBOT_IDS.get(name, -1)}"] = name
    return aliases


def normalize_panoramic(raw: Any, robots: Sequence[str]) -> Any:
    if isinstance(raw, str):
        return raw.strip()
    if not isinstance(raw, dict):
        return {}
    aliases = _robot_aliases(robots)
    out: Dict[str, str] = {}
    for key, value in raw.items():
        robot = aliases.get(str(key).strip().lower()) or aliases.get(str(key).strip())
        if robot is None:
            continue
        out[robot] = str(value or "").strip()
    return out


def empty_team_fields() -> Dict[str, Any]:
    return {
        "panoramic_observation": {},
        "map_understanding": "",
        "team_reasoning": "",
        "image_goal_scenes": {},
        "la_time_ms": None,
    }


def parse_t0_team_fields(data: Dict[str, Any], robots: Sequence[str]) -> Dict[str, Any]:
    fields = empty_team_fields()
    if not isinstance(data, dict):
        return fields
    fields["panoramic_observation"] = normalize_panoramic(data.get("panoramic_observation"), robots)
    fields["map_understanding"] = str(data.get("map_understanding") or "").strip()
    fields["team_reasoning"] = str(data.get("team_reasoning") or "").strip()
    fields["image_goal_scenes"] = normalize_image_goal_scenes(data.get("image_goal_scenes"))
    return fields


def parse_t0_assignments(
    text: str, robots: Sequence[str]
) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
    data = parse_json_object(text)
    if not isinstance(data, dict):
        return {}, empty_team_fields()
    blob = data.get("assignments") if isinstance(data.get("assignments"), dict) else data
    fields = parse_t0_team_fields(data, robots)
    if not fields["team_reasoning"] and isinstance(blob, dict):
        fields["team_reasoning"] = str(blob.get("team_reasoning") or "").strip()
    if not fields["image_goal_scenes"] and isinstance(blob, dict):
        fields["image_goal_scenes"] = normalize_image_goal_scenes(blob.get("image_goal_scenes"))
    if not isinstance(blob, dict):
        return {}, fields
    aliases = _robot_aliases(robots)
    out: Dict[str, Dict] = {}
    for key, value in blob.items():
        robot = aliases.get(str(key).strip().lower()) or aliases.get(str(key).strip())
        if robot is None or not isinstance(value, dict):
            continue
        la = normalize_la(value, robot)
        if la:
            out[robot] = la
    return out, fields


def rasterize_map_2d(
    scene: Optional[Dict[str, Any]],
    poses: Dict[str, Any],
    size: int = T0_MAP_DIM,
) -> Image.Image:
    """Top-down occupancy image with an English legend painted on the canvas."""
    legend_h = 92
    img = Image.new("RGB", (size, size + legend_h), (154, 165, 175))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    legend = (
        "map_2d T0  gray=unknown  black=occupied  purple=shared occupancy  "
        "configured agent markers; stroke=heading"
    )
    draw.rectangle((0, 0, size, legend_h), fill=(23, 35, 38))
    draw.text(
        (8, 8), "map_2d legend (top-down, metres, +X right, +Y up)", fill=(237, 244, 237), font=font
    )
    draw.text((8, 28), legend, fill=(185, 240, 75), font=font)
    draw.text(
        (8, 48),
        "Markers are robot poses at T0. Black cells are fused obstacles.",
        fill=(174, 187, 183),
        font=font,
    )
    draw.text(
        (8, 68),
        "Unknown space stays mid-gray. This image is T0 context only.",
        fill=(174, 187, 183),
        font=font,
    )

    fmap = (scene or {}).get("map") if scene and "map" in (scene or {}) else (scene or {})
    fused = list((fmap or {}).get("fused") or [])
    overlap = list((fmap or {}).get("overlap") or [])
    origin = list((fmap or {}).get("origin") or [0.0, 0.0])
    res = float((fmap or {}).get("resolution") or 0.1) or 0.1

    cells = [(int(c[0]), int(c[1])) for c in fused if len(c) >= 2]
    shared = {(int(c[0]), int(c[1])) for c in overlap if len(c) >= 2}
    pose_xy = {name: pose_xyyaw(pose) for name, pose in poses.items()}
    pose_xy = {k: v for k, v in pose_xy.items() if v}

    if cells:
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        for xy in pose_xy.values():
            xs.append(int(math.floor((xy[0] - origin[0]) / res)))
            ys.append(int(math.floor((xy[1] - origin[1]) / res)))
        min_ix, max_ix = min(xs) - 2, max(xs) + 2
        min_iy, max_iy = min(ys) - 2, max(ys) + 2
    elif pose_xy:
        xs = [int(math.floor((p[0] - origin[0]) / res)) for p in pose_xy.values()]
        ys = [int(math.floor((p[1] - origin[1]) / res)) for p in pose_xy.values()]
        min_ix, max_ix = min(xs) - 8, max(xs) + 8
        min_iy, max_iy = min(ys) - 8, max(ys) + 8
    else:
        draw.text(
            (12, legend_h + 16),
            "map_2d empty: no occupancy and no poses yet",
            fill=(23, 35, 38),
            font=font,
        )
        return img

    span = max(max_ix - min_ix + 1, max_iy - min_iy + 1, 1)
    cell = max(1, int((size - 16) / span))
    ox = 8
    oy = legend_h + 8

    def cell_px(ix: int, iy: int) -> Tuple[int, int, int, int]:
        x0 = ox + (ix - min_ix) * cell
        y0 = oy + (max_iy - iy) * cell
        return x0, y0, x0 + cell, y0 + cell

    for ix, iy in cells:
        box = cell_px(ix, iy)
        color = (180, 147, 228) if (ix, iy) in shared else (20, 20, 20)
        draw.rectangle(box, fill=color)

    for name, xyz in pose_xy.items():
        ix = int(math.floor((xyz[0] - origin[0]) / res))
        iy = int(math.floor((xyz[1] - origin[1]) / res))
        x0, y0, x1, y1 = cell_px(ix, iy)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        color = ROBOT_COLORS.get(name, (255, 255, 255))
        r = max(4, cell)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline=(23, 35, 38))
        hx = cx + int(math.cos(xyz[2]) * (r + 8))
        hy = cy - int(math.sin(xyz[2]) * (r + 8))
        draw.line((cx, cy, hx, hy), fill=color, width=2)
        label = f"agent_{ROBOT_IDS.get(name, '?')} {name}"
        draw.text((cx + r + 3, cy - 6), label, fill=color, font=font)
    return img


def fetch_scene(nav_url: str, timeout: float = 4.0) -> Dict[str, Any]:
    if not nav_url:
        return {}
    opener = build_opener(ProxyHandler({}))
    req = Request(
        nav_url.rstrip("/") + "/v1/scene", method="GET", headers={"Cache-Control": "no-store"}
    )
    try:
        with opener.open(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode() or "{}")
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.warning("T0 map_2d scene fetch failed: %s", exc)
        return {}


def collect_poses(observes: Dict[str, Dict], scene: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    poses: Dict[str, Any] = {}
    scene_poses = (scene or {}).get("poses") or {}
    for name in t0_robots():
        if name in scene_poses:
            poses[name] = scene_poses[name]
    for name, rec in (observes or {}).items():
        xy = pose_xyyaw((rec or {}).get("pose"))
        if xy:
            poses[name] = {"x": xy[0], "y": xy[1], "yaw": xy[2]}
    return poses


def build_t0_messages(
    observes: Dict[str, Dict],
    board_text: str,
    map_image: Image.Image,
    reference_images: Optional[Iterable[Tuple[str, Image.Image]]] = None,
    robots: Optional[Sequence[str]] = None,
) -> List[Dict]:
    robots = list(robots or t0_robots())
    min_z = view_mask_min_depth_m()
    legal_by_robot: Dict[str, List[str]] = {}
    blocked_notes: Dict[str, str] = {}
    blocked_by_robot: Dict[str, List[str]] = {}
    depths_by_robot: Dict[str, Dict[str, Optional[float]]] = {}
    for name in robots:
        rec = observes.get(name) or {}
        default_dirs = list(tuple(profile(name)["directions"]))
        dirs = [str(d) for d in (rec.get("dirs") or default_dirs) if str(d)]
        if not dirs:
            dirs = default_dirs
        blocked, depths = view_mask_blocked(rec.get("frames") or [], dirs, min_z)
        blocked_by_robot[name] = blocked
        depths_by_robot[name] = depths
        legal_by_robot[name] = legal_la_directions(dirs, blocked, exclude_rear=True)
        note = blocked_prompt_note(blocked, depths, min_z)
        if note:
            blocked_notes[name] = note
            log.info(
                "T0 view mask robot=%s blocked=%s legal=%s", name, blocked, legal_by_robot[name]
            )
    content: List[Dict] = [
        {
            "type": "text",
            "text": t0_joint_la_prompt(
                robots,
                board_text,
                legal_by_robot,
                blocked_notes,
            ),
        }
    ]
    refs = list(reference_images or [])
    if refs:
        content.append({"type": "text", "text": IMAGE_GOAL_SECTION_LABEL})
        for label, img in refs:
            content.append(
                {"type": "text", "text": f"{label} image-goal photo (identity and scene)"}
            )
            content.append(image_part(img))
    content.append({"type": "text", "text": T0_MAP2D_CAPTION})
    content.append(image_part(map_image, T0_MAP_DIM))
    for name in robots:
        rec = observes.get(name) or {}
        frames = rec.get("frames") or []
        pose = pose_xyyaw(rec.get("pose"))
        pose_txt = (
            "unknown"
            if pose is None
            else f"(x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[2]:.2f} rad)"
        )
        content.append(
            {
                "type": "text",
                "text": (
                    f"=== {name} / agent_{ROBOT_IDS.get(name, -1)} current panorama ===\n"
                    f"Reported map pose at T0: {pose_txt}"
                ),
            }
        )
        if not frames:
            content.append({"type": "text", "text": f"{name}: no T0 images uploaded."})
            continue
        for fr in frames:
            view = str((fr or {}).get("name") or "unknown")
            caption = t0_view_caption(name, view)
            z = depths_by_robot.get(name, {}).get(view)
            blocked_here = view in blocked_by_robot.get(name, [])
            if z is not None:
                caption += " Center-region mean depth is %.2f m." % float(z)
                if blocked_here:
                    caption += " Under 1.5 m, so this direction is omitted from the action list."
            content.append({"type": "text", "text": caption})
            img = fr.get("image")
            if img is None and fr.get("jpeg_b64"):
                img = jpeg_to_pil(fr["jpeg_b64"])
            if img is None:
                content.append({"type": "text", "text": f"{name} view {view}: NO IMAGE"})
                continue
            content.append(image_part(img))
    return [{"role": "user", "content": content}]


def run_t0_la(
    llm,
    observes: Dict[str, Dict],
    board_text: str,
    map_image: Image.Image,
    reference_images: Optional[Iterable[Tuple[str, Image.Image]]] = None,
    robots: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Dict], Dict[str, Any], str]:
    robots = list(robots or t0_robots())
    messages = build_t0_messages(observes, board_text, map_image, reference_images, robots)
    raw = ""
    parsed: Dict[str, Dict] = {}
    team = empty_team_fields()
    for attempt in range(2):
        started = time.perf_counter()
        note = getattr(llm, "note_record", None)
        if callable(note):
            note("t0_la", 0)
        raw = llm.generate_la(messages, max_tokens=4096)
        la_time_ms = round((time.perf_counter() - started) * 1000.0, 1)
        parsed, team = parse_t0_assignments(raw, robots)
        team["la_time_ms"] = la_time_ms
        if parsed:
            break
        log.warning("T0 joint LA parse failed attempt=%s", attempt + 1)
    return parsed, team, raw
