"""Build T0 observe payloads from the same JPEGs the webpage already has."""

from __future__ import annotations

import base64

from lavira_m3.profiles import AGENTS

# Webpage names → M3 T0 view names. Aloha keeps its own 7-view capture.
DOG_T0_VIEWS = {
    n: tuple(("rear" if d == "behind" else d, d) for d in p["directions"])
    for n, p in AGENTS.items()
}


def pose_xyyaw(pose):
    if not pose or not pose.get("live") or not pose.get("position"):
        return None
    pos = pose["position"]
    if len(pos) < 2:
        return None
    return [float(pos[0]), float(pos[1]), float(pose.get("yaw") or 0.0)]


def t0_frames_from_jpegs(jpegs, robot):
    mapping = DOG_T0_VIEWS.get(robot) or ()
    frames = []
    for cam, t0_name in mapping:
        data = jpegs.get(cam)
        if not data:
            return None
        frames.append(
            {
                "name": t0_name,
                "jpeg_b64": base64.b64encode(data).decode("ascii"),
            }
        )
    return frames or None


def build_t0_observe(robot, jpegs, pose):
    frames = t0_frames_from_jpegs(jpegs, robot)
    xyz = pose_xyyaw(pose)
    if not frames or xyz is None:
        return None
    return {
        "robot": robot,
        "pose": xyz,
        "dirs": [name for _cam, name in DOG_T0_VIEWS[robot]],
        "frames": frames,
    }


def fresh_dog_jpegs(images, robot, now, ttl):
    mapping = DOG_T0_VIEWS.get(robot) or ()
    out = {}
    for cam, _name in mapping:
        rec = images.get((robot, cam)) or {}
        data = rec.get("data")
        stamp = rec.get("updated_at")
        if not data or stamp is None or now - float(stamp) > ttl:
            return None
        out[cam] = data
    return out or None


def maybe_t0_payload(robot, images, pose, now, ttl):
    """Same four JPEGs the webpage shows, plus a live pose, become one T0 observe."""
    return build_t0_observe(robot, fresh_dog_jpegs(images, robot, now, ttl) or {}, pose)
