"""Hub-side live gate: lidar map upload plus all four cameras."""

from __future__ import annotations

from lavira_m3.profiles import AGENTS

CAMERAS = {
    n: tuple("rear" if d == "behind" else d for d in p["directions"]) for n, p in AGENTS.items()
}
CAMERA_LABELS = {
    "front": "front camera",
    "topdown": "top-down camera",
    "left": "left camera",
    "right": "right camera",
    "rear": "rear camera",
}


def _age(now, stamp):
    if stamp is None:
        return None
    return max(0.0, float(now) - float(stamp))


def _camera_stamp(images, robot, camera):
    rec = images.get((robot, camera))
    if not rec:
        return None
    if isinstance(rec, dict):
        return rec.get("updated_at")
    return rec


def build_device_status(now, camera_images, map_heard, robots=None, image_ttl=8.0, lidar_ttl=5.0):
    robots = tuple(robots or CAMERAS)
    out = {}
    for robot in robots:
        names = CAMERAS[robot]
        missing = []
        cameras = {}
        lidar_age = _age(now, map_heard.get(robot))
        lidar_ok = lidar_age is not None and lidar_age <= lidar_ttl
        if not lidar_ok:
            missing.append("lidar")
        for camera in names:
            age = _age(now, _camera_stamp(camera_images, robot, camera))
            ok = age is not None and age <= image_ttl
            cameras[camera] = {
                "live": ok,
                "age_s": None if age is None else round(age, 1),
            }
            if not ok:
                missing.append(CAMERA_LABELS.get(camera, camera))
        live = not missing
        out[robot] = {
            "overall": "live" if live else "error",
            "live": live,
            "lidar": {
                "live": lidar_ok,
                "age_s": None if lidar_age is None else round(lidar_age, 1),
            },
            "cameras": cameras,
            "missing": missing,
            "message": "" if live else ("missing: " + ", ".join(missing)),
        }
    return out
