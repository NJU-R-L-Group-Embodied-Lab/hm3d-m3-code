"""Ephemeral teammate footprints and the shared motion interlock; never map uploads."""

import math
import time

from lavira_m3.profiles import AGENTS

# Rear/front/half-width about the advertised navigation body origin, metres.
ENVELOPE = {n: tuple(p["envelope"]) for n, p in AGENTS.items()}
# Shared-state grace period; keep original ages through missing updates.
FRESH = 5.0
NEAR = 3.0


class PeerCache:
    def __init__(self):
        self.poses = {}

    def observe(self, state, now=None):
        now = time.monotonic() if now is None else now
        for name, source in state.get("sources", {}).items():
            if name not in ENVELOPE:
                continue
            p = source.get("pose", {})
            xyz, yaw, age = p.get("position", []), p.get("yaw"), p.get("age_s")
            if (
                len(xyz) != 3
                or yaw is None
                or age is None
                or not all(math.isfinite(v) for v in (*xyz, yaw, age))
                or age < 0
                or age > FRESH
            ):
                continue
            self.poses[name] = dict(name=name, x=xyz[0], y=xyz[1], yaw=yaw, age=age, received=now)

    def view(self, robot, now=None):
        now = time.monotonic() if now is None else now
        poses = {}
        for n, p in self.poses.items():
            age = p["age"] + max(0, now - p["received"])
            # Keep expired footprints so the existing nearby-peer interlock can stop.
            poses[n] = dict(p, age=age)
        return dict(
            robot=robot, own=poses.get(robot), peers=[p for n, p in poses.items() if n != robot]
        )


def bounds(p):
    rear, front, half = ENVELOPE[p["name"]]
    r = math.hypot(max(rear, front), half)
    return p["x"] - r, p["y"] - r, p["x"] + r, p["y"] + r


def inside_envelope(px, py, p):
    """Same yaw-oriented rectangle the webpage draws. Origin is the advertised body frame."""
    name = p.get("name")
    if name not in ENVELOPE:
        return False
    rear, front, half = ENVELOPE[name]
    c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
    dx, dy = px - p["x"], py - p["y"]
    forward = c * dx + s * dy
    lateral = -s * dx + c * dy
    return -rear <= forward <= front and abs(lateral) <= half


def occupied_cells(view, geometry):
    w, h, res, ox, oy = geometry
    cells = set()
    for p in view.get("peers", []):
        if p.get("name") not in ENVELOPE:
            continue
        x0, y0, x1, y1 = bounds(p)
        ix0 = max(0, int(math.floor((x0 - ox) / res)))
        iy0 = max(0, int(math.floor((y0 - oy) / res)))
        ix1 = min(w - 1, int(math.floor((x1 - ox) / res)))
        iy1 = min(h - 1, int(math.floor((y1 - oy) / res)))
        for iy in range(iy0, iy1 + 1):
            py = oy + (iy + 0.5) * res
            for ix in range(ix0, ix1 + 1):
                px = ox + (ix + 0.5) * res
                if inside_envelope(px, py, p):
                    cells.add((ix, iy))
    return cells


def occupied_xy(view, res=0.1):
    peers = view.get("peers", [])
    if not peers:
        return []
    boxes = [bounds(p) for p in peers]
    ox = math.floor(min(b[0] for b in boxes) / res) * res - res
    oy = math.floor(min(b[1] for b in boxes) / res) * res - res
    w = math.ceil((max(b[2] for b in boxes) - ox) / res) + 2
    h = math.ceil((max(b[3] for b in boxes) - oy) / res) + 2
    return [
        (ox + (x + 0.5) * res, oy + (y + 0.5) * res)
        for x, y in occupied_cells(view, (w, h, res, ox, oy))
    ]


def overlaps(a, b, padding=0):
    """Separating-axis test for two oriented, possibly off-centre body rectangles."""

    def box(p, extra):
        rear, front, half = ENVELOPE[p["name"]]
        c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
        shift = (front - rear) * 0.5
        return (
            (p["x"] + c * shift, p["y"] + s * shift),
            ((c, s), (-s, c)),
            ((rear + front) * 0.5 + extra, half + extra),
        )

    ac, aa, ah = box(a, 0)
    bc, ba, bh = box(b, padding)
    dx, dy = ac[0] - bc[0], ac[1] - bc[1]
    for ux, uy in (*aa, *ba):
        ar = sum(ah[i] * abs(ux * aa[i][0] + uy * aa[i][1]) for i in (0, 1))
        br = sum(bh[i] * abs(ux * ba[i][0] + uy * ba[i][1]) for i in (0, 1))
        if abs(dx * ux + dy * uy) > ar + br:
            return False
    return True


def allowed(view, age, vx, vy, wz):
    if not view or not all(math.isfinite(v) for v in (age, vx, vy, wz)) or not 0 <= age <= FRESH:
        return False
    own = view.get("own")
    if not own or own["age"] + age > FRESH:
        return False
    peers = view.get("peers", [])
    for p in peers:
        if p["age"] + age > FRESH and math.hypot(p["x"] - own["x"], p["y"] - own["y"]) < NEAR:
            return False
    pose = dict(own)
    # Bounded 1-second braking/look-ahead envelope; no command is sent by this helper.
    for _ in range(11):
        if any(overlaps(pose, p) for p in peers):
            return False
        c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
        pose["x"] += (c * vx - s * vy) * 0.1
        pose["y"] += (s * vx + c * vy) * 0.1
        pose["yaw"] += wz * 0.1
    return True


class PeerGuard:
    def __init__(self, robot):
        self.robot = robot
        self.snapshot = None

    def update(self, message):
        import json

        try:
            v = json.loads(message.data)
            assert v["robot"] == self.robot
            for p in ([v["own"]] if v.get("own") else []) + v["peers"]:
                assert (
                    p["name"] in ENVELOPE
                    and all(math.isfinite(p[k]) for k in ("x", "y", "yaw", "age"))
                    and p["age"] >= 0
                )
            assert not v.get("own") or v["own"]["name"] == self.robot
            assert all(p["name"] != self.robot for p in v["peers"])
            self.snapshot = (v, time.monotonic())
        except (ValueError, TypeError, KeyError, AssertionError):
            self.snapshot = None

    def permits(self, vx, vy=0.0, wz=0.0):
        snapshot = self.snapshot
        return bool(snapshot and allowed(snapshot[0], time.monotonic() - snapshot[1], vx, vy, wz))
