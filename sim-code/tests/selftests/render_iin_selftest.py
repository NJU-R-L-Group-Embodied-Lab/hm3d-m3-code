"""Offline checks for IIN square crop / resize (no Habitat)."""
from __future__ import annotations

import numpy as np

from benchmark import config as _cfg
from benchmark.render_iin import (
    _crop_around,
    _iter_orbit_agents,
    _project_points,
    _score_stretch512_view,
    _standoff_agent,
    _to_square_size,
    _wall_facing_stats,
    stretch512_issue_tags,
)


def _black_row_frac(img: np.ndarray, thresh: float = 8.0) -> tuple:
    dark = img.astype(np.float32).mean(axis=2) < thresh
    row = dark.mean(axis=1)
    top = 0
    for v in row:
        if v > 0.95:
            top += 1
        else:
            break
    bot = 0
    for v in row[::-1]:
        if v > 0.95:
            bot += 1
        else:
            break
    return top, bot


def main() -> None:
    assert _cfg.IIN_CAMERA_STRETCH["width_px"] == 360
    assert _cfg.IIN_CAMERA_STRETCH["height_px"] == 640
    assert _cfg.IIN_CAMERA_STRETCH_512["width_px"] == 512
    assert _cfg.IIN_CAMERA_STRETCH_512["height_px"] == 512
    assert abs(float(_cfg.IIN_CAMERA_STRETCH_512["hfov"]) - 42.0) < 1e-6
    assert abs(float(_cfg.IIN_CAMERA_STRETCH_512["height_m"]) - 1.31) < 1e-6

    wide = np.zeros((136, 344, 3), dtype=np.uint8)
    wide[:, :] = (40, 120, 200)
    wide[40:96, 120:224] = (220, 80, 40)
    out = _to_square_size(wide, 512)
    assert out.shape == (512, 512, 3), out.shape
    top, bot = _black_row_frac(out)
    assert top == 0 and bot == 0, (top, bot)

    cam = dict(_cfg.IIN_CAMERA_STRETCH)
    frame = np.zeros((640, 360, 3), dtype=np.uint8)
    frame[:, :] = (10, 80, 30)
    # rug-like: intended 344x344 hangs off the bottom -> old crop is 344x136
    uvz = (180.0, 676.2, 1.8)
    strip, box, _ = _crop_around(frame, uvz, cam, keep_square=False)
    assert box[2] - box[0] == 344 and box[3] - box[1] == 136, box
    sq, sbox, _ = _crop_around(frame, uvz, cam, keep_square=True)
    assert sq.shape[0] == sq.shape[1] == 344, (sq.shape, sbox)
    assert sbox[3] <= 640 and sbox[1] >= 0

    hits = _project_points([[0.0, 0.0, -1.8]], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], cam)
    assert hits and hits[0] is not None
    u, v, depth = hits[0]
    assert abs(depth - 1.8) < 1e-6
    assert abs(u - 180.0) < 1e-4
    assert abs(v - 320.0) < 1e-4

    class _PF:
        def __init__(self, collapse_18=True):
            self.collapse_18 = collapse_18

        def snap_point(self, p):
            p = np.asarray(p, dtype=np.float32)
            if self.collapse_18 and abs(float(p[2]) - 1.8) < 0.05:
                return np.array([0.0, 0.1, 1.16], dtype=np.float32)
            return p

        def is_navigable(self, _p):
            return True

    view = (0.0, 0.1, 0.55)
    obj = (0.0, 0.1, 0.0)
    collapsed = _standoff_agent(_PF(), view, obj, 1.8)
    assert abs(float(collapsed[2]) - 1.16) < 1e-3
    held = _standoff_agent(_PF(), view, obj, 1.8, ladder=(2.4, 3.0))
    assert abs(float(held[2]) - 2.4) < 1e-3

    flat = np.full((64, 64, 3), 160, dtype=np.uint8)
    flat = flat + np.random.default_rng(0).integers(-1, 2, size=flat.shape)
    w = _wall_facing_stats(flat)
    assert w["flat_rgb"] and w["wall_facing"]
    tex = np.zeros((64, 64, 3), dtype=np.uint8)
    tex[::8, :] = 200
    tex[:, ::8] = 40
    w = _wall_facing_stats(tex)
    assert not w["flat_rgb"]
    close = np.full((32, 32), 0.35, dtype=np.float32)
    w = _wall_facing_stats(tex, close, object_depth_m=2.0)
    assert w["close_plane"] and w["occluded"] and w["wall_facing"]

    blank = _wall_facing_stats(flat, np.full((64, 64), 0.08, dtype=np.float32), 1.9)
    assert "flat_rgb" in stretch512_issue_tags(blank)
    assert "inside_mesh" in stretch512_issue_tags(blank)
    mesh = {
        "rgb_std": 40.0,
        "rgb_edge": 4.0,
        "flat_rgb": False,
        "depth_med": 0.05,
        "object_depth_m": 2.0,
    }
    assert "inside_mesh" in stretch512_issue_tags(mesh)
    ok = {
        "rgb_std": 48.0,
        "rgb_edge": 5.0,
        "flat_rgb": False,
        "depth_med": 1.5,
        "object_depth_m": 1.9,
    }
    assert stretch512_issue_tags(ok) == []
    assert _score_stretch512_view(ok) > _score_stretch512_view(mesh)
    assert _score_stretch512_view(ok) > _score_stretch512_view(blank)

    class _OrbitPF:
        def snap_point(self, p):
            return np.asarray(p, dtype=np.float32)

        def is_navigable(self, _p):
            return True

    poses = list(_iter_orbit_agents(_OrbitPF(), (0.0, 0.1, 0.0), 0.1, (1.8,), (0, 90)))
    assert len(poses) == 2
    assert abs(poses[0][3] - 1.8) < 1e-3

    print("[render_iin_selftest] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
