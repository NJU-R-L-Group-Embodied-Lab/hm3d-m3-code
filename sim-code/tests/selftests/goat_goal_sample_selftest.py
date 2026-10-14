"""Offline checks for GOAT / Krantz goal-image grids (no Habitat)."""
from __future__ import annotations

import numpy as np

from benchmark import config as _cfg
from benchmark import render_iin as g


def main() -> None:
    assert _cfg.IIN_CAMERA_OFFICIAL["width_px"] == 512
    assert _cfg.IIN_CAMERA_OFFICIAL["height_px"] == 512
    assert _cfg.TN_CAMERA_GOAT["width_px"] == 512
    assert _cfg.TN_CAMERA_GOAT["height_px"] == 512
    assert g.count_iin_grid() == 4 * 36
    assert g.count_tn_grid() == 3 * 36
    assert _cfg.IIN_OFFICIAL_THETA_STEP_DEG == 10.0
    assert _cfg.TN_GOAT_THETA_STEP_DEG == 10.0
    assert _cfg.IIN_OFFICIAL_RADII_M == (0.5, 1.0, 1.5, 2.0)
    assert _cfg.TN_GOAT_RADII_M == (1.0, 1.5, 2.0)

    rng = np.random.default_rng(0)
    iin = list(g.iter_iin_cameras([0.0, 0.0, 0.0], 0.0, rng))
    assert len(iin) == 144
    hs = [c["position"][1] for c in iin]
    assert min(hs) >= 0.8 - 1e-6 and max(hs) <= 1.5 + 1e-6
    fov = [c["hfov"] for c in iin]
    assert min(fov) >= 60.0 - 1e-6 and max(fov) <= 120.0 + 1e-6

    tn = list(g.iter_tn_cameras([1.0, 0.2, -3.0], 0.5))
    assert len(tn) == 108
    assert all(abs(c["position"][1] - (0.5 + 1.31)) < 1e-6 for c in tn)
    assert all(abs(c["hfov"] - 42.0) < 1e-6 for c in tn)

    assert abs(g.cf_threshold(0.0) - 0.02) < 1e-9
    assert abs(g.cf_threshold(1.0) - (0.0232 + 0.02)) < 1e-9
    assert g.iin_gates(0.71, 0.05, 1.0) is True
    assert g.iin_gates(0.50, 0.20, 1.0) is False
    assert g.iin_gates(0.80, 0.01, 1.0) is False

    sem = np.zeros((8, 8), dtype=np.int32)
    sem[:2, :2] = 7
    assert abs(g.frame_coverage(sem, 7) - 4.0 / 64.0) < 1e-9

    recs = [
        {"ok": False, "frame_coverage": 0.4, "object_coverage": 0.9},
        {"ok": True, "frame_coverage": 0.2, "object_coverage": 0.8},
        {"ok": True, "frame_coverage": 0.3, "object_coverage": 0.7},
    ]
    assert g.pick_iin_index(recs) == 2
    assert g.pick_tn_index(recs) == 0

    a = g.goal_rng({"object_id": "abc"})
    b = g.goal_rng({"object_id": "abc"})
    assert a.integers(1000) == b.integers(1000)
    print("[goat_goal_sample_selftest] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
