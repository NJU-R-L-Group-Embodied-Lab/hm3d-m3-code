"""Offline selftest for the DM3-official port (no sim / no GPU needed).

Checks the adapter math (shared-frame GPS/compass vs the fork's sensor
formulas), depth preprocessing, official config completeness, and vocabulary
construction.

    python -m tests.selftests.dm3_runner_selftest
"""
from __future__ import annotations

import math
import numpy as np

from benchmark.dm3_official.runner import (DM3OfficialRunner, build_dm3_config, resolve_iin_path,
                     MIN_DEPTH_REPLACEMENT_VALUE, MAX_DEPTH_REPLACEMENT_VALUE)


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_pose_math():
    import quaternion  # noqa: F401  (installs np.quaternion dtype)

    class _R:  # minimal runner mixin access without __init__
        pass

    r = _R()
    fn = DM3OfficialRunner._pose_to_gps_compass

    origin = np.array([1.0, 0.5, 2.0])
    q_frame = np.quaternion(1.0, 0.0, 0.0, 0.0)  # identity frame

    # identity rotation, at origin -> gps (0,0), compass 0
    gps, comp = fn(r, origin, [0, 0, 0, 1], origin, q_frame)
    _check(np.allclose(gps, [0, 0], atol=1e-6), f"gps@origin={gps}")
    _check(abs(comp[0]) < 1e-6, f"compass@identity={comp}")

    # move 2 m forward (habitat forward = -Z) -> home_robot gps x = +2
    gps, _ = fn(r, origin + np.array([0, 0, -2.0]), [0, 0, 0, 1], origin, q_frame)
    _check(np.allclose(gps, [2.0, 0.0], atol=1e-5), f"gps forward={gps}")

    # strafe left (+X in habitat world? left of -Z is -X; use +X -> right side)
    gps, _ = fn(r, origin + np.array([1.5, 0, 0]), [0, 0, 0, 1], origin, q_frame)
    _check(np.allclose(gps, [0.0, -1.5], atol=1e-5), f"gps strafe={gps}")

    # turn left 90 deg (yaw +90 about +Y): compass should be +pi/2 (left-positive)
    yaw = math.pi / 2
    q_left = [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]  # xyzw
    _, comp = fn(r, origin, q_left, origin, q_frame)
    _check(abs(comp[0] - math.pi / 2) < 1e-5, f"compass left90={comp}")

    # non-identity frame: frame rotated +90 (facing -X); agent 2 m along frame
    # forward => world pos origin + (-2, 0, 0); gps should read x=+2
    qf_xyzw = [0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]
    qf = np.quaternion(qf_xyzw[3], qf_xyzw[0], qf_xyzw[1], qf_xyzw[2])
    gps, comp = fn(r, origin + np.array([-2.0, 0, 0]), qf_xyzw, origin, qf)
    _check(np.allclose(gps, [2.0, 0.0], atol=1e-5), f"gps framed={gps}")
    _check(abs(comp[0]) < 1e-5, f"compass framed={comp}")
    print("[selftest] pose math OK")


def test_depth():
    class _R:
        _dmin = 0.1
        _dmax = 5.0

    fn = DM3OfficialRunner._depth_to_home_robot
    # lavira obs are normalized [0,1] over [0.1, 5.0] m
    d = np.array([[0.0, (1.0 - 0.1) / 4.9, 1.0]], dtype=np.float32)
    out = fn(_R(), d)
    _check(out.shape == (1, 3), f"depth shape {out.shape}")
    _check(out[0, 0] == MIN_DEPTH_REPLACEMENT_VALUE, "zero depth -> MIN sentinel")
    _check(abs(out[0, 1] - 1.0) < 1e-4, f"1m metric roundtrip, got {out[0,1]}")
    _check(out[0, 2] == MAX_DEPTH_REPLACEMENT_VALUE, "max -> MAX sentinel")
    print("[selftest] depth OK")


def test_config():
    cfg = build_dm3_config(dump_location="/tmp/x", exp_name="t")
    # every attribute the vendored stack reads (spot-check the critical ones)
    assert cfg.AGENT.max_steps == 500
    assert cfg.AGENT.COMMUNICATION.radius == 5
    assert cfg.AGENT.COMMUNICATION.cooldown == 10
    assert cfg.AGENT.SEMANTIC_MAP.map_size_cm == 4800
    assert cfg.AGENT.SEMANTIC_MAP.map_resolution == 5
    assert cfg.AGENT.SEMANTIC_MAP.min_obs_height_cm == 20
    assert cfg.AGENT.SUPERGLUE.score_thresh_image == 24.5
    assert cfg.AGENT.SUPERGLUE.score_thresh_lang == 0.24
    assert cfg.AGENT.cat_match_threshold == 0.4
    assert cfg.AGENT.panorama_start == 1
    assert cfg.habitat.simulator.turn_angle == 30
    assert cfg.habitat.simulator.agents.agent0.sim_sensors.depth_sensor.hfov == 79
    assert cfg.habitat.task.type == "MultiAgentGoat-v1"
    assert cfg.ENVIRONMENT.max_subtasks_per_episode == 10
    assert cfg.SEQ == 0 and cfg.USE_YOLO == 0 and cfg.GROUND_TRUTH_SEMANTICS == 0
    print("[selftest] config OK")


def test_vocab():
    class _Ep:
        def __init__(self, cats):
            self.goals_set = [{"object_category": c} for c in cats]

    vocab = DM3OfficialRunner.build_vocabulary([_Ep(["bed", "chair"]), _Ep(["oven and stove", "bed"])])
    _check(vocab == ["bed", "chair", "oven_and_stove"], f"vocab={vocab}")
    print("[selftest] vocab OK")


def test_iin_path(tmp_path=None):
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        name = "ep__g0.png"
        full = os.path.join(td, name)
        open(full, "wb").close()
        got = resolve_iin_path(f"relative/dir/{name}", td)
        _check(os.path.samefile(got, full), f"resolve basename {got}")
        got = resolve_iin_path(full, td)
        _check(os.path.samefile(got, full), f"resolve abs {got}")
        _check(resolve_iin_path(None, td) is None, "resolve None")
    print("[selftest] iin path OK")


if __name__ == "__main__":
    test_config()
    test_vocab()
    test_depth()
    test_pose_math()
    test_iin_path()
    print("[dm3_official.runner_selftest] ALL CHECKS PASSED")
