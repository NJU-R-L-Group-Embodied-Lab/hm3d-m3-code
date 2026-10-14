"""CPU tests for OWL stride, shm RGB, and skip-after-detect. No Habitat."""
from __future__ import annotations

import tempfile

import numpy as np

from benchmark.owl_shm import F32ShmSlot, RgbShmSlot, read_f32, read_rgb
from benchmark.ovon_od import (
    STOP_RADIUS,
    OvonODSkill,
    approach_depth,
    expand_box,
    face_box_turn,
    iin_sticky_ok,
    should_run_owl,
    sticky_stop_radius,
)


def test_should_run_owl():
    assert should_run_owl(0, 4) is True
    assert should_run_owl(4, 4) is True
    assert should_run_owl(1, 4) is False
    assert should_run_owl(1, 4, force=True) is True
    assert should_run_owl(3, 1) is True


def test_shm_roundtrip():
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    with tempfile.TemporaryDirectory() as td:
        slot = RgbShmSlot(td, "a0")
        meta = slot.write(rgb)
        got = read_rgb(td, meta["shm_name"], expect_seq=meta["seq"])
        assert got is not None
        assert got.shape == (2, 2, 3)
        assert np.array_equal(got, rgb)
        rgb2 = rgb + 1
        meta2 = slot.write(rgb2)
        got2 = read_rgb(td, "a0", expect_seq=meta2["seq"])
        assert np.array_equal(got2, rgb2)
        slot.close()


def test_f32_shm_roundtrip():
    depth = np.linspace(0.1, 5.0, 12, dtype=np.float32).reshape(3, 4)
    with tempfile.TemporaryDirectory() as td:
        slot = F32ShmSlot(td, "d0")
        meta = slot.write(depth)
        got = read_f32(td, meta["shm_name"], expect_seq=meta["seq"])
        assert got is not None and got.shape == (3, 4)
        assert np.allclose(got, depth)


def test_skip_detect_when_already_found():
    skill = OvonODSkill(owl_url="")
    skill.detected = True
    skill.goal_xz = (1.0, 2.0)

    class Boom:
        def detect(self, *a, **k):
            raise AssertionError("OWL should not run after commit")

    skill.client = Boom()
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    xz = skill.update(rgb, depth, {"owl_query": "chair", "modality": "ON"}, (0.0, 0.0), 0.0)
    assert xz == (1.0, 2.0)


def _depth_norm(meters: float, dmin: float = 0.1, dmax: float = 5.0) -> float:
    return (float(meters) - dmin) / (dmax - dmin)


def _iin_skill(match):
    skill = OvonODSkill(owl_url="")
    skill._goal_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    scores = list(match) if isinstance(match, (list, tuple)) else [float(match)]

    class Fake:
        def __init__(self):
            self.last_crop = None
            self.scores = list(scores)

        def detect(self, *a, **k):
            return [([16.0, 16.0, 48.0, 48.0], 0.8)]

        def match_image(self, crop, *a, **k):
            self.last_crop = crop
            if self.scores:
                return float(self.scores.pop(0))
            return 0.0

    skill.client = Fake()
    return skill


def _rgb_depth(meters: float):
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.full((64, 64), _depth_norm(meters), dtype=np.float32)
    return rgb, depth


def test_iin_far_miss_approaches():
    skill = _iin_skill(3.0)
    rgb, depth = _rgb_depth(3.4)
    xz = skill.update(rgb, depth, {"owl_query": "chair", "modality": "IIN"}, (0.0, 0.0), 0.0)
    assert xz is not None
    assert skill.detected is False
    assert skill.approaching is True
    assert skill.n_approach == 1
    assert skill.n_reject == 0


def test_iin_close_miss_rejects():
    skill = _iin_skill(3.0)
    rgb, depth = _rgb_depth(1.2)
    xz = skill.update(rgb, depth, {"owl_query": "chair", "modality": "IIN"}, (0.0, 0.0), 0.0)
    assert skill.detected is False
    assert skill.approaching is False
    assert skill.n_reject == 1
    assert xz is None
    xz2 = skill.update(rgb, depth, {"owl_query": "chair", "modality": "IIN"}, (0.0, 0.0), 0.0)
    assert xz2 is None
    assert skill.n_reject == 1


def test_iin_far_pass_still_approaches():
    skill = _iin_skill(30.0)
    rgb, depth = _rgb_depth(3.4)
    xz = skill.update(rgb, depth, {"owl_query": "chair", "modality": "IIN"}, (0.0, 0.0), 0.0)
    assert xz is not None
    assert skill.detected is False
    assert skill.approaching is True
    assert skill.n_reject == 0


def test_expand_box_and_standoff():
    out = expand_box([16, 16, 48, 48], 64, 64, scale=2.0)
    assert out[0] < 16 and out[2] > 48
    assert approach_depth(3.4) == 2.4
    assert approach_depth(1.2) == 0.8


def test_iin_far_walks_in_front_of_box():
    skill = _iin_skill(30.0)
    rgb, depth = _rgb_depth(3.4)
    xz = skill.update(rgb, depth, {"owl_query": "chair", "modality": "IIN"}, (0.0, 0.0), 0.0)
    assert skill.approaching is True
    assert xz is not None
    dist = (xz[0] ** 2 + xz[1] ** 2) ** 0.5
    assert dist < 3.2


def test_iin_far_uses_larger_sg_crop():
    skill = _iin_skill(3.0)
    rgb, depth = _rgb_depth(3.4)
    skill.update(rgb, depth, {"owl_query": "chair", "modality": "IIN"}, (0.0, 0.0), 0.0)
    crop = skill.client.last_crop
    assert crop is not None
    assert crop.shape[0] * crop.shape[1] > 32 * 32


def test_iin_close_pass_commits():
    skill = _iin_skill(30.0)
    rgb, depth = _rgb_depth(1.2)
    xz = skill.update(rgb, depth, {"owl_query": "chair", "modality": "IIN"}, (0.0, 0.0), 0.0)
    assert skill.detected is True
    assert skill.approaching is False
    assert xz is not None


def test_iin_hold_and_standoff_stop():
    assert sticky_stop_radius(24.5) == STOP_RADIUS
    assert sticky_stop_radius(24.4) == STOP_RADIUS


def test_iin_sticky_survives_close_miss():
    assert iin_sticky_ok(24.5) is True
    assert iin_sticky_ok(24.4) is False
    skill = _iin_skill([30.0, 3.0])
    far_rgb, far_depth = _rgb_depth(3.4)
    skill.update(far_rgb, far_depth, {"owl_query": "chair", "modality": "IIN"}, (0.0, 0.0), 0.0)
    assert skill.detected is False
    assert skill.approaching is True
    assert skill.box_xz is not None
    close_rgb, close_depth = _rgb_depth(1.2)
    xz = skill.update(
        close_rgb, close_depth, {"owl_query": "chair", "modality": "IIN"}, (0.0, 0.0), 0.0,
    )
    assert skill.detected is True
    assert skill.n_reject == 0
    assert xz is not None


def test_face_box_turn():
    assert face_box_turn(0.4) == "left"
    assert face_box_turn(-0.4) == "right"
    assert face_box_turn(0.0) == "left"
