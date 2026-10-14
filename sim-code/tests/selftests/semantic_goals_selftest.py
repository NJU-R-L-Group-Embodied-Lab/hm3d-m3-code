"""Habitat-free checks for semantic goal mining helpers."""
from __future__ import annotations

import math

from benchmark import render_iin as sg
from tests.selftests.benchmark_selftest import MockNav


def test_category_filter():
    assert sg.is_goal_category("chair")
    assert sg.is_goal_category("TV Monitor")
    assert not sg.is_goal_category("wall")
    assert not sg.is_goal_category("CEILING")
    assert not sg.is_goal_category("")


def test_ring_and_yaw():
    offs = sg.ring_offsets(radii_m=(1.0,), n_angles=4)
    assert len(offs) == 4
    pts = sg.propose_ring_points([0.0, 1.0, 0.0])
    assert len(pts) >= 8
    yaw = sg.yaw_look_at([1.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    assert math.isfinite(yaw)
    q = sg.quat_xyzw_from_yaw(0.0)
    assert abs(q[3] - 1.0) < 1e-6


def test_viewpoints_and_scene_goals():
    nav = MockNav(extent=12.0, seed=0)
    recs = [
        {"category": "chair", "object_id": "c0", "object_name": "chair",
         "position": [3.0, 0.5, 3.0], "sizes": [0.5, 0.8, 0.5]},
        {"category": "plant", "object_id": "p0", "object_name": "plant",
         "position": [8.0, 0.4, 2.0], "sizes": [0.4, 1.0, 0.4]},
        {"category": "bed", "object_id": "b0", "object_name": "bed",
         "position": [2.0, 0.4, 9.0], "sizes": [2.0, 0.6, 1.5]},
        {"category": "wall", "object_id": "w0", "object_name": "wall",
         "position": [1.0, 1.0, 1.0], "sizes": [4.0, 3.0, 0.2]},
    ]
    scene = sg.make_scene_goals("mp3d/vyrNrziPKCB/vyrNrziPKCB.glb", nav, recs)
    assert scene is not None
    assert scene.scene_key == "vyrNrziPKCB"
    assert "chair" in scene.by_category
    assert "wall" not in scene.by_category
    assert scene.by_category["chair"][0].view_points


def test_dataset_config():
    assert sg.scene_dataset_config_for("data/scene_datasets/mp3d/vyrNrziPKCB/vyrNrziPKCB.glb") == ""
    assert sg.scene_dataset_config_for(
        "data/scene_datasets/hm3d-1.0/hm3d_v0.2/train/00792-4tdJ3qe1x7P/4tdJ3qe1x7P.basis.glb"
    ) == ""
    cfg = sg.scene_dataset_config_for(
        "hm3d/val/00800-x/x.basis.glb", "data/scene_datasets/hm3d-1.0"
    )
    assert cfg.endswith("hm3d_annotated_basis.scene_dataset_config.json")
    assert sg.scene_dataset_config_for("data/qual_scenes/00792-4tdJ3qe1x7P/x.basis.glb") == ""


def test_instances_from_records():
    by = sg.instances_from_records([
        {"category": "Sofa", "object_id": 1, "position": [0, 0, 0]},
        {"category": "void", "object_id": 2, "position": [1, 0, 1]},
    ])
    assert list(by) == ["sofa"]


def main() -> None:
    test_category_filter()
    test_ring_and_yaw()
    test_viewpoints_and_scene_goals()
    test_dataset_config()
    test_instances_from_records()
    print("semantic_goals_selftest: ok")


if __name__ == "__main__":
    main()
