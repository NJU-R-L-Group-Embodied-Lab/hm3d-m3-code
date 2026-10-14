"""Observed occupancy requires repeated, spatially supported returns."""

import math

from lavira_m3.mapping.builder import HAG_MAX, HAG_MIN, SENSOR_AGL, Builder2D, height_agl
from lavira_m3.mapping.fusion import Fusion2D
from lavira_m3.mapping.protocol import Meta, Record


def wall(x=2.05, z=0.4):
    return [(x, y, z) for y in (1.05, 1.15, 1.25)]


def confirm(builder, points, body=(0, 0, 0), start=1.0):
    for t in (start, start + 0.3, start + 0.6):
        assert builder.scan(points, (0, 0, 0.2), t, body)


def test_height_is_relative_to_configured_sensor_mount():
    assert (HAG_MIN, HAG_MAX) == (0.05, 1.1)
    for name in SENSOR_AGL:
        assert math.isclose(height_agl(0.2, 0.2, name), SENSOR_AGL[name])
        b = Builder2D(name)
        confirm(b, wall(z=2))
        assert not b.cells
        confirm(b, wall(z=0.2), start=2.0)
        assert b.cells


def test_single_scan_and_isolated_hit_are_not_obstacles():
    b = Builder2D("small_a")
    b.scan(wall(), (0, 0, 0.2), 1.0, (0, 0, 0))
    assert not b.cells
    confirm(b, [(3.05, 1.05, 0.4)], start=2.0)
    assert not b.cells
    confirm(b, wall(), start=3.0)
    assert set(b.cells) == {(20, 10), (20, 11), (20, 12)}
    assert all(row.iz == 0 and row.state == 2 for row in b.snapshot_rows())


def test_self_and_peer_returns_are_filtered():
    for own in ("small_a", "large"):
        b = Builder2D(own)
        confirm(b, [(0.1, y, 0.3) for y in (0, 0.1, 0.2)])
        assert not b.cells
    b = Builder2D("small_a")
    peers = [dict(name="large", x=2.05, y=1.1, yaw=0)]
    for t in (1.0, 1.3, 1.6):
        b.scan(wall(), (0, 0, 0.2), t, (0, 0, 0), peers)
    assert not b.cells


def test_missing_scans_do_not_invent_free_space():
    b = Builder2D("small_a", decay_sec=0.8)
    confirm(b, wall())
    b.scan([], (0, 0, 0.2), 2.0, (0, 0, 0))
    assert b.cells
    b.scan([], (0, 0, 0.2), 2.5, (0, 0, 0))
    assert b.cells
    assert all(row.state == 2 for row in b.delta_rows())


def test_fusion_occupied_wins_until_every_source_clears():
    f = Fusion2D()
    cells = [Record(10, i, 0, 2) for i in range(3)]
    assert f.apply("small_a", Meta(type=1, seq=1, map_id="one"), cells)
    assert f.apply(
        "large", Meta(type=1, seq=1, map_id="two"), [Record(10, i, 0, 1) for i in range(3)]
    )
    assert f.fused.cells[(10, 0)] == 2
