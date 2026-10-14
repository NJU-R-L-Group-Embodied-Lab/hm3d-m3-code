import math
import os
import sys
import unittest

from lavira_m3.mapping.inflate import (
    HARD_INFLATE_M,
    INFLATE_M,
    PLAN_INFLATE,
    PLAN_SOFT_INFLATE,
    SOFT_INFLATE_M,
    inflate_layers,
    open_inflate_start,
)
from lavira_m3.mapping.observed_grid import full_grids
from lavira_m3.mapping.peers import ENVELOPE
from lavira_m3.mapping.protocol import Meta, Record

#!/usr/bin/env python3


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class InflateTest(unittest.TestCase):
    def test_hard_only_for_ego(self):
        self.assertEqual(HARD_INFLATE_M, 0.80)
        self.assertEqual(SOFT_INFLATE_M, 0.0)
        self.assertEqual(INFLATE_M, HARD_INFLATE_M)
        self.assertEqual(PLAN_INFLATE, {"large": 0.80, "small_a": 0.4, "small_b": 0.35})
        self.assertEqual(PLAN_SOFT_INFLATE, {"large": 0.0, "small_a": 0.0, "small_b": 0.0})

    def test_layers_mark_hard_band_only_by_default(self):
        width = height = 21
        data = [0] * (width * height)
        data[10 * width + 10] = 100
        hard, combined = inflate_layers(data, width, height, 0.1)
        self.assertEqual(hard[10 * width + 10], 100)
        self.assertEqual(hard[10 * width + 15], 100)
        self.assertEqual(hard[10 * width + 16], 100)
        self.assertEqual(hard[10 * width + 17], 100)
        self.assertEqual(hard[10 * width + 18], 100)
        self.assertEqual(hard[10 * width + 19], 0)
        self.assertEqual(hard[12 * width + 16], 100)
        self.assertEqual(hard[13 * width + 16], 100)
        self.assertEqual(hard[11 * width + 17], 100)
        self.assertEqual(hard[14 * width + 16], 100)  # sqrt(.6^2+.4^2) < .80
        self.assertEqual(combined, hard)

    def test_layers_can_still_paint_an_explicit_soft_band(self):
        width = height = 21
        data = [0] * (width * height)
        data[10 * width + 10] = 100
        hard, combined = inflate_layers(data, width, height, 0.1, 0.65, 0.30)
        self.assertEqual(hard[10 * width + 16], 100)
        self.assertTrue(0 < combined[10 * width + 17] < 100)
        self.assertGreater(combined[10 * width + 17], combined[10 * width + 18])
        self.assertEqual(combined[10 * width + 20], 0)

    def test_full_grids_inflates_environment_hard_only(self):
        meta = Meta(resolution=0.1, origin=(0.0, 0.0, 0.0), frame="map")
        geometry, _, peers, environment, planning = full_grids(
            meta, [Record(0, 0, 0, 2)], {"peer_view": {"own": None, "peers": []}}
        )
        width, height = geometry[0], geometry[1]
        ox = int(round(-geometry[3] / 0.1))
        oy = int(round(-geometry[4] / 0.1))
        self.assertEqual(environment[(oy * width) + ox], 100)
        self.assertEqual(environment[(oy * width) + ox + 5], 100)
        self.assertEqual(environment[(oy * width) + ox + 6], 100)
        self.assertEqual(environment[(oy * width) + ox + 7], 100)
        self.assertEqual(environment[(oy * width) + ox + 8], 100)
        self.assertEqual(environment[(oy * width) + ox + 9], 0)
        self.assertEqual(planning[(oy * width) + ox + 6], 100)
        self.assertTrue(all(v < 100 for v in peers))
        self.assertGreaterEqual(width, 101)
        self.assertGreaterEqual(height, 101)
        self.assertAlmostEqual(geometry[3], -5.0)
        self.assertAlmostEqual(geometry[4], -5.0)
        self.assertEqual(environment[0], 0)
        self.assertTrue(all(v in (0, 100) for v in environment))

    def _cell(self, geometry, x, y):
        width, _height, res, ox, oy = geometry
        return math.floor((y - oy) / res) * width + math.floor((x - ox) / res)

    def test_halo_start_opens_escape_only(self):
        meta = Meta(resolution=0.1, origin=(0.0, 0.0, 0.0), frame="map")
        own = {"name": "large", "x": 0.40, "y": 0.0, "yaw": 0.0}
        geometry, thin, _peers, environment, planning = full_grids(
            meta, [Record(0, 0, 0, 2)], {"peer_view": {"own": own, "peers": []}}
        )
        start = self._cell(geometry, 0.40, 0.0)
        raw = self._cell(geometry, 0.0, 0.0)
        side = self._cell(geometry, 0.0, 0.40)
        self.assertEqual(thin[raw], 100)
        self.assertEqual(environment[start], 100)
        self.assertEqual(planning[start], 0)
        self.assertEqual(planning[raw], 100)
        self.assertEqual(planning[side], 100)

    def test_raw_occupied_start_stays_blocked(self):
        meta = Meta(resolution=0.1, origin=(0.0, 0.0, 0.0), frame="map")
        own = {"name": "large", "x": 0.0, "y": 0.0, "yaw": 0.0}
        geometry, thin, _peers, _environment, planning = full_grids(
            meta, [Record(0, 0, 0, 2)], {"peer_view": {"own": own, "peers": []}}
        )
        raw = self._cell(geometry, 0.0, 0.0)
        self.assertEqual(thin[raw], 100)
        self.assertEqual(planning[raw], 100)

    def test_open_inflate_start_helper_ignores_raw(self):
        width = height = 11
        planning = [0] * (width * height)
        thin = [0] * (width * height)
        planning[5 * width + 5] = 100
        thin[5 * width + 5] = 100
        open_inflate_start(planning, thin, width, height, 0.1, -0.5, -0.5, 0.0, 0.0)
        self.assertEqual(planning[5 * width + 5], 100)

    def test_only_shortest_connected_corridor_is_cleared(self):
        width, height = 7, 3
        planning = [100] * (width * height)
        thin = [100] * len(planning)
        for x in range(1, 6):
            thin[width + x] = 0
        planning[width + 4] = 0
        before, raw = list(planning), list(thin)
        open_inflate_start(planning, thin, width, height, 1.0, 0.0, 0.0, 1.5, 1.5)
        self.assertEqual(
            {i for i, (a, b) in enumerate(zip(before, planning)) if a != b}, {8, 9, 10}
        )
        self.assertEqual(planning[width + 5], 100)
        self.assertEqual(thin, raw)

    def test_unknown_and_soft_cells_are_not_exits(self):
        for value in (-1, 20, 99):
            planning = [100, value, 0]
            open_inflate_start(planning, [0, 0, 0], 3, 1, 1.0, 0.0, 0.0, 0.5, 0.5)
            self.assertEqual(planning, [100, value, 0])

    def test_cannot_escape_diagonally_between_true_obstacles(self):
        planning = [100, 100, 100, 0]
        open_inflate_start(planning, [0, 100, 100, 0], 2, 2, 1.0, 0.0, 0.0, 0.5, 0.5)
        self.assertEqual(planning, [100, 100, 100, 0])

    def test_no_exit_does_not_open_any_cells(self):
        planning = [100] * 9
        open_inflate_start(planning, [0] * 9, 3, 3, 1.0, 0.0, 0.0, 1.5, 1.5)
        self.assertEqual(planning, [100] * 9)

    def test_free_or_outside_start_leaves_grid_unchanged(self):
        for x, y in ((1.5, 0.5), (-0.1, 0.5), (3.0, 0.5), (0.5, float("nan"))):
            planning = [100, 0, 100]
            open_inflate_start(planning, [0] * 3, 3, 1, 1.0, 0.0, 0.0, x, y)
            self.assertEqual(planning, [100, 0, 100])

    def test_teammate_raw_occupancy_is_never_opened(self):
        own = dict(name="small_b", x=0.05, y=0.05, yaw=0.0)
        peer = dict(name="small_a", x=0.0, y=0.0, yaw=0.0)
        geometry, thin, peers, environment, planning = full_grids(
            None, [], {"peer_view": {"own": own, "peers": [peer]}}, inflate_m=0.3, soft_m=0.0
        )
        start = self._cell(geometry, own["x"], own["y"])
        self.assertEqual(thin[start], 0)
        self.assertEqual(environment[start], 0)
        self.assertEqual(peers[start], 100)
        self.assertEqual(planning[start], 100)

    def test_teammate_halo_exit_disappears_after_pose_moves_to_free(self):
        peer = dict(name="small_a", x=0.0, y=0.0, yaw=0.0)
        halo_x = ENVELOPE["small_a"][1] + 0.25
        own = dict(name="small_b", x=halo_x, y=0.05, yaw=0.0)
        view = {"own": own, "peers": [peer]}
        geometry, _, peers, _, planning = full_grids(
            None, [], {"peer_view": view}, inflate_m=0.3, soft_m=0.0
        )
        start = self._cell(geometry, halo_x, 0.05)
        self.assertEqual(peers[start], 100)
        self.assertEqual(planning[start], 0)
        view["own"] = dict(own, x=halo_x + 1.1)
        geometry2, _, peers2, _, planning2 = full_grids(
            None, [], {"peer_view": view}, inflate_m=0.3, soft_m=0.0
        )
        self.assertEqual(planning2[self._cell(geometry2, halo_x, 0.05)], 100)
        self.assertEqual(planning2, peers2)


if __name__ == "__main__":
    unittest.main()
