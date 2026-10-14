import os
import sys
import unittest

from lavira_m3.mapping.inflate import PLAN_INFLATE
from lavira_m3.mapping.observed_grid import full_grids
from lavira_m3.mapping.peers import ENVELOPE, PeerCache, allowed, occupied_cells
from lavira_m3.mapping.protocol import Meta

#!/usr/bin/env python3


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cell(geometry, x, y):
    w, h, res, ox, oy = geometry
    return int((x - ox) / res), int((y - oy) / res)


class PeerEnvelopeTest(unittest.TestCase):
    def test_missing_updates_preserve_age_and_nearby_peer_protection(self):
        cache = PeerCache()
        own = {"pose": {"position": [0, 0, 0], "yaw": 0, "age_s": 0}}
        peer = {"pose": {"position": [2, 0, 0], "yaw": 0, "age_s": 0}}
        cache.observe({"sources": {"small_a": own, "small_b": peer}}, now=10)
        cache.observe({"sources": {}}, now=14)
        view = cache.view("small_a", now=15)
        self.assertEqual(view["peers"][0]["age"], 5)
        self.assertTrue(allowed(view, 0, 0, 0, 0))
        cache.observe({"sources": {"small_a": own}}, now=15.01)
        view = cache.view("small_a", now=15.01)
        self.assertEqual(view["peers"][0]["x"], 2)
        self.assertGreater(view["peers"][0]["age"], 5)
        self.assertFalse(allowed(view, 0, 0, 0, 0))

    def test_display_sizes_unchanged(self):
        self.assertEqual(ENVELOPE["large"], (0.60, 0.60, 0.475))
        self.assertEqual(ENVELOPE["small_a"], (0.60, 0.20, 0.20))
        self.assertEqual(ENVELOPE["small_b"], (0.60, 0.20, 0.20))
        self.assertEqual(PLAN_INFLATE["large"], 0.80)
        self.assertEqual(PLAN_INFLATE["small_a"], 0.4)
        self.assertEqual(PLAN_INFLATE["small_b"], 0.35)

    def test_occupied_cells_use_envelope_not_a_point(self):
        geometry = (40, 40, 0.1, -2.0, -2.0)
        small_a = {"name": "small_a", "x": 0.0, "y": 0.0, "yaw": 0.0}
        cells = occupied_cells({"peers": [small_a]}, geometry)
        self.assertIn(_cell(geometry, 0.15, 0.0), cells)
        self.assertIn(_cell(geometry, -0.55, 0.0), cells)
        self.assertNotIn(_cell(geometry, 0.35, 0.0), cells)
        large = {"name": "large", "x": 0.0, "y": 0.0, "yaw": 0.0}
        big = occupied_cells({"peers": [large]}, geometry)
        self.assertIn(_cell(geometry, 0.50, 0.0), big)
        self.assertNotIn(_cell(geometry, 0.50, 0.0), cells)

    def test_own_body_is_not_an_obstacle(self):
        geometry = (80, 40, 0.1, -2.0, -2.0)
        view = {
            "own": {"name": "small_b", "x": 0.0, "y": 0.0, "yaw": 0.0},
            "peers": [{"name": "large", "x": 2.0, "y": 0.0, "yaw": 0.0}],
        }
        cells = occupied_cells(view, geometry)
        self.assertNotIn(_cell(geometry, 0.0, 0.0), cells)
        self.assertIn(_cell(geometry, 2.0, 0.0), cells)

    def test_robot_inflate_uses_own_radius(self):
        meta = Meta(resolution=0.1, origin=(0.0, 0.0, 0.0), frame="map")
        poses = {
            "peer_view": {
                "own": None,
                "peers": [{"name": "small_a", "x": 0.0, "y": 0.0, "yaw": 0.0}],
            }
        }
        geo75, _, peers75, _, _ = full_grids(meta, [], poses, inflate_m=0.75)
        geo40, _, peers40, _, _ = full_grids(meta, [], poses, inflate_m=0.40)
        probe = 0.80

        def value(geo, data, x, y):
            w, _h, res, ox, oy = geo
            ix = int((x - ox) // res)
            iy = int((y - oy) // res)
            return data[iy * w + ix]

        self.assertEqual(value(geo75, peers75, probe, 0.0), 100)
        self.assertNotEqual(value(geo40, peers40, probe, 0.0), 100)


if __name__ == "__main__":
    unittest.main()
