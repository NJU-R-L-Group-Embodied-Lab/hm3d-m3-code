import os
import sys
import unittest

from lavira_m3.mapping.inflate import PLAN_INFLATE, PLAN_SOFT_INFLATE
from lavira_m3.mapping.protocol import Meta
from lavira_m3.mapping.remote_planning import layers_for_robot

#!/usr/bin/env python3


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _status(*poses):
    sources = {}
    for name, x, y, yaw in poses:
        sources[name] = {
            "pose": {
                "position": [x, y, 0.0],
                "yaw": yaw,
                "age_s": 0.0,
            }
        }
    return {"sources": sources}


def _value(payload, layer, x, y):
    w = payload["width"]
    res = payload["resolution"]
    ox, oy = payload["origin"]
    ix = int((x - ox) // res)
    iy = int((y - oy) // res)
    return payload[layer][iy * w + ix]


class RemotePlanningTest(unittest.TestCase):
    def test_small_b_uses_ego_hard_band_only(self):
        meta = Meta(resolution=0.1, origin=(0.0, 0.0, 0.0), frame="map")
        payload = layers_for_robot(
            _status(("small_b", 0.0, 0.0, 0.0)), {}, meta, "small_b", now=0.0
        )
        self.assertEqual(payload["hard_inflate_m"], PLAN_INFLATE["small_b"])
        self.assertEqual(payload["soft_inflate_m"], PLAN_SOFT_INFLATE["small_b"])
        self.assertEqual(payload["hard_inflate_m"], 0.35)
        self.assertEqual(payload["soft_inflate_m"], 0.0)

    def test_small_b_hard_boundary_has_no_soft_band(self):
        meta = Meta(resolution=0.1, origin=(0.0, 0.0, 0.0), frame="map")
        payload = layers_for_robot(
            _status(("small_b", 0.0, 0.0, 0.0)), {(0, 0): 2}, meta, "small_b", now=0.0
        )
        self.assertEqual(_value(payload, "environment", 0.25, 0.05), 100)
        self.assertEqual(_value(payload, "environment", 0.35, 0.05), 100)
        self.assertEqual(_value(payload, "environment", 0.35, 0.15), 100)  # sqrt(.3^2+.1^2) < .35
        self.assertLessEqual(_value(payload, "environment", 0.35, 0.25), 0)  # sqrt(.3^2+.2^2) > .35
        self.assertLessEqual(_value(payload, "environment", 0.45, 0.05), 0)

    def test_small_a_uses_envelope_and_own_inflate(self):
        meta = Meta(resolution=0.1, origin=(0.0, 0.0, 0.0), frame="map")
        status = _status(("large", 2.0, 0.0, 0.0), ("small_a", 0.0, 0.0, 0.0))
        payload = layers_for_robot(status, {}, meta, "small_a", now=0.0)
        self.assertTrue(payload["available"])
        self.assertEqual(payload["inflate_m"], PLAN_INFLATE["small_a"])
        self.assertEqual(_value(payload, "peers", 2.0, 0.0), 100)
        self.assertEqual(_value(payload, "peers", 2.50, 0.0), 100)
        self.assertEqual(_value(payload, "peers", 2.85, 0.0), 100)
        self.assertNotEqual(_value(payload, "peers", 3.20, 0.0), 100)
        self.assertNotEqual(_value(payload, "peers", 0.0, 0.0), 100)

    def test_compact_overlay_separates_halo(self):
        meta = Meta(resolution=0.1, origin=(0.0, 0.0, 0.0), frame="map")
        status = _status(("large", 2.0, 0.0, 0.0), ("small_a", 0.0, 0.0, 0.0))
        payload = layers_for_robot(status, {(0, 0): 2}, meta, "small_a", now=0.0)
        occupied = {tuple(cell) for cell in payload["occupied"]}
        halo = {tuple(cell) for cell in payload["inflate"]}
        self.assertIn((0, 0), occupied)
        self.assertIn((20, 0), occupied)
        self.assertTrue(halo)
        self.assertTrue(occupied.isdisjoint(halo))
        soft = {tuple(cell) for cell in payload["soft"]}
        self.assertFalse(soft)
        self.assertIn((3, 0), halo)

    def test_own_body_excluded_from_small_a_peers(self):
        meta = Meta(resolution=0.1, origin=(0.0, 0.0, 0.0), frame="map")
        status = _status(("small_a", 0.0, 0.0, 0.0), ("small_b", 3.0, 0.0, 0.0))
        payload = layers_for_robot(status, {}, meta, "small_a", now=0.0)
        self.assertNotEqual(_value(payload, "peers", 0.0, 0.0), 100)
        self.assertEqual(_value(payload, "peers", 3.0, 0.0), 100)


if __name__ == "__main__":
    unittest.main()
