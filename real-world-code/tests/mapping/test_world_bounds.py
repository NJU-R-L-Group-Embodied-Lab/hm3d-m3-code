#!/usr/bin/env python3
"""Fixed 100 m canvas: world-absolute IDs, drop out-of-map cells."""

import os
import sys
import unittest

from lavira_m3.mapping.builder import Builder2D
from lavira_m3.mapping.fusion import Fusion2D
from lavira_m3.mapping.protocol import Meta, Record
from lavira_m3.mapping.world import IX_MAX, IX_MIN, in_world_ix, in_world_xy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WorldBoundsTest(unittest.TestCase):
    def test_origin_is_small_a_birth(self):
        self.assertTrue(in_world_xy(0.0, 0.0))
        self.assertTrue(in_world_xy(2.5, 0.0))
        self.assertTrue(in_world_xy(2.5, 2.0))
        self.assertTrue(in_world_xy(-49.99, 49.99))
        self.assertFalse(in_world_xy(50.0, 0.0))
        self.assertFalse(in_world_xy(0.0, -50.01))
        self.assertTrue(in_world_ix(IX_MIN, IX_MAX))
        self.assertFalse(in_world_ix(IX_MAX + 1, 0))

    def test_builder_drops_out_of_map_hits(self):
        b = Builder2D("small_a")
        wall = ((1.05, 0.05, 0.4), (1.05, 0.15, 0.4), (1.05, 0.25, 0.4))
        far = ((60.05, 0.05, 0.4), (60.05, 0.15, 0.4), (60.05, 0.25, 0.4))
        for t in (1.0, 1.3, 1.6):
            b.scan(wall + far, (0.0, 0.0, 0.2), t, (0.0, 0.0, 0.0))
        self.assertTrue({(10, 0), (10, 1), (10, 2)} <= set(b.cells))
        self.assertFalse(any(ix >= 500 or ix <= -501 for ix, _iy in b.cells))

    def test_fusion_drops_out_of_map_rows(self):
        f = Fusion2D()
        self.assertTrue(
            f.apply(
                "small_a",
                Meta(type=1, seq=1, map_id="g-1"),
                [
                    Record(10, 0, 0, 2),
                    Record(10, 1, 0, 2),
                    Record(10, 2, 0, 2),
                    Record(600, 0, 0, 2),
                    Record(-600, 1, 0, 2),
                ],
            )
        )
        self.assertEqual(f.fused.cells[(10, 0)], 2)
        self.assertNotIn((600, 0), f.fused.cells)
        self.assertNotIn((-600, 1), f.fused.cells)


if __name__ == "__main__":
    unittest.main()
