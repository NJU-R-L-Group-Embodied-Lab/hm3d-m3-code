import os
import sys
import unittest

from lavira_m3.mapping.fusion import Fusion2D
from lavira_m3.mapping.protocol import Meta, Record

#!/usr/bin/env python3


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DeltaTest(unittest.TestCase):
    def test_delta_needs_matching_base(self):
        f = Fusion2D()
        self.assertTrue(f.apply("small_a", Meta(type=1, seq=1, map_id="g-1"), [Record(0, 0, 0, 2)]))
        self.assertFalse(
            f.apply("small_a", Meta(type=2, seq=3, base=1, map_id="g-1"), [Record(1, 0, 0, 2)])
        )
        self.assertTrue(
            f.apply("small_a", Meta(type=2, seq=2, base=1, map_id="g-1"), [Record(1, 0, 0, 2)])
        )
        self.assertEqual(f.sources["small_a"].cells[(1, 0)], 2)

    def test_drop_nonzero_z(self):
        f = Fusion2D()
        f.apply(
            "small_a", Meta(type=1, seq=1, map_id="g-1"), [Record(0, 0, 3, 2), Record(1, 1, 0, 2)]
        )
        self.assertNotIn((0, 0), f.sources["small_a"].cells)
        self.assertIn((1, 1), f.sources["small_a"].cells)

    def test_stale_pose_keeps_last_map(self):
        f = Fusion2D()
        meta = Meta(
            type=1,
            seq=1,
            map_id="g-1",
            pose_valid=True,
            position=(1.0, 2.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        rows = [Record(3, 0, 0, 2), Record(3, 1, 0, 2), Record(3, 2, 0, 2)]
        self.assertTrue(f.apply("small_a", meta, rows))
        f.heard["small_a"] -= 6.0
        st = f.status()["sources"]["small_a"]
        self.assertFalse(st["pose"]["live"])
        self.assertEqual(f.sources["small_a"].cells[(3, 0)], 2)
        self.assertEqual(f.fused.cells[(3, 0)], 2)
        self.assertEqual(len(f.live_peers()), 0)
        sc = f.scene()
        self.assertIn([3, 0], sc["fused"])
        self.assertIn([3, 0], sc["sources"]["small_a"])

    def test_dog_grid_uses_live_go_footprint_only(self):
        f = Fusion2D()
        f.set_pose("small_a", [1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        f.set_pose("large", [8.0, 8.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        res, pts = f.occupied_xy()
        self.assertAlmostEqual(res, 0.1)
        self.assertTrue(any(abs(x - 1.0) < 0.45 and abs(y - 2.0) < 0.25 for x, y in pts))
        self.assertTrue(any(abs(x - 8.0) < 0.3 and abs(y - 8.0) < 0.3 for x, y in pts))
        f.heard["small_a"] -= 6.0
        f.heard["large"] -= 6.0
        _, stale = f.occupied_xy()
        self.assertEqual(stale, [])

    def test_scene_lists_occupied_cells(self):
        f = Fusion2D()
        f.apply(
            "small_a",
            Meta(type=1, seq=1, map_id="g-1"),
            [
                Record(4, 5, 0, 2),
                Record(4, 6, 0, 2),
                Record(4, 7, 0, 2),
            ],
        )
        sc = f.scene()
        self.assertEqual(sc["resolution"], 0.1)
        self.assertIn([4, 5], sc["fused"])
        self.assertIn([4, 5], sc["sources"]["small_a"])
        self.assertEqual(sc["overlap"], [])
        self.assertEqual(sc["aabb"], [-4.6, -4.5, 5.5, 5.8])

    def test_clear_empties_maps_and_rotates_epoch(self):
        f = Fusion2D()
        epoch = f.epoch
        f.apply("large", Meta(type=1, seq=1, map_id="a-1"), [Record(4, 0, 0, 2)])
        self.assertTrue(f.sources["large"].cells)
        f.clear()
        self.assertNotEqual(f.epoch, epoch)
        self.assertFalse(f.sources["large"].ready)
        self.assertEqual(f.sources["large"].cells, {})
        self.assertEqual(f.fused.cells, {})
        self.assertFalse(
            f.apply("large", Meta(type=2, seq=2, base=1, map_id="a-1"), [Record(5, 0, 0, 2)])
        )

    def test_overlap_needs_two_robots(self):
        f = Fusion2D()
        f.apply(
            "small_a", Meta(type=1, seq=1, map_id="g-1"), [Record(1, 0, 0, 2), Record(2, 0, 0, 2)]
        )
        f.apply(
            "large", Meta(type=1, seq=1, map_id="a-1"), [Record(2, 0, 0, 2), Record(3, 0, 0, 2)]
        )
        sc = f.scene()
        self.assertEqual(sc["overlap"], [[2, 0]])
        self.assertIn([1, 0], sc["sources"]["small_a"])
        self.assertIn([3, 0], sc["sources"]["large"])


if __name__ == "__main__":
    unittest.main()
