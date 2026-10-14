#!/usr/bin/env python3
"""Dog pose jumps must not smear occupied cells into the shared fused map."""

import os
import sys
import unittest

from lavira_m3.mapping.builder import Builder2D
from lavira_m3.mapping.fusion import Fusion2D
from lavira_m3.mapping.protocol import Meta, Record

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


WALL = ((1.05, 1.05, 0.4), (1.05, 1.15, 0.4), (1.05, 1.25, 0.4))
FAR = ((2.05, 1.05, 0.4), (2.05, 1.15, 0.4), (2.05, 1.25, 0.4))
NEAR_KEYS = {(10, 10), (10, 11), (10, 12)}
FAR_KEYS = {(20, 10), (20, 11), (20, 12)}
NEAR_ROWS = [Record(10, 0, 0, 2), Record(10, 1, 0, 2), Record(10, 2, 0, 2)]
FAR_ROWS = [Record(22, 0, 0, 2), Record(22, 1, 0, 2), Record(22, 2, 0, 2)]
ALOHA_ROWS = [Record(4, 0, 0, 2), Record(4, 1, 0, 2), Record(4, 2, 0, 2)]
DOGS = ("small_a", "small_b")


def _confirm(builder, points, sensor, t0, body=(0.0, 0.0, 0.0)):
    for t in (t0, t0 + 0.3, t0 + 0.6):
        builder.scan(points, sensor, t, body)
    return builder


class DogMapStabilityTest(unittest.TestCase):
    def test_dogs_still_mark_occupied(self):
        for name in DOGS:
            with self.subTest(name):
                b = Builder2D(name)
                _confirm(b, WALL, (0.0, 0.0, 0.2), 1.0)
                self.assertTrue(NEAR_KEYS <= set(b.cells))

    def test_dogs_can_mark_while_walking(self):
        for name in DOGS:
            with self.subTest(name):
                b = Builder2D(name)
                for i in range(6):
                    x = 0.20 * i
                    b.scan(FAR, (x, 0.0, 0.2), 1.0 + 0.2 * i, (x, 0.0, 0.0))
                self.assertTrue(FAR_KEYS <= set(b.cells))

    def test_large_can_mark_while_moving(self):
        b = Builder2D("large")
        for i in range(6):
            x = 0.20 * i
            b.scan(FAR, (x, 0.0, 0.2), 1.0 + 0.2 * i, (x, 0.0, 0.0))
        self.assertTrue(FAR_KEYS <= set(b.cells))

    def test_pose_jump_rejects_frame_and_keeps_map(self):
        for name in ("small_a", "small_b", "large"):
            with self.subTest(name):
                b = Builder2D(name)
                _confirm(b, WALL, (0.0, 0.0, 0.2), 1.0)
                self.assertTrue(NEAR_KEYS <= set(b.cells))
                old_id = b.meta.map_id
                last = b._last_body
                self.assertFalse(b.scan(FAR, (1.2, 0.0, 0.2), 1.7, (1.2, 0.0, 0.0)))
                self.assertEqual(b.meta.map_id, old_id)
                self.assertEqual(b._last_body, last)
                self.assertTrue(NEAR_KEYS <= set(b.cells))
                self.assertFalse(FAR_KEYS & set(b.cells))

    def test_fusion_rejects_jump_without_resetting_store(self):
        for name in ("small_a", "small_b", "large"):
            with self.subTest(name):
                f = Fusion2D()
                first = Meta(
                    type=1,
                    seq=1,
                    map_id="%s-1" % name,
                    pose_valid=True,
                    position=(0.0, 0.0, 0.0),
                    orientation=(0.0, 0.0, 0.0, 1.0),
                )
                self.assertTrue(f.apply(name, first, NEAR_ROWS))
                self.assertEqual(f.fused.cells[(10, 0)], 2)
                jumped = Meta(
                    type=2,
                    seq=2,
                    base=1,
                    map_id="%s-1" % name,
                    pose_valid=True,
                    position=(1.2, 0.0, 0.0),
                    orientation=(0.0, 0.0, 0.0, 1.0),
                )
                self.assertFalse(f.apply(name, jumped, FAR_ROWS))
                self.assertEqual(f.last_reject, "pose_jump")
                self.assertTrue(f.sources[name].ready)
                self.assertEqual(f.fused.cells[(10, 0)], 2)
                self.assertNotIn((22, 0), f.fused.cells)
                self.assertEqual(f.poses[name]["position"][:2], [0.0, 0.0])

    def test_fusion_keeps_aloha_cells_when_dog_jumps(self):
        for name in DOGS:
            with self.subTest(name):
                f = Fusion2D()
                self.assertTrue(f.apply("large", Meta(type=1, seq=1, map_id="a-1"), ALOHA_ROWS))
                first = Meta(
                    type=1,
                    seq=1,
                    map_id="%s-1" % name,
                    pose_valid=True,
                    position=(0.0, 0.0, 0.0),
                    orientation=(0.0, 0.0, 0.0, 1.0),
                )
                self.assertTrue(f.apply(name, first, NEAR_ROWS))
                jumped = Meta(
                    type=2,
                    seq=2,
                    base=1,
                    map_id="%s-1" % name,
                    pose_valid=True,
                    position=(1.2, 0.0, 0.0),
                    orientation=(0.0, 0.0, 0.0, 1.0),
                )
                self.assertFalse(f.apply(name, jumped, FAR_ROWS))
                self.assertEqual(f.fused.cells[(4, 0)], 2)
                self.assertEqual(f.fused.cells[(10, 0)], 2)
                self.assertNotIn((22, 0), f.fused.cells)


if __name__ == "__main__":
    unittest.main()
