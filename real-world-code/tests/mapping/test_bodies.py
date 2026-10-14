import math
import os
import sys
import unittest

from lavira_m3.mapping.bodies import (
    FOOTPRINT,
    inside_filter,
    inside_robot,
    raster_footprint,
)

#!/usr/bin/env python3


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BodyTest(unittest.TestCase):
    def test_sizes_differ(self):
        self.assertGreater(FOOTPRINT["large"][0], FOOTPRINT["small_a"][0])
        self.assertGreater(FOOTPRINT["large"][1], FOOTPRINT["small_a"][1])
        self.assertEqual(FOOTPRINT["small_a"], FOOTPRINT["small_b"])

    def test_small_a_box_not_as_wide_as_aloha(self):
        self.assertTrue(inside_robot(0.35, 0.0, "small_a", 0.0, 0.0, 0.0))
        self.assertFalse(inside_robot(0.55, 0.0, "small_a", 0.0, 0.0, 0.0))
        self.assertTrue(inside_robot(0.55, 0.0, "large", 0.0, 0.0, 0.0))

    def test_raster_small_a_is_smaller_than_aloha(self):
        small_a = raster_footprint(0.0, 0.0, 0.0, *FOOTPRINT["small_a"], 0.1)
        aloha = raster_footprint(0.0, 0.0, 0.0, *FOOTPRINT["large"], 0.1)
        self.assertTrue(small_a)
        self.assertGreater(len(aloha), len(small_a))
        self.assertTrue(all(abs(y) <= 0.20 + 1e-6 for _, y in small_a))

    def test_large_own_filter_covers_parked_arms(self):
        self.assertTrue(inside_filter(-1.0, 0.0, "large", 0.0, 0.0, 0.0, 0.4, own=True))
        self.assertFalse(inside_filter(-1.4, 0.0, "large", 0.0, 0.0, 0.0, 0.4, own=True))
        self.assertFalse(inside_filter(0.80, 0.0, "large", 0.0, 0.0, 0.0, 0.4, own=True))
        self.assertTrue(inside_filter(0.0, -1.0, "large", 0.0, 0.0, math.pi / 2, 0.4, own=True))


if __name__ == "__main__":
    unittest.main()
