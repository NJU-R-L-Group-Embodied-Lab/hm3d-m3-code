import math
import os
import sys
import unittest

from lavira_m3.mapping.frames import body_xy_yaw_to_base

#!/usr/bin/env python3


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FrameTest(unittest.TestCase):
    def test_body_origin_base_is_offset_and_minus_90(self):
        x, y, yaw = body_xy_yaw_to_base(0.0, 0.0, 0.0)
        self.assertAlmostEqual(yaw, -math.pi / 2, places=5)
        self.assertAlmostEqual(x, -0.070, places=3)
        self.assertAlmostEqual(y, 0.290, places=3)

    def test_forward_body_moves_base_left_in_lidar_frame(self):
        x0, y0, _ = body_xy_yaw_to_base(0.0, 0.0, 0.0)
        x1, y1, _ = body_xy_yaw_to_base(1.0, 0.0, 0.0)
        self.assertAlmostEqual(x1 - x0, 1.0, places=5)
        self.assertAlmostEqual(y1 - y0, 0.0, places=5)
