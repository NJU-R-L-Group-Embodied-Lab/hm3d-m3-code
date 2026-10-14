import os
import sys
import unittest

from lavira_m3.mapping.device_status import CAMERAS, build_device_status
from lavira_m3.mapping.fusion import Fusion2D
from lavira_m3.mapping.protocol import Meta, Record

#!/usr/bin/env python3


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DeviceStatusTest(unittest.TestCase):
    def test_live_needs_lidar_and_four_cameras(self):
        now = 100.0
        images = {(robot, cam): {"updated_at": now} for robot in CAMERAS for cam in CAMERAS[robot]}
        heard = {robot: now for robot in CAMERAS}
        view = build_device_status(now, images, heard, image_ttl=3, lidar_ttl=5)
        for robot in CAMERAS:
            self.assertTrue(view[robot]["live"])
            self.assertEqual(view[robot]["overall"], "live")
            self.assertEqual(view[robot]["message"], "")

    def test_missing_lidar_is_error(self):
        now = 100.0
        images = {("large", cam): {"updated_at": now} for cam in CAMERAS["large"]}
        view = build_device_status(now, images, {}, robots=["large"], image_ttl=3, lidar_ttl=5)
        self.assertFalse(view["large"]["live"])
        self.assertEqual(view["large"]["overall"], "error")
        self.assertIn("lidar", view["large"]["missing"])
        self.assertTrue(view["large"]["message"].startswith("missing: "))

    def test_stale_or_missing_camera_is_error(self):
        now = 100.0
        images = {("large", cam): {"updated_at": now} for cam in CAMERAS["large"]}
        images[("large", "front")] = {"updated_at": now - 4.0}
        view = build_device_status(
            now, images, {"large": now}, robots=["large"], image_ttl=3, lidar_ttl=5
        )
        self.assertFalse(view["large"]["live"])
        self.assertIn("front camera", view["large"]["missing"])
        del images[("large", "left")]
        view = build_device_status(
            now, images, {"large": now}, robots=["large"], image_ttl=3, lidar_ttl=5
        )
        self.assertIn("left camera", view["large"]["missing"])

    def test_default_image_ttl_holds_brief_gaps(self):
        now = 100.0
        images = {("large", cam): {"updated_at": now - 6.0} for cam in CAMERAS["large"]}
        view = build_device_status(now, images, {"large": now}, robots=["large"])
        self.assertTrue(view["large"]["live"])
        images[("large", "front")] = {"updated_at": now - 9.0}
        view = build_device_status(now, images, {"large": now}, robots=["large"])
        self.assertFalse(view["large"]["live"])

    def test_pose_upload_does_not_count_as_lidar(self):
        fusion = Fusion2D()
        fusion.set_pose("large", [1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        self.assertNotIn("large", fusion.map_heard)
        self.assertTrue(
            fusion.apply("large", Meta(type=1, seq=1, map_id="a-1"), [Record(0, 0, 0, 2)])
        )
        self.assertIn("large", fusion.map_heard)


if __name__ == "__main__":
    unittest.main()
