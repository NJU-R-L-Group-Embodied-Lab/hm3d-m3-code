import os
import sys
import unittest

from lavira_m3.mapping.t0_cameras import build_t0_observe, maybe_t0_payload

#!/usr/bin/env python3


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


JPEG = b"\xff\xd8\x00\x01\xff\xd9"


class T0CamerasTest(unittest.TestCase):
    def test_rear_becomes_behind(self):
        payload = build_t0_observe(
            "small_a",
            {"front": JPEG, "left": JPEG, "rear": JPEG, "right": JPEG},
            {"live": True, "position": [1.2, 3.4, 0.3], "yaw": 0.5},
        )
        self.assertEqual(payload["robot"], "small_a")
        self.assertEqual(payload["pose"], [1.2, 3.4, 0.5])
        self.assertEqual(payload["dirs"], ["front", "left", "behind", "right"])
        self.assertEqual(
            [fr["name"] for fr in payload["frames"]], ["front", "left", "behind", "right"]
        )

    def test_missing_cam_or_stale_pose_is_none(self):
        pose = {"live": True, "position": [0.0, 0.0], "yaw": 0.0}
        self.assertIsNone(
            build_t0_observe("small_a", {"front": JPEG, "left": JPEG, "right": JPEG}, pose)
        )
        self.assertIsNone(
            build_t0_observe(
                "small_a",
                {"front": JPEG, "left": JPEG, "rear": JPEG, "right": JPEG},
                {"live": False, "position": [0.0, 0.0], "yaw": 0.0},
            )
        )

    def test_does_not_fill_aloha(self):
        self.assertIsNone(
            build_t0_observe(
                "large",
                {"front": JPEG, "left": JPEG, "rear": JPEG, "right": JPEG},
                {"live": True, "position": [0.0, 0.0], "yaw": 0.0},
            )
        )

    def test_maybe_payload_needs_four_fresh_jpegs(self):
        now = 100.0
        images = {
            ("small_a", cam): {"data": JPEG, "updated_at": now}
            for cam in ("front", "left", "rear", "right")
        }
        pose = {"live": True, "position": [0.1, 0.2], "yaw": 1.0}
        payload = maybe_t0_payload("small_a", images, pose, now, 3.0)
        self.assertEqual(payload["dirs"][2], "behind")
        images[("small_a", "rear")] = {"data": JPEG, "updated_at": now - 4.0}
        self.assertIsNone(maybe_t0_payload("small_a", images, pose, now, 3.0))


if __name__ == "__main__":
    unittest.main()
