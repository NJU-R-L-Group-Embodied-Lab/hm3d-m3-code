import os
import sys
import unittest

from lavira_m3.mapping.protocol import Meta, Record, decode, encode

#!/usr/bin/env python3


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ProtocolTest(unittest.TestCase):
    def test_roundtrip_snapshot(self):
        meta = Meta(type=1, seq=1, base=0, timestamp=7, map_id="large-1", mode=1)
        rows = [Record(2, -1, 0, 2), Record(3, 0, 0, 1)]
        again, out = decode(encode(meta, rows))
        self.assertEqual(again.seq, 1)
        self.assertEqual(again.resolution, 0.1)
        self.assertEqual([(r.ix, r.iy, r.iz, r.state) for r in out], [(2, -1, 0, 2), (3, 0, 0, 1)])

    def test_pose_v2(self):
        meta = Meta(
            type=1,
            seq=1,
            map_id="small_a-1",
            pose_valid=True,
            position=(2.5, 0.0, 0.2),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        again, _ = decode(encode(meta, []))
        self.assertTrue(again.pose_valid)
        self.assertAlmostEqual(again.position[0], 2.5, places=4)
        self.assertAlmostEqual(again.orientation[3], 1.0, places=3)

    def test_2d_z_is_zero(self):
        meta = Meta(type=1, seq=1, map_id="a-1")
        _, rows = decode(encode(meta, [Record(1, 2, 0, 2)]))
        self.assertEqual(rows[0].iz, 0)


if __name__ == "__main__":
    unittest.main()
