import os
import struct
import sys
import unittest

from lavira_m3.mapping.grid_cloud import lethal_xy, pack_xyz, xyz_from_xy

#!/usr/bin/env python3


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GridCloudTest(unittest.TestCase):
    def test_lethal_cells_become_cell_centers(self):
        data = [0, 100, 50, -1]
        self.assertEqual(lethal_xy(data, 2, 2, 0.1, (1.0, 2.0)), [(1.15, 2.05)])

    def test_xyz_and_pack(self):
        xyz = xyz_from_xy([(1.0, 2.0)])
        self.assertEqual(xyz, [(1.0, 2.0, 0.05)])
        self.assertEqual(pack_xyz(xyz), struct.pack("<fff", 1.0, 2.0, 0.05))


if __name__ == "__main__":
    unittest.main()
