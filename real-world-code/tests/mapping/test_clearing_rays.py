import unittest
from dataclasses import replace

from lavira_m3.mapping.builder import Builder2D
from lavira_m3.mapping.fusion import Fusion2D


class ClearingRays(unittest.TestCase):
    def build(self):
        b = Builder2D("small_a")
        sensor = (0.05, 0.05, 0.05)
        for t in (1.0, 1.3, 1.6):
            b.scan([(2.05, y, 0.55) for y in (0.05, 0.15, 0.25)], sensor, t, (0, 0, 0))
        self.assertIn((20, 0), b.cells)
        return b, sensor

    def test_far_high_return_clears_after_30_seconds(self):
        b, sensor = self.build()
        # Same ray, but endpoint beyond 5 m and above the marking height.
        for i in range(100):
            b.scan([(8.05, y, 2.05) for y in (0.05, 0.45, 0.85)], sensor, 2 + i * 0.3, (0, 0, 0))
        self.assertIn((20, 0), b.cells)
        b.scan([(8.05, y, 2.05) for y in (0.05, 0.45, 0.85)], sensor, 32.0, (0, 0, 0))
        self.assertNotIn((20, 0), b.cells)
        self.assertNotIn((80, 0), b.cells)
        self.assertEqual(b.pending[(20, 0)], 0)

    def test_occlusion_missing_and_wrong_height_do_not_clear(self):
        for returns in ([], [(1.05, 0.05, 0.30)], [(8.05, 0.05, 0.05)]):
            b, sensor = self.build()
            for i in range(50):
                b.scan(returns, sensor, 2 + i * 0.3, (0, 0, 0))
            self.assertIn((20, 0), b.cells)

    def test_new_hit_resets_timer(self):
        b, sensor = self.build()
        for i in range(70):
            b.scan([(8.05, y, 2.05) for y in (0.05, 0.45, 0.85)], sensor, 2 + i * 0.3, (0, 0, 0))
        b.scan([(2.05, y, 0.55) for y in (0.05, 0.15, 0.25)], sensor, 23.0, (0, 0, 0))
        for i in range(70):
            b.scan([(8.05, y, 2.05) for y in (0.05, 0.45, 0.85)], sensor, 23.3 + i * 0.3, (0, 0, 0))
        self.assertIn((20, 0), b.cells)

    def test_fusion_deletion_propagates_only_after_sources_clear(self):
        b, sensor = self.build()
        f = Fusion2D()
        for name in ("small_a", "small_b"):
            self.assertTrue(
                f.apply(name, replace(b.meta, type=1, seq=1, base=0), b.snapshot_rows())
            )
        b.pending.clear()
        for i in range(102):
            b.scan([(8.05, y, 2.05) for y in (0.05, 0.45, 0.85)], sensor, 2 + i * 0.3, (0, 0, 0))
        rows = b.delta_rows()
        meta = replace(b.meta, type=2, seq=2, base=1)
        self.assertTrue(f.apply("small_a", meta, rows))
        self.assertIn((20, 0), f.fused.cells)
        self.assertTrue(f.apply("small_b", meta, rows))
        self.assertNotIn((20, 0), f.fused.cells)
        self.assertTrue(any(r.state == 0 and (r.ix, r.iy) == (20, 0) for r in rows))


if __name__ == "__main__":
    unittest.main()
