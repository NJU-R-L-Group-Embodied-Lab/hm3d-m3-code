import unittest
from unittest.mock import patch

from lavira_m3.mapping.builder import Builder2D
from lavira_m3.mapping.components import remove_small_components
from lavira_m3.mapping.fusion import Fusion2D
from lavira_m3.mapping.protocol import Meta, Record, decode, encode


class ComponentsTest(unittest.TestCase):
    def test_sizes_diagonals_and_free_cells(self):
        cells = {(0, 0): 2, (2, 0): 2, (3, 0): 2, (8, 8): 2, (9, 9): 2, (10, 10): 2, (1, 0): 1}
        remove_small_components(cells)
        self.assertEqual(cells, {(8, 8): 2, (9, 9): 2, (10, 10): 2, (1, 0): 1})

    def test_real_evidence_can_grow_into_visible_cluster(self):
        b = Builder2D("small_b")
        pair = [(2.05, 0.05, 0.4), (2.15, 0.05, 0.4)]
        for stamp in (10.0, 10.25, 10.5):
            b.scan(pair, (0.0, 0.0, 0.0), stamp, (0.0, 0.0, 0.0))
        self.assertEqual(len(b.evidence.voxels), 2)
        self.assertEqual(b.snapshot_rows(), [])
        self.assertEqual(b.delta_rows(), [])
        for stamp in (10.75, 11.0, 11.25):
            b.scan(pair + [(2.25, 0.05, 0.4)], (0.0, 0.0, 0.0), stamp, (0.0, 0.0, 0.0))
        self.assertEqual(len(b.snapshot_rows()), 3)
        self.assertEqual(len(b.delta_rows()), 3)

    def test_source_add_remove_and_recovery_snapshots(self):
        for robot in ("large", "small_a", "small_b"):
            with self.subTest(robot=robot):
                b = Builder2D(robot)
                cluster = {(20, 0), (21, 0), (22, 0)}
                # Drive confirmed evidence to isolate grid-output/delta behavior.
                with patch.object(b.evidence, "update", return_value=True):
                    b.evidence.voxels = {(x, y, 4): {} for x, y in cluster}
                    b.scan([], (0, 0, 0), 10.0, (0, 0, 0))
                    self.assertEqual(set(b.cells), cluster)
                    f = Fusion2D()
                    meta = Meta(type=1, seq=1, map_id="test")
                    m, rows = decode(encode(meta, b.snapshot_rows()))
                    self.assertTrue(f.apply(robot, m, rows))
                    b.pending.clear()
                    revision = f.fused.seq
                    stream = getattr(f, "stream_id", "")
                    b.evidence.voxels.pop((22, 0, 4))
                    b.scan([], (0, 0, 0), 10.25, (0, 0, 0))
                    removed = b.delta_rows()
                    self.assertEqual(
                        {(r.ix, r.iy, r.state) for r in removed}, {(x, y, 0) for x, y in cluster}
                    )
                    self.assertEqual(b.snapshot_rows(), [])
                    self.assertEqual(len(b.evidence.voxels), 2)
                    m, rows = decode(encode(Meta(type=2, seq=2, base=1, map_id="test"), removed))
                    self.assertTrue(f.apply(robot, m, rows))
                    self.assertFalse(f.fused.cells)
                    if hasattr(f, "updates"):
                        update = f.updates(stream, revision)
                        self.assertEqual(update["type"], "delta")
                        self.assertEqual(
                            {tuple(r) for r in update["cells"]}, {(x, y, 0) for x, y in cluster}
                        )

    def test_body_mask_cannot_leave_small_fragments(self):
        b = Builder2D("small_b")
        cluster = {(20, 0), (21, 0), (22, 0)}
        with patch.object(b.evidence, "update", return_value=True):
            b.evidence.voxels = {(x, y, 4): {} for x, y in cluster}
            b.scan([], (0, 0, 0), 10.0, (0, 0, 0))
            b.pending.clear()
            with patch.object(b, "_blocked", side_effect=lambda x, y, bodies, z=None: x > 2.2):
                b.scan([], (0, 0, 0), 10.25, (0, 0, 0))
            self.assertFalse(b.cells)
            self.assertEqual(len(b.delta_rows()), 3)

    def test_shared_union_retains_other_robots_occupancy(self):
        f = Fusion2D()
        cluster = [(20, 0), (21, 0), (22, 0)]
        for name in ("large", "small_a", "small_b"):
            f.apply(
                name, Meta(type=1, seq=1, map_id=name), [Record(x, y, 0, 2) for x, y in cluster]
            )
        revision = f.fused.seq
        stream = getattr(f, "stream_id", "")
        f.apply(
            "small_b",
            Meta(type=2, seq=2, base=1, map_id="small_b"),
            [Record(x, y, 0, 0) for x, y in cluster],
        )
        self.assertEqual(set(f.fused.cells), set(cluster))
        if hasattr(f, "updates"):
            self.assertEqual(f.updates(stream, revision)["cells"], [])


if __name__ == "__main__":
    unittest.main()
