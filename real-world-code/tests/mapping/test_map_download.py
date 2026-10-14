import os
import sys
import unittest

from lavira_m3.mapping.fusion import Fusion2D
from lavira_m3.mapping.local_adapter import compose
from lavira_m3.mapping.map_download import MapDownload
from lavira_m3.mapping.protocol import Meta, Record

#!/usr/bin/env python3


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MapDownloadTest(unittest.TestCase):
    def test_snapshot_then_delta_add_and_delete(self):
        fusion = Fusion2D()
        fusion.apply(
            "large",
            Meta(type=1, seq=1, map_id="a-1"),
            [
                Record(10, 0, 0, 2),
                Record(11, 0, 0, 2),
                Record(12, 0, 0, 2),
            ],
        )
        client = MapDownload()
        self.assertTrue(client.apply(fusion.updates(None, -1)))
        self.assertEqual(client.cells[(11, 0)], 2)
        revision = fusion.fused.seq
        stream = fusion.stream_id
        fusion.apply(
            "large",
            Meta(type=2, seq=2, base=1, map_id="a-1"),
            [
                Record(13, 0, 0, 2),
                Record(10, 0, 0, 0),
            ],
        )
        update = fusion.updates(stream, revision)
        self.assertEqual(update["type"], "delta")
        self.assertTrue(client.apply(update))
        self.assertNotIn((10, 0), client.cells)
        self.assertEqual(client.cells[(13, 0)], 2)
        self.assertEqual(client.cells[(11, 0)], 2)

    def test_seq_gap_keeps_old_cells_and_requests_snapshot(self):
        client = MapDownload()
        client.apply(
            {
                "type": "snapshot",
                "map_id": "s1",
                "seq": 3,
                "ready": True,
                "resolution": 0.1,
                "origin": [0, 0, 0],
                "frame": "map",
                "cells": [[1, 0, 2], [2, 0, 2], [3, 0, 2]],
            }
        )
        kept = dict(client.cells)
        self.assertFalse(
            client.apply(
                {
                    "type": "delta",
                    "map_id": "s1",
                    "base": 1,
                    "seq": 5,
                    "cells": [[1, 0, 0]],
                }
            )
        )
        self.assertEqual(client.cells, kept)
        self.assertTrue(client.resync)
        self.assertEqual(client.query(), (None, -1))

    def test_compose_small_a_stamps_aloha_envelope(self):
        client = MapDownload()
        client.apply(
            {
                "type": "snapshot",
                "map_id": "s1",
                "seq": 1,
                "ready": True,
                "resolution": 0.1,
                "origin": [0, 0, 0],
                "frame": "map",
                "cells": [],
            }
        )
        status = {
            "sources": {
                "large": {"pose": {"position": [2.0, 0.0, 0.0], "yaw": 0.0, "age_s": 0.0}},
                "small_a": {"pose": {"position": [0.0, 0.0, 0.0], "yaw": 0.0, "age_s": 0.0}},
            }
        }
        payload = compose(client, status, "small_a", now=0.0)
        w, res = payload["width"], payload["resolution"]
        ox, oy = payload["origin"]

        def value(layer, x, y):
            return payload[layer][int((y - oy) // res) * w + int((x - ox) // res)]

        self.assertEqual(payload["inflate_m"], 0.40)
        self.assertEqual(value("peers", 2.0, 0.0), 100)
        self.assertNotEqual(value("peers", 0.0, 0.0), 100)
        self.assertEqual(payload["view"]["robot"], "small_a")
        self.assertEqual(payload["view"]["own"]["name"], "small_a")
        self.assertEqual(payload["view"]["peers"][0]["name"], "large")


if __name__ == "__main__":
    unittest.main()
