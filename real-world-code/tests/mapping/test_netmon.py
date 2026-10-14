import os
import sys
import unittest

from lavira_m3.mapping.netmon import NetMonitor, classify_path, pick_iface

#!/usr/bin/env python3


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NetmonTest(unittest.TestCase):
    def test_classify_share_paths(self):
        self.assertEqual(classify_path("/v1/map/small_a"), ("map_up", "small_a"))
        self.assertEqual(classify_path("/v1/map-updates?since=3"), ("map_down", None))
        self.assertEqual(classify_path("/v1/front-image/small_b"), ("image", "small_b"))
        self.assertEqual(classify_path("/v1/scene"), ("scene", None))
        self.assertEqual(classify_path("/v1/state"), ("state", None))
        self.assertEqual(classify_path("/v1/plan/small_a"), ("plan", "small_a"))
        self.assertEqual(classify_path("/api/board/init"), ("board", None))
        self.assertEqual(classify_path("/index.html"), ("web", None))

    def test_rates_and_history(self):
        nic_steps = [
            {
                "wlan0": {
                    "rx_bytes": 0,
                    "rx_packets": 0,
                    "rx_errs": 0,
                    "rx_drop": 0,
                    "tx_bytes": 0,
                    "tx_packets": 0,
                    "tx_errs": 0,
                    "tx_drop": 0,
                }
            },
            {
                "wlan0": {
                    "rx_bytes": 1_000_000,
                    "rx_packets": 100,
                    "rx_errs": 1,
                    "rx_drop": 0,
                    "tx_bytes": 2_000_000,
                    "tx_packets": 80,
                    "tx_errs": 0,
                    "tx_drop": 2,
                }
            },
        ]
        state = {"i": 0}

        def reader():
            rec = nic_steps[min(state["i"], 1)]
            state["i"] += 1
            return rec

        mon = NetMonitor(
            {"small_a": "127.0.0.1", "large": "127.0.0.1", "small_b": "127.0.0.1"},
            "127.0.0.1",
            iface="wlan0",
            history=10,
            sample_s=1.0,
            nic_reader=reader,
        )
        mon.record("127.0.0.1", "/v1/map/small_a", 50_000, 200, 12.0, 200, now=10.0)
        mon.record("127.0.0.1", "/v1/map-updates", 80, 80_000, 40.0, 200, now=10.2)
        mon.record("127.0.0.1", "/v1/scene", 0, 400_000, 80.0, 200, now=10.4)
        snap = mon.snapshot(now=11.1)
        self.assertEqual(snap["iface"], "wlan0")
        self.assertGreater(snap["robots"]["small_a"]["up_mbps"], 0.3)
        self.assertGreater(snap["robots"]["small_b"]["down_mbps"], 0.5)
        self.assertGreater(snap["channels"]["scene"]["tx_mbps"], 2.0)
        self.assertEqual(snap["robots"]["small_a"]["ip"], "127.0.0.1")
        self.assertIsNotNone(snap["robots"]["small_a"]["last"])
        self.assertTrue(snap["history"])
        self.assertTrue(any("NIC errors" in a or "drops" in a for a in snap["alerts"]))

    def test_skip_self_endpoint(self):
        mon = NetMonitor({"large": "127.0.0.1"}, "127.0.0.1", sample_s=1.0, nic_reader=lambda: {})
        mon.record("127.0.0.1", "/v1/net", 0, 900, 2.0, 200, now=1.0)
        snap = mon.snapshot(now=2.1)
        self.assertEqual(snap["http"]["rps"], 0.0)

    def test_pick_bind_iface(self):
        self.assertEqual(
            pick_iface("192.0.2.1", "wlan0", {"wlan0": {"rx_bytes": 1, "tx_bytes": 1}}), "wlan0"
        )

    def test_persist_and_summary(self):
        import tempfile

        nic_steps = [
            {
                "wlan0": {
                    "rx_bytes": 0,
                    "rx_packets": 0,
                    "rx_errs": 0,
                    "rx_drop": 0,
                    "tx_bytes": 0,
                    "tx_packets": 0,
                    "tx_errs": 0,
                    "tx_drop": 0,
                }
            },
            {
                "wlan0": {
                    "rx_bytes": 500_000,
                    "rx_packets": 40,
                    "rx_errs": 0,
                    "rx_drop": 0,
                    "tx_bytes": 800_000,
                    "tx_packets": 30,
                    "tx_errs": 0,
                    "tx_drop": 0,
                }
            },
        ]
        state = {"i": 0}

        def reader():
            rec = nic_steps[min(state["i"], 1)]
            state["i"] += 1
            return rec

        handle = tempfile.NamedTemporaryFile(prefix="map2d-net-", suffix=".jsonl", delete=False)
        path = handle.name
        handle.close()
        os.remove(path)
        try:
            mon = NetMonitor(
                {"large": "127.0.0.1"},
                "127.0.0.1",
                iface="wlan0",
                history=10,
                sample_s=1.0,
                nic_reader=reader,
                persist=path,
            )
            mon.record("127.0.0.1", "/v1/scene", 0, 200_000, 30.0, 200, now=20.0)
            snap = mon.snapshot(now=21.2)
            self.assertGreaterEqual(snap["summary"]["samples"], 1)
            self.assertIn("peak", snap["summary"])
            self.assertTrue(os.path.isfile(path))
            mon2 = NetMonitor(
                {"large": "127.0.0.1"},
                "127.0.0.1",
                iface="wlan0",
                history=10,
                sample_s=1.0,
                nic_reader=lambda: {},
                persist=path,
            )
            self.assertGreaterEqual(len(mon2.history), 1)
            self.assertGreater(mon2.snapshot(now=30.0)["summary"]["samples"], 0)
        finally:
            if os.path.isfile(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
