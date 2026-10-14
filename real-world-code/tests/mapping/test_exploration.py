import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as N

from lavira_m3.adapters.ros1.fastplanner_contracts import CommandGate, sample_trajectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters"))


def trajectory():
    # x=t^2+2t+3, y=-t+1: asymmetric coefficients expose reversed ordering.
    return N(
        order=5,
        duration=[2.0],
        coef_x=[0, 0, 0, 1, 2, 3],
        coef_y=[0, 0, 0, 0, -1, 1],
        coef_z=[0] * 6,
    )


class ContractTests(unittest.TestCase):
    def test_network_grace_keeps_local_timeouts(self):
        gate = CommandGate()
        for source in gate.inputs:
            gate.observe(source, 10 if source in ("shared_map", "peer_state") else 15)
        self.assertEqual(gate.stale(15), "")
        self.assertEqual(gate.stale(15.001), "shared_map")
        gate.observe("shared_map", 15)
        self.assertEqual(gate.stale(15.001), "peer_state")
        gate.observe("peer_state", 15)
        gate.observe("odometry", 14.49)
        self.assertEqual(gate.stale(15), "odometry")

    def test_polynomial_order_and_endpoint(self):
        path = sample_trajectory(trajectory())
        self.assertEqual(path[0], [3, 1])
        self.assertEqual(path[-1], [11, -1])
        self.assertEqual(path[10], [6, 0])

    def test_invalid_polynomial(self):
        for field, value in [
            ("coef_y", []),
            ("duration", [0]),
            ("coef_z", [math.nan] * 6),
            ("order", 3),
            ("duration", [math.inf]),
            ("duration", []),
        ]:
            msg = trajectory()
            setattr(msg, field, value)
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    sample_trajectory(msg)

    def ready(self):
        gate = CommandGate()
        gate.begin(11, execute=True, now=10)
        for source in gate.inputs:
            gate.observe(source, 10)
        gate.trajectory_until = 20
        return gate

    def test_cancel_old_command_and_new_request(self):
        gate = self.ready()
        self.assertTrue(gate.accept(11, [0.3, 0, 0, 0, 0, 0.7], 10, 10))
        gate.stop("cancelled")
        self.assertFalse(gate.accept(11, [0.3, 0, 0, 0, 0, 0], 10.01, 10.01))
        gate.begin(12, True, 10.02)
        gate.trajectory_until = 20
        self.assertFalse(gate.accept(11, [0.3, 0, 0, 0, 0, 0], 10.02, 10.02))

    def test_preview_never_outputs(self):
        gate = self.ready()
        gate.execute = False
        self.assertFalse(gate.accept(11, [0.3, 0, 0, 0, 0, 0], 10, 10))

    def test_stale_invalid_and_limits_latch_stop(self):
        for command in (
            [math.nan, 0, 0, 0, 0, 0],
            [0.41, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.81],
            [0, 0, 0.1, 0, 0, 0],
        ):
            gate = self.ready()
            self.assertFalse(gate.accept(11, command, 10, 10))
            self.assertIsNone(gate.active)
        for source in CommandGate.inputs:
            gate = self.ready()
            gate.received[source] = 4 if source in ("shared_map", "peer_state") else 9
            self.assertFalse(gate.accept(11, [0.3, 0, 0, 0, 0, 0], 10, 10))
            self.assertIsNone(gate.active)

    def test_watchdog_without_ros_clock(self):
        gate = self.ready()
        self.assertTrue(gate.accept(11, [0.3, 0, 0, 0, 0, 0], 10, 10))
        for source in gate.inputs:
            gate.observe(source, 10.35)
        self.assertEqual(gate.expired(10.35, 10), "command_timeout")
        gate.trajectory_until = 9
        self.assertEqual(gate.expired(10.35, 10), "trajectory_expired")


if __name__ == "__main__":
    unittest.main()
