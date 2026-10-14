"""CPU tests for the detector FOUND gate. No Habitat / OWL."""
from __future__ import annotations

from benchmark.od_found import ODFoundGate, PROTOCOL_NAME, harvest_od_hits, stop_ok
from benchmark.ovon_od import OvonODSkill, sticky_stop_radius


def test_protocol_name():
    assert PROTOCOL_NAME == "ovon_od_detector_stop"


def test_stop_ok_requires_detected_and_near():
    s = OvonODSkill(owl_url=None)
    s.detected = True
    s.goal_xz = (1.0, 0.0)
    s.best_match = 0.0
    assert stop_ok(s, (1.0, 0.0), 0.0) is True
    assert stop_ok(s, (10.0, 0.0), 0.0) is False
    s.detected = False
    assert stop_ok(s, (1.0, 0.0), 0.0) is False


def test_harvest_skips_resolved_and_far():
    near = OvonODSkill(owl_url=None)
    near.detected = True
    near.goal_xz = (0.2, 0.0)
    far = OvonODSkill(owl_url=None)
    far.detected = True
    far.goal_xz = (8.0, 0.0)
    hits = harvest_od_hits([near, far], [False, False], (0.0, 0.0), 0.0)
    assert hits == [0]
    hits = harvest_od_hits([near, far], [True, False], (0.0, 0.0), 0.0)
    assert hits == []


def test_iin_standoff_uses_sticky_radius():
    assert sticky_stop_radius(24.5) == 0.9
    assert sticky_stop_radius(0.0) == 0.9


def test_from_runner_requires_owl_url(monkeypatch=None):
    import os

    old = os.environ.pop("OWL_URL", None)
    try:
        raised = False
        try:
            ODFoundGate(1, [{"modality": "ON", "object_category": "chair", "goal_index": 0}])
        except RuntimeError as exc:
            raised = "OWL_URL" in str(exc)
        assert raised
    finally:
        if old is not None:
            os.environ["OWL_URL"] = old


def main():
    test_protocol_name()
    test_stop_ok_requires_detected_and_near()
    test_harvest_skips_resolved_and_far()
    test_iin_standoff_uses_sticky_radius()
    test_from_runner_requires_owl_url()
    print("od_found_selftest OK")


if __name__ == "__main__":
    main()
