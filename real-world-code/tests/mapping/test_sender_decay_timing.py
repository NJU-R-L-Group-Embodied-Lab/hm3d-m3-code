"""Clearing duration uses observed intervals, never elapsed outage time."""

from lavira_m3.mapping.occupancy import FREE_GAP, OccupancyEvidence


def test_long_gaps_do_not_count_as_free_observation():
    e = OccupancyEvidence(0.1, 30.0)
    for t in (1.0, 1.3, 1.6):
        e.update([(2.05, 0.05, 0.55)], (0.05, 0.05, 0.05), t)
    assert e.voxels
    for t in range(2, 25):
        e.update([], (0.05, 0.05, 0.05), float(t), ray_endpoints=[(8.05, 0.05, 2.05)])
    assert e.voxels
    e.update([], (0.05, 0.05, 0.05), 32.0)
    assert e.voxels  # Outage time is not free-space evidence.
    assert all(v["free_s"] == 0 for v in e.voxels.values())


def test_repeated_hits_survive_jittered_sampling():
    e = OccupancyEvidence(0.1, 30.0)
    t = 1.0
    for i in range(800):
        t += (0.10006, 0.09998, 0.10002, 0.09997)[i % 4]
        e.update([(2.05, 0.05, 0.55)], (0.05, 0.05, 0.05), t)
    assert e.voxels
    assert FREE_GAP >= 0.10006
