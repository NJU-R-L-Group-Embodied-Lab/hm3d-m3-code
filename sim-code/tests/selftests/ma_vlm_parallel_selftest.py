"""Habitat-free check that same-wave LA+VA HTTP jobs overlap in wall time."""
from __future__ import annotations

import time

import numpy as np

from vlnce_baselines.multi.ma_decision import (
    run_agent_api_parallel,
    run_la_va,
    run_la_va_parallel,
)


class _SleepVLM:
    def __init__(self, delay_s: float = 0.12) -> None:
        self.delay_s = delay_s
        self.la_calls = 0
        self.va_calls = 0

    def navigate_or_backtrack(self, **_kwargs):
        time.sleep(self.delay_s)
        self.la_calls += 1
        return {
            "action": "NAVIGATE",
            "direction": "forward",
            "progress_analysis": "ok",
            "reasoning_action": "go",
        }

    def query_llm(self, **_kwargs):
        time.sleep(self.delay_s)
        self.va_calls += 1
        return {"x1": 10, "x2": 20, "y1": 10, "y2": 20, "target": "chair"}

    def double_check_stop(self, *args, **_kwargs):
        time.sleep(self.delay_s)
        self.dcs_calls = getattr(self, "dcs_calls", 0) + 1
        return False, {"decision": "CONTINUE"}


class _BacktrackVLM(_SleepVLM):
    def navigate_or_backtrack(self, **_kwargs):
        time.sleep(self.delay_s)
        self.la_calls += 1
        return {"action": "BACKTRACK", "waypoint": 0}


def _jobs(n: int, delay_s: float = 0.12, vlm_cls=_SleepVLM):
    frame = {
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "depth": np.full((8, 8, 1), 0.2, dtype=np.float32),
    }
    jobs = []
    vlms = []
    for i in range(n):
        vlm = vlm_cls(delay_s)
        vlms.append(vlm)
        jobs.append({
            "vlm_agent": vlm,
            "instruction": "Find a chair.",
            "visited_targets": [{"panorama_frames": [frame] * 4}],
            "panorama_frames": [frame] * 4,
            "current_step": 0,
            "width": 8,
            "height": 8,
            "agent_id": i,
        })
    return jobs, vlms


def test_parallel_overlaps_wall_time():
    n = 3
    delay = 0.12
    jobs, vlms = _jobs(n, delay)
    t0 = time.time()
    raws = run_la_va_parallel(jobs, max_workers=n)
    elapsed = time.time() - t0
    assert len(raws) == n
    assert all(r.error is None for r in raws)
    assert all(r.bbox is not None for r in raws)
    assert all(v.la_calls == 1 and v.va_calls == 1 for v in vlms)
    serial_floor = n * 2 * delay
    # Three serial LA+VA would be ~0.72 s; overlap should beat ~2/3 of that.
    assert elapsed < serial_floor * 0.7, elapsed


def test_workers_one_stays_serial():
    n = 2
    delay = 0.10
    jobs, _ = _jobs(n, delay)
    t0 = time.time()
    raws = run_la_va_parallel(jobs, max_workers=1)
    elapsed = time.time() - t0
    assert all(r.error is None for r in raws)
    assert elapsed >= (n * 2 * delay) * 0.85, elapsed


def test_backtrack_skips_va():
    jobs, vlms = _jobs(1, delay_s=0.0, vlm_cls=_BacktrackVLM)
    raw = run_la_va(**jobs[0])
    assert raw.action == "BACKTRACK"
    assert raw.bbox is None
    assert vlms[0].la_calls == 1
    assert vlms[0].va_calls == 0


def test_empty_wave():
    assert run_la_va_parallel([]) == []


def test_dcs_then_la_va_parallel_across_agents():
    n = 2
    delay = 0.10
    jobs, vlms = _jobs(n, delay)
    for job in jobs:
        job["need_dcs"] = True
        job["claim_goal_id"] = "goal_0"
    t0 = time.time()
    results = run_agent_api_parallel(jobs, max_workers=n)
    elapsed = time.time() - t0
    assert len(results) == n
    assert all(r.dcs_should_stop is False for r in results)
    assert all(r.raw is not None and r.raw.error is None for r in results)
    assert all(getattr(v, "dcs_calls", 0) == 1 for v in vlms)
    # Same agent: DCS→LA→VA (3 sleeps). Two agents overlap, not 6 serial sleeps.
    assert elapsed < (n * 3 * delay) * 0.7, elapsed
    assert elapsed >= (3 * delay) * 0.85, elapsed


def main() -> None:
    test_empty_wave()
    test_backtrack_skips_va()
    test_workers_one_stays_serial()
    test_parallel_overlaps_wall_time()
    test_dcs_then_la_va_parallel_across_agents()
    print("ma_vlm parallel self-test passed")


if __name__ == "__main__":
    main()
