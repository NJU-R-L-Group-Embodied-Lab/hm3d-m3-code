"""Offline RA timing regression: no ROS, networking, or robot motion."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def check(root):
    source = (root / "agent_loop.py").read_text()
    tree = ast.parse(source)
    agent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LaviraM3Agent")
    method = next(
        n for n in agent.body if isinstance(n, ast.FunctionDef) and n.name == "_plan_ready"
    )
    clock = SimpleNamespace(perf_counter=lambda: 12.0)
    namespace = {"time": clock, "log": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(root), "exec"), namespace)
    trace = Mock()
    obj = SimpleNamespace(_ra_started=10.0, _ra_time_ms=None, _trace=trace)
    ready = namespace["_plan_ready"]
    ready(obj, [], [1, 0])
    assert obj._ra_time_ms is None
    ready(obj, [[0, 0], [1, 0]], [1, 0])
    assert obj._ra_time_ms == 2000.0
    assert trace.call_args.kwargs["ra_time_ms"] == 2000.0
    clock.perf_counter = lambda: 112.0  # 100 seconds of motion do not change RA.
    ready(obj, [[0, 0], [1, 0]], [1, 0])
    assert obj._ra_time_ms == 2000.0 and trace.call_count == 1
    obj._ra_started, obj._ra_time_ms = 112.0, None
    ready(obj, [], None)
    assert obj._ra_time_ms is None  # A failed new cycle cannot reuse old timing.
    assert "self._trace(ra_time_ms=None)" in source
    assert "time.perf_counter() - ra_started" not in source
    assert "self.runtime.nav.on_plan = self._plan_ready" in source
    for relative, end in (
        ("nav_http.py", "submit = {"),
        ("deploy/run_small_a_m3.py", "result = self.nav.navigate_to"),
        ("deploy/run_small_b_m3.py", "result = self.nav.navigate_to"),
    ):
        path = root / relative
        if path.exists():
            text = path.read_text()
            if "on_plan(" in text:
                assert text.index("on_plan(") < text.index(end), relative
    print("PASS: RA freezes at trajectory return, excludes motion, clears next cycle:", root)


if __name__ == "__main__":
    import sys

    check(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent)
