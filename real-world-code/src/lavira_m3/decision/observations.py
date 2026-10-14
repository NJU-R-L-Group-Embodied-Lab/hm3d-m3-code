"""Wait for complete camera observations before a decision may run."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    view_wait_s,
)

log = logging.getLogger(__name__)


class MissingViewsError(RuntimeError):
    """Required camera views are missing; stop LA/VA instead of deciding half-blind."""


def require_all_views(robot: str, frames: List[Dict[str, Any]], expected: Tuple[str, ...]) -> None:
    by_name = {str((fr or {}).get("name") or ""): fr for fr in frames or []}
    missing = []
    for name in expected:
        fr = by_name.get(name)
        if fr is None or fr.get("rgb") is None:
            missing.append(name)
    if missing:
        got = [n for n, fr in by_name.items() if fr is not None and fr.get("rgb") is not None]
        raise MissingViewsError(
            "%s missing views %s (have %s); refusing LA/VA" % (robot, missing, got)
        )


def wait_all_views(
    robot: str,
    capture_fn,
    expected: Tuple[str, ...],
    timeout_s: Optional[float] = None,
    poll_s: float = 0.4,
) -> List[Dict[str, Any]]:
    """Poll capture_fn until every expected view has rgb. Still fails if timeout elapses."""
    deadline = time.time() + max(0.0, float(timeout_s if timeout_s is not None else view_wait_s()))
    last: List[Dict[str, Any]] = []
    while True:
        try:
            last = list(capture_fn() or [])
            require_all_views(robot, last, expected)
            return last
        except MissingViewsError as exc:
            if time.time() >= deadline:
                raise
            log.warning("%s", exc)
            time.sleep(max(0.05, float(poll_s)))
