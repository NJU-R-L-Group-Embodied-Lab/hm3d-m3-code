"""Process stop flag. rospy's default SIGINT handler swallows Ctrl+C unless we own it."""

from __future__ import annotations

import os
import signal
import threading
import time

_stop = threading.Event()
_hits = 0
_last = 0.0
_on_stop = None


def requested() -> bool:
    if _stop.is_set():
        return True
    try:
        import rospy

        return bool(rospy.is_shutdown())
    except Exception:
        return False


def request(reason: str = "interrupt") -> None:
    _stop.set()
    try:
        import rospy

        rospy.signal_shutdown(reason)
    except Exception:
        pass


def on_stop(fn) -> None:
    """Register a callback that runs on the first SIGINT/SIGTERM (Aloha halt_navigation)."""
    global _on_stop
    _on_stop = fn


def install() -> None:
    """Call after rospy.init_node(..., disable_signals=True). First Ctrl+C stops; second exits."""

    def handler(signum, _frame):
        global _hits, _last
        now = time.monotonic()
        # Terminal Ctrl+C hits the whole process group; the wrapper trap may
        # send another INT within the same second. Treat those as one stop.
        if _hits and now - _last < 1.0:
            return
        _last = now
        _hits += 1
        request("signal %s" % signum)
        try:
            with open(os.environ.get("LAVIRA_ESTOP_FILE", "/tmp/lavira_estop"), "a"):
                pass
        except OSError:
            pass
        cb = _on_stop
        if cb is not None and _hits == 1:
            try:
                cb()
            except Exception:
                pass
        if _hits >= 2:
            os._exit(1)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
