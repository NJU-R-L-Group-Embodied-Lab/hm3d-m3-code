"""Middleware-independent navigation results and runtime interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple


@dataclass
class NavResult:
    ok: bool
    state: str
    message: str
    execution_id: Optional[int] = None
    preview_id: Optional[str] = None
    target: Optional[Tuple[float, float, float]] = None
    frame: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        extra = self.extra or {}
        path = extra.get("path")
        return {
            "ok": self.ok,
            "state": self.state,
            "message": self.message,
            "execution_id": self.execution_id,
            "preview_id": self.preview_id,
            "target": list(self.target) if self.target else None,
            "frame": self.frame,
            "path": path if isinstance(path, list) else None,
            "timing": dict(extra.get("timing") or {}),
            "completion": extra.get("completion"),
            "requested_goal": extra.get("requested_goal"),
            "accepted_goal": extra.get("accepted_goal"),
        }


class RobotRuntime(Protocol):
    def capture_panorama(self) -> List[Dict[str, Any]]:
        """Each item: name, rgb, depth?, fx, fy, cx, cy."""

    def pose(self) -> Optional[Tuple[float, float, float]]: ...

    def face_direction(self, name: str) -> NavResult: ...

    def capture_direction(self, name: str) -> Optional[Dict[str, Any]]: ...

    def navigate_to(self, x: float, y: float, yaw: Optional[float] = None) -> NavResult: ...

    def stop(self) -> None: ...
