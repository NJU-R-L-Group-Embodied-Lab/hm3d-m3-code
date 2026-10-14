"""File-locked work queue for independent Habitat workers (work-stealing).

Single-agent eval uses ``mp.Queue`` inside one ``run_mp.py`` process tree.
HM3D-M3 launches one ``ma_search`` per worker with its own
``CUDA_VISIBLE_DEVICES``, so the queue lives on disk and is claimed with
``flock``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional


def _atomic_write(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


class FileWorkQueue:
    def __init__(self, path: str):
        self.path = path
        self.lock_path = path + ".lock"

    @classmethod
    def init(
        cls,
        path: str,
        items: Iterable[Dict[str, str]],
        skip_ids: Optional[Iterable[str]] = None,
    ) -> "FileWorkQueue":
        skip = {str(x) for x in (skip_ids or [])}
        pending = []
        skipped = []
        for raw in items:
            item = {
                "episode_id": str(raw["episode_id"]),
                "scene_id": str(raw["scene_id"]),
            }
            if item["episode_id"] in skip:
                skipped.append(item["episode_id"])
            else:
                pending.append(item)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _atomic_write(
            path,
            {
                "pending": pending,
                "claimed": {},
                "done": [],
                "skipped_done": skipped,
            },
        )
        open(path + ".lock", "a").close()
        return cls(path)

    def _lock_fd(self):
        return open(self.lock_path, "a+")

    def claim(
        self,
        worker_id: str,
        prefer_scene: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        import fcntl

        with self._lock_fd() as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            with open(self.path) as qf:
                data = json.load(qf)
            pending: List[Dict[str, str]] = list(data.get("pending") or [])
            if not pending:
                return None
            idx = 0
            if prefer_scene:
                for i, item in enumerate(pending):
                    if item.get("scene_id") == prefer_scene:
                        idx = i
                        break
            item = pending.pop(idx)
            claimed = dict(data.get("claimed") or {})
            claimed[item["episode_id"]] = {
                "worker_id": str(worker_id),
                "claimed_at": time.time(),
                "scene_id": item.get("scene_id"),
            }
            data["pending"] = pending
            data["claimed"] = claimed
            _atomic_write(self.path, data)
            return item

    def mark_done(self, episode_id: str) -> None:
        import fcntl

        with self._lock_fd() as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            with open(self.path) as qf:
                data = json.load(qf)
            eid = str(episode_id)
            claimed = dict(data.get("claimed") or {})
            info = claimed.pop(eid, None)
            done = list(data.get("done") or [])
            if eid not in done:
                done.append(eid)
            data["claimed"] = claimed
            data["done"] = done
            _atomic_write(self.path, data)

    def release(self, episode_id: str) -> None:
        """Put a failed claim back on pending so another worker can retry."""
        import fcntl

        with self._lock_fd() as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            with open(self.path) as qf:
                data = json.load(qf)
            eid = str(episode_id)
            claimed = dict(data.get("claimed") or {})
            info = claimed.pop(eid, None) or {}
            pending: List[Dict[str, str]] = list(data.get("pending") or [])
            if not any(item.get("episode_id") == eid for item in pending):
                pending.append({
                    "episode_id": eid,
                    "scene_id": str(info.get("scene_id") or ""),
                })
            data["claimed"] = claimed
            data["pending"] = pending
            _atomic_write(self.path, data)


def existing_episode_ids(root: str) -> List[str]:
    ids = []
    if not root or not os.path.isdir(root):
        return ids
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("episode_") and name.endswith(".json"):
                ids.append(name[len("episode_") : -len(".json")])
            elif name == "assignment.json":
                ids.append(os.path.basename(dirpath))
    return ids


def _cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="HM3D-M3 file work queue")
    sub = p.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init")
    init.add_argument("--episodes", required=True)
    init.add_argument("--out", required=True)
    init.add_argument("--n", type=int, default=0)
    init.add_argument("--skip-root", default="")
    args = p.parse_args(argv)
    if args.cmd == "init":
        with open(args.episodes) as fh:
            eps = json.load(fh)["episodes"]
        if args.n:
            eps = eps[: args.n]
        items = [
            {"episode_id": str(e["episode_id"]), "scene_id": str(e["scene_id"])}
            for e in eps
        ]
        skip = existing_episode_ids(args.skip_root) if args.skip_root else []
        q = FileWorkQueue.init(args.out, items, skip_ids=skip)
        payload = json.load(open(q.path))
        print(
            f"[work-queue] init {q.path} pending={len(payload['pending'])} "
            f"skipped_done={len(payload['skipped_done'])}"
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
