"""Habitat-free FileWorkQueue claim / scene-prefer / uniqueness checks."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from benchmark.work_queue import FileWorkQueue, existing_episode_ids


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        qpath = os.path.join(td, "work_queue.json")
        items = [
            {"episode_id": "a1", "scene_id": "S1"},
            {"episode_id": "a2", "scene_id": "S1"},
            {"episode_id": "b1", "scene_id": "S2"},
            {"episode_id": "c1", "scene_id": "S3"},
            {"episode_id": "d1", "scene_id": "S4"},
        ]
        skip_root = os.path.join(td, "done")
        os.makedirs(os.path.join(skip_root, "w0"), exist_ok=True)
        Path(os.path.join(skip_root, "w0", "episode_c1.json")).write_text("{}")
        os.makedirs(os.path.join(skip_root, "shard0", "d1"), exist_ok=True)
        Path(os.path.join(skip_root, "shard0", "d1", "assignment.json")).write_text("{}")
        FileWorkQueue.init(qpath, items, skip_ids=existing_episode_ids(skip_root))
        payload = json.loads(Path(qpath).read_text())
        assert [x["episode_id"] for x in payload["pending"]] == ["a1", "a2", "b1"]
        assert set(payload["skipped_done"]) == {"c1", "d1"}

        q = FileWorkQueue(qpath)
        first = q.claim("w0", prefer_scene="S1")
        assert first["episode_id"] == "a1"
        second = q.claim("w1", prefer_scene="S1")
        assert second["episode_id"] == "a2"
        third = q.claim("w0", prefer_scene="S1")
        assert third["episode_id"] == "b1"
        assert q.claim("w2") is None
        q.mark_done("a1")
        done = json.loads(Path(qpath).read_text())["done"]
        assert done == ["a1"]
        q.release("a2")
        back = json.loads(Path(qpath).read_text())
        assert "a2" not in back["claimed"]
        assert any(item["episode_id"] == "a2" for item in back["pending"])

        qpath2 = os.path.join(td, "race.json")
        FileWorkQueue.init(
            qpath2,
            [{"episode_id": f"e{i}", "scene_id": "S"} for i in range(20)],
        )
        got = []
        lock = threading.Lock()

        def worker(wid: str) -> None:
            qq = FileWorkQueue(qpath2)
            while True:
                item = qq.claim(wid)
                if item is None:
                    return
                with lock:
                    got.append(item["episode_id"])

        threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert set(got) == {f"e{i}" for i in range(20)}, got
        assert len(got) == 20
        print("work_queue_selftest OK")


if __name__ == "__main__":
    main()
