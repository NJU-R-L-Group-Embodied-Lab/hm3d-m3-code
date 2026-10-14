"""Bounded dashboard recordings; replay reads snapshots without changing live state."""

import base64
import gzip
import json
import re
import sqlite3
import threading
import time
import uuid
import zipfile
from pathlib import Path


class DashboardRecordings:
    interval = 0.5
    max_bytes = 512 * 1024 * 1024
    max_frames = 7200

    def __init__(self, root, capture):
        self.root = Path(root)
        self.capture = capture
        self.lifecycle = threading.Lock()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker = None
        self.state = dict(active=False, id=None, samples=0, bytes=0, error="")

    def _path(self, recording_id):
        if not re.fullmatch(r"recording-[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", recording_id):
            raise ValueError("invalid recording identifier")
        return self.root / (recording_id + ".sqlite3")

    def status(self):
        with self.lock:
            return dict(self.state)

    def start(self):
        with self.lifecycle, self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError("a recording is already active")
            self.root.mkdir(parents=True, exist_ok=True)
            recording_id = (
                "recording-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            )
            with sqlite3.connect(self._path(recording_id)) as db:
                db.execute("CREATE TABLE frames (stamp REAL NOT NULL, payload BLOB NOT NULL)")
            self.state = dict(active=True, id=recording_id, samples=0, bytes=0, error="")
            self.stop_event.clear()
            self.worker = threading.Thread(target=self._record, daemon=True)
            self.worker.start()
            return self.status()

    def _record(self):
        try:
            with sqlite3.connect(self._path(self.state["id"])) as db:
                while not self.stop_event.is_set():
                    frame = self.capture()
                    payload = gzip.compress(json.dumps(frame, allow_nan=False).encode())
                    with self.lock:
                        if (
                            self.state["bytes"] + len(payload) > self.max_bytes
                            or self.state["samples"] >= self.max_frames
                        ):
                            self.state["error"] = "recording capacity reached"
                            break
                    db.execute("INSERT INTO frames VALUES (?, ?)", (frame["t"], payload))
                    db.commit()
                    with self.lock:
                        self.state["samples"] += 1
                        self.state["bytes"] += len(payload)
                    self.stop_event.wait(self.interval)
        except Exception as error:
            with self.lock:
                self.state["error"] = str(error)
        finally:
            with self.lock:
                self.state["active"] = False

    def stop(self):
        with self.lifecycle:
            self.stop_event.set()
            if self.worker:
                self.worker.join(timeout=10)
                if self.worker.is_alive():
                    raise RuntimeError("recording worker is still stopping")
            return self.status()

    def _open(self, recording_id):
        path = self._path(recording_id)
        if not path.is_file():
            raise ValueError("recording not found")
        return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)

    def list(self):
        result = []
        for path in sorted(self.root.glob("recording-*.sqlite3"), reverse=True)[:50]:
            with self._open(path.stem) as db:
                count, first, last = db.execute(
                    "SELECT COUNT(*), MIN(stamp), MAX(stamp) FROM frames"
                ).fetchone()
            result.append(dict(id=path.stem, total=count, duration_s=(last or 0) - (first or 0)))
        return result

    def frame(self, recording_id, index):
        if type(index) is not int or index < 0:
            raise ValueError("invalid frame index")
        with self._open(recording_id) as db:
            row = db.execute("SELECT payload FROM frames WHERE rowid = ?", (index + 1,)).fetchone()
        if row is None:
            raise ValueError("frame index out of range")
        return json.loads(gzip.decompress(row[0]))

    def export(self, recording_id, destination):
        """Stream a ZIP to a seekable temporary file; recorded JPEGs retain frame time."""
        with self._open(recording_id) as db, zipfile.ZipFile(
            destination, "w", zipfile.ZIP_STORED
        ) as archive:
            for index, (payload,) in enumerate(
                db.execute("SELECT payload FROM frames ORDER BY rowid")
            ):
                frame = json.loads(gzip.decompress(payload))
                images = frame.pop("images", {})
                stem = f"frame-{index:06d}"
                archive.writestr(stem + "/snapshot.json", json.dumps(frame, ensure_ascii=False))
                for agent, cameras in images.items():
                    for camera, url in cameras.items():
                        # Names are validated profiles; validate stored data as well.
                        if not re.fullmatch(r"[a-zA-Z0-9_-]+", agent + camera):
                            raise ValueError("invalid recorded image name")
                        archive.writestr(
                            stem + f"/{agent}-{camera}.jpg",
                            base64.b64decode(url.split(",", 1)[1], validate=True),
                        )
