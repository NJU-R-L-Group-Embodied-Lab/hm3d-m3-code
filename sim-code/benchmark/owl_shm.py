"""File-backed RGB frames for the OWL sidecar (no JPEG / base64).

Habitat runs in Docker with ipc=private, so POSIX /dev/shm is not shared.
A directory bind-mounted into both containers (default ``$PWD/.owl_shm``) is.
Writers and the sidecar must point OWL_SHM_DIR at the same directory.
"""
from __future__ import annotations

import os
import struct
from typing import Optional

import numpy as np

MAGIC = b"OWL1"
HEADER_FMT = "<4sIIIII"  # magic, h, w, c, seq, nbytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)
TRAILER_FMT = "<I"


def default_shm_dir() -> str:
    env = (os.environ.get("OWL_SHM_DIR") or "").strip()
    if env:
        return env
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".owl_shm"))


def slot_path(shm_dir: str, name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))
    return os.path.join(shm_dir, safe + ".rgb")


class RgbShmSlot:
    def __init__(self, shm_dir: str, name: str):
        self.shm_dir = shm_dir
        self.name = str(name)
        self.path = slot_path(shm_dir, self.name)
        self.seq = 0
        self._mm = None
        self._cap = 0

    def _ensure(self, nbytes: int) -> None:
        need = HEADER_SIZE + int(nbytes) + 4
        if self._mm is not None and self._cap >= need:
            return
        os.makedirs(self.shm_dir, exist_ok=True)
        if self._mm is not None:
            try:
                self._mm._mmap.close()
            except Exception:
                pass
            self._mm = None
        with open(self.path, "wb") as f:
            f.truncate(need)
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r+", shape=(need,))
        self._cap = need

    def write(self, rgb: np.ndarray) -> dict:
        img = np.ascontiguousarray(np.asarray(rgb)[..., :3])
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        h, w, c = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
        raw = np.ascontiguousarray(img).reshape(-1)
        self._ensure(int(raw.nbytes))
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        header = np.frombuffer(
            struct.pack(HEADER_FMT, MAGIC, h, w, c, self.seq, int(raw.nbytes)),
            dtype=np.uint8,
        )
        self._mm[:HEADER_SIZE] = header
        self._mm[HEADER_SIZE:HEADER_SIZE + raw.nbytes] = raw
        end = HEADER_SIZE + int(raw.nbytes)
        self._mm[end:end + 4] = np.frombuffer(struct.pack(TRAILER_FMT, self.seq), dtype=np.uint8)
        self._mm.flush()
        return {"shm_name": self.name, "h": h, "w": w, "c": c, "seq": int(self.seq)}

    def close(self) -> None:
        if self._mm is not None:
            try:
                self._mm._mmap.close()
            except Exception:
                pass
            self._mm = None


ND_MAGIC = b"ND01"
ND_HEADER_FMT = "<4sIIII"  # magic, h, w, seq, nbytes
ND_HEADER_SIZE = struct.calcsize(ND_HEADER_FMT)


class F32ShmSlot:
    """HxW float32 (depth) on the same bind-mount as RGB slots."""

    def __init__(self, shm_dir: str, name: str):
        self.shm_dir = shm_dir
        self.name = str(name)
        self.path = slot_path(shm_dir, self.name + "_f32")
        self.seq = 0
        self._mm = None
        self._cap = 0

    def _ensure(self, nbytes: int) -> None:
        need = ND_HEADER_SIZE + int(nbytes) + 4
        if self._mm is not None and self._cap >= need:
            return
        os.makedirs(self.shm_dir, exist_ok=True)
        if self._mm is not None:
            try:
                self._mm._mmap.close()
            except Exception:
                pass
            self._mm = None
        with open(self.path, "wb") as f:
            f.truncate(need)
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r+", shape=(need,))
        self._cap = need

    def write(self, arr: np.ndarray) -> dict:
        d = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        if d.ndim == 3:
            d = d[..., 0]
        if d.ndim != 2:
            raise ValueError("F32ShmSlot expects HxW")
        h, w = int(d.shape[0]), int(d.shape[1])
        raw = d.reshape(-1).view(np.uint8)
        self._ensure(int(raw.nbytes))
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        header = np.frombuffer(
            struct.pack(ND_HEADER_FMT, ND_MAGIC, h, w, self.seq, int(raw.nbytes)),
            dtype=np.uint8,
        )
        self._mm[:ND_HEADER_SIZE] = header
        self._mm[ND_HEADER_SIZE:ND_HEADER_SIZE + raw.nbytes] = raw
        end = ND_HEADER_SIZE + int(raw.nbytes)
        self._mm[end:end + 4] = np.frombuffer(struct.pack(TRAILER_FMT, self.seq), dtype=np.uint8)
        self._mm.flush()
        return {"shm_name": self.name, "h": h, "w": w, "seq": int(self.seq)}


def read_f32(shm_dir: str, name: str, expect_seq: Optional[int] = None) -> Optional[np.ndarray]:
    path = slot_path(shm_dir, str(name) + "_f32")
    if not os.path.isfile(path):
        return None
    data = np.memmap(path, dtype=np.uint8, mode="r")
    if data.size < ND_HEADER_SIZE + 4:
        return None
    magic, h, w, seq, nbytes = struct.unpack(ND_HEADER_FMT, bytes(data[:ND_HEADER_SIZE]))
    if magic != ND_MAGIC or h <= 0 or w <= 0:
        return None
    end = ND_HEADER_SIZE + int(nbytes)
    if data.size < end + 4:
        return None
    seq2 = struct.unpack(TRAILER_FMT, bytes(data[end:end + 4]))[0]
    if seq != seq2:
        return None
    if expect_seq is not None and int(expect_seq) != int(seq):
        return None
    raw = np.array(data[ND_HEADER_SIZE:end], dtype=np.uint8)
    return raw.view(np.float32).reshape(int(h), int(w)).copy()


def read_rgb(shm_dir: str, name: str, expect_seq: Optional[int] = None) -> Optional[np.ndarray]:
    path = slot_path(shm_dir, name)
    if not os.path.isfile(path):
        return None
    data = np.memmap(path, dtype=np.uint8, mode="r")
    if data.size < HEADER_SIZE + 4:
        return None
    magic, h, w, c, seq, nbytes = struct.unpack(HEADER_FMT, bytes(data[:HEADER_SIZE]))
    if magic != MAGIC or h <= 0 or w <= 0 or c not in (1, 3, 4):
        return None
    end = HEADER_SIZE + int(nbytes)
    if data.size < end + 4:
        return None
    seq2 = struct.unpack(TRAILER_FMT, bytes(data[end:end + 4]))[0]
    if seq != seq2:
        return None
    if expect_seq is not None and int(expect_seq) != int(seq):
        return None
    arr = np.array(data[HEADER_SIZE:end], dtype=np.uint8).reshape(int(h), int(w), int(c))
    return arr[..., :3]
