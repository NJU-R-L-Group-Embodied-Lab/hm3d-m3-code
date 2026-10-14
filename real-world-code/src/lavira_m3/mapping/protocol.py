"""VXMP v1/v2 codec. Same layout as voxel-share L0 voxel_protocol.hpp."""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass

HEADER = 160
RECORD = 13
MAX_CELLS = 2_000_000
MAGIC = b"VXMP"


def now_ns():
    return time.time_ns()


def _ident(s):
    if not s or len(s) >= 32:
        raise ValueError("invalid identifier")
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_/-" for c in s):
        raise ValueError("invalid identifier")
    return s


@dataclass
class Meta:
    type: int = 1
    seq: int = 1
    base: int = 0
    timestamp: int = 0
    resolution: float = 0.1
    origin: tuple = (0.0, 0.0, 0.0)
    map_id: str = "map2d"
    frame: str = "map"
    mode: int = 1
    pose_valid: bool = False
    position: tuple = (0.0, 0.0, 0.0)
    orientation: tuple = (0.0, 0.0, 0.0, 1.0)


@dataclass
class Record:
    ix: int
    iy: int
    iz: int = 0
    state: int = 2


def validate(m: Meta):
    if m.type not in (1, 2):
        raise ValueError("invalid message type")
    if not (0 < m.seq < 9007199254740991):
        raise ValueError("invalid sequence")
    if m.type == 1:
        if m.base != 0:
            raise ValueError("invalid base sequence")
    elif not (m.base > 0 and m.seq == m.base + 1):
        raise ValueError("invalid base sequence")
    if m.resolution not in (0.05, 0.1):
        raise ValueError("resolution must be 0.05 or 0.10")
    _ident(m.map_id)
    _ident(m.frame)
    if m.mode not in (1, 2, 3):
        raise ValueError("invalid source mode")


def index_xy(x, y, origin, res):
    return (
        int(math.floor((x - origin[0]) / res)),
        int(math.floor((y - origin[1]) / res)),
    )


def encode(meta: Meta, rows):
    validate(meta)
    if len(rows) > MAX_CELLS:
        raise ValueError("map exceeds 2M cells")
    ver = 2 if meta.pose_valid else 1
    buf = bytearray(HEADER + len(rows) * RECORD)
    buf[0:4] = MAGIC
    struct.pack_into("<H", buf, 4, ver)
    struct.pack_into("<H", buf, 6, meta.type)
    struct.pack_into("<I", buf, 8, len(rows) * RECORD)
    struct.pack_into("<I", buf, 12, len(rows))
    struct.pack_into("<Q", buf, 16, meta.seq)
    struct.pack_into("<Q", buf, 24, meta.base)
    struct.pack_into("<Q", buf, 32, meta.timestamp)
    struct.pack_into("<d", buf, 40, meta.resolution)
    struct.pack_into("<ddd", buf, 48, *meta.origin)
    mid = meta.map_id.encode("ascii")
    frm = meta.frame.encode("ascii")
    buf[72 : 72 + len(mid)] = mid
    buf[104 : 104 + len(frm)] = frm
    buf[136] = meta.mode
    if meta.pose_valid:
        buf[137] = 1
        struct.pack_into("<fff", buf, 138, *meta.position)
        for i, q in enumerate(meta.orientation):
            struct.pack_into("<h", buf, 150 + 2 * i, int(round(q * 32767)))
    for i, r in enumerate(rows):
        if r.state > 2:
            raise ValueError("invalid voxel state")
        o = HEADER + i * RECORD
        struct.pack_into("<iii", buf, o, r.ix, r.iy, r.iz)
        buf[o + 12] = r.state
    return bytes(buf)


def decode(body: bytes):
    if len(body) < HEADER or body[:4] != MAGIC:
        raise ValueError("invalid header")
    ver = struct.unpack_from("<H", body, 4)[0]
    if ver not in (1, 2):
        raise ValueError("unsupported protocol version")
    n = struct.unpack_from("<I", body, 12)[0]
    if n > MAX_CELLS or struct.unpack_from("<I", body, 8)[0] != n * RECORD:
        raise ValueError("invalid payload length")
    if len(body) != HEADER + n * RECORD:
        raise ValueError("invalid payload length")
    m = Meta()
    m.type = struct.unpack_from("<H", body, 6)[0]
    m.seq = struct.unpack_from("<Q", body, 16)[0]
    m.base = struct.unpack_from("<Q", body, 24)[0]
    m.timestamp = struct.unpack_from("<Q", body, 32)[0]
    m.resolution = struct.unpack_from("<d", body, 40)[0]
    m.origin = struct.unpack_from("<ddd", body, 48)

    def text(off):
        raw = body[off : off + 32]
        end = raw.find(b"\x00")
        if end < 0 or any(raw[end:]):
            raise ValueError("invalid identifier")
        return raw[:end].decode("ascii")

    m.map_id = text(72)
    m.frame = text(104)
    m.mode = body[136]
    if ver == 2:
        if body[137] != 1:
            raise ValueError("missing v2 pose")
        m.pose_valid = True
        m.position = struct.unpack_from("<fff", body, 138)
        qs = [struct.unpack_from("<h", body, 150 + 2 * i)[0] / 32767.0 for i in range(4)]
        m.orientation = tuple(qs)
    start = 158 if ver == 2 else 137
    if any(body[i] for i in range(start, HEADER)):
        raise ValueError("reserved bytes must be zero")
    validate(m)
    rows = []
    for i in range(n):
        o = HEADER + i * RECORD
        ix, iy, iz = struct.unpack_from("<iii", body, o)
        st = body[o + 12]
        if st > 2:
            raise ValueError("invalid voxel state")
        rows.append(Record(ix, iy, iz, st))
    return m, rows
