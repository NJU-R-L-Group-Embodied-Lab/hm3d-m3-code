"""Hub-side HTTP + NIC accounting for the live board."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import struct
import threading
import time
from collections import deque

from lavira_m3.profiles import AGENTS

CHANNELS = (
    "map_up",
    "map_down",
    "image",
    "scene",
    "state",
    "plan",
    "pose",
    "nav",
    "board",
    "web",
    "other",
)

SIOCGIFADDR = 0x8915


def classify_path(path):
    path = (path or "").split("?", 1)[0]
    if path.startswith("/v1/map/") and path.count("/") >= 3:
        return "map_up", path.rstrip("/").rsplit("/", 1)[-1]
    if path == "/v1/map-updates" or path in ("/v1/map", "/v1/view", "/v1/costmap2d"):
        return "map_down", None
    if path.startswith(("/v1/image/", "/v1/front-image/", "/v1/camera-image/")):
        parts = path.strip("/").split("/")
        robot = parts[2] if len(parts) >= 3 else None
        return "image", robot
    if path in ("/v1/scene", "/v1/dashboard"):
        return "scene", None
    if path.startswith("/v1/telemetry/"):
        return "nav", path.rsplit("/", 1)[-1]
    if path in ("/v1/planning", "/v1/planning-grid", "/v1/robot-plan"):
        return "plan", None
    if path in ("/v1/state", "/v1/poses", "/v1/status"):
        return "state", None
    if path == "/v1/plan" or path.startswith("/v1/plan/"):
        robot = path.rstrip("/").rsplit("/", 1)[-1] if path.startswith("/v1/plan/") else None
        return "plan", robot if robot in AGENTS else None
    if path.startswith("/v1/pose/"):
        return "pose", path.rstrip("/").rsplit("/", 1)[-1]
    if path in ("/v1/goal", "/v1/preview", "/v1/cancel", "/v1/nearest-traversable"):
        return "nav", None
    if path.startswith("/api/board") or path.startswith("/api/t0"):
        return "board", None
    if path in ("/", "/index.html", "/app.js", "/trajectory.js", "/costmap.html", "/lavira.html"):
        return "web", None
    return "other", None


def read_netdev(path="/proc/net/dev"):
    out = {}
    try:
        lines = open(path, "r").read().splitlines()
    except OSError:
        return out
    for line in lines[2:]:
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        cols = rest.split()
        if len(cols) < 16:
            continue
        out[name.strip()] = {
            "rx_bytes": int(cols[0]),
            "rx_packets": int(cols[1]),
            "rx_errs": int(cols[2]),
            "rx_drop": int(cols[3]),
            "tx_bytes": int(cols[8]),
            "tx_packets": int(cols[9]),
            "tx_errs": int(cols[10]),
            "tx_drop": int(cols[11]),
        }
    return out


def iface_ipv4():
    names = []
    try:
        names = os.listdir("/sys/class/net")
    except OSError:
        return {}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    out = {}
    try:
        for name in names:
            try:
                packed = struct.pack("256s", name.encode("utf-8")[:15])
                info = fcntl.ioctl(sock.fileno(), SIOCGIFADDR, packed)
                out[name] = socket.inet_ntoa(info[20:24])
            except OSError:
                continue
    finally:
        sock.close()
    return out


def pick_iface(bind_ip, preferred=None, nic=None):
    if preferred and (nic is None or preferred in nic):
        return preferred
    addrs = iface_ipv4()
    for name, ip in addrs.items():
        if ip == bind_ip:
            return name
    nic = nic or {}
    ranked = [
        (counters["rx_bytes"] + counters["tx_bytes"], name)
        for name, counters in nic.items()
        if name != "lo"
    ]
    ranked.sort(reverse=True)
    return ranked[0][1] if ranked else "lo"


def _empty_bucket():
    return {"bytes_in": 0, "bytes_out": 0, "n": 0, "err": 0, "ms_sum": 0.0, "ms_max": 0.0}


def _mbps(nbytes, dt):
    if dt <= 0:
        return 0.0
    return round(nbytes * 8.0 / dt / 1e6, 3)


def _rps(n, dt):
    if dt <= 0:
        return 0.0
    return round(n / dt, 2)


def _compact_sample(sample):
    channels = sample.get("channels") or {}
    return {
        "t": sample.get("t"),
        "dt_s": sample.get("dt_s"),
        "iface": sample.get("iface"),
        "nic_rx_mbps": sample.get("nic_rx_mbps", 0.0),
        "nic_tx_mbps": sample.get("nic_tx_mbps", 0.0),
        "http_rx_mbps": sample.get("http_rx_mbps", 0.0),
        "http_tx_mbps": sample.get("http_tx_mbps", 0.0),
        "http_rps": sample.get("http_rps", 0.0),
        "http_err_rps": sample.get("http_err_rps", 0.0),
        "scene_tx_mbps": (channels.get("scene") or {}).get(
            "tx_mbps", sample.get("scene_tx_mbps", 0.0)
        ),
        "map_up_mbps": (channels.get("map_up") or {}).get(
            "rx_mbps", sample.get("map_up_mbps", 0.0)
        ),
        "map_down_mbps": (channels.get("map_down") or {}).get(
            "tx_mbps", sample.get("map_down_mbps", 0.0)
        ),
        "image_tx_mbps": (channels.get("image") or {}).get(
            "tx_mbps", sample.get("image_tx_mbps", 0.0)
        ),
    }


def _stats(history, key):
    vals = [float(item.get(key) or 0.0) for item in history]
    if not vals:
        return {"avg": 0.0, "peak": 0.0}
    return {"avg": round(sum(vals) / len(vals), 3), "peak": round(max(vals), 3)}


class NetMonitor:
    def __init__(
        self, allow, bind_ip, iface=None, history=300, sample_s=1.0, nic_reader=None, persist=None
    ):
        self.allow = dict(allow or {})
        self.ip_to_name = {ip: name for name, ip in self.allow.items()}
        self.bind_ip = bind_ip
        self.iface_hint = iface
        self.history_n = int(history)
        self.sample_s = float(sample_s)
        self.nic_reader = nic_reader or read_netdev
        self.persist = persist
        self.lock = threading.Lock()
        self.window = self._new_window(time.time())
        self.history = deque(maxlen=self.history_n)
        self.latest = None
        self.last_seen = {name: None for name in self.allow}
        self.last_hit = {name: None for name in self.allow}
        self.last_nic = self.nic_reader() or {}
        self._load_persist()

    def _new_window(self, now):
        return {
            "t0": now,
            "total": _empty_bucket(),
            "channels": {name: _empty_bucket() for name in CHANNELS},
            "robots": {name: _empty_bucket() for name in self.allow},
        }

    def robot_for(self, peer, path_robot):
        if path_robot in self.allow:
            return path_robot
        if peer in self.ip_to_name:
            return self.ip_to_name[peer]
        if peer in (self.bind_ip, "127.0.0.1", "::1"):
            return next(iter(self.allow), None)
        return None

    def record(self, peer, path, bytes_in, bytes_out, ms, code, now=None):
        now = time.time() if now is None else float(now)
        channel, path_robot = classify_path(path)
        if channel == "other" and path == "/v1/net":
            return
        robot = self.robot_for(peer, path_robot)
        with self.lock:
            self._flush_locked(now)
            bucket = self.window["total"]
            self._add(bucket, bytes_in, bytes_out, ms, code)
            self._add(self.window["channels"][channel], bytes_in, bytes_out, ms, code)
            if robot:
                self._add(self.window["robots"][robot], bytes_in, bytes_out, ms, code)
                self.last_seen[robot] = now
                self.last_hit[robot] = {
                    "t": now,
                    "ms": round(ms, 1),
                    "code": int(code or 0),
                    "channel": channel,
                    "peer": peer,
                    "bytes_in": int(bytes_in),
                    "bytes_out": int(bytes_out),
                }

    def _add(self, bucket, bytes_in, bytes_out, ms, code):
        bucket["bytes_in"] += int(bytes_in or 0)
        bucket["bytes_out"] += int(bytes_out or 0)
        bucket["n"] += 1
        bucket["ms_sum"] += float(ms or 0.0)
        bucket["ms_max"] = max(bucket["ms_max"], float(ms or 0.0))
        if int(code or 0) >= 400:
            bucket["err"] += 1

    def _flush_locked(self, now):
        if now < self.window["t0"]:
            self.window["t0"] = now
        dt = now - self.window["t0"]
        if dt < self.sample_s:
            return
        nic = self.nic_reader() or {}
        iface = pick_iface(self.bind_ip, self.iface_hint, nic)
        nic_rates = {}
        if self.last_nic is not None:
            for name, cur in nic.items():
                prev = self.last_nic.get(name)
                if not prev:
                    continue
                nic_rates[name] = {
                    "rx_mbps": _mbps(cur["rx_bytes"] - prev["rx_bytes"], dt),
                    "tx_mbps": _mbps(cur["tx_bytes"] - prev["tx_bytes"], dt),
                    "rx_pps": _rps(cur["rx_packets"] - prev["rx_packets"], dt),
                    "tx_pps": _rps(cur["tx_packets"] - prev["tx_packets"], dt),
                    "rx_err_delta": cur["rx_errs"] - prev["rx_errs"],
                    "tx_err_delta": cur["tx_errs"] - prev["tx_errs"],
                    "rx_drop_delta": cur["rx_drop"] - prev["rx_drop"],
                    "tx_drop_delta": cur["tx_drop"] - prev["tx_drop"],
                }
        total = self.window["total"]
        sample = {
            "t": round(now, 3),
            "dt_s": round(dt, 3),
            "iface": iface,
            "http_rx_mbps": _mbps(total["bytes_in"], dt),
            "http_tx_mbps": _mbps(total["bytes_out"], dt),
            "http_rps": _rps(total["n"], dt),
            "http_err_rps": _rps(total["err"], dt),
            "nic_rx_mbps": (nic_rates.get(iface) or {}).get("rx_mbps", 0.0),
            "nic_tx_mbps": (nic_rates.get(iface) or {}).get("tx_mbps", 0.0),
            "channels": {
                name: {
                    "rx_mbps": _mbps(bucket["bytes_in"], dt),
                    "tx_mbps": _mbps(bucket["bytes_out"], dt),
                    "rps": _rps(bucket["n"], dt),
                    "err_rps": _rps(bucket["err"], dt),
                    "avg_ms": round(bucket["ms_sum"] / bucket["n"], 1) if bucket["n"] else 0.0,
                    "max_ms": round(bucket["ms_max"], 1),
                }
                for name, bucket in self.window["channels"].items()
            },
            "robots": {
                name: {
                    "up_mbps": _mbps(bucket["bytes_in"], dt),
                    "down_mbps": _mbps(bucket["bytes_out"], dt),
                    "rps": _rps(bucket["n"], dt),
                    "avg_ms": round(bucket["ms_sum"] / bucket["n"], 1) if bucket["n"] else 0.0,
                    "max_ms": round(bucket["ms_max"], 1),
                }
                for name, bucket in self.window["robots"].items()
            },
            "nic": nic_rates,
        }
        compact = _compact_sample(sample)
        self.latest = sample
        self.history.append(compact)
        self._append_persist(compact)
        self.last_nic = nic
        self.window = self._new_window(now)

    def _load_persist(self):
        path = self.persist
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r") as handle:
                lines = handle.read().splitlines()
        except OSError:
            return
        for line in lines[-self.history_n :]:
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict) and item.get("t") is not None:
                self.history.append(_compact_sample(item))

    def _append_persist(self, sample):
        path = self.persist
        if not path:
            return
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "a") as handle:
                handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            if os.path.getsize(path) > 2_000_000:
                with open(path, "r") as handle:
                    lines = handle.read().splitlines()[-self.history_n :]
                with open(path, "w") as handle:
                    handle.write("\n".join(lines) + ("\n" if lines else ""))
        except OSError:
            pass

    def snapshot(self, now=None):
        now = time.time() if now is None else float(now)
        with self.lock:
            self._flush_locked(now)
            latest = self.latest
            alerts = []
            if latest:
                iface = latest.get("iface")
                nic = (latest.get("nic") or {}).get(iface) or {}
                if (nic.get("rx_err_delta") or 0) > 0 or (nic.get("tx_err_delta") or 0) > 0:
                    alerts.append(
                        "%s NIC errors +rx%s +tx%s"
                        % (iface, nic.get("rx_err_delta", 0), nic.get("tx_err_delta", 0))
                    )
                if (nic.get("rx_drop_delta") or 0) > 0 or (nic.get("tx_drop_delta") or 0) > 0:
                    alerts.append(
                        "%s drops +rx%s +tx%s"
                        % (iface, nic.get("rx_drop_delta", 0), nic.get("tx_drop_delta", 0))
                    )
                if latest["http_err_rps"] >= 1:
                    alerts.append("HTTP errors %.1f/s" % latest["http_err_rps"])
                scene_mbps = ((latest.get("channels") or {}).get("scene") or {}).get("tx_mbps", 0)
                if scene_mbps >= 15:
                    alerts.append("dashboard /v1/scene downlink %.1f Mbps" % scene_mbps)
            robots = {}
            for name in self.allow:
                last = self.last_hit.get(name)
                ago = (
                    None
                    if self.last_seen.get(name) is None
                    else round(now - self.last_seen[name], 2)
                )
                if ago is not None and ago >= 5:
                    alerts.append("%s no HTTP for %.1fs" % (name, ago))
                row = dict(
                    (latest or {}).get("robots", {}).get(name)
                    or {
                        "up_mbps": 0.0,
                        "down_mbps": 0.0,
                        "rps": 0.0,
                        "avg_ms": 0.0,
                        "max_ms": 0.0,
                    }
                )
                row["ip"] = self.allow[name]
                row["last_ago_s"] = ago
                row["last"] = last
                robots[name] = row
            history = list(self.history)
            keys = (
                "nic_rx_mbps",
                "nic_tx_mbps",
                "http_rx_mbps",
                "http_tx_mbps",
                "scene_tx_mbps",
                "map_up_mbps",
                "map_down_mbps",
                "image_tx_mbps",
            )
            summary = {key: _stats(history, key) for key in keys}
            window_s = 0.0
            if len(history) >= 2:
                window_s = round(float(history[-1]["t"]) - float(history[0]["t"]), 1)
            return {
                "t": now,
                "iface": (latest or {}).get("iface")
                or pick_iface(self.bind_ip, self.iface_hint, self.last_nic),
                "bind": self.bind_ip,
                "http": {
                    "rx_mbps": (latest or {}).get("http_rx_mbps", 0.0),
                    "tx_mbps": (latest or {}).get("http_tx_mbps", 0.0),
                    "rps": (latest or {}).get("http_rps", 0.0),
                    "err_rps": (latest or {}).get("http_err_rps", 0.0),
                },
                "nic": (latest or {}).get("nic") or {},
                "channels": (latest or {}).get("channels")
                or {
                    name: {
                        "rx_mbps": 0.0,
                        "tx_mbps": 0.0,
                        "rps": 0.0,
                        "err_rps": 0.0,
                        "avg_ms": 0.0,
                        "max_ms": 0.0,
                    }
                    for name in CHANNELS
                },
                "robots": robots,
                "history": history[-180:],
                "summary": {
                    "samples": len(history),
                    "window_s": window_s,
                    "peak": {key: summary[key]["peak"] for key in keys},
                    "avg": {key: summary[key]["avg"] for key in keys},
                },
                "persist": self.persist,
                "alerts": alerts,
            }
