"""HTTP dashboard, replay integrity and bounded recording regressions."""

import base64
import io
import json
import threading
import time
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as N
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from lavira_m3 import hub
from lavira_m3.dashboard import Dashboard
from lavira_m3.dashboard_bridge import DashboardPublisher
from lavira_m3.decision import board_state
from lavira_m3.mapping.protocol import Meta, Record, encode
from lavira_m3.recordings import DashboardRecordings
from lavira_m3.settings import AGENT, PROFILE


def wait_until(check):
    until = time.monotonic() + 4
    while time.monotonic() < until:
        if check():
            return
        time.sleep(0.01)
    raise AssertionError("condition timed out")


def test_recording_persists_images_and_seek_after_restart(tmp_path):
    jpeg = b"\xff\xd8synthetic\xff\xd9"
    frame = dict(
        t=123.5,
        scene=dict(fused=[[1, 2]]),
        images=dict(large=dict(front="data:image/jpeg;base64," + base64.b64encode(jpeg).decode())),
    )
    recorder = DashboardRecordings(tmp_path, lambda: frame)
    recorder.interval = 0.01
    try:
        recording_id = recorder.start()["id"]
        with pytest.raises(ValueError, match="already active"):
            recorder.start()
        wait_until(lambda: recorder.status()["samples"] >= 2)
    finally:
        recorder.stop()
    reopened = DashboardRecordings(tmp_path, lambda: {})
    assert reopened.list()[0]["total"] >= 2
    assert reopened.frame(recording_id, 1) == frame
    with pytest.raises(ValueError):
        reopened.frame(recording_id, 999)
    output = io.BytesIO()
    reopened.export(recording_id, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.read("frame-000000/large-front.jpg") == jpeg
        saved = json.loads(archive.read("frame-000000/snapshot.json"))
        assert saved["scene"] == frame["scene"]
    assert "images" in frame


@pytest.mark.parametrize("limit", ["max_bytes", "max_frames"])
def test_recording_stops_at_capacity(tmp_path, limit):
    recorder = DashboardRecordings(tmp_path, lambda: dict(t=1))
    recorder.interval = 0.001
    setattr(recorder, limit, 1)
    try:
        recorder.start()
        wait_until(lambda: not recorder.status()["active"])
        state = recorder.status()
        assert state["error"] == "recording capacity reached"
        assert state["samples"] <= 1
    finally:
        recorder.stop()


def test_recording_failure_and_path_validation(tmp_path):
    recorder = DashboardRecordings(tmp_path, lambda: dict(t=float("nan")))
    try:
        recorder.start()
        wait_until(lambda: not recorder.status()["active"])
        assert recorder.status()["error"]
        assert recorder.status()["samples"] == 0
    finally:
        recorder.stop()
    for value in ("../outside", "missing", "recording-20261008-000000-12345678"):
        with pytest.raises(ValueError):
            recorder.frame(value, 0)
    with pytest.raises(ValueError):
        recorder.frame(recorder.status()["id"], -1)


@pytest.fixture
def dashboard_server(monkeypatch, tmp_path):
    store = hub.MapStore()
    dashboard = Dashboard(store, tmp_path / "recordings")
    monkeypatch.setattr(hub, "STORE", store)
    monkeypatch.setattr(hub, "DASHBOARD", dashboard)
    monkeypatch.setattr(board_state, "STORE", board_state.BoardStore())
    server = ThreadingHTTPServer(("127.0.0.1", 0), hub.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_port)

    def call(path, body=None, origin=None):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        data = None if body is None else json.dumps(body).encode()
        with urlopen(Request(base + path, data=data, headers=headers), timeout=5) as response:
            return response.read()

    yield dashboard, call
    dashboard.recordings.stop()
    server.shutdown()
    server.server_close()
    thread.join(2)


def test_dashboard_http_assets_and_snapshot_isolation(dashboard_server):
    dashboard, call = dashboard_server
    for asset in (Path(hub.__file__).parent / "web").iterdir():
        assert call("/" if asset.name == "index.html" else "/" + asset.name) == asset.read_bytes()
    dashboard.store.upload(
        "large",
        encode(
            Meta(timestamp=time.time_ns(), pose_valid=True), [Record(10, y) for y in range(10, 14)]
        ),
    )
    dashboard.store.images[("large", "front")] = (time.monotonic(), b"\xff\xd8image")
    dashboard.telemetry.ingest(
        "large",
        dict(
            session="test",
            seq=1,
            state=dict(phase="tracking"),
            reference=[[1, 2]],
            prediction=[[2, 3]],
        ),
    )
    frame = dashboard.snapshot(images=True)
    assert frame["telemetry"]["robots"]["large"]["reference"] == [[1, 2]]
    assert frame["images"]["large"]["front"].startswith("data:image/jpeg;")
    frame["scene"]["fused"].clear()
    assert dashboard.store.scene()["fused"]
    overlay = json.loads(call("/v1/planning-grid?robot=large"))
    assert overlay["available"] and "peers" not in overlay and "planning" not in overlay
    with pytest.raises(HTTPError) as error:
        call("/v1/record/start", {}, "https://example.invalid")
    assert error.value.code == 403
    recording_id = json.loads(call("/v1/record/start", {}))["id"]
    wait_until(lambda: dashboard.recordings.status()["samples"] >= 1)
    call("/v1/record/stop", {})
    before = dashboard.store.scene()
    recorded = json.loads(call("/v1/record/frame?id=" + recording_id + "&index=0"))
    assert recorded["scene"]["fused"]
    assert dashboard.store.scene() == before
    assert zipfile.is_zipfile(io.BytesIO(call("/v1/record/download?id=" + recording_id)))
    with pytest.raises(HTTPError) as error:
        call("/v1/record/frame?id=../bad&index=0")
    assert error.value.code == 400
    call("/v1/clear", {})
    assert dashboard.planning == {} and dashboard.store.scene()["fused"] == []


def test_snapshot_camera_freshness(dashboard_server):
    dashboard, call = dashboard_server
    dashboard.store.images[("large", "front")] = (time.monotonic() - 6, b"\xff\xd8old")
    frame = dashboard.snapshot(images=True)
    assert frame["cameras"]["large"]["front"]["online"] is False
    assert frame["images"]["large"] == {}
    with pytest.raises(HTTPError) as error:
        call("/v1/image/large/front")
    assert error.value.code == 404


def test_dashboard_publisher_validates_markers_and_publishes_jpeg():
    publisher = DashboardPublisher.__new__(DashboardPublisher)
    publisher.io = N(now_ns=lambda: 10_000_000_000)
    publisher.lock = threading.Lock()
    publisher.markers = {}
    publisher.session, publisher.sequence = "test", 0
    publisher.stopped = threading.Event()
    publisher.sensors = N(
        view=lambda _: dict(rgb=np.zeros((8, 8, 3), dtype=np.uint8)), pose=lambda: (1, 2, 0)
    )
    sent = []
    publisher.request = lambda *args: sent.append(args)
    message = N(
        header=N(frame_id="map", stamp=N(sec=10, nanosec=0)), action=0, type=4, points=[N(x=1, y=2)]
    )
    publisher.on_marker("reference", message)
    publisher.publish()
    assert len(sent) == len(PROFILE["directions"]) + 1
    assert all(item[1].startswith(b"\xff\xd8") for item in sent[:-1])
    assert sent[-1][0] == "/v1/telemetry/" + AGENT
    assert json.loads(sent[-1][1])["reference"] == [[1, 2]]
    publisher.io.now_ns = lambda: 12_000_000_000
    message.points = [N(x=float("nan"), y=0)]
    publisher.on_marker("prediction", message)
    publisher.publish()
    packet = json.loads(sent[-1][1])
    assert packet["reference"] == [] and "prediction" not in packet
    publisher.stopped.set()
    count = len(sent)
    publisher.publish()
    assert len(sent) == count
