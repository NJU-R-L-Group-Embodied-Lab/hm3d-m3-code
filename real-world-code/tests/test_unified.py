"""Public configuration, replay, sensor parsing and shared-map boundaries."""

import json
import time
from dataclasses import replace
from types import SimpleNamespace as N

import pytest

from lavira_m3.decision.replay import ReplayRouter, request_key
from lavira_m3.mapping.protocol import Meta, Record, encode
from lavira_m3.profiles import validate_agents
from lavira_m3.sensors import image_array


def test_arbitrary_agent_name_and_geometry():
    p = validate_agents(
        {"research_cart": dict(footprint=[1.1, 0.6], directions=["front", "right"])}
    )
    assert p["research_cart"]["footprint"] == [1.1, 0.6]
    for invalid in (
        {},
        {"../escape": {}},
        {"valid": {"arrival_tolerance": 0}},
        {"valid": {"inflate": float("nan")}},
        {"valid": {"directions": ["front", "front"]}},
    ):
        with pytest.raises(ValueError):
            validate_agents(invalid)


def test_replay_is_exact_and_has_no_live_fallback(tmp_path):
    messages = [dict(role="user", content="connectivity")]
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps([dict(request_sha256=request_key("la", messages), response="OK")]))
    router = ReplayRouter(path)
    assert router.generate_la(messages) == "OK"
    with pytest.raises(ValueError):
        router.generate_va(messages)
    with pytest.raises(ValueError):
        router.generate_la([dict(role="user", content="different")])


def test_padded_rgb_and_depth_are_decoded_without_driver_dependency():
    rgb = N(
        height=1, width=1, step=4, encoding="rgb8", is_bigendian=False, data=bytes([10, 20, 30, 0])
    )
    assert image_array(rgb).tolist() == [[[30, 20, 10]]]
    depth = N(height=1, width=1, step=2, encoding="16UC1", is_bigendian=False, data=bytes([232, 3]))
    assert image_array(depth)[0, 0] == 1.0
    rgb.step = 2
    with pytest.raises(ValueError):
        image_array(rgb)


def test_map_upload_rejects_stale_frames_and_unknown_sources():
    from lavira_m3.hub import MapStore

    store = MapStore()
    meta = Meta(frame="map", timestamp=time.time_ns(), map_id="probe")
    rows = [Record(10, y, 0, 2) for y in range(3)]
    assert store.upload("small_a", encode(meta, rows))["ok"]
    with pytest.raises(ValueError):
        store.upload("small_a", encode(meta, rows))
    with pytest.raises(ValueError):
        store.upload("unknown", encode(meta, rows))
    with pytest.raises(ValueError):
        store.upload("small_b", encode(replace(meta, frame="wrong"), rows))
    with pytest.raises(ValueError):
        store.upload("small_b", encode(replace(meta, timestamp=1), rows))
    assert store.scene()["fused"]


def test_hub_http_board_and_web_assets():
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from lavira_m3 import hub

    server = ThreadingHTTPServer(("127.0.0.1", 0), hub.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_port)
    try:
        for path in ("/", "/app.js", "/style.css", "/v1/scene", "/v1/status", "/api/board"):
            with urlopen(base + path, timeout=2) as response:
                assert response.status == 200 and response.read()
        req = Request(
            base + "/api/board/init",
            data=b"{}",
            headers={"Content-Type": "application/json", "Origin": "https://example.invalid"},
        )
        with pytest.raises(HTTPError) as caught:
            urlopen(req, timeout=2)
        assert caught.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_fixed_camera_selection_returns_navigation_result():
    from lavira_m3.runtime import Runtime

    runtime = Runtime.__new__(Runtime)
    assert runtime.face_direction("front").ok
    assert not runtime.face_direction("missing_camera").ok


def test_uploaded_map_obeys_shared_planning_contract():
    from lavira_m3.hub import MapStore
    from lavira_m3.mapping.remote_planning import layers_for_robot

    store = MapStore()
    meta = Meta(frame="map", timestamp=time.time_ns(), map_id="planning-probe")
    rows = [Record(10, y, 0, 2) for y in range(3)]
    for invalid in (replace(meta, resolution=0.05), replace(meta, origin=(1, 0, 0))):
        with pytest.raises(ValueError):
            store.upload("small_a", encode(invalid, rows))
    store.upload("small_a", encode(meta, rows))
    result = layers_for_robot(
        store.fusion.status(), store.fusion.fused.cells, store.fusion.fused.meta, "small_a"
    )
    assert result["available"] and 100 in result["planning"]
