"""GO1/GO2 local planning map: fused incremental download + peer envelopes.

This is not the upload map. Upload is own lidar with teammate bodies punched
out (sender_ros1). Planning treats teammates as dynamic obstacles here.
"""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

from .inflate import PLAN_INFLATE
from .peers import PeerCache
from .remote_planning import layers_for_robot


def fetch_state(host, port, timeout=3.0):
    req = Request(
        "http://%s:%s/v1/state" % (host, port),
        headers={"Cache-Control": "no-store"},
    )
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def peer_view(status, robot, now=None):
    cache = PeerCache()
    cache.observe(status, now=now)
    return cache.view(robot, now=now)


def compose(download, status, robot, now=None):
    if robot not in PLAN_INFLATE:
        raise ValueError("unknown planning robot")
    if not download.cells and not download.ready:
        return None
    layers = layers_for_robot(status, download.cells, download.meta(), robot, now=now)
    layers["view"] = peer_view(status, robot, now=now)
    return layers
