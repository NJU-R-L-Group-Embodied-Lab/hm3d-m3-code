"""Registered point clouds to shared maps; shared planning snapshots back to ROS."""

import json
import logging
import threading
import time
from dataclasses import replace
from urllib.request import ProxyHandler, Request, build_opener

from .adapters.ros_io import RosIO, stamp_ns
from .dashboard_bridge import DashboardPublisher
from .mapping.builder import Builder2D
from .mapping.grid_cloud import lethal_xy, pack_xyz, xyz_from_xy
from .mapping.peers import PeerCache
from .mapping.protocol import encode
from .sensors import Sensors
from .settings import AGENT, FRAME, SAFETY, SERVICES, TOPICS

log = logging.getLogger(__name__)


def request(path, body=None, content_type="application/json"):
    req = Request(
        SERVICES["map"].rstrip("/") + path, data=body, headers={"Content-Type": content_type}
    )
    with build_opener(ProxyHandler({})).open(req, timeout=2.0) as response:
        return json.loads(response.read())


class MapBridge:
    def __init__(self, ros):
        from nav_msgs.msg import OccupancyGrid
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import String

        self.io = RosIO(ros, "map_bridge")
        self.sensors = Sensors(self.io)
        self.dashboard = DashboardPublisher(self.io, self.sensors, request)
        self.builder = Builder2D(AGENT)
        self.builder.meta.frame = FRAME
        self.lock = threading.RLock()
        self.peers = PeerCache()
        self.cloud = None
        self.sequence = 0
        self.last_map_ns = 0
        self.last_cloud_ns = 0
        self.grid = self.io.publisher(OccupancyGrid, TOPICS["map"], durable=True)
        self.points = self.io.publisher(PointCloud2, "/map2d/planning_cloud", durable=True)
        self.peer_pub = self.io.publisher(String, "/map2d/peer_state")
        self.io.subscribe(PointCloud2, TOPICS["cloud"], self.on_cloud)

    def on_cloud(self, msg):
        stamp = stamp_ns(msg.header.stamp)
        if msg.header.frame_id != FRAME or not -0.05 <= (self.io.now_ns() - stamp) / 1e9 <= 0.5:
            return
        with self.lock:
            if stamp <= self.last_cloud_ns:
                return
            self.cloud = msg
            self.last_cloud_ns = stamp

    def send(self):
        with self.lock:
            msg = self.cloud
            self.cloud = None
        pose = self.sensors.pose()
        if (
            msg is None
            or pose is None
            or not -0.05 <= (self.io.now_ns() - stamp_ns(msg.header.stamp)) / 1e9 <= 0.5
        ):
            return
        if self.io.version == "ros1":
            from sensor_msgs.point_cloud2 import read_points
        else:
            from sensor_msgs_py.point_cloud2 import read_points
        points = [
            tuple(float(v) for v in row)
            for row in read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        ]
        odom = self.sensors.odometry
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        view = self.peers.view(AGENT)
        if not self.builder.scan(
            points, (p.x, p.y, p.z), stamp_ns(msg.header.stamp), pose, view["peers"]
        ):
            return
        self.builder.set_pose((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
        self.sequence += 1
        meta = replace(self.builder.meta, type=1, seq=self.sequence, base=0)
        request(
            "/v1/map/" + AGENT,
            encode(meta, self.builder.snapshot_rows()),
            "application/octet-stream",
        )

    def pull(self):
        from nav_msgs.msg import OccupancyGrid
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import String

        state = request("/v1/status")
        self.peers.observe(state)
        self.peer_pub.publish(String(data=json.dumps(self.peers.view(AGENT))))
        data = request("/v1/planning?robot=" + AGENT)
        ns = int(data["stamp_ns"])
        if (
            not data.get("available")
            or ns <= self.last_map_ns
            or not -0.05 <= (self.io.now_ns() - ns) / 1e9 <= SAFETY["map_timeout"]
        ):
            return
        width, height = int(data["width"]), int(data["height"])
        if (
            width <= 0
            or height <= 0
            or width * height > 2_000_000
            or len(data["planning"]) != width * height
        ):
            raise ValueError("invalid planning grid")
        grid = OccupancyGrid()
        grid.header.stamp = self.io.stamp(ns)
        grid.header.frame_id = FRAME
        grid.info.width = width
        grid.info.height = height
        grid.info.resolution = float(data["resolution"])
        grid.info.origin.position.x = float(data["origin"][0])
        grid.info.origin.position.y = float(data["origin"][1])
        grid.info.origin.orientation.w = 1.0
        grid.data = data["planning"]
        xyz = xyz_from_xy(lethal_xy(grid.data, width, height, grid.info.resolution, data["origin"]))
        cloud = PointCloud2()
        cloud.header = grid.header
        cloud.height = 1
        cloud.width = len(xyz)
        cloud.fields = [
            PointField(name=name, offset=i * 4, datatype=PointField.FLOAT32, count=1)
            for i, name in enumerate(("x", "y", "z"))
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = 12 * len(xyz)
        cloud.is_dense = True
        cloud.data = pack_xyz(xyz)
        self.grid.publish(grid)
        self.points.publish(cloud)
        self.last_map_ns = ns

    def run(self):
        self.dashboard.start()
        try:
            while True:
                try:
                    self.send()
                    self.pull()
                except (OSError, ValueError, KeyError) as error:
                    log.warning("map exchange unavailable: %s", error)
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.dashboard.close()
            self.io.close()
