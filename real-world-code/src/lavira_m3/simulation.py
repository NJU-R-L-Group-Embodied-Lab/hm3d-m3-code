"""Deterministic kinematic connectivity fixture for either ROS version."""

import math
import time

import numpy as np

from .adapters.ros_io import RosIO
from .mapping.grid_cloud import pack_xyz
from .settings import BODY_FRAME, FRAME, PROFILE, SAFETY, TOPICS


def run(ros):
    from geometry_msgs.msg import TransformStamped, Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
    from tf2_ros import TransformBroadcaster

    io = RosIO(ros, "connectivity_fixture")
    transform = TransformBroadcaster(io.node) if ros == "ros2" else TransformBroadcaster()
    odom = io.publisher(Odometry, TOPICS["odometry"])
    localization = io.publisher(Odometry, TOPICS["localization"])
    imu = io.publisher(Imu, TOPICS["imu"])
    cloud = io.publisher(PointCloud2, TOPICS["cloud"])
    cameras = {
        d: (
            io.publisher(Image, "/camera/" + d + "/color/image_raw"),
            io.publisher(CameraInfo, "/camera/" + d + "/color/camera_info"),
        )
        for d in PROFILE["directions"]
    }
    velocity = [0.0, 0.0, 0.0]

    def command(m):
        if all(math.isfinite(v) for v in (m.linear.x, m.angular.z)):
            velocity[:] = [
                max(-SAFETY["max_speed"], min(SAFETY["max_speed"], m.linear.x)),
                max(-SAFETY["max_yaw_rate"], min(SAFETY["max_yaw_rate"], m.angular.z)),
                time.monotonic(),
            ]

    io.subscribe(Twist, TOPICS["command"], command)
    x = y = yaw = 0.0
    last = time.monotonic()
    tick = 0
    try:
        while True:
            now = time.monotonic()
            dt = min(0.05, now - last)
            last = now
            v, w = velocity[:2] if now - velocity[2] <= SAFETY["command_timeout"] else (0.0, 0.0)
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += w * dt
            o = Odometry()
            o.header.stamp = io.stamp()
            o.header.frame_id = FRAME
            o.child_frame_id = BODY_FRAME
            o.pose.pose.position.x = x
            o.pose.pose.position.y = y
            o.pose.pose.orientation.z = math.sin(yaw / 2)
            o.pose.pose.orientation.w = math.cos(yaw / 2)
            o.twist.twist.linear.x = v * math.cos(yaw)
            o.twist.twist.linear.y = v * math.sin(yaw)
            o.twist.twist.angular.z = w
            odom.publish(o)
            localization.publish(o)
            t = TransformStamped()
            t.header = o.header
            t.child_frame_id = BODY_FRAME
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.rotation = o.pose.pose.orientation
            transform.sendTransform(t)
            m = Imu()
            m.header.stamp = o.header.stamp
            m.header.frame_id = BODY_FRAME
            m.orientation = o.pose.pose.orientation
            m.angular_velocity.z = w
            m.linear_acceleration.z = 9.81
            imu.publish(m)
            if tick % 5 == 0:
                points = [(4.0, j * 0.1, 0.4) for j in range(-20, 21)]
                p = PointCloud2()
                p.header = o.header
                p.height = 1
                p.width = len(points)
                p.point_step = 12
                p.row_step = 12 * len(points)
                p.fields = [
                    PointField(name=n, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                    for i, n in enumerate(("x", "y", "z"))
                ]
                p.is_dense = True
                p.data = pack_xyz(points)
                cloud.publish(p)
                for rgb, info in cameras.values():
                    im = Image()
                    im.header = o.header
                    im.height = 48
                    im.width = 64
                    im.encoding = "bgr8"
                    im.step = 192
                    im.data = np.full((48, 64, 3), 120, dtype=np.uint8).tobytes()
                    rgb.publish(im)
                    c = CameraInfo()
                    c.header = o.header
                    c.width = 64
                    c.height = 48
                    matrix = [60.0, 0.0, 32.0, 0.0, 60.0, 24.0, 0.0, 0.0, 1.0]
                    if ros == "ros1":
                        c.K = matrix
                    else:
                        c.k = matrix
                    info.publish(c)
            tick += 1
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        io.close()
