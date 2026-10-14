"""Compose middleware, observations and the shared LA/VA workflow."""

import fcntl
import math

from .adapters.ros_io import RosIO
from .contracts import NavResult
from .sensors import Sensors
from .settings import FRAME, PROFILE, lock_path


class Runtime:
    def __init__(self, ros, control=False):
        self.io = RosIO(ros)
        self.sensors = Sensors(self.io)
        self.owner = open(lock_path("owner"), "a")
        fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ros == "ros1":
            from .adapters.ros1.fastplanner_nav import FastPlannerNavClient

            self.nav = FastPlannerNavClient(frame=FRAME, enable_motion=control, pose_fn=self.pose)
        else:
            from .adapters.ros2.nav_client import NavClient

            self.nav = NavClient(self.io, control)

    def pose(self):
        return self.sensors.pose()

    def capture_direction(self, name):
        return self.sensors.view(name)

    def capture_panorama(self):
        return [v for d in PROFILE["directions"] for v in [self.sensors.view(d)] if v is not None]

    def face_direction(self, name):
        # A configured fixed camera observes that bearing without chassis rotation.
        return NavResult(name in PROFILE["directions"], "camera_selected", str(name))

    def navigate_to(self, x, y, yaw=None, **kwargs):
        pose = self.pose()
        if pose is None:
            return NavResult(False, "unavailable", "odometry stale")
        if yaw is None:
            yaw = math.atan2(y - pose[1], x - pose[0])
        if self.io.version == "ros1":
            return self.nav.navigate_to(x, y, yaw)
        return self.nav.navigate_to(x, y, yaw, **kwargs)

    def stop(self):
        return self.nav.cancel()

    def close(self):
        try:
            self.nav.close()
        finally:
            self.owner.close()
            self.io.close()
