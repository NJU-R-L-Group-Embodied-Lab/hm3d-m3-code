"""Standard ROS messages at one boundary; no middleware imports at package import."""

import threading


class RosIO:
    def __init__(self, version, name="lavira_m3"):
        self.version = version
        self.subscriptions = []
        if version == "ros1":
            import rospy

            self.ros = rospy
            if not rospy.core.is_initialized():
                rospy.init_node(name, disable_signals=True)
            self.node = None
        elif version == "ros2":
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.node import Node

            self.ros = rclpy
            if not rclpy.ok():
                rclpy.init()
            self.node = Node(name)
            self.executor = MultiThreadedExecutor(num_threads=4)
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.thread.start()
        else:
            raise ValueError("expected ros1 or ros2")

    def publisher(self, kind, topic, durable=False):
        if self.version == "ros1":
            return self.ros.Publisher(topic, kind, queue_size=1, latch=durable)
        from rclpy.qos import DurabilityPolicy, QoSProfile

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL if durable else DurabilityPolicy.VOLATILE,
        )
        return self.node.create_publisher(kind, topic, qos)

    def subscribe(self, kind, topic, callback, durable=False):
        if self.version == "ros1":
            sub = self.ros.Subscriber(topic, kind, callback, queue_size=1)
        else:
            from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data

            qos = (
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                if durable
                else qos_profile_sensor_data
            )
            sub = self.node.create_subscription(kind, topic, callback, qos)
        self.subscriptions.append(sub)
        return sub

    def now_ns(self):
        return (
            self.ros.Time.now().to_nsec()
            if self.version == "ros1"
            else self.node.get_clock().now().nanoseconds
        )

    def stamp(self, ns=None):
        ns = self.now_ns() if ns is None else int(ns)
        if self.version == "ros1":
            return self.ros.Time(ns // 10**9, ns % 10**9)
        from builtin_interfaces.msg import Time

        return Time(sec=ns // 10**9, nanosec=ns % 10**9)

    def close(self):
        if self.version == "ros2":
            self.executor.shutdown(timeout_sec=2)
            self.thread.join(timeout=2)
            self.node.destroy_node()
            self.ros.try_shutdown()
        else:
            self.ros.signal_shutdown("navigation closed")


def stamp_ns(stamp):
    if hasattr(stamp, "to_nsec"):
        return stamp.to_nsec()
    return stamp.sec * 10**9 + stamp.nanosec
