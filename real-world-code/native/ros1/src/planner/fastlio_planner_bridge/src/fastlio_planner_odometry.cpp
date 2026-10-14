#include <cmath>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>

// Validate the imported EKF output and share it with planning and future control.
// Its linear velocity is already estimated in world coordinates; do not differentiate.
int main(int argc, char** argv) {
  ros::init(argc, argv, "fastlio_planner_odometry");
  ros::NodeHandle nh("~");
  auto publisher = nh.advertise<nav_msgs::Odometry>("odom", 10);
  auto subscriber = nh.subscribe<nav_msgs::Odometry>(
      "input", 10, [&](const nav_msgs::Odometry::ConstPtr& input) {
        const auto& p = input->pose.pose.position;
        const auto& q = input->pose.pose.orientation;
        const auto& v = input->twist.twist.linear;
        const auto& w = input->twist.twist.angular;
        if (input->header.frame_id != "world" || input->header.stamp.isZero() ||
            !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
            !std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z) ||
            !std::isfinite(q.w) || q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w < 0.5 ||
            !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z) ||
            !std::isfinite(w.x) || !std::isfinite(w.y) || !std::isfinite(w.z)) {
          ROS_WARN_THROTTLE(2.0, "Rejecting invalid EKF world odometry");
          return;
        }
        nav_msgs::Odometry output = *input;
        // With identity Rr_i/tr_i and zero imu_trans, the EKF pose is the same
        // IMU origin that FAST-LIO calls body. Do not apply LiDAR extrinsics again.
        output.child_frame_id = "body";
        publisher.publish(output);
      });
  ros::spin();
}
