#include <cmath>
#include <geometry_msgs/PoseStamped.h>
#include <livox_ros_driver2/CustomMsg.h>
#include <nav_msgs/Odometry.h>
#include <plan_manage/Bspline.h>
#include <ros/master.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

// Integration probe: synthetic raw Livox + IMU -> unmodified FAST-LIO ->
// EKF -> Fast-Planner. This verifies wiring, not real-world trajectory quality.
int main(int argc, char** argv) {
  ros::init(argc, argv, "raw_pipeline_smoke");
  ros::NodeHandle nh;
  const bool synthetic_input = argc < 2 || std::string(argv[1]) != "--observe";
  auto imu_pub = nh.advertise<sensor_msgs::Imu>("/livox/imu", 100);
  auto lidar_pub = nh.advertise<livox_ros_driver2::CustomMsg>("/livox/lidar", 10);
  auto goal_pub = nh.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1);
  bool have_odom = false, have_cloud = false, have_map = false;
  bool have_spline = false, have_marker = false;
  size_t lio_count = 0, ekf_count = 0;
  ros::Time first_ekf_stamp, last_ekf_stamp;
  nav_msgs::Odometry pose;
  auto lio_sub = nh.subscribe<nav_msgs::Odometry>("/Odometry", 10,
      [&](const nav_msgs::Odometry::ConstPtr&) { ++lio_count; });
  auto odom_sub = nh.subscribe<nav_msgs::Odometry>("/odom_ekf", 100,
      [&](const nav_msgs::Odometry::ConstPtr& msg) {
        pose = *msg;
        have_odom = msg->header.frame_id == "world" &&
                    std::isfinite(msg->pose.pose.position.x) &&
                    std::isfinite(msg->twist.twist.linear.x) &&
                    std::isfinite(msg->twist.twist.linear.y) &&
                    std::isfinite(msg->twist.twist.linear.z);
        if (have_odom) {
          if (ekf_count++ == 0) first_ekf_stamp = msg->header.stamp;
          last_ekf_stamp = msg->header.stamp;
        }
      });
  auto cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_registered", 1,
      [&](const sensor_msgs::PointCloud2::ConstPtr& msg) {
        have_cloud = msg->header.frame_id == "world" && msg->width*msg->height > 0;
      });
  auto map_sub = nh.subscribe<sensor_msgs::PointCloud2>("/sdf_map/occupancy_inflate", 1,
      [&](const sensor_msgs::PointCloud2::ConstPtr& msg) {
        have_map = msg->header.frame_id == "world" && msg->width*msg->height > 0;
      });
  auto spline_sub = nh.subscribe<plan_manage::Bspline>("/planning/bspline", 10,
      [&](const plan_manage::Bspline::ConstPtr& msg) {
        bool valid = msg->order == 3 && msg->pos_pts.size() >= 4 &&
            msg->knots.size() == msg->pos_pts.size()+msg->order+1 &&
            !msg->start_time.isZero();
        for (const auto& p : msg->pos_pts)
          valid = valid && std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        for (size_t i = 0; i < msg->knots.size(); ++i)
          valid = valid && std::isfinite(msg->knots[i]) &&
              (i == 0 || msg->knots[i] >= msg->knots[i-1]);
        if (valid)
          valid = msg->knots[msg->pos_pts.size()] > msg->knots[msg->order];
        have_spline = have_spline || valid;
      });
  auto marker_sub = nh.subscribe<visualization_msgs::Marker>("/planning_vis/trajectory", 10,
      [&](const visualization_msgs::Marker::ConstPtr& msg) {
        have_marker = have_marker || (msg->header.frame_id == "world" &&
            msg->action == visualization_msgs::Marker::ADD && msg->points.size() > 1);
      });

  const auto start = ros::WallTime::now();
  ros::WallTime last_scan = start, last_goal = start;
  ros::WallRate rate(200);
  while (ros::ok() && (ros::WallTime::now()-start).toSec() < 45.0) {
    sensor_msgs::Imu imu;
    imu.header.stamp = ros::Time::now();
    imu.header.frame_id = "livox_frame";
    imu.orientation.w = 1.0;
    // Livox's recorded acceleration is in g; the imported EKF multiplies by 9.81.
    imu.linear_acceleration.z = 1.0;
    if (synthetic_input) imu_pub.publish(imu);
    const auto now = ros::WallTime::now();
    if (synthetic_input && (now-last_scan).toSec() >= 0.1 && lidar_pub.getNumSubscribers() > 0) {
      last_scan = now;
      livox_ros_driver2::CustomMsg scan;
      scan.header.stamp = ros::Time::now()-ros::Duration(0.1);
      scan.header.frame_id = "livox_frame";
      scan.timebase = scan.header.stamp.toNSec();
      // Three distant walls survive the original Mid-360 blind=5 m setting.
      for (int wall = 0; wall < 3; ++wall)
        for (int a = 0; a < 40; ++a)
          for (int b = 0; b < 15; ++b) {
            livox_ros_driver2::CustomPoint point;
            const float across = -3.0f+a*0.15f, height = -0.7f+b*0.2f;
            point.x = wall == 0 ? 5.2f : across;
            point.y = wall == 0 ? across : (wall == 1 ? 5.2f : -5.2f);
            point.z = height;
            point.reflectivity = 100;
            point.line = a%4;
            point.tag = 0x10;
            point.offset_time = static_cast<uint32_t>(scan.points.size()*55000);
            scan.points.push_back(point);
          }
      scan.point_num = scan.points.size();
      lidar_pub.publish(scan);
    }
    ros::spinOnce();
    if (have_odom && have_cloud && have_map && goal_pub.getNumSubscribers() > 0 &&
        (now-last_goal).toSec() > 3.0 && !have_spline) {
      last_goal = now;
      geometry_msgs::PoseStamped goal;
      goal.header.frame_id = "world";
      goal.header.stamp = ros::Time::now();
      goal.pose.position.x = pose.pose.pose.position.x+3.0;
      goal.pose.position.y = pose.pose.pose.position.y;
      goal.pose.position.z = 1.0;
      goal.pose.orientation.w = 1.0;
      goal_pub.publish(goal);
    }
    if (have_odom && have_cloud && have_map && have_spline && have_marker &&
        lio_count > 0 && ekf_count >= 5*lio_count && last_ekf_stamp > first_ekf_stamp) {
      ros::master::V_TopicInfo topics;
      if (!ros::master::getTopics(topics)) return 2;
      for (const auto& topic : topics)
        if (topic.name.find("cmd_vel") != std::string::npos) {
          ROS_ERROR("Unexpected velocity publisher: %s", topic.name.c_str());
          return 3;
        }
      ROS_INFO("PASS (%s input): FAST-LIO -> EKF %.1f Hz (%.1fx LIO message count), finite velocity -> planner map -> cubic Bspline + visualization; no cmd_vel publisher",
               synthetic_input ? "synthetic" : "external raw",
               (ekf_count-1)/(last_ekf_stamp-first_ekf_stamp).toSec(),
               static_cast<double>(ekf_count)/lio_count);
      return 0;
    }
    rate.sleep();
  }
  ROS_ERROR("FAIL: odom=%d cloud=%d map=%d spline=%d marker=%d",
            have_odom, have_cloud, have_map, have_spline, have_marker);
  return 1;
}
