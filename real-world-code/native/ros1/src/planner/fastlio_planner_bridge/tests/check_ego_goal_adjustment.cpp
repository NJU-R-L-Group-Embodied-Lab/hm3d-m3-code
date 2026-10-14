#include <plan_manage/goal_adjustment.h>
#include <plan_manage/ego_replan_fsm.h>
#include <pcl_conversions/pcl_conversions.h>
#include <limits>
#include <stdexcept>

using Eigen::Vector3d;
void require(bool value, const char *message)
{
  if (!value) throw std::runtime_error(message);
}

void checkGeometry()
{
  using namespace ego_planner;
  const auto free = [](const Vector3d &p)
  {
    require(p.z() == 0, "clearance must stay in XY, not hit virtual slab walls");
    return !(p.x() >= 0 && p.x() < 0.1 && p.y() >= 0 && p.y() < 0.1);
  };
  require(!hasGoalClearanceXY(Vector3d(0.15, 0.05, 0), 0.1, 0.1, free), "free voxel too close to inflated obstacle");
  require(hasGoalClearanceXY(Vector3d(0.25, 0.05, 0), 0.1, 0.1, free), "clearance measured from voxel face");
  require(!hasGoalClearanceXY(Vector3d::Zero(), 0, 0.1, free), "invalid resolution");
  const Vector3d start(-2, 0, 0), goal(0, 0, 0), lower(-3, -3, -1), upper(3, 3, 1);
  auto candidates = goalCandidatesXY(start, goal, lower, upper, 0.1,
      [](const Vector3d &p) { return p.y() > 0.5 && std::abs(p.x()) < 0.2; });
  require(!candidates.empty() && candidates.front().y() > 0.5, "must find lateral exit, not a single retreat ray");
  for (size_t i = 1; i < candidates.size(); ++i)
    require(candidates[i].squaredNorm() + 1e-12 >= candidates[i - 1].squaredNorm(), "nearest goal first");
  candidates = goalCandidatesXY(start, goal, lower, upper, 0.1,
      [](const Vector3d &p) { return p.x() > 2; });
  require(!candidates.empty() && candidates.front().norm() > 1, "no one-meter correction cap");
  require(goalCandidatesXY(start, goal, lower, upper, 0.1, [](const Vector3d &) { return false; }).empty(), "no safe cells");
  Vector3d invalid = goal;
  invalid.x() = std::numeric_limits<double>::quiet_NaN();
  require(goalCandidatesXY(start, invalid, lower, upper, 0.1, free).empty(), "reject non-finite goal");
  std::cout << "PASS: XY voxel-face clearance, lateral candidates, distance ordering, no radius cap and invalid input\n";
}

Vector3d endpoint(const traj_utils::PolyTraj &msg)
{
  require(!msg.duration.empty(), "trajectory must have pieces");
  Vector3d p = Vector3d::Zero();
  const size_t i = msg.duration.size() - 1;
  for (size_t j = 0; j < 6; ++j)
    p = p * msg.duration[i] + Vector3d(msg.coef_x[6 * i + j], msg.coef_y[6 * i + j], msg.coef_z[6 * i + j]);
  return p;
}

void checkRuntime(int argc, char **argv)
{
  ros::init(argc, argv, "check_ego_goal_adjustment");
  ros::NodeHandle nh("/drone_0_ego_planner_node");
  ego_planner::EGOReplanFSM fsm;
  fsm.init(nh);
  Vector3d accepted = Vector3d::Constant(99);
  int accepted_count = 0;
  traj_utils::PolyTraj last_traj;
  auto marker_sub = nh.subscribe<visualization_msgs::Marker>("goal_point", 10,
      [&](const visualization_msgs::Marker::ConstPtr &msg)
      {
        if (msg->id != 0) return;
        accepted = Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        ++accepted_count;
      });
  auto trajectory_sub = nh.subscribe<traj_utils::PolyTraj>("planning/trajectory", 10,
      [&](const traj_utils::PolyTraj::ConstPtr &msg) { last_traj = *msg; });
  auto odom_pub = nh.advertise<nav_msgs::Odometry>("odom_world", 1);
  auto map_odom_pub = nh.advertise<nav_msgs::Odometry>("grid_map/odom", 1);
  auto cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("grid_map/cloud", 1);
  auto goal_pub = nh.advertise<quadrotor_msgs::GoalSet>("/goal", 1);
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.push_back(pcl::PointXYZ(4, 4, 1));
  auto pump = [&](double seconds)
  {
    const ros::WallTime end = ros::WallTime::now() + ros::WallDuration(seconds);
    do
    {
      nav_msgs::Odometry odom;
      odom.header.stamp = ros::Time::now();
      odom.header.frame_id = "world";
      odom.pose.pose.orientation.w = 1;
      odom.pose.pose.position.z = 0.3;
      odom_pub.publish(odom);
      map_odom_pub.publish(odom);
      sensor_msgs::PointCloud2 msg;
      pcl::toROSMsg(cloud, msg);
      msg.header = odom.header;
      cloud_pub.publish(msg);
      ros::spinOnce();
      ros::WallDuration(0.01).sleep();
    } while (ros::ok() && ros::WallTime::now() < end);
    ros::spinOnce();
  };
  auto goal = [&](double x, double y)
  {
    quadrotor_msgs::GoalSet msg;
    msg.drone_id = 0;
    msg.goal = {{float(x), float(y), 0.0f}};
    goal_pub.publish(msg);
    pump(0.4);
  };
  // A wide, shallow obstacle: the nearest safe exits are sideways from the
  // robot-to-goal line. Points are inflated by the real GridMap implementation.
  for (int x = 18; x <= 22; ++x)
    for (int y = -1; y <= 0; ++y)
      cloud.push_back(pcl::PointXYZ((x + 0.5) * 0.1, (y + 0.5) * 0.1, 0.05));
  pump(0.5);
  const int before = accepted_count;
  goal(2.05, 0.05);
  require(accepted_count > before && std::abs(accepted.y() - 0.05) > 0.15, "first occupied goal needs lateral correction without a previous trajectory");
  require((endpoint(last_traj) - accepted).norm() < 1e-3, "publish trajectory to accepted candidate");
  for (size_t i = 0; i < last_traj.duration.size(); ++i)
  {
    require(std::isfinite(last_traj.duration[i]) && last_traj.duration[i] > 0, "finite positive durations");
    for (int k = 0; k <= 100; ++k)
    {
      const double t = last_traj.duration[i] * k / 100.0;
      Vector3d p = Vector3d::Zero();
      for (int j = 0; j < 6; ++j)
        p = p * t + Vector3d(last_traj.coef_x[6 * i + j], last_traj.coef_y[6 * i + j], last_traj.coef_z[6 * i + j]);
      require(p.allFinite() && std::abs(p.z()) < 0.05, "finite trajectory in thin layer");
      require(p.x() < 1.7 || p.x() >= 2.4 || p.y() < -0.2 || p.y() >= 0.2, "trajectory outside inflated obstacle");
    }
  }
  goal(1, 0);
  require((accepted - Vector3d(1, 0, 0)).norm() < 1e-6, "free goal should be unchanged");
  goal(2.05, 0.05);
  require(std::abs(accepted.y() - 0.05) > 0.15 &&
          (endpoint(last_traj) - accepted).norm() < 1e-3, "replace existing task with a verified corrected goal");
  const Vector3d previous = accepted;
  goal(60, 0);
  require((accepted - previous).norm() < 1e-9, "invalid new goal preserves previous goal");
  goal(20, 0);
  require((accepted - Vector3d(20, 0, 0)).norm() < 1e-6, "out-of-cache goal remains deferred");
  goal(0, 0);
  require(endpoint(last_traj).norm() < 1e-6, "goal at robot produces a valid stop trajectory");
  goal(2.05, 0.05);
  const Vector3d last_valid = accepted;
  // Occupy the robot so the next correction has no usable start.
  cloud.push_back(pcl::PointXYZ(0.05, 0.05, 0.05));
  pump(0.1);
  goal(2.05, 0.05);
  require((accepted - last_valid).norm() < 1e-9, "failed correction preserves accepted goal");
  cloud.push_back(pcl::PointXYZ(last_valid.x(), last_valid.y(), 0.05));
  pump(0.3);
  require(endpoint(last_traj).norm() < 1e-6, "invalid active goal should stop");
  std::cout << "PASS: real inflated map, first/existing goal correction, verified trajectory, rejection/defer and active-goal stop\n";
  ros::shutdown();
}

int main(int argc, char **argv)
{
  try
  {
    checkGeometry();
    if (argc <= 1 || std::string(argv[1]) != "--self-test") checkRuntime(argc, argv);
    return 0;
  }
  catch (const std::exception &e)
  {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
