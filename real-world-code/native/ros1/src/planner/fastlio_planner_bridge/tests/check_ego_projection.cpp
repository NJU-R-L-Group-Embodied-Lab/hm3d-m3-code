#include <path_searching/dyn_a_star.h>
#include <pcl_conversions/pcl_conversions.h>
#include <stdexcept>

using Eigen::Vector3d;
void require(bool value, const char *message)
{
  if (!value) throw std::runtime_error(message);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "check_ego_projection", ros::init_options::NoRosout);
  try
  {
    ros::NodeHandle nh("/projection_regression");
    nh.setParam("grid_map/resolution", 0.1);
    nh.setParam("grid_map/local_update_range_x", 2.5);
    nh.setParam("grid_map/local_update_range_y", 2.5);
    nh.setParam("grid_map/local_update_range_z", 2.0);
    nh.setParam("grid_map/obstacles_inflation", 0.1);
    nh.setParam("grid_map/virtual_ground", -1.0);
    nh.setParam("grid_map/virtual_ceil", 4.0);
    nh.setParam("grid_map/enable_virtual_wall", true);
    nh.setParam("grid_map/fading_time", 0.5);
    nh.setParam("planning/enable_height_limit", true);
    nh.setParam("planning/height", 0.25);
    nh.setParam("planning/half_height", 0.05);
    nh.setParam("planning/obstacle_projection_height", 2.0);
    GridMap::Ptr map(new GridMap);
    map->initMap(nh);
    map->occ_timer_.stop();
    map->vis_timer_.stop();
    map->fading_timer_.stop();
    require(map->getInflateOccupancy(Vector3d(0, 0, 0.25)) != 0, "uninitialized projection cannot be free");

    nav_msgs::Odometry::Ptr odom(new nav_msgs::Odometry);
    odom->pose.pose.orientation.w = 1;
    odom->pose.pose.position.z = 0.25;
    map->odomCallback(odom);
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.push_back(pcl::PointXYZ(0.05, 0.05, 1.25));   // Overhead obstacle.
    cloud.push_back(pcl::PointXYZ(1.05, 0.05, 2.35));   // Above the last included voxel.
    cloud.push_back(pcl::PointXYZ(1.55, 0.05, 2.25));   // Exactly two metres above the plane.
    cloud.push_back(pcl::PointXYZ(-1.05, 0.05, -0.25)); // Lower floor, not projected upward.
    cloud.push_back(pcl::PointXYZ(-0.55, 0.55, 0.15));  // Original near-plane collision.
    auto insert = [&]()
    {
      sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2);
      pcl::toROSMsg(cloud, *msg);
      map->cloudCallback(msg);
    };
    insert();
    const Vector3d overhead(0.05, 0.05, 0.25);
    require(map->getOccupancy(overhead) == 0 && map->getOccupancy(Vector3d(0.05, 0.05, 1.25)) == 1,
            "projection must preserve the raw 3D occupancy map");
    require(map->getInflateOccupancy(overhead) != 0, "overhead obstacle must block the planning plane");
    require(map->getInflateOccupancy(Vector3d(0.15, 0.05, 0.25)) != 0 &&
            map->getInflateOccupancy(Vector3d(0.25, 0.05, 0.25)) == 0,
            "projection must use the original XY inflation distance");
    require(map->getInflateOccupancy(Vector3d(1.05, 0.05, 0.25)) == 0 &&
            map->getInflateOccupancy(Vector3d(1.55, 0.05, 0.25)) != 0,
            "height interval includes its top voxel without additional Z inflation");
    require(map->getInflateOccupancy(Vector3d(-1.05, 0.05, 0.25)) == 0 &&
            map->getInflateOccupancy(Vector3d(-0.55, 0.55, 0.25)) != 0,
            "floor must not be projected upward; existing lower collisions must remain");
    map->mp_.obstacle_projection_height_ = 0;
    require(map->getInflateOccupancy(overhead) == 0, "zero height restores the original plane query");
    map->mp_.obstacle_projection_height_ = 2;
    map->mp_.enable_height_limit_ = false;
    require(map->getInflateOccupancy(overhead) == 0 &&
            map->getInflateOccupancy(Vector3d(0.05, 0.05, 1.25)) != 0, "3D mode remains unchanged");
    map->mp_.enable_height_limit_ = true;
    const int saved_upper_z = map->md_.ringbuffer_upbound3i_.z();
    map->md_.ringbuffer_upbound3i_.z() = 10;
    require(map->getInflateOccupancy(Vector3d(0.55, 1.05, 0.25)) != 0,
            "incomplete height coverage cannot alias a free column");
    map->md_.ringbuffer_upbound3i_.z() = saved_upper_z;
    require(map->getInflateOccupancy(Vector3d(2.45, 1.05, 0.25)) == 0,
            "partial XY footprint must preserve the original unknown-space policy without a border wall");
    require(map->getInflateOccupancy(Vector3d(60, 0, 0.25)) != 0 &&
            map->getInflateOccupancy(Vector3d(0, 0, 0.5)) != 0, "world and planning bounds remain enforced");

    pcl::PointCloud<pcl::PointXYZ> displayed;
    bool received = false;
    auto sub = nh.subscribe<sensor_msgs::PointCloud2>("grid_map/occupancy_inflate", 1,
        [&](const sensor_msgs::PointCloud2::ConstPtr &msg)
        {
          pcl::fromROSMsg(*msg, displayed);
          received = true;
        });
    const ros::WallTime deadline = ros::WallTime::now() + ros::WallDuration(2);
    while (!received && ros::WallTime::now() < deadline)
    {
      map->publishMapInflate();
      ros::spinOnce();
      ros::WallDuration(0.01).sleep();
    }
    require(received && !displayed.empty(), "RViz projection topic must publish");
    bool saw_overhead = false;
    for (const auto &p : displayed)
    {
      require(std::abs(p.z - 0.25) < 1e-6 && map->getInflateOccupancy(Vector3d(p.x, p.y, p.z)) != 0,
              "every displayed cell must match the planning-plane collision query");
      saw_overhead |= std::abs(p.x - 0.05) < 1e-5 && std::abs(p.y - 0.05) < 1e-5;
    }
    require(saw_overhead, "RViz must include the overhead obstacle projected to the plane");
    std::cout << "PASS: 2 m height column, XY inflation, floor exclusion, 3D preservation, bounds and RViz consistency\n";

    // A wall that exists only overhead must divert the real planar A* search.
    for (int y = -5; y < 5; ++y)
      cloud.push_back(pcl::PointXYZ(0.05, (y + 0.5) * 0.1, 1.25));
    insert();
    AStar search;
    search.initGridMap(map, Eigen::Vector3i(80, 80, 3), true);
    require(search.AstarSearch(0.1, Vector3d(-1, 0, 0.25), Vector3d(1, 0, 0.25)) == ASTAR_RET::SUCCESS,
            "A* must find a planar route around an overhead-only wall");
    double lateral = 0;
    for (const auto &p : search.getPath())
    {
      require(map->getInflateOccupancy(p) == 0 && std::abs(p.z() - 0.25) < 1e-9,
              "A* route must remain on the free planning plane");
      lateral = std::max(lateral, std::abs(p.y()));
    }
    require(lateral >= 0.59, "A* must detour, not pass underneath the obstacle");
    search.require_local_map_ = false;
    require(search.AstarSearch(0.1, Vector3d(1.8, 1.5, 0.25), Vector3d(3.0, 1.5, 0.25)) == ASTAR_RET::SUCCESS,
            "normal A* must retain its existing ability to cross the rolling-map boundary toward a distant target");
    const ros::WallTime begin = ros::WallTime::now();
    int occupied = 0;
    for (int i = 0; i < 20000; ++i)
      occupied += map->getInflateOccupancy(Vector3d(0.35 + (i % 10) * 0.1, 1.05, 0.25)) != 0;
    require(occupied == 0, "benchmark columns must be free");
    std::cout << "PASS: real A* overhead-wall avoidance; 20000 column queries in "
              << (ros::WallTime::now() - begin).toSec() * 1000 << " ms\n";

    map->fadingCallback(ros::TimerEvent());
    map->fadingCallback(ros::TimerEvent());
    require(map->getInflateOccupancy(overhead) == 0, "faded 3D obstacles must disappear from projection too");
    insert();
    map->md_.camera_pos_.x() = 0.4;
    map->moveRingBuffer();
    require(map->getInflateOccupancy(overhead) != 0, "buffer movement must preserve retained obstacles");
    map->md_.camera_pos_.x() += map->md_.ringbuffer_size3i_.x() * map->mp_.resolution_;
    map->moveRingBuffer();
    require(map->getInflateOccupancy(Vector3d(5.05, 0.05, 0.25)) == 0,
            "ring-buffer wrap must not alias an old obstacle column");
    for (double invalid_height : {-1.0, 4.0})
    {
      nh.setParam("planning/obstacle_projection_height", invalid_height);
      bool rejected = false;
      try { GridMap invalid; invalid.initMap(nh); }
      catch (const std::invalid_argument &) { rejected = true; }
      require(rejected, "negative or out-of-map projection height must be rejected");
    }
    std::cout << "PASS: fading, moving ring buffer, stale-column rejection and invalid configuration\n";
    ros::shutdown();
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
