#include <iostream>
#include <limits>
#include <plan_env/grid_map.h>

int main() {
  MappingParameters map;
  bool passed = map.isInMap(Eigen::Vector3d::Zero());
  for (int axis = 0; axis < 3; ++axis) {
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    p(axis) = map.map_min_boundary_(axis);
    passed &= !map.isInMap(p);
    p(axis) += 1e-5;
    passed &= map.isInMap(p);
    p(axis) = map.map_max_boundary_(axis);
    passed &= !map.isInMap(p);
    p(axis) -= 1e-5;
    passed &= map.isInMap(p);
    p(axis) = std::numeric_limits<double>::quiet_NaN();
    passed &= !map.isInMap(p);
    p(axis) = std::numeric_limits<double>::infinity();
    passed &= !map.isInMap(p);
  }
  map.enable_height_limit_ = true;
  passed &= map.isInPlanningBounds(Eigen::Vector3d::Zero());
  for (double z : {-0.05, 0.05, -0.2, 0.2, 1.0}) {
    const Eigen::Vector3d p(0, 0, z);
    passed &= map.isInMap(p) && !map.isInPlanningBounds(p);
  }
  for (double z : {-0.04999, 0.04999})
    passed &= map.isInPlanningBounds(Eigen::Vector3d(0, 0, z));
  passed &= !map.isInPlanningBounds(Eigen::Vector3d(50, 0, 0));
  passed &= !map.isInPlanningBounds(Eigen::Vector3d(0, 0, std::numeric_limits<double>::quiet_NaN()));
  passed &= !map.isInPlanningBounds(Eigen::Vector3d(0, 0, std::numeric_limits<double>::infinity()));
  map.planning_height_ = 1.0;
  map.planning_half_height_ = 0.1;
  passed &= map.isInPlanningBounds(Eigen::Vector3d(0, 0, 1));
  passed &= !map.isInPlanningBounds(Eigen::Vector3d::Zero());
  map.enable_height_limit_ = false;
  passed &= map.isInPlanningBounds(Eigen::Vector3d::Zero());
  passed &= map.isInPlanningBounds(Eigen::Vector3d(0, 0, 2));
  // Exercise a shifted/custom extent as well as the launch defaults.
  map.map_min_boundary_ = Eigen::Vector3d(10, 20, 30);
  map.map_max_boundary_ = Eigen::Vector3d(11, 21, 31);
  passed &= map.isInMap(Eigen::Vector3d(10.5, 20.5, 30.5));
  passed &= !map.isInMap(Eigen::Vector3d::Zero());
  std::cout << (passed ? "PASS" : "FAIL") << ": world bounds, independent planning slab, disabled/custom height and invalid coordinates\n";
  return passed ? 0 : 1;
}
