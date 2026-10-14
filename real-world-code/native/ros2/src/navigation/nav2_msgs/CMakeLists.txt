cmake_minimum_required(VERSION 3.16)
project(nav2_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ComputePathToPose.action" "action/NavigateToPose.action"
  DEPENDENCIES geometry_msgs nav_msgs builtin_interfaces)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
