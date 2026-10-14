cmake_minimum_required(VERSION 3.0.2)
project(mpc)

find_package(catkin REQUIRED COMPONENTS roscpp nav_msgs std_msgs geometry_msgs visualization_msgs traj_utils message_generation)
find_package(Eigen3 REQUIRED)

# Keep dependency options scoped to MPC; catkin's other libraries stay shared.
function(add_mpc_solver)
  set(solver_sources "${CMAKE_CURRENT_SOURCE_DIR}/../../../deps")
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_TESTING OFF)
  set(OSQP_BUILD_SHARED_LIB OFF)
  set(OSQP_BUILD_STATIC_LIB ON)
  set(OSQP_BUILD_DEMO_EXE OFF)
  set(OSQP_BUILD_UNITTESTS OFF)
  set(FETCHCONTENT_SOURCE_DIR_QDLDL "${solver_sources}/qdldl")
  set(FETCHCONTENT_FULLY_DISCONNECTED ON)
  add_subdirectory("${solver_sources}/osqp" "${CMAKE_CURRENT_BINARY_DIR}/solver/osqp" EXCLUDE_FROM_ALL)
  add_library(osqp::osqpstatic ALIAS osqpstatic)
  # The vendored OSQP has the final v1 API; try_compile cannot link build targets.
  set(OSQP_IS_V1 TRUE)
  set(OSQP_IS_V1_FINAL TRUE)
  add_subdirectory("${solver_sources}/osqp-eigen" "${CMAKE_CURRENT_BINARY_DIR}/solver/osqp-eigen" EXCLUDE_FROM_ALL)
endfunction()
add_mpc_solver()

add_message_files(FILES Polynome.msg)
generate_messages(DEPENDENCIES geometry_msgs)
catkin_package(CATKIN_DEPENDS roscpp nav_msgs std_msgs geometry_msgs visualization_msgs traj_utils message_runtime)

add_executable(mpc_controller_node src/mpc_node.cpp src/mpc.cpp src/cubic_spline_planner.cpp)
target_compile_features(mpc_controller_node PRIVATE cxx_std_14)
target_include_directories(mpc_controller_node PRIVATE include ${catkin_INCLUDE_DIRS})
target_link_libraries(mpc_controller_node ${catkin_LIBRARIES} Eigen3::Eigen OsqpEigen::OsqpEigen)
add_dependencies(mpc_controller_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(check_mpc_tracking EXCLUDE_FROM_ALL
  ../../planner/fastlio_planner_bridge/tests/check_mpc_tracking.cpp
  src/mpc.cpp src/cubic_spline_planner.cpp)
# Exercise the real optimizer without starting command publishers or timers.
target_compile_options(check_mpc_tracking PRIVATE -fno-access-control)
target_compile_features(check_mpc_tracking PRIVATE cxx_std_14)
target_include_directories(check_mpc_tracking PRIVATE include ${catkin_INCLUDE_DIRS})
target_link_libraries(check_mpc_tracking ${catkin_LIBRARIES} Eigen3::Eigen OsqpEigen::OsqpEigen)
add_dependencies(check_mpc_tracking ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

if(CATKIN_ENABLE_TESTING)
  add_executable(check_mpc_solver EXCLUDE_FROM_ALL ../../planner/fastlio_planner_bridge/tests/check_mpc_solver.cpp)
  target_link_libraries(check_mpc_solver Eigen3::Eigen OsqpEigen::OsqpEigen)
endif()
install(TARGETS mpc_controller_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
