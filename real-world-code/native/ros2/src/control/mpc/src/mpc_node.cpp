#include "mpc.h"

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("mpc_node", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  MPC controller;
  controller.init(node);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
