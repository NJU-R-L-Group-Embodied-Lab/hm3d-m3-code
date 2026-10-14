#include <mpc.h>
#include <stdexcept>

void require(bool value, const char *message)
{
  if (!value) throw std::runtime_error(message);
}

void checkYawBranch()
{
  MPC controller;
  controller.node_=std::make_shared<rclcpp::Node>("mpc_contract_test");
  controller.active_task_=1;
  controller.T = 100;
  controller.dt = 0.02;
  controller.control_a = false;
  controller.max_speed = 0.5;
  controller.max_omega = 1.0;
  controller.max_cv = 0.02;
  controller.max_comega = 0.04;
  controller.Q = {0, 0, 2.5, 2};
  controller.R = {2, 2};
  controller.Rd = {0.5, 0.3};
  controller.xref = Eigen::MatrixXd::Zero(4, 500);
  controller.dref = controller.output = Eigen::MatrixXd::Zero(2, 500);
  const double rad = M_PI / 180.0;
  auto unwrap = [&](double heading, double reference, double rate)
  {
    controller.now_state.theta = heading * rad;
    controller.now_state.yaw_rate = rate;
    controller.xref.row(3).setConstant(reference * rad);
    controller.smooth_yaw();
    return (controller.xref(3, 0) - controller.now_state.theta) / rad;
  };
  require(std::abs(unwrap(179, -179, 0) - 2) < 1e-9,
          "absolute +/-pi crossing must remain a two-degree error");
  require(std::abs(unwrap(0, 179.5, 0.6) - 179.5) < 1e-9, "initialize left turn");
  for (int i = 0; i < 10; ++i)
    require(std::abs(unwrap(0, i % 2 ? 179.5 : -179.5, i % 2 ? 0.01 : -0.01) -
                         (i % 2 ? 179.5 : 180.5)) < 1e-9,
            "small reference/gyro sign changes must retain the chosen branch");

  traj_utils::msg::PolyTraj::SharedPtr replacement(new traj_utils::msg::PolyTraj);
  replacement->header.frame_id="map";replacement->header.stamp=controller.node_->now();
  replacement->start_time=controller.node_->now();replacement->task_id=1;replacement->execute=true;
  replacement->order = 5;
  replacement->traj_id = 42;
  replacement->duration = {1.0};
  replacement->coef_x.assign(6, 0.0);
  replacement->coef_y.assign(6, 0.0);
  replacement->coef_z.assign(6, 0.0);
  controller.polyTrajCallback(replacement);
  require(controller.receive_traj_,"valid identified trajectory rejected");
  auto bad=std::make_shared<traj_utils::msg::PolyTraj>(*replacement);bad->duration={1e20};bad->traj_id=43;
  controller.polyTrajCallback(bad);require(controller.traj_id_==42,"unbounded duration accepted");
  bad->duration={1};bad->traj_id=41;controller.polyTrajCallback(bad);require(controller.traj_id_==42,"old trajectory accepted");
  require(controller.yaw_turn_direction_ == 1 && unwrap(0, -179.5, -0.1) > 180,
          "trajectory replacement must retain branch memory");
  require(std::abs(unwrap(0, -150, 0.6) + 150) < 1e-9 && controller.yaw_turn_direction_ == 0,
          "outside the ambiguity region the shortest turn must be available");
  require(std::abs(unwrap(0, 179.5, -0.6) + 180.5) < 1e-9 &&
          std::abs(unwrap(0, -179.5, 0) + 179.5) < 1e-9,
          "measured right turn must select the mirrored branch");
  unwrap(0, 0, 0);
  require(std::abs(unwrap(179, 0, 0.6) - 181) < 1e-9 &&
          std::abs(unwrap(-179, 0, 0.6) - 179) < 1e-9,
          "branch must survive the odometry yaw crossing +/-pi too");
  controller.now_state.theta = 0;
  for (int i = 0; i < controller.T; ++i)
  {
    double angle = (179.0 + 0.04 * i) * rad;
    controller.normlize_theta(angle);
    controller.xref(3, i) = angle;
  }
  controller.smooth_yaw();
  for (int i = 0; i < controller.T; ++i)
    require(std::abs(controller.xref(3, i) - (179.0 + 0.04 * i) * rad) < 1e-9,
            "the complete prediction horizon must use the same continuous branch");
  controller.allow_reverse = true;
  controller.xref.row(2).setConstant(0.2);
  require(std::abs(unwrap(0, 179, 0.6) + 1) < 1e-9 &&
          controller.xref(2, 0) < 0 && controller.yaw_turn_direction_ == 0,
          "legacy reverse mode must retain its original behavior");
  controller.allow_reverse = false;
  controller.xref.row(2).setZero();
  auto command = [&](double reference, double rate)
  {
    unwrap(0, reference, rate);
    controller.output.setZero();
    controller.predictMotion();
    controller.solveMPCV();
    require(controller.output.row(1).head(controller.T).allFinite() &&
            controller.output.row(1).head(controller.T).cwiseAbs().maxCoeff() <= 1.00001,
            "branch selection must preserve finite commands and angular speed cap");
    return controller.output(1, 0);
  };
  require(command(179.5, 0.6) > 0.1 && command(-179.5, 0.6) > 0.1,
          "actual MPC must keep turning left across the reference branch cut");
  require(command(-150, 0.6) < -0.1,
          "actual MPC must still turn right when the reference clearly requires it");
  require(command(179.5, -0.6) < -0.1 && command(-179.5, -0.6) < -0.1,
          "actual MPC right-turn behavior must mirror left-turn behavior");
  controller.max_speed=-1.;controller.solveMPCV();
  require(!controller.solve_ok_,"infeasible QP must not reuse previous command as a new solution");
  controller.emergencyStopCallback(std_msgs::msg::Empty());
  require(controller.yaw_turn_direction_ == 0, "emergency stop must clear branch memory");
  std::cout << "PASS: yaw branch crossings, jitter, replans, odometry wrap, horizon, release, reverse mode and actual MPC commands\n";
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try
  {
    MPC controller;
    controller.node_=std::make_shared<rclcpp::Node>("mpc_optimizer_test");
    checkYawBranch();
    // Call the real velocity optimizer directly; do not start publishers/timers.
    controller.control_a = false;
    controller.T = 100;
    controller.dt = 0.02;
    controller.max_speed = 0.5;
    controller.max_omega = 1.0;
    controller.max_cv = 0.02;
    controller.max_comega = 0.02;
    controller.xref = Eigen::MatrixXd::Zero(4, 500);
    controller.dref = controller.output = Eigen::MatrixXd::Zero(2, 500);
    // Isolate the speed cost so a position error cannot hide a zero reference.
    controller.Q[0] = controller.Q[1] = 0;
    auto solve = [&](double reference_speed, int delay)
    {
      controller.delay_num = delay;
      controller.output_buff.assign(delay, Eigen::Vector2d::Zero());
      controller.output.setZero();
      controller.xref.row(2).setConstant(reference_speed);
      controller.predictMotion();
      controller.solveMPCV();
      const auto speeds = controller.output.row(0).segment(delay, controller.T - delay);
      require(speeds.allFinite() && speeds.minCoeff() >= -1e-5 && speeds.maxCoeff() <= 0.50001,
              "velocity solution must respect forward-only and speed limits");
      return controller.output(0, delay);
    };
    const double moving = solve(0.4, 0);
    require(moving > 0.38 && moving < 0.41, "nonzero trajectory speed must drive motion without position error");
    require(solve(1.0, 0) > 0.49, "catch-up speed capped at 0.5 m/s");
    require(std::abs(solve(0.0, 0)) < 1e-5, "zero reference must stop");
    require(std::abs(solve(-0.4, 0)) < 1e-5, "forward-only constraint preserved");
    require(solve(0.4, 2) > 0.38, "delay mode must use trajectory speed too");
    controller.xref.setZero();
    controller.Q[3] = 0;
    auto turn = [&](double measured_rate, int delay, double queued_rate, double yaw_reference)
    {
      controller.delay_num = delay;
      controller.now_state.yaw_rate = measured_rate;
      controller.output_buff.assign(delay, Eigen::Vector2d(0, queued_rate));
      controller.output.setZero();
      controller.xref.row(3).setConstant(yaw_reference);
      controller.predictMotion();
      controller.solveMPCV();
      const auto rates = controller.output.row(1).segment(delay, controller.T - delay);
      require(rates.allFinite() && rates.cwiseAbs().maxCoeff() <= 1.00001, "angular speed cap preserved");
      for (int k = delay + 1; k < controller.T; ++k)
        require(std::abs(controller.output(1, k) - controller.output(1, k - 1)) <= controller.max_comega + 1e-5,
                "existing inter-step angular acceleration limit preserved");
      return controller.output(1, delay);
    };
    const double left = turn(0.6, 0, 0, 0), right = turn(-0.6, 0, 0, 0);
    require(left > 0.01 && right < -0.01 && std::abs(left + right) < 1e-5,
            "first control soft cost must preserve current turn preference symmetrically");
    require(turn(0.6, 2, -0.4, 0) < -0.01, "delay mode must join the last queued command");
    controller.Q[3] = 20;
    require(turn(0.6, 0, 0, -M_PI) < 0.6 - controller.max_comega - 1e-3,
            "new first-step continuity must be soft, not a new hard bound");
    nav_msgs::msg::Odometry::SharedPtr odom(new nav_msgs::msg::Odometry);
    odom->header.frame_id = "map";
    odom->child_frame_id = "base_link";
    odom->header.stamp = controller.node_->now();
    odom->pose.pose.orientation.w = 1;
    odom->twist.twist.angular.z = -0.4;
    controller.rcvOdomCallBack(odom);
    require(controller.has_odom, "MPC must accept odometry in map/base_link frames");
    require(std::abs(controller.now_state.yaw_rate + 0.4) < 1e-9, "MPC must consume measured angular velocity");
    auto wrong_frame = std::make_shared<nav_msgs::msg::Odometry>(*odom);
    wrong_frame->child_frame_id = "sensor_frame";
    wrong_frame->twist.twist.angular.z = 0.8;
    controller.rcvOdomCallBack(wrong_frame);
    require(!controller.has_odom && std::abs(controller.now_state.yaw_rate + 0.4) < 1e-9,
            "MPC must reject odometry outside the canonical body frame");
    controller.control_a = true;
    controller.delay_num = 0;
    controller.now_state.yaw_rate = 0.6;
    controller.Q[3] = 0;
    controller.xref.setZero();
    controller.output.setZero();
    controller.predictMotion();
    controller.solveMPCA();
    require(controller.output(1, 0) > 0.01, "acceleration mode shares the soft angular continuity cost");
    std::cout << "PASS: soft first-command turn continuity, reversible preference, delay and unchanged hard limits\n";
    std::cout << "PASS: actual MPC velocity optimizer, v_ref=0.4 -> v=" << moving
              << ", speed cap 0.5, zero/negative references and delay indexing\n";
    rclcpp::shutdown();
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
