#include <plan_manage/ego_replan_fsm.h>
#include <optimizer/planar_motion_cost.h>
#include <limits>
#include <stdexcept>

void require(bool value, const char *message)
{
  if (!value) throw std::runtime_error(message);
}

void checkErrorTrigger()
{
  ego_planner::TrackingErrorTrigger trigger;
  require(!trigger.update(0.3, 0.0), "threshold alone must not trigger");
  require(!trigger.update(0.4, 0.1), "spike must wait for confirmation");
  require(!trigger.update(0.1, 0.2), "small error cancels confirmation");
  require(!trigger.update(0.4, 0.3), "new episode needs full confirmation");
  require(!trigger.update(0.4, 0.49), "must wait 0.20 seconds");
  require(trigger.update(0.4, 0.51), "persistent error must trigger");
  require(!trigger.update(0.4, 0.6), "recovery cooldown");
  require(!trigger.update(0.4, 1.50), "recovery interval is at least one second");
  require(trigger.update(0.4, 1.52), "persistent error retriggers after cooldown");
  trigger.reset(2.0);
  require(!trigger.update(0.4, 2.1) && !trigger.update(0.4, 2.9), "successful recovery starts cooldown");
  require(trigger.update(0.4, 3.01), "success does not suppress errors forever");
  require(!trigger.update(0.4, 1.0), "clock rollback resets timers");
  require(!trigger.update(0.4, 1.21), "rollback preserves cooldown");
  require(trigger.update(0.4, 2.01), "clock rollback recovers");
  trigger.reset(4.0);
  require(!trigger.update(0.4, 5.0), "start confirmation");
  require(!trigger.update(std::numeric_limits<double>::quiet_NaN(), 5.1), "invalid sample cancels confirmation");
  require(!trigger.update(0.4, 5.21), "invalid sample must not confirm");
  bool rejected = false;
  try { ego_planner::TrackingErrorTrigger invalid(0.0); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "invalid thresholds must be rejected");
  std::cout << "PASS: error threshold, continuous confirmation, cooldown, reset, clock rollback and invalid input\n";
}

void checkPlanarMotionCosts()
{
  using Eigen::Vector3d;
  using namespace ego_planner;
  const Eigen::Quaterniond q = Eigen::AngleAxisd(0.7, Vector3d::UnitZ()) *
      Eigen::AngleAxisd(0.3, Vector3d::UnitY()) * Eigen::AngleAxisd(0.2, Vector3d::UnitX());
  const double expected = (-0.2 * std::sin(0.2) + 0.6 * std::cos(0.2)) / std::cos(0.3);
  require(std::abs(traj_utils::planarYawRate(q, Vector3d(0.1, -0.2, 0.6)) - expected) < 1e-10,
          "convert body angular velocity to yaw rate with roll/pitch");
  require(!std::isfinite(traj_utils::planarYawRate(Eigen::Quaterniond(0, 0, 0, 0), Vector3d::Zero())),
          "reject invalid orientation");
  Vector3d gv, ga;
  double gt;
  require(headingAlignmentCost(Vector3d(0, 0.4, 0), M_PI / 2, 0, 0, 1, gv, gt) < 1e-12,
          "world Y motion is forward when yaw is pi/2");
  require(headingAlignmentCost(Vector3d(-0.4, 0, 0), 0, 0, 0, 1, gv, gt) > 0.1,
          "backward motion receives a soft penalty");
  for (const Vector3d velocity : {Vector3d(0.3, 0.1, 0), Vector3d(-0.2, 0.1, 0), Vector3d::Zero().eval()})
  {
    const Vector3d acceleration(0.1, 0.2, 0);
    const double time = 0.4, eps = 1e-6;
    auto cost = [&](Vector3d v, Vector3d a, double t)
    {
      Vector3d dv, da, dh;
      double dt, dht;
      return 10 * yawRateContinuityCost(v, a, 0.5, t, 1, dv, da, dt) +
             50 * headingAlignmentCost(v, 0.2, 0.5, t, 1, dh, dht);
    };
    yawRateContinuityCost(velocity, acceleration, 0.5, time, 1, gv, ga, gt);
    Vector3d gh;
    double ght;
    headingAlignmentCost(velocity, 0.2, 0.5, time, 1, gh, ght);
    gv = 10 * gv + 50 * gh;
    ga *= 10;
    gt = 10 * gt + 50 * ght;
    for (int axis = 0; axis < 3; ++axis)
    {
      Vector3d delta = Vector3d::Zero(); delta[axis] = eps;
      const double dv = (cost(velocity + delta, acceleration, time) - cost(velocity - delta, acceleration, time)) / (2 * eps);
      const double da = (cost(velocity, acceleration + delta, time) - cost(velocity, acceleration - delta, time)) / (2 * eps);
      require(std::abs(dv - gv[axis]) < 1e-4 && std::abs(da - ga[axis]) < 1e-4,
              "motion cost gradients must match finite differences, including at rest");
    }
    require(std::abs((cost(velocity, acceleration, time + eps) - cost(velocity, acceleration, time - eps)) /
                        (2 * eps) - gt) < 1e-4, "time gradient includes predicted heading and decay");
  }
  const Vector3d velocity(0.3, 0.1, 0);
  const Vector3d turn = traj_utils::turningAcceleration(velocity, 0.6);
  require(yawRateContinuityCost(velocity, turn, 0.6, 0, 1, gv, ga, gt) < 1e-12,
          "measured turn has zero yaw-rate mismatch");
  require(yawRateContinuityCost(velocity, -turn, 0.6, 0, 1, gv, ga, gt) > 0.5,
          "reversing a measured turn incurs a cost");
  std::cout << "PASS: body/world frames, soft alignment, turn direction, zero speed and analytic gradients\n";
}

void checkOptimizerMotionGradient()
{
  using namespace ego_planner;
  PolyTrajOptimizer optimizer;
  SwarmTrajData swarm;
  optimizer.swarm_trajs_ = &swarm;
  optimizer.piece_num_ = 2;
  optimizer.cps_num_prePiece_ = 5;
  optimizer.cps_.resize_cp(11);
  optimizer.touch_goal_ = true;
  optimizer.wei_obs_ = optimizer.wei_obs_soft_ = optimizer.wei_swarm_ = optimizer.wei_swarm_mod_ = 0;
  optimizer.wei_feas_ = optimizer.wei_sqrvar_ = optimizer.wei_time_ = 0;
  optimizer.max_vel_ = optimizer.max_acc_ = optimizer.max_jer_ = 100;
  Eigen::Matrix3d head = Eigen::Matrix3d::Zero(), tail = head;
  head(0, 1) = 0.3;
  tail(0, 0) = 1.0;
  tail(0, 1) = 0.3;
  optimizer.jerkOpt_.reset(head, tail, 2);
  Eigen::VectorXd variables(5), gradient(5);
  variables << 0.5, 0.2, 0, 0.4, 0.6;
  optimizer.setCurrentPlanarMotion(0, 0.5);
  auto added_cost = [&](const Eigen::VectorXd &x, Eigen::VectorXd &g)
  {
    optimizer.iter_num_ = 0; // No obstacle rebound in this isolated derivative test.
    optimizer.heading_weight_ = 50;
    optimizer.yaw_rate_weight_ = 10;
    const double with_cost = PolyTrajOptimizer::costFunctionCallback(&optimizer, x.data(), g.data(), x.size());
    optimizer.iter_num_ = 0;
    optimizer.heading_weight_ = optimizer.yaw_rate_weight_ = 0;
    Eigen::VectorXd baseline_gradient(5);
    const double baseline = PolyTrajOptimizer::costFunctionCallback(&optimizer, x.data(), baseline_gradient.data(), x.size());
    g -= baseline_gradient;
    return with_cost - baseline;
  };
  const double left = added_cost(variables, gradient);
  for (int index = 0; index < variables.size(); ++index)
  {
    const double eps = 1e-5;
    Eigen::VectorXd plus = variables, minus = variables, scratch(5);
    plus[index] += eps; minus[index] -= eps;
    const double numeric = (added_cost(plus, scratch) - added_cost(minus, scratch)) / (2 * eps);
    require(std::abs(numeric - gradient[index]) < 1e-3 * std::max(1.0, std::abs(numeric)),
            "actual MINCO gradient must include control points and all piece durations");
  }
  Eigen::VectorXd mirrored = variables, scratch(5);
  mirrored[1] *= -1;
  const double right = added_cost(mirrored, scratch);
  require(left < right, "actual planner objective must prefer continuing a measured left turn");
  optimizer.setCurrentPlanarMotion(0, -0.5);
  require(added_cost(mirrored, scratch) < added_cost(variables, scratch), "right turn preference must mirror left");
  std::cout << "PASS: actual MINCO position/time gradients and mirrored left/right route preference\n";
}

void checkWarmStart()
{
  using Eigen::Vector3d;
  ego_planner::EGOPlannerManager manager;
  manager.pp_.max_vel_ = 0.4;
  manager.pp_.polyTraj_piece_length = 0.4;
  poly_traj::MinJerkOpt warm;
  // Prime the existing first-call initializer before testing route reuse.
  require(manager.computeInitState(Vector3d::Zero(), Vector3d::Zero(), Vector3d::Zero(),
      Vector3d(1.4, 0, 0), Vector3d::Zero(), true, false, 1, warm), "initial trajectory construction");
  Eigen::Matrix3d head = Eigen::Matrix3d::Zero(), tail = head;
  head(0, 1) = 0.3;
  tail(0, 0) = 1.4;
  Eigen::MatrixXd inner(3, 1);
  inner.col(0) = Vector3d(0.7, 0.5, 0);
  poly_traj::MinJerkOpt previous;
  previous.reset(head, tail, 2);
  previous.generate(inner, Eigen::VectorXd::Constant(2, 2.0));
  auto &local = manager.traj_.local_traj;
  local.traj = previous.getTraj();
  local.start_time = 10;
  local.duration = 4;
  manager.traj_.global_traj.last_glb_t_of_lc_tgt = manager.traj_.global_traj.glb_t_of_lc_tgt = 10;
  ros::Time::setNow(ros::Time(10.5));
  const Vector3d measured_position(-0.1, -0.1, 0), measured_velocity(0.2, 0, 0), measured_acceleration(0, 0.12, 0);
  require(manager.computeInitState(measured_position, measured_velocity, measured_acceleration,
      Vector3d(1.4, 0, 0), Vector3d::Zero(), false, false, 1, warm), "reuse old route with measured start");
  const auto trajectory = warm.getTraj();
  require((trajectory.getPos(0) - measured_position).norm() < 1e-9 &&
          (trajectory.getVel(0) - measured_velocity).norm() < 1e-9 &&
          (trajectory.getAcc(0) - measured_acceleration).norm() < 1e-9,
          "warm start must not restore the old nominal head");
  const double first_sample = 0.5 + 3.5 / trajectory.getPieceNum();
  require((trajectory.getPositions().col(1) - local.traj.getPos(first_sample)).norm() < 1e-9 &&
          trajectory.getPositions()(1, 1) > 0, "warm start retains the previous left-side route");
  std::cout << "PASS: old route reuse with actual position, velocity and turning acceleration\n";
}

void checkStartStates()
{
  ego_planner::EGOReplanFSM fsm;
  fsm.planner_manager_.reset(new ego_planner::EGOPlannerManager);
  fsm.have_odom_ = true;
  // No target: exercise real start selection and failed retries without
  // optimizing, publishing a trajectory or commanding hardware.
  fsm.have_target_ = false;
  fsm.odom_pos_ = Eigen::Vector3d(-1, 0, 0);
  fsm.odom_vel_ = Eigen::Vector3d(0.07, 0, 0);
  fsm.odom_stamp_ = ros::Time(10.0);
  Eigen::Matrix3d head = Eigen::Matrix3d::Zero(), tail = head;
  head.row(0) << 0, 0.4, 0.2;
  tail.row(0) << 3, 0, 0;
  poly_traj::MinJerkOpt polynomial;
  polynomial.reset(head, tail, 1);
  polynomial.generate(Eigen::MatrixXd(3, 0), Eigen::VectorXd::Constant(1, 10.0));
  auto &info = fsm.planner_manager_->traj_.local_traj;
  info.traj = polynomial.getTraj();
  info.duration = 10;
  info.start_time = 10;
  info.traj_id = 1;
  ros::Time::setNow(ros::Time(10.5));
  require(!fsm.planFromLocalTraj(), "no target cannot plan");
  require((fsm.start_pt_ - info.traj.getPos(0.5)).norm() < 1e-9 &&
          (fsm.start_vel_ - info.traj.getVel(0.5)).norm() < 1e-9 &&
          (fsm.start_acc_ - info.traj.getAcc(0.5)).norm() < 1e-9,
          "ordinary replan must retain trajectory position, velocity and acceleration");
  fsm.replan_from_odometry_ = true;
  require(!fsm.planFromLocalTraj(), "recovery without target fails");
  require((fsm.start_pt_ - fsm.odom_pos_).norm() < 1e-9 &&
          (fsm.start_vel_ - fsm.odom_vel_).norm() < 1e-9 && fsm.start_acc_.isZero(),
          "recovery must use actual position and velocity, zero acceleration");
  fsm.odom_pos_.x() = -0.9;
  fsm.odom_vel_.x() = 0.09;
  require(!fsm.planFromLocalTraj() && fsm.replan_from_odometry_, "failed recovery keeps source");
  require(fsm.start_pt_.x() == -0.9 && fsm.start_vel_.x() == 0.09, "retry uses newest odometry");
  fsm.replan_from_odometry_ = false;
  ros::Time::setNow(ros::Time(20.0));
  require(!fsm.planFromLocalTraj() && fsm.replan_from_odometry_, "expired trajectory falls back to odometry");
  require((fsm.start_pt_ - fsm.odom_pos_).norm() < 1e-9, "must not extrapolate expired trajectory");
  fsm.replan_from_odometry_ = false;
  ros::Time::setNow(ros::Time(9.0));
  require(!fsm.planFromLocalTraj() && fsm.replan_from_odometry_, "negative trajectory time falls back");
  fsm.odom_vel_.x() = std::numeric_limits<double>::quiet_NaN();
  require(!fsm.planFromLocalTraj(), "invalid measured state cannot plan");
  std::cout << "PASS: trajectory p/v/a, recovery odometry, retry source, expired/negative time and invalid state\n";

  // Drive the real FSM callbacks without a map, optimizer or robot connection.
  ros::NodeHandle nh("/replanning_regression");
  fsm.heartbeat_pub_ = nh.advertise<std_msgs::Empty>("heartbeat", 1);
  fsm.data_disp_pub_ = nh.advertise<traj_utils::DataDisp>("display", 1);
  fsm.planner_manager_->grid_map_.reset(new GridMap);
  auto &map = *fsm.planner_manager_->grid_map_;
  map.mp_.have_initialized_ = false;
  map.md_.flag_depth_odom_timeout_ = false;
  map.mp_.enable_height_limit_ = true;
  map.mp_.planning_height_ = 0;
  nav_msgs::Odometry::Ptr odom(new nav_msgs::Odometry);
  odom->header.stamp = ros::Time(9.0);
  odom->pose.pose.orientation.z = std::sin(M_PI / 4);
  odom->pose.pose.orientation.w = std::cos(M_PI / 4);
  odom->twist.twist.linear.y = 0.3;
  odom->twist.twist.angular.z = 0.6;
  fsm.odometryCallback(odom);
  require(!fsm.planFromLocalTraj() && std::abs(fsm.start_acc_.x() + 0.18) < 1e-9 &&
          std::abs(fsm.start_acc_.y()) < 1e-9, "recovery must preserve measured turning acceleration");
  require(std::abs(fsm.odom_yaw_ - M_PI / 2) < 1e-9 && std::abs(fsm.odom_yaw_rate_ - 0.6) < 1e-9,
          "FSM must consume orientation and angular velocity");
  fsm.planner_manager_->pp_.drone_id = 0;
  fsm.have_target_ = fsm.have_trigger_ = true;
  fsm.enable_ground_height_measurement_ = false;
  fsm.target_type_ = ego_planner::EGOReplanFSM::MANUAL_TARGET;
  fsm.final_goal_ = fsm.local_target_pt_ = Eigen::Vector3d(100, 0, 0);
  fsm.replan_thresh_ = fsm.emergency_time_ = 1.0;
  fsm.tracking_error_trigger_ = ego_planner::TrackingErrorTrigger();
  fsm.replan_from_odometry_ = false;
  fsm.exec_state_ = ego_planner::EGOReplanFSM::EXEC_TRAJ;
  ros::Time::setNow(ros::Time(10.1));
  fsm.odom_pos_ = info.traj.getPos(0.1) - Eigen::Vector3d(0.4, 0, 0);
  fsm.execFSMCallback(ros::TimerEvent());
  require(fsm.exec_state_ == ego_planner::EGOReplanFSM::EXEC_TRAJ, "FSM must confirm tracking error");
  ros::Time::setNow(ros::Time(10.31));
  fsm.execFSMCallback(ros::TimerEvent());
  require(fsm.exec_state_ == ego_planner::EGOReplanFSM::REPLAN_TRAJ && fsm.replan_from_odometry_,
          "confirmed error must select recovery from odometry");
  fsm.exec_state_ = ego_planner::EGOReplanFSM::EXEC_TRAJ;
  fsm.replan_from_odometry_ = false;
  ros::Time::setNow(ros::Time(11.1));
  fsm.odom_pos_ = info.traj.getPos(1.1);
  fsm.execFSMCallback(ros::TimerEvent());
  require(fsm.exec_state_ == ego_planner::EGOReplanFSM::REPLAN_TRAJ && !fsm.replan_from_odometry_,
          "normal timer must preserve trajectory sampling");
  fsm.exec_state_ = ego_planner::EGOReplanFSM::EXEC_TRAJ;
  ros::Time::setNow(ros::Time(20.0));
  fsm.execFSMCallback(ros::TimerEvent());
  require(fsm.exec_state_ == ego_planner::EGOReplanFSM::REPLAN_TRAJ && fsm.replan_from_odometry_,
          "expired trajectory must trigger measured-state recovery");
  fsm.enable_fail_safe_ = true;
  map.md_.flag_depth_odom_timeout_ = true;
  fsm.checkCollisionCallback(ros::TimerEvent());
  require(fsm.exec_state_ == ego_planner::EGOReplanFSM::EMERGENCY_STOP && !fsm.enable_fail_safe_,
          "map timeout must stop and disable automatic recovery");
  map.md_.flag_depth_odom_timeout_ = false;
  fsm.checkCollisionCallback(ros::TimerEvent());
  require(fsm.exec_state_ == ego_planner::EGOReplanFSM::EMERGENCY_STOP, "safety callback must preserve stop");
  std::cout << "PASS: actual FSM error/timer/expiry transitions and emergency-stop priority\n";
}

int main(int argc, char **argv)
{
  try
  {
    checkErrorTrigger();
    checkPlanarMotionCosts();
    checkOptimizerMotionGradient();
    ros::init(argc, argv, "check_ego_replanning", ros::init_options::NoRosout);
    ros::NodeHandle nh;
    checkWarmStart();
    checkStartStates();
    ros::shutdown();
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
