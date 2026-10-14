#pragma once

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <string.h>
#include <algorithm>
 
#include <OsqpEigen/OsqpEigen.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/create_timer.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <quadrotor_msgs/msg/goal_set.hpp>
#include <traj_utils/msg/motion_command.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "cubic_spline_planner.h"
//#include "diablo_sdk/Diablo_Ctrl.h"


#include "traj_utils/msg/poly_traj.hpp"
#include <traj_utils/planar_motion.h>
#include "mpc_utils/poly_traj_utils.hpp"
using namespace std;
using namespace Eigen;

class MPCState
{
public:
    double x = 0;
    double y = 0;
    double v = 0; 
    double theta = 0;
    double yaw_rate = 0;
};

class MPC
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    uint64_t active_task_=0, retired_task_=0;
    int64_t map_stamp_=0;
    rclcpp::Time odom_time_{0,0,RCL_ROS_TIME};
    rclcpp::SubscriptionBase::SharedPtr task_sub_,cancel_sub_;
    void publishCommand();
    // parameters
    /// algorithm param
    double du_th = 0.1;
    double dt = 0.2;
    int T = 5;
    int max_iter = 3;
    int delay_num = 0;
    vector<double> Q = {10, 10, 2.5, 2.0};
    vector<double> R = {0.01, 0.01};
    vector<double> Rd = {0.01, 1.0};
    /// constraints
    double max_omega = M_PI / 2;
    double max_domega = M_PI / 6;
    double max_comega = M_PI / 6 * 0.2;
    double max_speed = 55.0 / 3.6;
    double max_cv = 0.2;
    double max_accel = 1.0;
    // When false the MPC always tracks the reference heading directly. When true
    // (legacy behavior) smooth_yaw() may flip the reference yaw by pi and negate
    // the reference speed if the heading error exceeds pi/2 — useful for
    // bidirectional differential drives, but causes spurious U-turns when the
    // chassis only drives forward.
    bool allow_reverse = false;
    // Retained across trajectory replacements, cleared outside the antipodal
    // region or on stop. This selects an equivalent reference, not a bound.
    int yaw_turn_direction_ = 0;

    // MPC dataset
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    Eigen::VectorXd C;
    MPCState xbar[500];
    Eigen::MatrixXd xref;
    Eigen::MatrixXd dref;
    Eigen::MatrixXd output;
    Eigen::MatrixXd last_output;
    std::vector<Eigen::Vector2d> output_buff;

    //Bspline about
    // vector<UniformBspline> traj_;
    std::shared_ptr<mpc_utils::Trajectory> traj_;
    rclcpp::Time start_time_{0, 0, RCL_ROS_TIME}, true_start_time_{0, 0, RCL_ROS_TIME};
    int traj_id_;
    bool back_flag = false, end_flag = false;
    Eigen::Vector3d final_pos_;

    // control data
    bool has_odom;
    bool receive_traj_ = false;
    bool control_a = true;
    double traj_duration_;
    double t_track = 0.0;
    MPCState now_state;

    // ros interface
	rclcpp::Node::SharedPtr node_;
    rclcpp::TimerBase::SharedPtr cmd_timer_;
    rclcpp::Publisher<traj_utils::msg::MotionCommand>::SharedPtr pos_cmd_pub_; rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr vis_pub; rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr predict_pub; rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr ref_pub;
    rclcpp::SubscriptionBase::SharedPtr odom_sub_, traj_sub_, trigger_sub_,bspline_sub_, emergency_stop_sub_;

    bool emergency_stop_flag = false;
    bool solve_ok_ = false;

    //diablo_sdk::Diablo_Ctrl cmd;
    geometry_msgs::msg::Twist cmd;
    void cmdCallback();
    void rcvOdomCallBack(nav_msgs::msg::Odometry::SharedPtr msg);
    void rcvTriggerCallBack(const geometry_msgs::msg::PoseStamped msg);
    void emergencyStopCallback(const std_msgs::msg::Empty msg);

    void polyTrajCallback(const traj_utils::msg::PolyTraj::ConstSharedPtr& msg);
    // void bsplineCallback(traj_utils::msg::Bspline::ConstSharedPtr msg);

    // for test tracking performance
    bool in_test = false;
    //cubic_spline_planner csp;
    string test_traj = "";
    vector<Eigen::Vector3d> csp_path;

    // MPC function
    void getLinearModel(const MPCState& s);
    void stateTrans(MPCState& s, double a, double yaw_dot);
    void predictMotion(void);
    void predictMotion(MPCState *b);
    void solveMPCV(void);
    void solveMPCA(void);
    void getCmd(void);

    // utils
    MPCState xopt[500];
    void normlize_theta(double& th)
    {
        while (th > M_PI)
            th -= M_PI * 2;
        while (th <= -M_PI)
            th += M_PI * 2;
    }
    void smooth_yaw(void)
    {
        // Unwrap within the horizon and retain the chosen turn across control
        // cycles near the opposite heading. E.g. +179 -> -179 becomes +181
        // after choosing a positive turn, instead of reversing that turn.
        double dyaw = xref(3, 0) - now_state.theta;
        normlize_theta(dyaw);
        constexpr double branch_margin = 15.0 * M_PI / 180.0;
        constexpr double rate_deadband = 0.05; // rad/s, ignore stationary gyro noise
        if (!allow_reverse && std::abs(dyaw) >= M_PI - branch_margin)
        {
            if (yaw_turn_direction_ == 0)
            {
                const double direction = std::abs(now_state.yaw_rate) > rate_deadband
                    ? now_state.yaw_rate : dyaw;
                yaw_turn_direction_ = direction >= 0.0 ? 1 : -1;
                RCLCPP_INFO(rclcpp::get_logger("mpc"), "[MPC_YAW_BRANCH] direction=%d shortest_error=%.3f yaw_rate=%.3f",
                         yaw_turn_direction_, dyaw, now_state.yaw_rate);
            }
            if (dyaw * yaw_turn_direction_ < 0.0)
                dyaw += yaw_turn_direction_ * 2.0 * M_PI;
        }
        else
            yaw_turn_direction_ = 0;
        if (allow_reverse && (dyaw > M_PI / 2 || dyaw < -M_PI / 2))
        {
            dyaw = (dyaw > 0) ? dyaw - M_PI : dyaw + M_PI;
            xref(2, 0) = -xref(2, 0);
        }
        xref(3, 0) = now_state.theta + dyaw;

        for (int i=0; i<T-1; i++)
        {
            dyaw = xref(3, i+1) - xref(3, i);
            normlize_theta(dyaw);
            if (allow_reverse && (dyaw > M_PI / 2 || dyaw < -M_PI / 2))
            {
                dyaw = (dyaw > 0) ? dyaw - M_PI : dyaw + M_PI;
                xref(2, i+1) = -xref(2, i+1);
            }
            xref(3, i+1) = xref(3, i) + dyaw;
        }
    }
    // double get_nearest(void)
    // {
    //     Eigen::Vector3d track_point = csp.get_state(t_track);
    //     Eigen::Vector2d now_point(now_state.x, now_state.y);
    //     double min_dist = (now_point-track_point.head(2)).norm();
    //     double track_temp = t_track;
    //     double interval = now_state.v * dt;
        
    //     for (double t_temp = track_temp + interval, i=1.0; i < 10; i += 1.0, t_temp += interval)
    //     // for (double t_temp = track_temp + 1.0, i=1.0; i < 10; i += 1.0, t_temp += 1.0)
    //     {
    //         if (t_temp > traj_duration_)
    //         {
    //             t_temp = traj_duration_;
    //         }
    //         Eigen::Vector3d temp = csp.get_state(t_temp);
    //         double dist = (temp.head(2) - now_point).norm();
         
    //         if (dist<min_dist)
    //         {
    //             min_dist = dist;
    //             t_track = t_temp;
    //         }
    //     }
    //     // RCLCPP_INFO(rclcpp::get_logger("mpc"), "t_track=%f,traj_duration=%f",t_track,traj_duration_);

    //     return t_track;
    // }
    void drawFollowPath(void)
    {
        int id = 0;
        visualization_msgs::msg::Marker sphere, line_strip;
        sphere.header.frame_id = line_strip.header.frame_id = "map";
        sphere.header.stamp = line_strip.header.stamp = node_->now();
        sphere.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        line_strip.type = visualization_msgs::msg::Marker::LINE_STRIP;
        sphere.action = line_strip.action = visualization_msgs::msg::Marker::ADD;
        sphere.id = id;
        line_strip.id = id + 1000;

        sphere.pose.orientation.w = line_strip.pose.orientation.w = 1.0;
        sphere.color.r = line_strip.color.r = 1;
        sphere.color.g = line_strip.color.g = 0;
        sphere.color.b = line_strip.color.b = 1;
        sphere.color.a = line_strip.color.a = 1;
        sphere.scale.x = 0.04;
        sphere.scale.y = 0.04;
        sphere.scale.z = 0.04;
        line_strip.scale.x = 0.04 / 2;
        geometry_msgs::msg::Point pt;
        
        for (auto p:csp_path)
        {
            pt.x = p(0);
            pt.y = p(1);
            pt.z = 0.0;
            line_strip.points.push_back(pt);
        }
        vis_pub->publish(line_strip);
    }
    void drawPredictPath(MPCState *b)
    {
        int id = 0;
        double sc = 0.05;
        visualization_msgs::msg::Marker sphere, line_strip;
        sphere.header.frame_id = line_strip.header.frame_id = "map";
        sphere.header.stamp = line_strip.header.stamp = node_->now();
        sphere.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        line_strip.type = visualization_msgs::msg::Marker::LINE_STRIP;
        sphere.action = line_strip.action = visualization_msgs::msg::Marker::ADD;
        sphere.id = id;
        line_strip.id = id + 1000;

        sphere.pose.orientation.w = line_strip.pose.orientation.w = 1.0;
        sphere.color.r = line_strip.color.r = 0;
        sphere.color.g = line_strip.color.g = 1;
        sphere.color.b = line_strip.color.b = 0;
        sphere.color.a = line_strip.color.a = 1;
        sphere.scale.x = sc;
        sphere.scale.y = sc;
        sphere.scale.z = sc;
        line_strip.scale.x = sc / 2;
        geometry_msgs::msg::Point pt;
        
        for (int i=0; i<T; i++)
        {
            pt.x = b[i].x;
            pt.y = b[i].y;
            pt.z = 0.0;
            line_strip.points.push_back(pt);
        }
        predict_pub->publish(line_strip);
    }
    void drawRefPath(void)
    {
        int id = 0;
        double sc = 0.05;
        visualization_msgs::msg::Marker sphere, line_strip;
        sphere.header.frame_id = line_strip.header.frame_id = "map";
        sphere.header.stamp = line_strip.header.stamp = node_->now();
        sphere.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        line_strip.type = visualization_msgs::msg::Marker::LINE_STRIP;
        sphere.action = line_strip.action = visualization_msgs::msg::Marker::ADD;
        sphere.id = id;
        line_strip.id = id + 1000;

        sphere.pose.orientation.w = line_strip.pose.orientation.w = 1.0;
        sphere.color.r = line_strip.color.r = 0;
        sphere.color.g = line_strip.color.g = 0;
        sphere.color.b = line_strip.color.b = 1;
        sphere.color.a = line_strip.color.a = 1;
        sphere.scale.x = sc;
        sphere.scale.y = sc;
        sphere.scale.z = sc;
        line_strip.scale.x = sc / 2;
        geometry_msgs::msg::Point pt;
        
        for (int i=0; i<T; i++)
        {
            pt.x = xref(0, i);
            pt.y = xref(1, i);
            pt.z = 0.0;
            line_strip.points.push_back(pt);
        }
        ref_pub->publish(line_strip);
    }
    // Eigen::Vector4d getBsplineState(const double &u)
    // {
    //     Eigen::Vector3d pos, vel;
    //     Eigen::Vector4d state;
    //     pos = traj_[0].evaluateDeBoorT(u);
    //     vel = traj_[1].evaluateDeBoorT(u);
    //     double v_temp = vel.norm();
    //     state(0) = pos(0);
    //     state(1) = pos(1);
    //     state(2) = v_temp;
    //     state(3) = atan2(vel(1),vel(0));
    //     return state;
    // }

    Eigen::Vector4d getPolyTrajState(const double &u)
    {
        Eigen::Vector3d pos, vel;
        Eigen::Vector4d state;
        pos = traj_->getPos(u);
        vel = traj_->getVel(u);
        double v_temp = vel.norm();
        state(0) = pos(0);
        state(1) = pos(1);
        state(2) = v_temp;
        state(3) = atan2(vel(1),vel(0));

        // std::cout << "\n\n\n\n cur_vel: " << vel.transpose() << std::endl;
        return state;
    } 

public:
	MPC(){}//: csp(vector<double>{0.0, 3.0, 2.7, 1.5, 0.3, 0, 3.0, 2.7, 1.5, 0.3, 0.0}, vector<double>{0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 6.0, 4.0, 3.0, 2.0, 0.0}, 0.1) {}
    void init(rclcpp::Node::SharedPtr &nh);
	~MPC() {}
};
