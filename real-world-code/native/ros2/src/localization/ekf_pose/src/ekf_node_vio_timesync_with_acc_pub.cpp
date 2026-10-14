#include "ekf.h"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/create_timer.hpp>
#include <stdexcept>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/range.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
// #include <geometry_msgs/msg/accel.hpp>
#include "conversion.h"
#include <cstdlib>

using namespace std;
using namespace Eigen;

template <typename T>
static inline void readEssentialParam(const rclcpp::Node::SharedPtr &nh, const std::string &name, T &val)
{
    if (!nh->get_parameter(name, val))
    {
        RCLCPP_FATAL_STREAM(rclcpp::get_logger("ekf"), "[" << nh->get_name() << "] required param missing: " << name);
        rclcpp::shutdown();
        std::exit(EXIT_FAILURE);
    }
}
//20200531: time synchronization
//20200105: ekf_node_vio.cpp and ekf_node_mocap.cpp merge into one (ekf_node_vio.cpp) and the differences between them is the odom format
//X_state: p q v gb ab   with time stamp aligned between imu and img
/*
    EKF model
    prediction:
    xt~ = xt-1 + dt*f(xt-1, ut, 0)
    sigmat~ = Ft*sigmat-1*Ft' + Vt*Qt*Vt'
    Update:
    Kt = sigmat~*Ct'*(Ct*sigmat~*Ct' + Wt*Rt*Wt')^-1
    xt = xt~ + Kt*(zt - g(xt~,0))
    sigmat = sigmat~ - Kt*Ct*sigmat~
*/
/*
   -pi ~ pi crossing problem:
   1. the model prpagation: X_state should be limited to [-pi,pi] after predicting and updating
   2. inovation crossing: (measurement - g(X_state)) should also be limited to [-pi,pi] when getting the inovation.
   z_measurement is normally in [-pi~pi]
*/

// imu frame is imu body frame

//odom: pose px,py pz orientation qw qx qy qz
//imu: acc: x y z gyro: wx wy wz

#define TimeSync 0 //time synchronize or not
#define RePub 0 //re publish the odom when repropagation

rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr cam_odom_pub;
rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr acc_filtered_pub;

//state
geometry_msgs::msg::Pose pose;
Vector3d position, orientation, velocity;

// Now set up the relevant matrices
//states X [p q pdot]  [px,py,pz, wx,wy,wz, vx,vy,vz]
size_t stateSize;                                // x = [p q pdot bg ba]
size_t stateSize_pqv;                                // x = [p q pdot]
size_t measurementSize;                          // z = [p q]
size_t inputSize;                                // u = [w a]
VectorXd X_state(stateSize);                     // x (in most literature)
VectorXd u_input;
VectorXd Z_measurement;                          // z
MatrixXd StateCovariance;                        // sigma
MatrixXd Kt_kalmanGain;                          // Kt
VectorXd X_state_correct(stateSize);                     // x (in most literature)
MatrixXd StateCovariance_correct;                        // sigma
// MatrixXd Ct_stateToMeasurement;                  // Ct
//  VectorXd innovation;                         // z - Hx

MatrixXd Qt;
MatrixXd Rt;
Vector3d u_gyro = Vector3d::Zero();
Vector3d u_acc = Vector3d::Zero();
bool output_body_pose = false, initialize_gravity_from_imu = false;
Vector3d initial_acc_mean = Vector3d::Zero();
size_t initial_acc_count = 0;
double initial_pose_since = -1., initial_pose_last = -1.;
double initial_imu_stamp = -1.;
Vector3d initial_pose_position = Vector3d::Zero();
Quaterniond initial_pose_orientation = Quaterniond::Identity();
string initial_pose_frame;
Vector3d gravity(0., 0., -10.2); //need to estimate the bias 9.8099
// Vector3d gravity(0., 0., -9.81); //need to estimate the bias 9.8099
Vector3d bg_0(0., 0., 0); //need to estimate the bias
Vector3d ba_0(0., 0., 0); //need to estimate the bias  0.1
Vector3d ng(0., 0., 0.);
Vector3d na(0., 0., 0.);
Vector3d nbg(0., 0., 0.);
Vector3d nba(0., 0., 0.);

Vector3d q_last;
Vector3d bg_last;
Vector3d ba_last;

//Qt imu covariance matrix  smaller believe system(imu) more
double imu_trans_x = 0.0;
double imu_trans_y = 0.0;
double imu_trans_z = 0.0;
double gyro_cov = 0.01;
double acc_cov = 0.01;
//Rt visual odomtry covariance smaller believe measurement more
double position_cov = 0.1;
double q_rp_cov = 0.1;
double q_yaw_cov = 0.1;

double dt = 0.01; //second
double t_last, t_now;  
bool first_frame_imu = true;
bool first_frame_tag_odom = true;
bool test_odomtag_call = false;
bool odomtag_call = false;

double time_now, time_last;
double time_odom_tag_now;
// double diff_time;

string world_frame_id = "world";

//world frame points velocity
deque<pair<VectorXd, sensor_msgs::msg::Imu>> sys_seq;
deque<MatrixXd> cov_seq;
double dt_0_rp; //the dt for the first frame in repropagation
void seq_keep(const sensor_msgs::msg::Imu::SharedPtr &imu_msg)
{
    #define seqsize 60
    if(sys_seq.size() < seqsize)
    {
        sys_seq.push_back(make_pair(X_state, *imu_msg));  //X_state before propagation and imu at that time
        cov_seq.push_back(StateCovariance);
    }
    else
    {
        sys_seq.pop_front();
        sys_seq.push_back(make_pair(X_state, *imu_msg));
        cov_seq.pop_front();
        cov_seq.push_back(StateCovariance);
    }
    // ensure that the later frame time > the former one
}
//choose the coordinate frame imu for the measurement
bool search_proper_frame(double odom_time)
{
    if(sys_seq.size() <= 1)
    {
        dt_0_rp = 0;
        return false;
    }
        
    size_t rightframe = sys_seq.size()-1;
    bool find_proper_frame = false;
    for(size_t i = 1; i < sys_seq.size(); i++)  // TODO: it better to search from the middle instead in the front
    {
        double time_before = odom_time - rclcpp::Time(sys_seq[i-1].second.header.stamp).seconds();
        double time_after  = odom_time - rclcpp::Time(sys_seq[i].second.header.stamp).seconds();
        if((time_before >= 0) && (time_after < 0))
        {
            if(abs(time_before) > abs(time_after))
            {
                rightframe = i;
            }
            else
            {
                rightframe = i-1;
            }

            if(rightframe != 0)
            {
                dt_0_rp = rclcpp::Time(sys_seq[rightframe].second.header.stamp).seconds() - rclcpp::Time(sys_seq[rightframe-1].second.header.stamp).seconds();
            }
            else
            {// if rightframe is the first frame in the seq, set dt_0_rp as the next dt
                dt_0_rp = rclcpp::Time(sys_seq[rightframe+1].second.header.stamp).seconds() - rclcpp::Time(sys_seq[rightframe].second.header.stamp).seconds();
            }
            
            find_proper_frame = true;
            break;
        }
    }
    if(!find_proper_frame)
    {
        if((odom_time - rclcpp::Time(sys_seq[0].second.header.stamp).seconds()) <= 0)  //if odom time before the first frame, set first frame
        {
            rightframe = 0;
            // if rightframe is the first frame in the seq, set dt_0_rp as the next dt
            dt_0_rp = rclcpp::Time(sys_seq[rightframe+1].second.header.stamp).seconds() - rclcpp::Time(sys_seq[rightframe].second.header.stamp).seconds();
        }
        if((odom_time - rclcpp::Time(sys_seq[sys_seq.size()-1].second.header.stamp).seconds()) >= 0)  //if odom time after the last frame, set last frame
        {
            rightframe = sys_seq.size()-1;
            dt_0_rp = rclcpp::Time(sys_seq[rightframe].second.header.stamp).seconds() - rclcpp::Time(sys_seq[rightframe-1].second.header.stamp).seconds();
        }
        //no process, set the latest one
    }

    //set the right frame as the first frame in the queue
    for(size_t i = 0; i < rightframe; i++)  
    {
        sys_seq.pop_front();
        cov_seq.pop_front();
    }
    
    if(find_proper_frame)
    {
        return true;
    }
    else
    {
        return false;
    }
}
void re_propagate()
{
    for(size_t i=1; i < sys_seq.size(); i++)
    {
        //re-prediction for the rightframe 
        dt = rclcpp::Time(sys_seq[i].second.header.stamp).seconds() - rclcpp::Time(sys_seq[i-1].second.header.stamp).seconds();

        u_gyro(0) = sys_seq[i].second.angular_velocity.x;
        u_gyro(1) = sys_seq[i].second.angular_velocity.y;
        u_gyro(2) = sys_seq[i].second.angular_velocity.z;
        u_acc(0)  = sys_seq[i].second.linear_acceleration.x;
        u_acc(1)  = sys_seq[i].second.linear_acceleration.y;
        u_acc(2)  = sys_seq[i].second.linear_acceleration.z;

        MatrixXd Ft;
        MatrixXd Vt;

        q_last = sys_seq[i].first.segment<3>(3);  // last X2
        bg_last = sys_seq[i].first.segment<3>(9);  //last X4
        ba_last = sys_seq[i].first.segment<3>(12);  //last X5
        Ft = MatrixXd::Identity(stateSize, stateSize) + dt*diff_f_diff_x(q_last, u_gyro, u_acc, bg_last, ba_last);
        
        Vt = dt*diff_f_diff_n(q_last);

        X_state += dt*F_model(u_gyro, u_acc);
        if(X_state(3) > PI)  X_state(3) -= 2*PI;
        if(X_state(3) < -PI) X_state(3) += 2*PI;
        if(X_state(4) > PI)  X_state(4) -= 2*PI;
        if(X_state(4) < -PI) X_state(4) += 2*PI;
        if(X_state(5) > PI)  X_state(5) -= 2*PI;
        if(X_state(5) < -PI) X_state(5) += 2*PI;
        StateCovariance = Ft*StateCovariance*Ft.transpose() + Vt*Qt*Vt.transpose();

        #if RePub
        system_pub(sys_seq[i].second.header.stamp);  // choose to publish the repropagation or not
        #endif

    }
}


void imu_callback(sensor_msgs::msg::Imu::SharedPtr msg)
{
    // seq_keep(msg);
    // nav_msgs::msg::Odometry odom_fusion;
    //your code for propagation
    msg->linear_acceleration.z *= 9.81;
    msg->linear_acceleration.x *= 9.81;
    msg->linear_acceleration.y *= 9.81;

    if (first_frame_tag_odom && initialize_gravity_from_imu) {
        const double stamp = rclcpp::Time(msg->header.stamp).seconds();
        if (stamp <= initial_imu_stamp) return;
        if (initial_imu_stamp > 0. && stamp-initial_imu_stamp > .1) {
            initial_acc_count = 0;
            initial_acc_mean.setZero();
            initial_pose_since = -1.;
        }
        initial_imu_stamp = stamp;
        const Vector3d acc(msg->linear_acceleration.x, msg->linear_acceleration.y,
                           msg->linear_acceleration.z);
        const Vector3d gyro(msg->angular_velocity.x, msg->angular_velocity.y,
                            msg->angular_velocity.z);
        // Calibration requires a stationary start, just like FAST-LIO initialization.
        if (!acc.allFinite() || !gyro.allFinite() || acc.norm() < 5. || acc.norm() > 15. ||
            gyro.norm() > 0.15 || (initial_acc_count && (acc-initial_acc_mean).norm() > 0.5)) {
            initial_acc_count = 0;
            initial_acc_mean.setZero();
            initial_pose_since = -1.;
        } else {
            initial_acc_mean += (acc-initial_acc_mean) / static_cast<double>(++initial_acc_count);
        }
    }

    // std::cout <<msg2->linear_acceleration.z <<std::endl;
    
    if(!first_frame_tag_odom)
    { //get the initial pose and orientation in the first frame of measurement 
        if(first_frame_imu)
        {
            u_gyro << msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z;
            u_acc << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
            first_frame_imu = false;
            time_now = rclcpp::Time(msg->header.stamp).seconds();
            time_last = time_now;
            #if TimeSync
            seq_keep(msg);//keep before propagation
            #endif

            system_pub(msg->header.stamp);
        }
        else
        {
            #if TimeSync
            seq_keep(msg);//keep before propagation
            #endif

            time_now = rclcpp::Time(msg->header.stamp).seconds();
            dt = time_now - time_last;

            if(odomtag_call)
            {
                odomtag_call = false;
                // diff_time = time_now - time_odom_tag_now;
                // if(diff_time<0)
                // {
                //     cout << "diff time: " << diff_time << endl;  //???!!! exist !!!???
                //     cout << "timeimu: " << time_now - 1.60889e9 << " time_odom: " << time_odom_tag_now - 1.60889e9 << endl;
                //     // cout << "diff time: " << diff_time << endl;  //about 30ms
                // }
            }
            MatrixXd Ft;
            MatrixXd Vt;

            u_gyro(0) = msg->angular_velocity.x;
            u_gyro(1) = msg->angular_velocity.y;
            u_gyro(2) = msg->angular_velocity.z;
            u_acc(0)  = msg->linear_acceleration.x;
            u_acc(1)  = msg->linear_acceleration.y;
            u_acc(2)  = msg->linear_acceleration.z;

            q_last = X_state.segment<3>(3);  // last X2
            bg_last = X_state.segment<3>(9);  //last X4
            ba_last = X_state.segment<3>(12);  //last X5
            Ft = MatrixXd::Identity(stateSize, stateSize) + dt*diff_f_diff_x(q_last, u_gyro, u_acc, bg_last, ba_last);
         
            Vt = dt*diff_f_diff_n(q_last);

            
            X_state += dt*F_model(u_gyro, u_acc);
            if(X_state(3) > PI)  X_state(3) -= 2*PI;
            if(X_state(3) < -PI) X_state(3) += 2*PI;
            if(X_state(4) > PI)  X_state(4) -= 2*PI;
            if(X_state(4) < -PI) X_state(4) += 2*PI;
            if(X_state(5) > PI)  X_state(5) -= 2*PI;
            if(X_state(5) < -PI) X_state(5) += 2*PI;
            StateCovariance = Ft*StateCovariance*Ft.transpose() + Vt*Qt*Vt.transpose();
 
            time_last = time_now;
            
            // if(test_odomtag_call) //no frequency boost
            // {
            //     test_odomtag_call = false;
            //     system_pub(msg->header.stamp);
            // }
            system_pub(msg->header.stamp);

            acc_f_pub(u_gyro, u_acc, msg->header.stamp);
        }
    }
  
}

//Rotation from the camera frame to the IMU frame
Matrix3d Rc_i;     
Vector3d tc_i;  //  cam in imu frame
int cnt = 0;
Vector3d INNOVATION_;
Matrix3d Rr_i;     
Vector3d tr_i;  //  rigid body in imu frame
//msg is imu in world
VectorXd get_pose_from_VIOodom(const nav_msgs::msg::Odometry::ConstSharedPtr &msg)
{
    Matrix3d Rr_w;    //rigid body in world
    Vector3d tr_w;
    Matrix3d Ri_w;  
    Vector3d ti_w;
    Vector3d p_temp;
    p_temp(0) = msg->pose.pose.position.x;
    p_temp(1) = msg->pose.pose.position.y;
    p_temp(2) = msg->pose.pose.position.z;
    //quaternion2euler:  ZYX  roll pitch yaw
    Quaterniond q;
    q.w() = msg->pose.pose.orientation.w;
    q.x() = msg->pose.pose.orientation.x;
    q.y() = msg->pose.pose.orientation.y;
    q.z() = msg->pose.pose.orientation.z;
    
    //Euler transform
    // Ri_w = q.toRotationMatrix();
    // ti_w = p_temp;
    Rr_w = q.toRotationMatrix();
    tr_w = p_temp;
    Ri_w = Rr_w * Rr_i.inverse();
    ti_w = tr_w - Ri_w*tr_i;
    Vector3d euler = mat2euler(Ri_w);

    VectorXd pose = VectorXd::Random(6);
    pose.segment<3>(0) = ti_w;
    pose.segment<3>(3) = euler;

    return pose;
}
void vioodom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr &msg)
{//assume that the odom_tag from camera is sychronized with the imus and without delay. !!!

    //your code for update
    if(first_frame_tag_odom)
    {//system begins in first odom frame
        if (initialize_gravity_from_imu) {
            // A first prior-map match is not a settled world pose. Calibrating
            // gravity there locks its registration error into every prediction.
            const double stamp = rclcpp::Time(msg->header.stamp).seconds();
            const auto& p = msg->pose.pose.position;
            const auto& q = msg->pose.pose.orientation;
            const Vector3d position(p.x, p.y, p.z);
            const Quaterniond orientation(q.w, q.x, q.y, q.z);
            if (!position.allFinite() || !orientation.coeffs().allFinite() ||
                fabs(orientation.squaredNorm()-1.) > .01 || stamp <= 0. ||
                msg->header.frame_id.empty()) {
                initial_pose_since = -1.;
                initial_acc_count = 0;
                initial_acc_mean.setZero();
                return;
            }
            if (stamp <= initial_pose_last) return;
            const bool restart = initial_pose_since < 0. || stamp-initial_pose_last > .25 ||
                initial_pose_frame != msg->header.frame_id ||
                (position-initial_pose_position).norm() > .02 ||
                initial_pose_orientation.angularDistance(orientation) > .01;
            initial_pose_last = stamp;
            if (restart) {
                initial_pose_since = stamp;
                initial_pose_position = position;
                initial_pose_orientation = orientation.normalized();
                initial_pose_frame = msg->header.frame_id;
                initial_acc_count = 0;
                initial_acc_mean.setZero();
                return;
            }
            // Both pose and raw IMU must remain stationary for this window.
            if (stamp-initial_pose_since < 1. || initial_acc_count < 50 ||
                stamp-initial_imu_stamp > .05 || initial_imu_stamp-stamp > .25) return;
        }
        first_frame_tag_odom = false;
        time_odom_tag_now = rclcpp::Time(msg->header.stamp).seconds();

        VectorXd odom_pose = get_pose_from_VIOodom(msg);
        X_state.segment<3>(0) = odom_pose.segment<3>(0);
        X_state.segment<3>(3) = odom_pose.segment<3>(3);

        if (initialize_gravity_from_imu) {
            // The public world is the initial body frame, which need not be perfectly level.
            // Use the measured SI acceleration, matching this EKF's raw-IMU scale.
            gravity = -euler2mat(X_state.segment<3>(3)) * initial_acc_mean;
            RCLCPP_INFO(rclcpp::get_logger("ekf"),
                        "Initialized world gravity [%.6f %.6f %.6f] from %zu stationary IMUs after stable localization",
                        gravity.x(), gravity.y(), gravity.z(), initial_acc_count);
        }
        world_frame_id = msg->header.frame_id;

    }
    else
    {
        time_odom_tag_now = rclcpp::Time(msg->header.stamp).seconds();
        //    double t = clock();
        MatrixXd Ct;
        MatrixXd Wt;

        VectorXd odom_pose = get_pose_from_VIOodom(msg);

        #if TimeSync
        //call back to the proper time
        search_proper_frame(time_odom_tag_now);
        #endif

        Z_measurement.segment<3>(0) = odom_pose.segment<3>(0);
        Z_measurement.segment<3>(3) = odom_pose.segment<3>(3);
        // cam_system_pub(msg->header.stamp);
       
        #if !TimeSync //no aligned
        Ct = diff_g_diff_x();
        Wt = diff_g_diff_v();

        Kt_kalmanGain = StateCovariance*Ct.transpose() * (Ct*StateCovariance*Ct.transpose() + Wt*Rt*Wt.transpose()).inverse();
        VectorXd gg = g_model();
        VectorXd innovation = Z_measurement - gg;
        VectorXd innovation_t = gg;
    
        //Prevent innovation changing suddenly when euler from -Pi to Pi
        
        if(innovation(3) > 6)  innovation(3) -= 2*PI;
        if(innovation(3) < -6) innovation(3) += 2*PI;
        if(innovation(4) > 6)  innovation(4) -= 2*PI;
        if(innovation(4) < -6) innovation(4) += 2*PI;
        if(innovation(5) > 6)  innovation(5) -= 2*PI;
        if(innovation(5) < -6) innovation(5) += 2*PI;
        INNOVATION_ = innovation_t.segment<3>(3);
        X_state += Kt_kalmanGain*(innovation);
        if(X_state(3) > PI)  X_state(3) -= 2*PI;
        if(X_state(3) < -PI) X_state(3) += 2*PI;
        if(X_state(4) > PI)  X_state(4) -= 2*PI;
        if(X_state(4) < -PI) X_state(4) += 2*PI;
        if(X_state(5) > PI)  X_state(5) -= 2*PI;
        if(X_state(5) < -PI) X_state(5) += 2*PI;
        StateCovariance = StateCovariance - Kt_kalmanGain*Ct*StateCovariance;

        // RCLCPP_INFO(rclcpp::get_logger("ekf"), "time cost: %f\n", (clock() - t) / CLOCKS_PER_SEC);
        // cout << "z " << Z_measurement(2) << " k " << Kt_kalmanGain(2) << " inn " << innovation(2) << endl;
        
        cam_system_pub(msg->header.stamp);
        test_odomtag_call = true;
        odomtag_call = true;
        
        if(INNOVATION_(0)>6 || INNOVATION_(1)>6 || INNOVATION_(2)>6)
        cout << "\ninnovation: \n" << INNOVATION_ << endl; 
        if(INNOVATION_(0)<-6 || INNOVATION_(1)<-6 || INNOVATION_(2)<-6)
        cout << "\ninnovation: \n" << INNOVATION_ << endl; 
        //monitor the position changing
        if((innovation(0)>1.5) || (innovation(1)>1.5) || (innovation(2)>1.5) ||
           (innovation(0)<-1.5) || (innovation(1)<-1.5) || (innovation(2)<-1.5))
           RCLCPP_ERROR(rclcpp::get_logger("ekf"), "posintion diff too much between measurement and model prediction!!!");
        if(cnt == 10||cnt==50||cnt==90)
        {
            // cout << "Ct: \n" << Ct << "\nWt:\n" << Wt << endl; 
            // cout << "Kt_kalmanGain: \n" << Kt_kalmanGain << endl; 
            // cout << "\ninnovation: \n" << Kt_kalmanGain*innovation  << "\ndt:\n" << dt << endl;
            // cout << "\ninnovation: \n" << Kt_kalmanGain*innovation  << endl;
            // cout << "\ninnovation: \n" << INNOVATION_ << endl; 
        } 
        cnt++;
        if(cnt>100) cnt=101;

        #else  // time sync

        //re-prediction for the rightframe 
        dt = dt_0_rp;

        u_gyro(0) = sys_seq[0].second.angular_velocity.x;
        u_gyro(1) = sys_seq[0].second.angular_velocity.y;
        u_gyro(2) = sys_seq[0].second.angular_velocity.z;
        u_acc(0)  = sys_seq[0].second.linear_acceleration.x;
        u_acc(1)  = sys_seq[0].second.linear_acceleration.y;
        u_acc(2)  = sys_seq[0].second.linear_acceleration.z;

        MatrixXd Ft;
        MatrixXd Vt;

        X_state = sys_seq[0].first;
        StateCovariance  = cov_seq[0];

        q_last = sys_seq[0].first.segment<3>(3);  // last X2
        bg_last = sys_seq[0].first.segment<3>(9);  //last X4
        ba_last = sys_seq[0].first.segment<3>(12);  //last X5
        Ft = MatrixXd::Identity(stateSize, stateSize) + dt*diff_f_diff_x(q_last, u_gyro, u_acc, bg_last, ba_last);
        
        Vt = dt*diff_f_diff_n(q_last);

        X_state += dt*F_model(u_gyro, u_acc);
        if(X_state(3) > PI)  X_state(3) -= 2*PI;
        if(X_state(3) < -PI) X_state(3) += 2*PI;
        if(X_state(4) > PI)  X_state(4) -= 2*PI;
        if(X_state(4) < -PI) X_state(4) += 2*PI;
        if(X_state(5) > PI)  X_state(5) -= 2*PI;
        if(X_state(5) < -PI) X_state(5) += 2*PI;
        StateCovariance = Ft*StateCovariance*Ft.transpose() + Vt*Qt*Vt.transpose();

        //re-update for the rightframe 
        Ct = diff_g_diff_x();
        Wt = diff_g_diff_v();

        Kt_kalmanGain = StateCovariance*Ct.transpose() * (Ct*StateCovariance*Ct.transpose() + Wt*Rt*Wt.transpose()).inverse();

        VectorXd gg = g_model();
        VectorXd innovation = Z_measurement - gg;
        VectorXd innovation_t = gg;
        //Prevent innovation changing suddenly when euler from -Pi to Pi
        if(innovation(3) > 6)  innovation(3) -= 2*PI;
        if(innovation(3) < -6) innovation(3) += 2*PI;
        if(innovation(4) > 6)  innovation(4) -= 2*PI;
        if(innovation(4) < -6) innovation(4) += 2*PI;
        if(innovation(5) > 6)  innovation(5) -= 2*PI;
        if(innovation(5) < -6) innovation(5) += 2*PI;
        INNOVATION_ = innovation_t.segment<3>(3);

        X_state += Kt_kalmanGain*(innovation);
        if(X_state(3) > PI)  X_state(3) -= 2*PI;
        if(X_state(3) < -PI) X_state(3) += 2*PI;
        if(X_state(4) > PI)  X_state(4) -= 2*PI;
        if(X_state(4) < -PI) X_state(4) += 2*PI;
        if(X_state(5) > PI)  X_state(5) -= 2*PI;
        if(X_state(5) < -PI) X_state(5) += 2*PI;
        StateCovariance = StateCovariance - Kt_kalmanGain*Ct*StateCovariance;

        system_pub(sys_seq[0].second.header.stamp);  // choose to publish the repropagation or not

        re_propagate();

        #endif
    }
    
}

Quaterniond q_gt, q_gt0;
bool first_gt = true;
void gt_callback(const nav_msgs::msg::Odometry::ConstSharedPtr &msg)
{
    q_gt.w() = msg->pose.pose.orientation.w;
    q_gt.x() = msg->pose.pose.orientation.x;
    q_gt.y() = msg->pose.pose.orientation.y;
    q_gt.z() = msg->pose.pose.orientation.z;

    if(first_gt&&!first_frame_tag_odom)
    {
        first_gt = false;
        q_gt0 = q_gt;
        // q_gt0 = q_gt0.normalized();
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("ekf", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
    auto n = node;
    rclcpp::SubscriptionBase::SharedPtr s1 = n->create_subscription<sensor_msgs::msg::Imu>("~/imu", rclcpp::SensorDataQoS().keep_last(1000), imu_callback);
    rclcpp::SubscriptionBase::SharedPtr s2 = n->create_subscription<nav_msgs::msg::Odometry>("~/bodyodometry", rclcpp::SensorDataQoS().keep_last(40), vioodom_callback);
    rclcpp::SubscriptionBase::SharedPtr s4 = n->create_subscription<nav_msgs::msg::Odometry>("~/gt_", rclcpp::SensorDataQoS().keep_last(40), gt_callback);
    odom_pub = n->create_publisher<nav_msgs::msg::Odometry>("~/ekf_odom", 1000);   //freq = imu freq
    cam_odom_pub = n->create_publisher<nav_msgs::msg::Odometry>("~/cam_ekf_odom", 1000);
    acc_filtered_pub = n->create_publisher<sensor_msgs::msg::Imu>("~/acc_filtered", 1000);
    
    readEssentialParam(n, "gyro_cov", gyro_cov);
    readEssentialParam(n, "acc_cov", acc_cov);
    readEssentialParam(n, "position_cov", position_cov);
    readEssentialParam(n, "q_rp_cov", q_rp_cov);
    readEssentialParam(n, "q_yaw_cov", q_yaw_cov);
    readEssentialParam(n, "imu_trans_x", imu_trans_x);
    readEssentialParam(n, "imu_trans_y", imu_trans_y);
    readEssentialParam(n, "imu_trans_z", imu_trans_z);

    cout << "Q:" << gyro_cov << " " << acc_cov << " R: " << position_cov << " " << q_rp_cov << " " << q_yaw_cov << endl;

    std::vector<double> Rri = {1.0, 0.0, 0.0, 0.0};
    std::vector<double> tri = {0.0, 0.0, 0.0};
    readEssentialParam(n, "Rr_i", Rri);
    readEssentialParam(n, "tr_i", tri);
    Rr_i = Quaterniond(Rri.at(0), Rri.at(1), Rri.at(2), Rri.at(3)).toRotationMatrix();
    tr_i << tri.at(0), tri.at(1), tri.at(2);
    cout << "Rr_i: " << endl << Rr_i << endl;
    cout << "tr_i: " << endl << tr_i << endl;

    n->get_parameter_or<bool>("output_body_pose", output_body_pose, false);
    n->get_parameter_or<bool>("initialize_gravity_from_imu", initialize_gravity_from_imu, false);
    if (!Rr_i.allFinite() || !tr_i.allFinite() ||
        (Rr_i.transpose()*Rr_i-Matrix3d::Identity()).norm() > 1e-6 ||
        fabs(Rr_i.determinant()-1.) > 1e-6) {
        RCLCPP_FATAL(n->get_logger(), "Invalid body/IMU extrinsics");
        return 1;
    }
    if (output_body_pose)
        RCLCPP_INFO(n->get_logger(), "Publishing calibrated body pose, world linear velocity and body angular velocity");
    initsys();
    cout << "initsys" << endl;

    // cout << "======================" << endl;
    // double r = atan2(1,-100);
    // double p = asin(-0.707);
    // double y = atan2(-1, -100);   
    // cout << "r: " << r << " p: " << p << " y: " << y << endl;
    // cout << "======================" << endl;

    rclcpp::spin(node);
    odom_pub.reset();
    cam_odom_pub.reset();
    acc_filtered_pub.reset();
    rclcpp::shutdown();
}

void get_filtered_acc_gyro(Vector3d &gyro, Vector3d &acc)
{
    Vector3d q, ba, bg;
    q = X_state.segment<3>(3);
    ba = X_state.segment<3>(12);
    bg = X_state.segment<3>(9);
    // return (euler2mat(q)*(acc-ba-na));
    // return (q_gt.toRotationMatrix()*(acc-ba-na));  //false
    acc = acc - ba - na;
    gyro = gyro - bg - ng;
    // return ((acc-ba-na));
    // return ((acc-na));
    // return (euler2mat(q)*(acc));
}

void acc_f_pub(Vector3d gyro, Vector3d acc, rclcpp::Time stamp)
{
    sensor_msgs::msg::Imu Accel_filtered;
    Accel_filtered.header.frame_id = world_frame_id;
    Accel_filtered.header.stamp = stamp;
    // Accel_filtered.header.stamp = node->now();
    Vector3d raw_gyro = gyro;
    Vector3d raw_acc = acc;
    get_filtered_acc_gyro(raw_gyro, raw_acc);
    Accel_filtered.linear_acceleration.x = raw_acc[0];
    Accel_filtered.linear_acceleration.y = raw_acc[1];
    Accel_filtered.linear_acceleration.z = raw_acc[2];

    // Quaterniond q;
    // q = euler2quaternion(X_state.segment<3>(3));
    // q = q.normalized();
    // Accel_filtered.angular_velocity.x = raw_gyro[0];
    // Accel_filtered.angular_velocity.y = raw_gyro[1];
    // Accel_filtered.angular_velocity.z = raw_gyro[2];


    Quaterniond q;
    q = euler2quaternion(X_state.segment<3>(3));
    q = q.normalized();
    Accel_filtered.orientation.w = q.w();
    Accel_filtered.orientation.x = q.x();
    Accel_filtered.orientation.y = q.y();
    Accel_filtered.orientation.z = q.z();

    // cout << "q_gt0: " << quaternion2euler(q_gt0) << endl;
    // cout << "q_gt: " << quaternion2euler(q_gt) << endl;
    // cout << "q_vio: " << mat2euler(q_gt0.toRotationMatrix() * euler2quaternion(X_state.segment<3>(3)).toRotationMatrix()) << endl;
    // cout << "q_gt0*vio != q_gt: " << quaternion2euler(q_gt0 * q) << endl;   
    // cout << "q_gt0*vio*q_gt0^-1 = q_gt: " << quaternion2euler(q_gt0 * q * q_gt0.inverse()) << endl;   //q1*euler2quaternion(V)*q1.inverse()  = q1.toRotationMatrix() * V  TODO why?

    acc_filtered_pub->publish(Accel_filtered);
}
void system_pub(rclcpp::Time stamp)
{
    nav_msgs::msg::Odometry odom_fusion;
    odom_fusion.header.stamp = stamp;
    odom_fusion.header.frame_id = world_frame_id;
    // Frame is inherited from the actual localized body odometry.

    Quaterniond q;
    Eigen::Matrix3d R1, R2;
    R1 << 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0;
    R2 << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

    const Matrix3d R_world_imu = euler2mat(X_state.segment<3>(3));
    const Vector3d offset = output_body_pose ? tr_i : Vector3d(imu_trans_x, imu_trans_y, imu_trans_z);
    const Vector3d position = X_state.segment<3>(0) + R_world_imu * offset;
    Vector3d velocity = X_state.segment<3>(6);
    Vector3d angular = u_gyro;
    q = Quaterniond(R_world_imu);
    if (output_body_pose) {
        const Vector3d omega_imu = u_gyro - X_state.segment<3>(9);
        q = Quaterniond(R_world_imu * Rr_i);
        velocity += R_world_imu * omega_imu.cross(offset);
        angular = Rr_i.transpose() * omega_imu;
    }
    odom_fusion.child_frame_id = "go2_imu";
    odom_fusion.pose.pose.orientation.w = q.w();
    odom_fusion.pose.pose.orientation.x = q.x();
    odom_fusion.pose.pose.orientation.y = q.y();
    odom_fusion.pose.pose.orientation.z = q.z();
    odom_fusion.pose.pose.position.x = position.x();
    odom_fusion.pose.pose.position.y = position.y();
    odom_fusion.pose.pose.position.z = position.z();
    odom_fusion.twist.twist.linear.x = velocity.x();
    odom_fusion.twist.twist.linear.y = velocity.y();
    odom_fusion.twist.twist.linear.z = velocity.z();
    odom_fusion.twist.twist.angular.x = angular.x();
    odom_fusion.twist.twist.angular.y = angular.y();
    odom_fusion.twist.twist.angular.z = angular.z();

    odom_pub->publish(odom_fusion);

    q = Quaterniond(euler2quaternion(X_state.segment<3>(3)).toRotationMatrix()*R1*R2);
    odom_fusion.pose.pose.orientation.w = q.w();
    odom_fusion.pose.pose.orientation.x = q.x();
    odom_fusion.pose.pose.orientation.y = q.y();
    odom_fusion.pose.pose.orientation.z = q.z();

    cam_odom_pub->publish(odom_fusion);
}
void cam_system_pub(rclcpp::Time stamp)
{
    nav_msgs::msg::Odometry odom_fusion;
    odom_fusion.header.stamp = stamp;
    odom_fusion.header.frame_id = world_frame_id;
    // Frame is inherited from the actual localized body odometry.
    // odom_fusion.header.frame_id = "imu";
    // odom_fusion.pose.pose.position.x = Z_measurement(0);
    // odom_fusion.pose.pose.position.y = Z_measurement(1);
    // odom_fusion.pose.pose.position.z = Z_measurement(2);
    odom_fusion.pose.pose.position.x = Z_measurement(0);
    odom_fusion.pose.pose.position.y = Z_measurement(1);
    odom_fusion.pose.pose.position.z = Z_measurement(2);
    Quaterniond q;
    // Quaterniond q1(0.707, 0.0, 0.707, 0.0);
    // Quaterniond q2(0.707, 0.0, 0.0, -0.707);
    Eigen::Matrix3d R1, R2;
    R1 << 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0;
    R2 << 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    // Eigen::Vector3d euler(-Z_measurement.segment<3>(3)(1), -Z_measurement.segment<3>(3)(2), Z_measurement.segment<3>(3)(0));
    q = Quaterniond(euler2quaternion(Z_measurement.segment<3>(3)).toRotationMatrix()*R1*R2);
    odom_fusion.pose.pose.orientation.w = q.w();
    odom_fusion.pose.pose.orientation.x = q.x();
    odom_fusion.pose.pose.orientation.y = q.y();
    odom_fusion.pose.pose.orientation.z = q.z();
    odom_fusion.twist.twist.linear.x = Z_measurement(3);
    odom_fusion.twist.twist.linear.y = Z_measurement(4);
    odom_fusion.twist.twist.linear.z = Z_measurement(5);
    
    // odom_fusion.twist.twist.angular.x = INNOVATION_(0);
    // odom_fusion.twist.twist.angular.y = INNOVATION_(1);
    // odom_fusion.twist.twist.angular.z = INNOVATION_(2);
    Vector3d pp, qq, v, bg, ba;
    getState(pp, qq, v, bg, ba);
    odom_fusion.twist.twist.angular.x = ba(0);
    odom_fusion.twist.twist.angular.y = ba(1);///??????why work??????????//////
    odom_fusion.twist.twist.angular.z = ba(2); 
    // odom_fusion.twist.twist.angular.x = diff_time;
    // odom_fusion.twist.twist.angular.y = dt;
    // cam_odom_pub->publish(odom_fusion);
}

//process model
void initsys()
{
    //  camera position in the IMU frame = (0.05, 0.05, 0)
    // camera orientaion in the IMU frame = Quaternion(0, 1, 0, 0); w x y z, respectively
    //					   RotationMatrix << 1, 0, 0,
    //							             0, -1, 0,
    //                                       0, 0, -1;
    //set the cam2imu params
    Rc_i = Quaterniond(0, 1, 0, 0).toRotationMatrix();
    // cout << "R_cam" << endl << Rc_i << endl;
    tc_i << 0.05, 0.05, 0; 

    
    //  rigid body position in the IMU frame = (0, 0, 0.04)
    // rigid body orientaion in the IMU frame = Quaternion(1, 0, 0, 0); w x y z, respectively
    //					   RotationMatrix << 1, 0, 0,
    //						 	             0, 1, 0,
    //                                       0, 0, 1; 

    //states X [p q pdot bg ba]  [px,py,pz, wx,wy,wz, vx,vy,vz bgx,bgy,bgz bax,bay,baz]
    stateSize = 15;                                                                 // x = [p q pdot bg ba]
    stateSize_pqv = 9;                                                              // x = [p q pdot]
    measurementSize = 6;                                                            // z = [p q]
    inputSize = 6;                                                                  // u = [w a]
    X_state = VectorXd::Zero(stateSize);                                            // x 
    //velocity
    X_state(6) = 0;
    X_state(7) = 0;
    X_state(8) = 0; 
    //bias
    X_state.segment<3>(9) = bg_0;
    X_state.segment<3>(12) = ba_0;
    u_input = VectorXd::Zero(inputSize);
    Z_measurement = VectorXd::Zero(measurementSize);                                // z
    StateCovariance = MatrixXd::Identity(stateSize, stateSize);                     // sigma
    Kt_kalmanGain = MatrixXd::Identity(stateSize, measurementSize);                 // Kt
    // Ct_stateToMeasurement = MatrixXd::Identity(stateSize, measurementSize);         // Ct
    X_state_correct = X_state;
    StateCovariance_correct = StateCovariance;

    Qt = MatrixXd::Identity(inputSize, inputSize);  //6x6 input [gyro acc]covariance
    Rt = MatrixXd::Identity(measurementSize, measurementSize); //6x6 measurement [p q]covariance
    // MatrixXd temp_Rt = MatrixXd::Identity(measurementSize, measurementSize);

    // You should also tune these parameters
    // Q imu covariance matrix; Rt visual odomtry covariance matrix
    // //Rt visual odomtry covariance smaller believe measurement more
    Qt.topLeftCorner(3, 3) = gyro_cov * Qt.topLeftCorner(3, 3);
    Qt.bottomRightCorner(3, 3) = acc_cov * Qt.bottomRightCorner(3, 3);
    Rt.topLeftCorner(3, 3) = position_cov * Rt.topLeftCorner(3, 3);
    Rt.bottomRightCorner(3, 3) = q_rp_cov * Rt.bottomRightCorner(3, 3);
    Rt.bottomRightCorner(1, 1) = q_yaw_cov * Rt.bottomRightCorner(1, 1);
}

void getState(Vector3d& p, Vector3d& q, Vector3d& v, Vector3d& bg, Vector3d& ba)
{
	p = X_state.segment<3>(0);
	q = X_state.segment<3>(3);
	v = X_state.segment<3>(6);
    bg = X_state.segment<3>(9);
    ba = X_state.segment<3>(12);
}

VectorXd F_model(Vector3d gyro, Vector3d acc)
{
    // IMU is in FLU frame
    // Transform IMU frame into "world" frame whose original point is FLU's original point and the XOY plain is parallel with the ground and z axis is up
    VectorXd f(VectorXd::Zero(stateSize));
    Vector3d p, q, v, bg, ba;
    getState(p, q, v, bg, ba);
    f.segment<3>(0) = v;
    f.segment<3>(3) = w_Body2Euler(q)*(gyro-bg-ng);
    // Eigen::Vector3d gra = Eigen::Vector3d(0.0, 0.0, -1.0) * acc.norm();
    f.segment<3>(6) = gravity + euler2mat(q)*(acc-ba-na);
    f.segment<3>(9) = nbg;
    f.segment<3>(12) = nba;

    return f;
}

VectorXd g_model()
{
    VectorXd g(VectorXd::Zero(measurementSize));

    g.segment<6>(0) = X_state.segment<6>(0);

    // if(g(3) > PI)  g(3) -= 2*PI;
    // if(g(3) < -PI) g(3) += 2*PI;
    // if(g(4) > PI)  g(4) -= 2*PI;
    // if(g(4) < -PI) g(4) += 2*PI;
    // if(g(5) > PI)  g(5) -= 2*PI;
    // if(g(5) < -PI) g(5) += 2*PI;

    return g;
}

//F_model G_model Jocobian
//diff_f()/diff_x (x_t-1  ut  noise=0)   At     Ft = I+dt*At
MatrixXd diff_f_diff_x(Vector3d q_last, Vector3d gyro, Vector3d acc, Vector3d bg_last, Vector3d ba_last)
{
    double cr = cos( q_last(0));
    double sr = sin( q_last(0));
    double cp = cos( q_last(1));
    double sp = sin( q_last(1));
    double cy = cos( q_last(2));
    double sy = sin( q_last(2));
 
    // ng na = 0 nbg nba = 0
    double Ax = acc(0) - ba_last(0);
    double Ay = acc(1) - ba_last(1);
    double Az = acc(2) - ba_last(2);
    // double Wx = gyro(0) - bg_last(0);
    double Wy = gyro(1) - bg_last(1);
    double Wz = gyro(2) - bg_last(2);

    MatrixXd diff_f_diff_x_jacobian(MatrixXd::Zero(stateSize, stateSize));
    MatrixXd diff_f_diff_x_jacobian_pqv(MatrixXd::Zero(stateSize_pqv, stateSize_pqv));

    diff_f_diff_x_jacobian_pqv <<  0, 0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 1,  
      0, 0, 0, (sp*(Wy*cr - Wz*sr))/cp, (Wz*cr + Wy*sr)/(cp*cp), 0, 0, 0, 0,  
      0, 0, 0, (- Wz*cr - Wy*sr), 0, 0, 0, 0, 0, 
      0, 0, 0, (Wy*cr - Wz*sr)/cp, (sp*(Wz*cr + Wy*sr))/(cp*cp), 0, 0, 0, 0, 
      0, 0, 0, (Ay*(sr*sy + cr*cy*sp) + Az*(cr*sy - cy*sp*sr)), (Az*cp*cr*cy - Ax*cy*sp + Ay*cp*cy*sr), (Az*(cy*sr - cr*sp*sy) - Ay*(cr*cy + sp*sr*sy) - Ax*cp*sy), 0, 0, 0, 
      0, 0, 0, (- Ay*(cy*sr - cr*sp*sy) - Az*(cr*cy + sp*sr*sy)), (Az*cp*cr*sy - Ax*sp*sy + Ay*cp*sr*sy), (Az*(sr*sy + cr*cy*sp) - Ay*(cr*sy - cy*sp*sr) + Ax*cp*cy), 0, 0, 0,
      0, 0, 0, (Ay*cp*cr - Az*cp*sr), (- Ax*cp - Az*cr*sp - Ay*sp*sr), 0, 0, 0, 0 ;
    
    diff_f_diff_x_jacobian.block<9, 9>(0, 0) = diff_f_diff_x_jacobian_pqv;
    diff_f_diff_x_jacobian.block<3, 3>(3, 9) = -w_Body2Euler(q_last);
    diff_f_diff_x_jacobian.block<3, 3>(6, 12) = -euler2mat(q_last);

    return diff_f_diff_x_jacobian;

    // cp != 0 pitch != 90° !!!!!!!!!
}
//diff_f()/diff_n (x_t-1  ut  noise=0)  Ut    Vt = dt*Ut
MatrixXd diff_f_diff_n(Vector3d q_last)
{
    MatrixXd diff_f_diff_n_jacobian(MatrixXd::Zero(stateSize, inputSize));
    diff_f_diff_n_jacobian.block<3,3>(3,0) = -w_Body2Euler(q_last);
    diff_f_diff_n_jacobian.block<3,3>(6,3) = -euler2mat(q_last);

    return diff_f_diff_n_jacobian;
}
//diff_g()/diff_x  (xt~ noise=0)  Ct 
MatrixXd diff_g_diff_x()
{
    MatrixXd diff_g_diff_x_jacobian(MatrixXd::Zero(measurementSize, stateSize));
    diff_g_diff_x_jacobian.block<3,3>(0,0) = MatrixXd::Identity(3,3);
    diff_g_diff_x_jacobian.block<3,3>(3,3) = MatrixXd::Identity(3,3);

    return diff_g_diff_x_jacobian;
}
//diff_g()/diff_v  (xt~ noise=0) Wt
MatrixXd diff_g_diff_v()
{
    MatrixXd diff_g_diff_v_jacobian(MatrixXd::Identity(measurementSize, measurementSize));

    return diff_g_diff_v_jacobian;
}
