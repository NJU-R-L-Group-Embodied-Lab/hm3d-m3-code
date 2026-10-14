// Exercises the actual estimator callbacks without sensors or chassis control.
#define main ekf_node_main
#include "ekf_node_vio_timesync_with_acc_pub.cpp"
#undef main
#include <iomanip>

int main(int argc, char **argv) {
    const std::string scenario = argc > 1 ? argv[1] : "prior_settling";
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("ekf_initialization_check");
    odom_pub = node->create_publisher<nav_msgs::msg::Odometry>("/offline/odom", 1);
    cam_odom_pub = node->create_publisher<nav_msgs::msg::Odometry>("/offline/cam", 1);
    acc_filtered_pub = node->create_publisher<sensor_msgs::msg::Imu>("/offline/acc", 1);
    output_body_pose = initialize_gravity_from_imu = true;
    gyro_cov=.02; acc_cov=.5; position_cov=.01; q_rp_cov=q_yaw_cov=.001;
    Rr_i = Quaterniond(.9935718556974359,0.,-.11320321358492341,0.).toRotationMatrix();
    tr_i << -.17514363249534007,-.02329,-.0761877633040974;
    initsys();
    double max_position=0., max_velocity=0., initialized_at=-1.;
    Vector3d true_gravity(.02,-.01,-9.74);
    constexpr double step=.005;
    for (int i=0;i<24000;++i) {
        double t=i*step;
        // Stand while the prior-map pose settles, then rotate and stand again.
        double yaw=t<90.?0.:(t<96.?-(t-90.)*.48:-2.88);
        double rate=t>=90. && t<96.?-.48:0.;
        Matrix3d body=euler2mat(Vector3d(0.,0.,yaw));
        double body_x=0., body_vx=0., body_ax=0.;
        if (scenario=="translation") {
            if (t>=102. && t<104.) {body_ax=.15;body_vx=.15*(t-102.);body_x=.075*(t-102.)*(t-102.);}
            else if (t>=104. && t<110.) {body_vx=.3;body_x=.3+.3*(t-104.);}
            else if (t>=110. && t<112.) {body_ax=-.15;body_vx=.3-.15*(t-110.);body_x=2.1+.3*(t-110.)-.075*(t-110.)*(t-110.);}
            else if (t>=112.) body_x=2.4;
        }
        Matrix3d imu=body*Rr_i.transpose();
        Vector3d wb(0.,0.,rate), wi=Rr_i*wb;
        Vector3d offset=-Rr_i.transpose()*tr_i;
        Vector3d sensor_acceleration=Vector3d(body_ax,0.,0.)+body*wb.cross(wb.cross(offset));
        Vector3d measured=imu.transpose()*(sensor_acceleration-true_gravity)/9.81;
        auto sample=std::make_shared<sensor_msgs::msg::Imu>();
        sample->header.stamp=rclcpp::Time(static_cast<int64_t>((1000.+t)*1e9));
        sample->linear_acceleration.x=measured.x();
        sample->linear_acceleration.y=measured.y();
        sample->linear_acceleration.z=measured.z();
        sample->angular_velocity.x=wi.x(); sample->angular_velocity.y=wi.y(); sample->angular_velocity.z=wi.z();
        if (!(scenario=="imu_gap" && t>=.4 && t<2.)) imu_callback(sample);
        if (i%20==0) {
            auto pose=std::make_shared<nav_msgs::msg::Odometry>();
            pose->header.stamp=sample->header.stamp; pose->header.frame_id="map";
            pose->pose.pose.position.x=body_x;
            // First LiDAR registration still reflects the approximate initial pose.
            Matrix3d observed=scenario=="prior_settling" && t<1.?euler2mat(Vector3d(.01,.065,0.)):body;
            Quaterniond q(observed);
            pose->pose.pose.orientation.x=q.x(); pose->pose.pose.orientation.y=q.y();
            pose->pose.pose.orientation.z=q.z(); pose->pose.pose.orientation.w=q.w();
            if (scenario=="stale_pose" && t<3.)
                pose->header.stamp=rclcpp::Time(static_cast<int64_t>((999.5+t)*1e9));
            if (scenario=="invalid_pose" && t<2.)
                pose->pose.pose.orientation.w=std::numeric_limits<double>::quiet_NaN();
            if (scenario=="moving_start" && t<2.)
                pose->pose.pose.position.x=.2*t;
            vioodom_callback(pose);
        }
        if (!first_frame_tag_odom && initialized_at<0.) initialized_at=t;
        if (t>98. && !first_frame_imu) {
            Matrix3d rotation=euler2mat(X_state.segment<3>(3));
            Vector3d position=X_state.segment<3>(0)+rotation*tr_i;
            Vector3d velocity=X_state.segment<3>(6)+rotation*(u_gyro-X_state.segment<3>(9)).cross(tr_i);
            max_position=std::max(max_position,(position-Vector3d(body_x,0.,0.)).norm());
            max_velocity=std::max(max_velocity,(velocity-Vector3d(body_vx,0.,0.)).norm());
        }
    }
    std::cout << std::setprecision(8) << "scenario=" << scenario << " initialized_at=" << initialized_at << " gravity=" << gravity.transpose()
              << " max_stationary_position=" << max_position
              << " max_stationary_velocity=" << max_velocity << std::endl;
    bool pass=!first_frame_imu && max_position<.05 && max_velocity<.05;
    double earliest=scenario=="prior_settling"?2.:1.;
    if (scenario=="imu_gap" || scenario=="invalid_pose" || scenario=="moving_start" || scenario=="stale_pose") earliest=3.;
    pass=pass && initialized_at>=earliest;
    odom_pub.reset(); cam_odom_pub.reset(); acc_filtered_pub.reset(); node.reset();
    rclcpp::shutdown();
    return pass?0:1;
}
