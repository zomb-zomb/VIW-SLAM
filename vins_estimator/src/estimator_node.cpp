/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Original Author: Qin Tong (qintonguav@gmail.com)
 * Remodified Author: Hu(rhuag@connect.ust.hk) at HKUST, https://blog.csdn.net/iwanderu
 *******************************************************/

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "../include/estimator.h"
#include "../include/parameters.h"
#include "../include/utility/visualization.h"
#include <cmath>

#include <custom_msgs/Encoder.h>

// @param main vio operator
Estimator estimator;

// @param buffer
std::condition_variable con;
double current_time = -1;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
deque<custom_msgs::EncoderConstPtr> encoder_buf; 

int sum_of_wait = 0;

// @param mutex for buf, status value and vio processing
std::mutex m_buf;
std::mutex m_state; 
//std::mutex i_buf;   // TODO seems like useless
std::mutex m_estimator;
std::mutex e_buf; 

// @param temp status values
double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;

// @param flags
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;
double last_encoder_t = 0; 

// @brief predict status values: Ps/Vs/Rs
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// @brief update status values: Ps/Vs/Rs
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

// @brief take and align measurement from feature frames and IMU measurement

std::vector<std::tuple<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr, 
    std::vector<custom_msgs::EncoderConstPtr>>>
getMeasurements()
{
    std::vector<std::tuple<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr, 
        std::vector<custom_msgs::EncoderConstPtr>>>  measurements;

    while (true)
    {
        
        if (imu_buf.empty() || feature_buf.empty() || encoder_buf.empty())
            return measurements;

        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        
        // @param align encoder
        if (!(encoder_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec()))
        {
            ROS_WARN("wait for encoder.");
            sum_of_wait++;
            return measurements;
        }
        if (!(encoder_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec()))
        {
            ROS_WARN("throw img, only should happen at the beginning.");
            feature_buf.pop();
            continue;
        }

        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");

        // align encoder and IMU.
        std::vector<custom_msgs::EncoderConstPtr> encoders;
        if (IMUs.size() > 1)
        {
            while (encoder_buf.front()->header.stamp.toSec() < IMUs[IMUs.size()-2]->header.stamp.toSec())
            {
                encoders.emplace_back(encoder_buf.front());
                encoder_buf.pop_front();
            }
        }

        // add a new IMU data.
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        
        // compair encoder and the last IMU time, and align encoder.
        for (auto iter = encoder_buf.begin(); iter != encoder_buf.end(); iter++)
        {
            if ((*iter)->header.stamp.toSec() < IMUs.back()->header.stamp.toSec())
            {
                encoders.emplace_back(*iter);
            }
            else
            {
                encoders.emplace_back(*iter);
                break;
            }
        }
        if (encoders.empty())
            ROS_WARN("no encoder between two image.");
        measurements.emplace_back(IMUs, img_msg, encoders);
    }
    return measurements;
}

// @brief put IMU measurement in buffer and publish status values: Ps/Vs/Rs
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);// TODO useless?
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

// @brief put encoder measurement in buffer
void encoder_callback(const custom_msgs::EncoderConstPtr &encoder_msg)
{
    // ROS_DEBUG("encoder callback!");
    if (encoder_msg->header.stamp.toSec() <= last_encoder_t)
    {
        ROS_WARN("encoder message in disorder!");
    }

    last_encoder_t = encoder_msg->header.stamp.toSec();

    e_buf.lock();
    encoder_buf.push_back(encoder_msg);
    e_buf.unlock();
    con.notify_one();

    last_encoder_t = encoder_msg->header.stamp.toSec(); 
}

// @brief put feature measurement in buffer
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    ROS_DEBUG("feature callback!");
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

// @brief restart
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

// @brief put relocalization flag in buffer
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// @brief IMU pre-integration and get pre-optimized status values Ps/Vs/Rs
void processIMU(sensor_msgs::ImuConstPtr &imu_msg,
                sensor_msgs::PointCloudConstPtr &img_msg)
{
    double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
    double t = imu_msg->header.stamp.toSec();
    double img_t = img_msg->header.stamp.toSec() + estimator.td;
    if (t <= img_t)
    { 
        if (current_time < 0)
            current_time = t;
        double dt = t - current_time;
        ROS_ASSERT(dt >= 0);
        current_time = t;
        dx = imu_msg->linear_acceleration.x;
        dy = imu_msg->linear_acceleration.y;
        dz = imu_msg->linear_acceleration.z;
        rx = imu_msg->angular_velocity.x;
        ry = imu_msg->angular_velocity.y;
        rz = imu_msg->angular_velocity.z;
        estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));                    
        //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
    }
    else
    {
        double dt_1 = img_t - current_time;
        double dt_2 = t - img_t;
        current_time = img_t;
        ROS_ASSERT(dt_1 >= 0);
        ROS_ASSERT(dt_2 >= 0);
        ROS_ASSERT(dt_1 + dt_2 > 0);
        double w1 = dt_2 / (dt_1 + dt_2);
        double w2 = dt_1 / (dt_1 + dt_2);
        dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
        dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
        dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
        rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
        ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
        rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
        estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
        //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
    }
}

// @brief setReloFrame
void setReloFrame(sensor_msgs::PointCloudConstPtr &relo_msg)
{
    //sensor_msgs::PointCloudConstPtr relo_msg = NULL;
    while (!relo_buf.empty())
    {
        relo_msg = relo_buf.front();
        relo_buf.pop();
    }
    if (relo_msg != NULL)
    {
        vector<Vector3d> match_points;
        double frame_stamp = relo_msg->header.stamp.toSec();
        for (unsigned int i = 0; i < relo_msg->points.size(); i++)
        {
            Vector3d u_v_id;
            u_v_id.x() = relo_msg->points[i].x;
            u_v_id.y() = relo_msg->points[i].y;
            u_v_id.z() = relo_msg->points[i].z;
            match_points.push_back(u_v_id);
        }
        Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
        Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
        Matrix3d relo_r = relo_q.toRotationMatrix();
        int frame_index;
        frame_index = relo_msg->channels[0].values[7];
        estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
    }    
}

// @brief main vio function, including initialization and optimization
void processVIO(sensor_msgs::PointCloudConstPtr& img_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
    for (unsigned int i = 0; i < img_msg->points.size(); i++)
    {
        int v = img_msg->channels[0].values[i] + 0.5;
        int feature_id = v / NUM_OF_CAM;
        int camera_id = v % NUM_OF_CAM;
        double x = img_msg->points[i].x;
        double y = img_msg->points[i].y;
        double z = img_msg->points[i].z;
        double p_u = img_msg->channels[1].values[i];
        double p_v = img_msg->channels[2].values[i];
        double velocity_x = img_msg->channels[3].values[i];
        double velocity_y = img_msg->channels[4].values[i];
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    estimator.processImage(image, img_msg->header);
}

// @brief visualization
void visualize(sensor_msgs::PointCloudConstPtr &relo_msg,std_msgs::Header &header)
{
    pubOdometry(estimator, header);
    pubKeyPoses(estimator, header);
    pubCameraPose(estimator, header);
    pubPointCloud(estimator, header);
    pubTF(estimator, header);
    pubKeyframe(estimator);
    if (relo_msg != NULL)
        pubRelocalization(estimator);
    //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
}

// @brief find the closest encoder velocity around the imu message
Eigen::Vector3d findClosetEncoderVelocity(double t, 
    const std::vector<std::pair<double, Eigen::Vector3d>> &encoder_velocities)
{
    double t_1 = 0, t_2 = 0;
    Eigen::Vector3d encoder_velocity;
    // the first encoder velocity
    if (!encoder_velocities.empty())
    {
        encoder_velocity = encoder_velocities[0].second;
    }
    else
    {
        encoder_velocity = estimator.Vs[WINDOW_SIZE];
    }

    std::pair<double, Eigen::Vector3d> enc_vel_0, enc_vel_1;

    // find the encoder velocity that is closest to the time.
    for (auto &enc_vel : encoder_velocities)
    {
        if (enc_vel.first <= t)
        {
            t_1 = enc_vel.first;
            enc_vel_0 = enc_vel;
        }
        else
        {
            t_2 = enc_vel.first;
            enc_vel_1 = enc_vel;
            break;
        }
    }

    // interpolate encoder velocity
    if (t_1 > 0 && t_2 > 0)
    {
    
        double dt_1 = t_2 - t;
        double dt_2 = t - t_1;
        ROS_ASSERT(dt_1 >= 0);
        ROS_ASSERT(dt_2 >= 0);
        ROS_ASSERT(dt_1 + dt_2 > 0);
        double w1 = dt_2 / (dt_1 + dt_2);
        double w2 = dt_1 / (dt_1 + dt_2);
        encoder_velocity = w1 * enc_vel_0.second + w2 * enc_vel_1.second;
    }
    return encoder_velocity;
}

// TODO: need to complete
// @brief main vio function, including initialization and optimization
void processMeasurement(std::vector<std::tuple<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr, 
            std::vector<custom_msgs::EncoderConstPtr>>> & measurements)
{
    ROS_DEBUG("processing measurement");
    for (auto &measurement : measurements)
    {
        auto img_msg = std::get<1>(measurement);

        double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0, vx = 0, vy = 0, vz = 0;
        
        // get encoder velocities
        auto encoder_measurement = std::get<2>(measurement);
        std::vector<std::pair<double, Eigen::Vector3d>> encoder_velocities;
        for (size_t i = 1; i < std::get<2>(measurement).size(); i++)
        {
            auto begin_encoder_msg = std::get<2>(measurement)[i-1];
            auto end_encoder_msg = std::get<2>(measurement)[i];
            double dt = end_encoder_msg->header.stamp.toSec() - begin_encoder_msg->header.stamp.toSec();
            double enc_vel_left = (double)(end_encoder_msg->left_encoder - begin_encoder_msg->left_encoder) / ENC_RESOLUTION * M_PI * LEFT_D / dt;
            double enc_vel_right = (double)(end_encoder_msg->right_encoder - begin_encoder_msg->right_encoder) / ENC_RESOLUTION * M_PI * RIGHT_D / dt;
            double enc_v = 0.5 * (enc_vel_left + enc_vel_right);
            double enc_omega = (enc_vel_right - enc_vel_left) / WHEELBASE; 

            Eigen::Vector3d tmp_enc_vel(enc_v, 0, 0);
            Eigen::AngleAxisd tmp_rot_vec(enc_omega * dt, Eigen::Vector3d::UnitY());

            Eigen::Vector3d enc_vel(tmp_enc_vel);  
            double timestamp = 0.5 * (begin_encoder_msg->header.stamp.toSec() + end_encoder_msg->header.stamp.toSec());
            encoder_velocities.emplace_back(timestamp, enc_vel);
        }         

        // get status value:Rs/Vs/Ps
        // for (auto &imu_msg : std::get<0>(measurement))
        //     processIMU(imu_msg, img_msg, );
        for (auto &imu_msg : std::get<0>(measurement))
        {
            double t = imu_msg->header.stamp.toSec();
            Eigen::Vector3d encoder_velocity = findClosetEncoderVelocity(t, encoder_velocities);
            double img_t = img_msg->header.stamp.toSec() + estimator.td;
            if (t <= img_t)
            { 
                if (current_time < 0)
                    current_time = t;
                double dt = t - current_time;
                ROS_ASSERT(dt >= 0);
                current_time = t;
                dx = imu_msg->linear_acceleration.x;
                dy = imu_msg->linear_acceleration.y;
                dz = imu_msg->linear_acceleration.z;
                rx = imu_msg->angular_velocity.x;
                ry = imu_msg->angular_velocity.y;
                rz = imu_msg->angular_velocity.z;
                vx = encoder_velocity.x();
                vy = encoder_velocity.y();
                vz = encoder_velocity.z();
                estimator.processIMUEncoder(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz), Vector3d(vx, vy, vz));
            }
            else
            {
                double dt_1 = img_t - current_time;
                double dt_2 = t - img_t;
                current_time = img_t;
                ROS_ASSERT(dt_1 >= 0);
                ROS_ASSERT(dt_2 >= 0);
                ROS_ASSERT(dt_1 + dt_2 > 0);
                double w1 = dt_2 / (dt_1 + dt_2);
                double w2 = dt_1 / (dt_1 + dt_2);
                dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                vx = w1 * vx + w2 * encoder_velocity.x();
                vy = w1 * vy + w2 * encoder_velocity.y();
                vz = w1 * vz + w2 * encoder_velocity.z();
                estimator.processIMUEncoder(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz), Vector3d(vx, vy, vz));
            }
            
        }
        encoder_velocities.clear(); // 清空内存

        // set relocalization frame
        sensor_msgs::PointCloudConstPtr relo_msg = NULL;
        setReloFrame(relo_msg);
        ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

        // main function for vio
        TicToc t_s;
        processVIO(img_msg);

        double whole_t = t_s.toc();
        printStatistics(estimator, whole_t);
        std_msgs::Header header = img_msg->header;
        header.frame_id = "world";

        // show in rviz
        visualize(relo_msg, header);
    }
}

// @brief main vio function, including initialization and optimization
// thread: visual-inertial odometry
void process()
{
    while (true)
    {
        // get measurement in buf and make them aligned
        std::vector<std::tuple<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr, 
            std::vector<custom_msgs::EncoderConstPtr>>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);        
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();

        // main function of vio
        m_estimator.lock();
        processMeasurement(measurements);
        m_estimator.unlock();

        // update status value Rs/Ps/Vs
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

// @brief main function 
int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_encoder = n.subscribe(ENCODER_TOPIC, 2000, encoder_callback, ros::TransportHints().tcpNoDelay()); // subsribe encoder
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
