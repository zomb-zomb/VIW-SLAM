#include <ros/ros.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <set>

#include "../include/feature_tracker.h"
#include "../include/parameters.h"
#include "../include/tic_toc.h"

#define SHOW_UNDISTORTION 0

using std::vector;
using std::set;

ros::Publisher pub_img, pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];

bool first_image_flag = true;
double first_image_time;
double last_image_time = 0;
int pub_count = 1;
bool init_pub = 0;

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    // First image processing
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = first_image_time;
        return;
    }
    // Detect unstable camera stream
    if(img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("Unstable camera stream detected. Resetting feature tracker.");
        first_image_flag = true;
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }

    // Renew time stamp
    last_image_time = img_msg->header.stamp.toSec();

    // Frequency control
    if(round(1.0 * pub_count / (last_image_time - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // Reset the frequency control
        if(abs(1.0 * pub_count / (last_image_time - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;
    ROS_DEBUG("PUB_THIS_FRAME: %d", PUB_THIS_FRAME);
    // Convert ROS image to OpenCV image
    cv_bridge::CvImageConstPtr ptr;
    if(img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    cv::Mat show_img = ptr->image;
    TicToc t_r;

    // Read image and track features
    for(int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("Processing camera %d", i);
        if(i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        else
        {
            if(EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }
#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif       
    }

    // Renew global feature ids
    for(unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;        
    }

    // Publish features
    if (PUB_THIS_FRAME)
    {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);                    
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                    
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("Publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        
        // Skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);

        // Show track
        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    
                    // //Draw speed line
                    // Eigen::Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    // Eigen::Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    // Eigen::Vector3d tmp_prev_un_pts;
                    // tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    // tmp_prev_un_pts.z() = 1;
                    // Eigen::Vector2d tmp_prev_uv;
                    // trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    // cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);                     
                }
            }
            // cv::imshow("vis", stereo_img);
            // cv::waitKey(5);            
            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %.2fms", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    ROS_INFO("Starting feature tracker node");    
    for(int i = 0; i < NUM_OF_CAM; i++)
    {
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);
    }

    if(FISHEYE)
    {
        for(int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_ERROR("Fisheye mask is empty");
                ROS_BREAK();
            }
            else
                ROS_INFO("Fisheye mask is loaded");
        }
    }

    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);
    
    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);

    ros::spin();
    return 0;
}

