#ifndef _FEATURE_TRACKER_H_
#define _FEATURE_TRACKER_H_

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

class FeatureTracker
{
private:
    /* data */
public:
    FeatureTracker(/* args */);
    ~FeatureTracker();

    cv::Mat mask;
    cv::Mat fisheye_mask;

    cv::Mat prev_img, cur_img, forw_img;
    std::vector<cv::Point2f> n_pts;
    std::vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    std::vector<cv::Point2f> prev_un_pts, cur_un_pts;
    std::vector<cv::Point2f> pts_velocity;
    std::vector<int> ids;
    std::vector<int> track_cnt;
    std::map<int, cv::Point2f> cur_un_pts_map;
    std::map<int, cv::Point2f> prev_un_pts_map;

    camodocal::CameraPtr m_camera;

    double cur_time;
    double prev_time;

    static int n_id;

};




#endif