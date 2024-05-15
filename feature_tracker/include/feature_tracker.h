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
    
    /**
     * @brief Read cameras' parameters
     * @param Input const string &calib_file
     * @return nothing
     */
    void readIntrinsicParameter(const std::string &calib_file);

    /**
     * @brief Main process of feature tracking
     *
     * @param Input const cv::Mat &_img
     * @param Input double _cur_time
     * @return nothing
     */
    void readImage(const cv::Mat &_img, double _cur_time);

    /**
     * @brief Equalize the input image
     *
     * @param Input const cv::Mat &img
     * @param IntputOutput cv::Mat &img_out
     * @return nothing
     */
    void equalize(const cv::Mat &img, cv::Mat &img_out);

    /**
     * @brief Track existing features and delete failures
     * @return nothing
     */
    void flowTrack();

    /**
     * @brief Track new features in forward image
     * @return nothing
     */
    void trackNew();

    /**
     * @brief Use Fundamental matrix to delete outliers
     * @return nothing
     */
    void rejectWithF();

    /**
     * @brief Rank features according to their existing times and setMask for high rank features
     * to avoid crowded layout of features
     */    
    void setMask();

    /**
     * @brief Add new features to the feature vector
     * @return nothing
     */
    void addPoints();

    /**
     * @brief Get undistorted normalized coordinates of features and calculate the velocity of features
     * @return nothing
     */
    void undistortedPoints();

    /**
     * @brief Show the undistorted image
     * @param Input const std::string &name
     * @return nothing
     */
    void showUndistortion(const std::string &name);

    /**
     * @brief Update ID for features
     * @return bool
     */
    bool updateID(unsigned int i);


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