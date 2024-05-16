#include <ros/ros.h>
#include <opencv2/opencv.hpp>

#include "../include/feature_tracker.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "../include/tic_toc.h"
#include "../include/parameters.h"

using std::vector;
using std::pair;

int FeatureTracker::n_id = 0;
FeatureTracker::FeatureTracker(){}


bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for(int i = 0; i < int(v.size()); i++)
    {
        if(status[i])
            v[j++] = v[i];
    }
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const std::string &calib_file)
{
    ROS_INFO("Reading intrinsic parameters of camera from %s", calib_file.c_str());
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const std::string &name)
{
    cv::Mat undistorted_img(ROW + 600, COL + 600, CV_8UC3, cv::Scalar(0));
    vector<Eigen::Vector2d> distorted_pts, undistorted_pts;
    for(int i = 0; i < COL; i++)
    {
        for(int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d p(i, j);
            Eigen::Vector3d p_u;
            m_camera->liftProjective(p, p_u);
            distorted_pts.push_back(p);
            undistorted_pts.push_back(Eigen::Vector2d(p_u.x() / p_u.z(), p_u.y() / p_u.z()));
        }
    }
    for(int i = 0; i < int(distorted_pts.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = distorted_pts[i].x() *  FOCAL_LENGTH + COL / 2.0;
        pp.at<float>(1, 0) = distorted_pts[i].y() * FOCAL_LENGTH + ROW / 2.0;
        pp.at<float>(2, 0) = 1;
        if(pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < undistorted_img.rows
            && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < undistorted_img.cols)
        {
            undistorted_img.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distorted_pts[i].y(), distorted_pts[i].x());
        }
    }
    cv::imshow(name, undistorted_img);
    cv::waitKey(0);    
}

void FeatureTracker::equalize(const cv::Mat &img, cv::Mat &img_out)
{
    if(EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        TicToc t_c;
        clahe->apply(img, img_out);
        ROS_DEBUG("CLAHE costs: %f", t_c.toc());
    }
    else
        img_out = img;
}

void FeatureTracker::flowTrack()
{
    if(cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for(int i = 0; i < int(forw_pts.size()); i++)
        {
            if(status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        }
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(pts_velocity, status);
        ROS_DEBUG("Temporal optical flow costs: %fms", t_o.toc());
    }
}

void FeatureTracker::trackNew()
{
    if(PUB_THIS_FRAME)
    {
        rejectWithF();
        ROS_DEBUG("Set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("Set mask costs: %fms", t_m.toc());
        
        ROS_DEBUG("Detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if(n_max_cnt > 0)
        {
            if(mask.empty())
                ROS_INFO("Mask is empty");
            if(mask.type() != CV_8UC1)
                ROS_INFO("Mask type is wrong");
            if(mask.size() != forw_img.size())
                ROS_INFO("Mask size is wrong");
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("Detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("Add new feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("Select features costs: %fms", t_a.toc());
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    equalize(_img, img);

    if(forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();

    flowTrack();

    for(auto &n : track_cnt)
        n++;
    
    trackNew();

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    if(forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        // Use undistorted point matches to compute Fundamental matrix
        for(std::size_t i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            // Get undistorted points in normailized plane
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }
        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());        
    }
}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

    // Prefer to keep features that are tracked for a long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for(std::size_t i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.emplace_back(track_cnt[i], std::make_pair(forw_pts[i], ids[i]));
    
    std::sort(cnt_pts_id.begin(), cnt_pts_id.end(), 
                [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
                {
                    return a.first > b.first;
                });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for(auto &it : cnt_pts_id)
    {
        if(mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::undistortedPoints()
{
    ROS_DEBUG("Undistorted points begins");
    TicToc t_u;
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    for(std::size_t i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map[ids[i]] = cv::Point2f(b.x() / b.z(), b.y() / b.z());
    }
    // Calculate the velocity of tracked features
    if(!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for(std::size_t i = 0; i < cur_pts.size(); i++)
        {
            if(ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if(it != prev_un_pts_map.end())
                {
                    double dx = (cur_un_pts[i].x - it->second.x) / dt;
                    double dy = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(dx, dy));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }    
    // prev_un_pts_map = cur_un_pts_map;
    prev_un_pts_map.swap(cur_un_pts_map);

    ROS_DEBUG("Undistorted points costs: %fms", t_u.toc());
}