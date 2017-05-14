#pragma once
// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu

#ifndef __TYPE_CONVERSION_H_INCLUDED__
#define __TYPE_CONVERSION_H_INCLUDED__

#include "myheader.h"

#include <pcl/point_types.h>

class pts_2d_conf;
class pts_3d_conf;


// intra-class conversion
std::vector<cv::Point2d> float2double_vec_pts2d(std::vector<cv::Point2f>& pts_src);
std::vector<cv::Point2f> double2float_vec_pts2d(std::vector<cv::Point2d>& pts_src);
std::vector<cv::Point2d> int2double_vec_pts2d(std::vector<cv::Point2i>& pts_src);
std::vector<cv::Point2i> double2int_vec_pts2d(std::vector<cv::Point2d>& pts_src);


std::vector<cv::Point3d> float2double_vec_pts3d(std::vector<cv::Point3f>& pts_src);
std::vector<cv::Point3f> double2float_vec_pts3d(std::vector<cv::Point3d>& pts_src);
std::vector<cv::Point3d> int2double_vec_pts3d(std::vector<cv::Point3i>& pts_src);
std::vector<cv::Point3i> double2int_vec_pts3d(std::vector<cv::Point3d>& pts_src);


cv::Point2d float2double_pts2d(cv::Point2f& pts_src);
cv::Point2f double2float_pts2d(cv::Point2d& pts_src);
cv::Point2d int2double_pts2d(cv::Point2i& pts_src);
cv::Point2i double2int_pts2d(cv::Point2d& pts_src);


cv::Point3d float2double_pts3d(cv::Point3f& pts_src);
cv::Point3f double2float_pts3d(cv::Point3d& pts_src);
cv::Point3d int2double_pts3d(cv::Point3i& pts_src);
cv::Point3i double2int_pts3d(cv::Point3d& pts_src);



__declspec(dllexport) std::vector<double> float2double_vec(std::vector<float>& pts_src);
__declspec(dllexport) std::vector<float> double2float_vec(std::vector<double>& pts_src);
std::vector<double> int2double_vec(std::vector<int>& pts_src);
std::vector<int> double2int_vec(std::vector<double>& pts_src);




// inter-class conversion
__declspec(dllexport) cv::Point2d pcl2cv_pts2d(pcl::PointXY& pts_src);
__declspec(dllexport) cv::Point3d pcl2cv_pts3d(pcl::PointXYZ& pts_src);
pcl::PointXY cv2pcl_pts2d(cv::Point2d& pts_src);
pcl::PointXYZ cv2pcl_pts3d(cv::Point3d& pts_src);


std::vector<cv::Point2d> conf2cv_vec_pts2d(std::vector<pts_2d_conf>& pts_src);
std::vector<cv::Point3d> conf2cv_vec_pts3d(std::vector<pts_3d_conf>& pts_src);
std::vector<pts_2d_conf> cv2conf_vec_pts2d(std::vector<cv::Point2d>& pts_src);
std::vector<pts_3d_conf> cv2conf_vec_pts3d(std::vector<cv::Point3d>& pts_src);


cv::Point3d vec2cv_pts3d(std::vector<double>& pts_src);
cv::Point2d vec2cv_pts2d(std::vector<double>& pts_src);
std::vector<double> cv2vec_pts2d(cv::Point2d& pts_src);
__declspec(dllexport) std::vector<double> cv2vec_pts3d(cv::Point3d& pts_src);

std::vector<double> mat2vec(cv::Mat pts_src);

#endif