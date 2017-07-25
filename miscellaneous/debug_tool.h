// Author: Xinshuo Weng
// Email: xinshuow@andrew.cmu.edu
#pragma once

#ifndef __DEBUG_TOOL_H_INCLUDED__
#define __DEBUG_TOOL_H_INCLUDED__

#include "myheader.h"

class pts_2d_conf;
class pts_3d_conf;

/************************************** print function ********************************************/
void print_mat(cv::Mat& mat, int prec = default_precision);
void print_mat_info(cv::Mat& mat);

void print_sca(double sca);
void print_sca(float sca);
void print_sca(int sca);

void print_vec_pts3d(std::vector<cv::Point3d>& v);
void print_vec_pts3d(std::vector<cv::Point3f>& v);
void print_vec_pts3d(std::vector<cv::Point3i>& v);
void print_vec_pts2d(std::vector<cv::Point2d>& v);
void print_vec_pts2d(std::vector<cv::Point2f>& v);
void print_vec_pts2d(std::vector<cv::Point2i>& v);

void print_vec(std::vector<double>& v, int prec = default_precision);
void print_vec(std::vector<float>& v, int prec = default_precision);
void print_vec(std::vector<int>& v, int prec = default_precision);

void print_pts2d(cv::Point2d& pts);
void print_pts2d(cv::Point2f& pts);
void print_pts2d(cv::Point2i& pts);
void print_pts3d(cv::Point3d& pts);
void print_pts3d(cv::Point3f& pts);
void print_pts3d(cv::Point3i& pts);

void print_vec_pts_2d_conf(std::vector<pts_2d_conf> pts);
void print_vec_pts_3d_conf(std::vector<pts_3d_conf> pts);

/************************************** check function ********************************************/
bool CHECK_VEC_EQ(std::vector<double>& vec1, std::vector<double>& vec2);
bool CHECK_VEC_EQ(std::vector<float>& vec1, std::vector<float>& vec2);
bool CHECK_VEC_EQ(std::vector<int>& vec1, std::vector<int>& vec2);

// this function check the numeric equality between two cv::Mat. Only CV_32F and CV_64F are supported now.
// the size and dimension should be also equal for two mat while two mat could have different type
bool CHECK_MAT_EQ(cv::Mat& mat1, cv::Mat& mat2);		

bool CHECK_CV_PTS_EQ(cv::Point3d& pts1, cv::Point3d& pts2);
bool CHECK_CV_PTS_EQ(cv::Point3f& pts1, cv::Point3f& pts2);
bool CHECK_CV_PTS_EQ(cv::Point3i& pts1, cv::Point3i& pts2);
bool CHECK_CV_PTS_EQ(cv::Point2d& pts1, cv::Point2d& pts2);
bool CHECK_CV_PTS_EQ(cv::Point2f& pts1, cv::Point2f& pts2);
bool CHECK_CV_PTS_EQ(cv::Point2i& pts1, cv::Point2i& pts2);

bool CHECK_SCALAR_EQ(double sca1, double sca2);
bool CHECK_SCALAR_EQ(float sca1, float sca2);
bool CHECK_SCALAR_EQ(int sca1, int sca2);
bool CHECK_SCALAR_EQ(double sca1, int sca2);
bool CHECK_SCALAR_EQ(int sca1, double sca2);

bool CHECK_VEC_CV_PTS_EQ(std::vector<cv::Point2d>& pts1, std::vector<cv::Point2d>& pts2);
bool CHECK_VEC_CV_PTS_EQ(std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2);
bool CHECK_VEC_CV_PTS_EQ(std::vector<cv::Point2i>& pts1, std::vector<cv::Point2i>& pts2);
bool CHECK_VEC_CV_PTS_EQ(std::vector<cv::Point3d>& pts1, std::vector<cv::Point3d>& pts2);
bool CHECK_VEC_CV_PTS_EQ(std::vector<cv::Point3f>& pts1, std::vector<cv::Point3f>& pts2);
bool CHECK_VEC_CV_PTS_EQ(std::vector<cv::Point3i>& pts1, std::vector<cv::Point3i>& pts2);
#endif