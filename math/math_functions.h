#pragma once
// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu

#ifndef __MATH_FUNCTIONS_H_INCLUDED__
#define __MATH_FUNCTIONS_H_INCLUDED__

#include <myheader.h>

// pcl library
#include <pcl/point_types.h>

// Doc: 
//		all 2d lines in this file are denoted by using three variables: ax + by + c = 0. There are only two degree of freedom though.
//		all 3d planes in this file are denoted by using three variables: ax + by + cz + d = 0. There are only three degree of freedom though.


/*********************************************** I/O ******************************************************/
void read_matrix(FILE *califp, cv::Mat& m);

/*********************************************** basics ******************************************************/
double get_median(std::vector<double>& a);											// get the median from an array or list by sorting

// this function generates a set of random number based on weights provided
// the default size of random number generated is 1
// the seed is fixed by default for debug
// the output is a set of integer index for reference as index for further manipulation
// this weighting is strict, which means it's impossible to choose the item which has 0 weight
// this function assumes all of the weights are >= 0
// the returned bool value is to show if all of the weights are zero
// the samples must be pre-allocated, then the size of the samples is equal to the size required
bool generate_weighted_randomization(std::vector<double>& weights, std::vector<int>& samples, int seed = SEED);

// this function generates a set of random number with equal probability
// the default size of random number generated is 1
// the seed is fixed by default for debug
// the output is a set of integer index for reference as index for further manipulation
std::vector<int> random_sample(std::vector<double>& samples, int size = 1, int seed = SEED);

/*********************************************** algebra ******************************************************/
double l2_norm(std::vector<double>& vec);
double l2_norm(std::vector<float>& vec);
std::vector<double> cross(std::vector<double>& a, std::vector<double>& b);			// return the cross product of two vectors
std::vector<float> cross(std::vector<float>& a, std::vector<float>& b);				// return the cross product of two vectors
double inner(std::vector<double>& a, std::vector<double>& b);						// return the inner product of two vectors
float inner(std::vector<float>& a, std::vector<float>& b);							// return the inner product of two vectors


/*********************************************** geometry ******************************************************/
bool inside_polygon(cv::Mat &M, pcl::PointXYZ& p, std::vector<pcl::PointXY>& m, int s, int e);
void get_2d_line(pcl::PointXY& a, pcl::PointXY& b, cv::Mat& line);						// get line from two vectors. The first two dimensions is the direction of the line
void get_2d_line(cv::Point2d& a, cv::Point2d& b, std::vector<double>& line);			// get line from two vectors. The first two dimensions is the direction of the line
std::vector<double> normalize_line_plane(std::vector<double>& src);						// normalize the 2d or 3d line or plane with fixing the direction vector of the line is 1, and also set a as positive value
double get_x_from_2d_line(std::vector<double>& line, double y);							// given a 2d line and one coordinate, compute the another coordinate
double get_y_from_2d_line(std::vector<double>& line, double x);							// given a 2d line and one coordinate, compute the another coordinate

// this function calculate the intersection point given two 2d lines
void get_intersection_pts_from_2d_lines(std::vector<double>& line1, std::vector<double>& line2, cv::Point2d& pts_dst);

// given a 2d point and a 2d line, find the projected 2d point along that line
void get_projected_pts_on_2d_line(cv::Point2d& pts_src, std::vector<double>& line, cv::Point2d& pts_dst);

// when three points are in the same line, return 0
// the unit coordinate for the plane ax + by + cz + d = 0
double get_3d_plane(cv::Point3d& a, cv::Point3d& b, cv::Point3d& c, std::vector<double>& p);
double get_3d_plane(pcl::PointXYZ& a, pcl::PointXYZ& b, pcl::PointXYZ& c, std::vector<double>& p);

// check is a 3d point is inside a 3d triangle space
bool point_triangle_test_3d(std::vector<double>& pts, std::vector<double>& tri_a, std::vector<double>& tri_b, std::vector<double>& tri_c);

/*********************************************** algorithm ******************************************************/
void mean_shift(std::vector<std::vector<double>>& pts_estimate, pcl::PointXYZ& out);	// mean shift for a single keypoint in all cameras


// deprecated
//bool point_triangle_test_3d(std::vector<double>& pts, std::vector<double>& tri_a, std::vector<double>& tri_b, std::vector<double>& tri_c, int debug_flag);

#endif
