// Author: Xinshuo Weng
// Email: xinshuow@andrew.cmu.edu
#ifndef __CAMERA_GEOMETRY_H_INCLUDED__
#define __CAMERA_GEOMETRY_H_INCLUDED__


#include <myheader.h>


class mycamera;
class pts_2d_conf;
class pts_3d_conf;

/*************************************************************************************************** multi-view processing ********************************************************************************************************/
// this function takes a single point from multiple view and do optimization (ransac and two view triangulation) and then project the point back to 2d.
void multiview_optimization_single_point(std::map<std::string, pts_2d_conf>& pts_src, std::vector<mycamera>& camera_cluster, std::map<std::string, pts_2d_conf>& pts_2d, const bool consider_dist = default_consider_dist, const int num_ransac = default_ransac);

// this function takes a set of points from multiple view and do optimization (ransac and two view triangulation) and then project all points back to 2d.
void multiview_optimization_multiple_points(std::map<std::string, std::vector<pts_2d_conf>>& pts_src, std::vector<mycamera>& camera_cluster, std::map<std::string, std::vector<pts_2d_conf>>& pts_2d, const bool consider_dist = default_consider_dist, const int num_ransac = default_ransac);

// this function receive two sets of correspondent points and their correspondent camera model, and then propagate these points to all other cameras given all other camera paramters
// input pts_src1, pts_src2 needs to be std::vector of 2d points with double type
// input camera model must be following the class of my camera
// output pts_dst must be std::vector of std::vector of points, which has the same number of std::vector of points as number of cameras in camera_cluster. In each std::vector of points, the dimension is Nx2, N is the number of points from source camera
// if distorted is true, then it will run undistortion inside the propagation
void triangulation_from_two_views(std::vector<pts_2d_conf>& pts_src1, std::vector<pts_2d_conf>& pts_src2, mycamera& camera1, mycamera& camera2, std::vector<pts_3d_conf>& pts_dst, const bool consider_dist = default_consider_dist);
void triangulation_from_two_views(std::vector<cv::Point2d>& pts_src1, std::vector<cv::Point2d>& pts_src2, mycamera& camera1, mycamera& camera2, std::vector<cv::Point3d>& pts_dst, const bool consider_dist = default_consider_dist);

// this function takes a single 2d point and its corresponding camera and also all camera parameter from other views
// then return the epipolar line for all other views
// each 2d line is denoted by using ax + by + c = 0
void epipolar_2d_lines_from_anchor_point(pts_2d_conf& pts_anchor, mycamera& camera_anchor, std::vector<mycamera>& camera_cluster, std::map<std::string, std::vector<double>>& line, const bool consider_dist = default_consider_dist);


// support multiple points
// propagate the 3D point to all 2D image
// camera with distortion could also be considered
void multi_view_projection(std::vector<cv::Point3d>& pts_src, std::vector<mycamera>& camera_cluster, std::map<std::string, std::vector<cv::Point2d>>& pts_dst, const bool consider_dist = default_consider_dist);

// support single point
// camera with distortion could also be considered
void multi_view_projection(cv::Point3d& pts_src, std::vector<mycamera>& camera_cluster, std::map<std::string, cv::Point2d>& pts_dst, const bool consider_dist = default_consider_dist, const bool debug_mode = DEBUG_MODE);





// calculate total pixel error when projecting 3d points to 2d for all view
// support multiple points
// confidence supported	
// camera with distortion could also be considered
//double calculate_projection_error(std::vector<cv::Point3d>& pts_src, std::vector<mycamera>& camera_cluster, std::map<std::string, std::vector<pts_2d_conf>>& pts_dst, const bool consider_dist = default_consider_dist);

// calculate total pixel error when projecting 3d points to 2d for all view
// support single point
// confidence supported	
// camera with distortion could also be considered
double calculate_projection_error(cv::Point3d& pts_src, std::vector<mycamera>& camera_cluster, std::map<std::string, pts_2d_conf>& pts_2d_given, const bool consider_dist = default_consider_dist);





// this function takes a single 2d point from multiple view, then using 2 view triangulation and ransac to find the best 3d point vis optimization
// the optimization is done using calculate_projection_error
// confidence supported	
// camera with distortion could also be considered
void multiview_ransac_single_point(std::map<std::string, pts_2d_conf>& pts_2d_given, std::vector<mycamera>& camera_cluster, cv::Point3d& pts_dst, const bool consider_dist = default_consider_dist, const int num_ransac = default_ransac);

// this function takes a set of 2d points from multiple view, then using 2 view triangulation and ransac to find a set of best 3d point vis optimization
// the optimization is done using calculate_projection_error
// confidence supported	
// camera with distortion could also be considered
void multiview_ransac_multiple_points(std::map<std::string, std::vector<pts_2d_conf>>& pts_2d_given, std::vector<mycamera>& camera_cluster, std::vector<cv::Point3d>& pts_dst, const bool consider_dist = default_consider_dist, const int num_ransac = default_ransac);


/****************************************************************************************************** miscellaneous ***********************************************************************************************************/
// this function takes a 2d single point with confidence from multiple view as input
// and return two weighted ranom camera id based on confidence
void get_two_random_camera_id(std::map<std::string, pts_2d_conf>& pts_2d_given, std::vector<int>& view_selected, int seed = SEED);

// this function takes a single 2d point and its corresponding camera and then find the ray (a 3d starting point and a line for direction) in 3d space via back projection
void get_3d_ray(pts_2d_conf& pts_2d, mycamera& mycamera, cv::Point3d& C, std::vector<double>& ray, const bool consider_dist = default_consider_dist);

// this function takes a single 2d point and its corresponding camera and then return the undistorted point
void undistort_single_point(pts_2d_conf& pts_src, mycamera& camera, pts_2d_conf& pts_dst, const bool consider_dist = default_consider_dist);

// this function takes a set of 2d points from 1 view and then return a set of undistorted 2d points
void undistort_single_point(std::vector<pts_2d_conf>& pts_src, mycamera& camera, std::vector<pts_2d_conf>& pts_dst, const bool consider_dist = default_consider_dist);

#endif
