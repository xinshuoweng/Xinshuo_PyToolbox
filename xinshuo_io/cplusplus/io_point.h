
// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu


#ifndef __IO_POINT_H_INCLUDED__
#define __IO_POINT_H_INCLUDED__


#include "myheader.h"

class mycamera;
class pts_2d_conf;

// save a set of points to path
void save_point(const char* path, std::vector<cv::Point2d>& pts_src);

// save a set of points to path and corresponding confidence
void save_points_with_conf(const char* folder_path, const char* file_name, std::vector<pts_2d_conf>& pts_src);

// save a set of points for multiple view to path
// frame denote the frame number which will be saved
void save_points_with_conf_multiview(std::string path, int frame, std::map<std::string, std::vector<pts_2d_conf>>& pts_src);

// load a set of points from path and corresponding confidence
void load_point(const char* path, std::vector<cv::Point2d>& pts_dst, std::vector<double>& conf_vec);

// load a set of points from path from one view and corresponding confidence
// the resize factor is to control the resize of the original image
// the bigger the resize factor is, the larger coordinate will be saved. In other words, the scale of the pts will be enlarged
// one could also control how many points to load. By default all points will be loaded
void load_points_with_conf(const char* path, std::vector<pts_2d_conf>& pts_dst, double resize_factor = 1, int number_pts_load = INT_MAX);	// the resize factor is for downsample prediction

// load a set of points from path from multiple view and corresponding confidence
// the resize factor is to control the resize of the original image
// the bigger the resize factor is, the larger coordinate will be saved. In other words, the scale of the pts will be enlarged
// one could also control how many points to load. By default all points will be loaded
// string in subfolder_list contains the name of all camera id
// frame denote the frame number which will be loaded
void load_points_with_conf_multiview(std::string path, std::vector<mycamera> camera_cluster, int frame, std::map<std::string, std::vector<pts_2d_conf>>& pts_dst, double resize_factor = 1, int number_pts_load = INT_MAX);

void fileparts(std::string str, std::string separator, std::string *path, std::string *filename, std::string *extension);

#endif