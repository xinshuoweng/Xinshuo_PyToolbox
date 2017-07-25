// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu

#pragma once


#ifndef __MESH_VISUALIZATION_H_INCLUDED__
#define __MESH_VISUALIZATION_H_INCLUDED__


#include <myheader.h>

// pcl library
#include <pcl/visualization/pcl_visualizer.h>

// visualize the mesh and 3d keypoint found
boost::shared_ptr<pcl::visualization::PCLVisualizer> keypoint_mesh_visualization(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr keypoints, pcl::PointCloud<pcl::PointXYZ>::ConstPtr mesh);
boost::shared_ptr<pcl::visualization::PCLVisualizer> rgb_cloud_visualization(std::map<std::string, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> cloud_map);
boost::shared_ptr<pcl::visualization::PCLVisualizer> keypoint_line_mesh_visualization(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr keypoints, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr line, pcl::PointCloud<pcl::PointXYZ>::ConstPtr mesh);
void get_cloud_from_points(std::vector<std::vector<double>> points, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
void get_cloud_from_points(std::vector<std::vector<float>> points, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
void get_cloud_from_line(std::vector<double> line, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointXYZ pts_start = pcl::PointXYZ(0, 0, 0), uint32_t range = 10000, double interval = 1);
void get_cloud_from_line(std::vector<float> line, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointXYZ pts_start = pcl::PointXYZ(0, 0, 0), uint32_t range = 10000, double interval = 1);
void get_cloud_from_lines(std::vector<std::vector<double>> lines, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::vector<pcl::PointXYZ> pts_start, std::vector<uint32_t> range, double interval = 1);
void get_cloud_from_lines(std::vector<std::vector<float>> lines, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::vector<pcl::PointXYZ> pts_start, std::vector<uint32_t> range, double interval = 1);
#endif
