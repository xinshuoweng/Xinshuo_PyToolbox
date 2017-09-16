#pragma once

// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu
#ifndef __PTS_3D_CONF_H_INCLUDED__
#define __PTS_3D_CONF_H_INCLUDED__

#include <myheader.h>
#include <computer_vision/geometry/pts_2d_conf.h>

#define	default_z	0.0

class pts_3d_conf : public pts_2d_conf{
public:
	double z;

	pts_3d_conf()																: pts_2d_conf(), z(default_z) {}
	pts_3d_conf(double tx, double ty, double tz, double tconf)					: pts_2d_conf(tx, ty, tconf), z(tz) {}
	pts_3d_conf(double tx, double ty, double tz)								: pts_2d_conf(tx, ty), z(tz) {}
	pts_3d_conf(float tx, float ty, float tz)									: pts_2d_conf(tx, ty), z(tz) {}
	pts_3d_conf(float tx, float ty, float tz, double tconf)						: pts_2d_conf(tx, ty, tconf), z(tz) {}
	pts_3d_conf(int tx, int ty, int tz)											: pts_2d_conf(tx, ty), z(tz) {}
	pts_3d_conf(int tx, int ty, int tz, double tconf)							: pts_2d_conf(tx, ty, tconf), z(tz) {}
	pts_3d_conf(cv::Point3i pts)												: pts_2d_conf(pts.x, pts.y, default_conf), z(double(pts.z)) {}
	pts_3d_conf(cv::Point3d pts)												: pts_2d_conf(pts.x, pts.y, default_conf), z(double(pts.z)) {}
	pts_3d_conf(cv::Point3f pts)												: pts_2d_conf(pts.x, pts.y, default_conf), z(double(pts.z)) {}
	pts_3d_conf(cv::Point3i pts, double tconf)									: pts_2d_conf(pts.x, pts.y, tconf), z(double(pts.z)) {}
	pts_3d_conf(cv::Point3d pts, double tconf)									: pts_2d_conf(pts.x, pts.y, tconf), z(double(pts.z)) {}
	pts_3d_conf(cv::Point3f pts, double tconf)									: pts_2d_conf(pts.x, pts.y, tconf), z(double(pts.z)) {}

	cv::Point3d convert_to_point3d();
	std::vector<double> convert_to_pts_vec();
	std::vector<double> convert_to_pts_vec_conf();
	void print(int prec = default_precision);
};


#endif
