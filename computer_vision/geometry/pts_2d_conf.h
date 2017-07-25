#pragma once

// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu
#ifndef __PTS_2D_CONF_H_INCLUDED__
#define __PTS_2D_CONF_H_INCLUDED__

#include "myheader.h"


class pts_2d_conf {
public:
	double x, y;
	double conf;	// confidence of this 2d point prediction

	pts_2d_conf()														: x(default_x), y(default_y), conf(default_conf) {}
	pts_2d_conf(double tx, double ty, double tconf)						: x(tx), y(ty) { this->conf = tconf; }
	pts_2d_conf(double tx, double ty)									: x(tx), y(ty), conf(default_conf) {}
	pts_2d_conf(float tx, float ty)										: x(double(tx)), y(double(ty)), conf(default_conf) {}
	pts_2d_conf(float tx, float ty, double tconf)						: x(double(tx)), y(double(ty)) { this->conf = tconf; }
	pts_2d_conf(int tx, int ty)											: x(double(tx)), y(double(ty)), conf(default_conf) {}
	pts_2d_conf(int tx, int ty, double tconf)							: x(double(tx)), y(double(ty)) { this->conf = tconf; }
	pts_2d_conf(cv::Point pts)											: x(pts.x), y(pts.y), conf(default_conf) {}
	pts_2d_conf(cv::Point2d pts)										: x(pts.x), y(pts.y), conf(default_conf) {}
	pts_2d_conf(cv::Point2f pts)										: x(pts.x), y(pts.y), conf(default_conf) {}
	pts_2d_conf(cv::Point pts, double tconf)							: x(pts.x), y(pts.y) { this->conf = tconf; }
	pts_2d_conf(cv::Point2d pts, double tconf)							: x(pts.x), y(pts.y) { this->conf = tconf; }
	pts_2d_conf(cv::Point2f pts, double tconf)							: x(pts.x), y(pts.y) { this->conf = tconf; }

	cv::Point2d convert_to_point2d();
	std::vector<double> convert_to_pts_vec();
	std::vector<double> convert_to_pts_vec_conf();
	void print(int prec = default_precision);
};


#endif