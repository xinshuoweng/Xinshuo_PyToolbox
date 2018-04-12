#pragma once

// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu

#ifndef __PTS_2D_TOOL_H_INCLUDED__
#define __PTS_2D_TOOL_H_INCLUDED__


#include <myheader.h>
#include <xinshuo_vision/geometry/pts_2d_conf.h>



class pts_2d_tool: public pts_2d_conf {
public:
	bool anchor;

	pts_2d_tool()														: pts_2d_conf(), anchor(default_anchor) {}
	pts_2d_tool(double tx, double ty, double tconf, bool tanchor)		: pts_2d_conf(tx, ty, tconf), anchor(tanchor) {}
	pts_2d_tool(double tx, double ty, bool tanchor)						: pts_2d_conf(tx, ty, default_conf), anchor(tanchor) {}
	pts_2d_tool(float tx, float ty, bool tanchor)						: pts_2d_conf(tx, ty, default_conf), anchor(tanchor) {}
	pts_2d_tool(float tx, float ty, double tconf, bool tanchor)			: pts_2d_conf(tx, ty, tconf), anchor(tanchor) {}
	pts_2d_tool(int tx, int ty, bool tanchor)							: pts_2d_conf(tx, ty, default_conf), anchor(tanchor) {}
	pts_2d_tool(int tx, int ty, double tconf, bool tanchor)				: pts_2d_conf(tx, ty, tconf), anchor(tanchor) {}
	pts_2d_tool(double tx, double ty, double tconf)						: pts_2d_conf(tx, ty, tconf), anchor(default_anchor) {}
	pts_2d_tool(double tx, double ty)									: pts_2d_conf(tx, ty, default_conf), anchor(default_anchor) {}
	pts_2d_tool(float tx, float ty)										: pts_2d_conf(tx, ty, default_conf), anchor(default_anchor) {}
	pts_2d_tool(float tx, float ty, double tconf)						: pts_2d_conf(tx, ty, tconf), anchor(default_anchor) {}
	pts_2d_tool(int tx, int ty)											: pts_2d_conf(tx, ty, default_conf), anchor(default_anchor) {}
	pts_2d_tool(int tx, int ty, double tconf)							: pts_2d_conf(tx, ty, tconf), anchor(default_anchor) {}
	pts_2d_tool(cv::Point& pts, bool tanchor)							: pts_2d_conf(pts), anchor(tanchor) {}
	pts_2d_tool(cv::Point& pts)											: pts_2d_conf(pts), anchor(default_anchor) {}
	pts_2d_tool(cv::Point2d& pts, bool tanchor)							: pts_2d_conf(pts), anchor(tanchor) {}
	pts_2d_tool(cv::Point2d& pts)										: pts_2d_conf(pts), anchor(default_anchor) {}
	pts_2d_tool(cv::Point2f& pts, bool tanchor)							: pts_2d_conf(pts), anchor(tanchor) {}
	pts_2d_tool(cv::Point2f& pts)										: pts_2d_conf(pts), anchor(default_anchor) {}
	pts_2d_tool(pts_2d_conf& pts, bool tanchor)							: pts_2d_conf(pts), anchor(tanchor) {}
	pts_2d_tool(pts_2d_conf& pts)										: pts_2d_conf(pts), anchor(default_anchor) {}

	pts_2d_conf convert_to_pts_2d_conf();
	void print(int prec = default_precision);
};


#endif
