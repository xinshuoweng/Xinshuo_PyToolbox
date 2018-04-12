#pragma once
// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu

#ifndef __PTS_ON_MESH_H_INCLUDED__
#define __PTS_ON_MESH_H_INCLUDED__


#include <myheader.h>
#include <xinshuo_vision/geometry/pts_3d_conf.h>

class pts_3d_conf;

class pts_on_mesh : public pts_3d_conf {
public:
	int vertice_id;						// vertice id
	int triangle_id;					// triangle id

	pts_on_mesh(int pts_id, int tri_id, double tx, double ty, double tz, double tconf)			: pts_3d_conf(tx, ty, tz, tconf), vertice_id(pts_id), triangle_id(tri_id) {}
	pts_on_mesh(int pts_id, int tri_id, double tx, double ty, double tz)						: pts_3d_conf(tx, ty, tz), vertice_id(pts_id), triangle_id(tri_id) {}
	pts_on_mesh(int pts_id, int tri_id, float tx, float ty, float tz)							: pts_3d_conf(tx, ty, tz), vertice_id(pts_id), triangle_id(tri_id) {}
	pts_on_mesh(int pts_id, int tri_id, float tx, float ty, float tz, double tconf)				: pts_3d_conf(tx, ty, tz, tconf), vertice_id(pts_id), triangle_id(tri_id) {}
	pts_on_mesh(int pts_id, int tri_id, int tx, int ty, int tz)									: pts_3d_conf(tx, ty, tz), vertice_id(pts_id), triangle_id(tri_id) {}
	pts_on_mesh(int pts_id, int tri_id, int tx, int ty, int tz, double tconf)					: pts_3d_conf(tx, ty, tz, tconf), vertice_id(pts_id), triangle_id(tri_id) {}

	cv::Point3d convert_to_point3d();
	pts_3d_conf convert_to_pts_3d_conf();
	std::vector<double> convert_to_pts_vec();
	std::vector<double> convert_to_pts_vec_conf();
	void print(int prec = default_precision);
};

#endif