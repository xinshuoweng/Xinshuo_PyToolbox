// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu
#ifndef __MYMESH_H_INCLUDED__
#define __MYMESH_H_INCLUDED__


#include <myheader.h>

// pcl library
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

class mycamera;
class pts_on_mesh;
class pts_2d_conf;
class pts_3d_conf;

class MyMesh{
public:
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_original;			// 3D coordinate indexed by point id
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;			// 3D coordinate indexed by point id
	std::vector<std::vector<int>> planes_of_vertices;	// planes_of_vertices[x] represent which plane xth vertices belongs to
	std::vector<int> first_plane_of_vertices;
	std::vector<std::vector<double>> planes;			// 4D vector representing a plane indexed by plane id
	std::vector<std::vector<int>> plane_pts_idx;		// plane_pts_idx[x] is a 3d vector, representing three points around this plane (triangle)
	std::vector<Eigen::Matrix3f> plane_projection;		// 3x3 matrix, transfer plane from original euclidean to B space, plane_projection * A = B
	std::vector<bool> plane_projection_good;
	pcl::PolygonMesh::Ptr poly_ptr;

	MyMesh(char* filename, int scale);
	~MyMesh();

	// reproject 2d ray to 3d mesh and find the intersection point
	// ray intersection to find points on mesh
	pts_on_mesh* get_pts_on_mesh(cv::Point3d C, std::vector<double>& ray, double conf);
    pts_on_mesh* get_pts_on_mesh(cv::Point3d C, std::vector<double>& ray, double conf, cv::Point3d C_ref);

    int get_pts_with_mesh_heuristic(cv::Point3d C_src, std::vector<double>& ray, double conf, cv::Point3d pts_3d_ref, pts_3d_conf& pts_3d_out);
    pts_on_mesh* get_pts_on_mesh_heuristic(cv::Point3d C, std::vector<double>& ray, double conf, cv::Point3d C_ref, pts_3d_conf& pts_3d_out);

	// given an arbitraty 3d point in 3d space, find the closest one on the mesh
	// given a 3d point, find closest 3d point on mesh
	pts_on_mesh* find_closest_pts_on_mesh(pts_3d_conf& pts_3d);

	// given an arbitraty 3d point in 3d space and a given plane, find the "almost" closest one on the mesh (better efficiency), current algorithm only evaluate the points in the given plane
	pts_on_mesh* find_closest_pts_on_mesh(pts_3d_conf& pts_3d, int plane_id);		

	// pts_2d may include confidence inside it, support only one point
	// no optimization
	pts_on_mesh* pts_back_projection_single_view(pts_2d_conf& pts_2d, mycamera& camera_src, const bool consider_dist = true);
	pts_on_mesh* pts_back_projection_single_view(pts_2d_conf& pts_2d, pts_2d_conf& pts_2d_ref, mycamera& camera_src, mycamera& camera_ref, pts_3d_conf& pts_3d, const bool consider_dist = true);

	// optimize the multiview triangulation in 3d space, current strategy is to select one best 3d point from all view
	// support only one point
	// optimization involved
	pts_on_mesh* pts_back_projection_multiview(std::map<std::string, pts_2d_conf>& pts_2d, std::vector<mycamera>& camera_cluster, const bool consider_dist = true);	

	// multiple points from all views
	// each pts_2d could be only x, y or includes confidence
	// optimization involved
	std::vector<pts_on_mesh*> pts_back_projection_multiview(std::vector<std::map<std::string, pts_2d_conf>>& pts_2d, std::vector<mycamera>& camera_cluster, const bool consider_dist = true);


	// deprecated
	//pts_on_mesh* get_pts_on_mesh(std::vector<double>& C, std::vector<double>& ray, double conf, bool debug);
	//pts_on_mesh* pts_back_projection_multiview(std::map<std::string, pts_2d_conf>& pts_2d, std::map<std::string, mycamera>& camera_cluster);// single point from all views
	//std::vector<pts_on_mesh*> pts_back_projection_multiview(std::vector<std::map<std::string, pts_2d_conf>>& pts_2d, std::map<std::string, mycamera>& camera_cluster);// multiple points from all views
	//pts_on_mesh* project_prediction(double x, double y, cv::Mat& M_mat, double conf, std::vector<double>& ray_found, std::vector<double>& camera_center);
	//pts_on_mesh* project_prediction(double x, double y, cv::Mat& M_mat, double conf, std::vector<double>& ray_found, std::vector<double>& camera_center, bool debug);

};



#endif