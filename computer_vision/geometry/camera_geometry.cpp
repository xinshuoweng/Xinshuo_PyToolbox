// Author: Xinshuo Weng
// Email: xinshuow@andrew.cmu.edu


// in-project library
#include "camera_geometry.h"
#include "debug_tool.h"
#include "mycamera.h"
#include "pts_2d_conf.h"
#include "pts_3d_conf.h"

// self-contained library
#include "math_functions.h"
#include "type_conversion.h"

// opencv library
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// TODO: optimization using only corrected points
// TODO: eliminate conf mode for calculating projection error

/*************************************************************************************************** multi-view processing ********************************************************************************************************/
void triangulation_from_two_views(std::vector<cv::Point2d>& pts_src1, std::vector<cv::Point2d>& pts_src2, mycamera& camera1, \
mycamera& camera2, std::vector<cv::Point3d>& pts_dst, const bool consider_dist) {
	ASSERT_WITH_MSG(pts_src1.size() > 0, "The size of 2d points from two view should be larger than 0 while doing two view triangulation.");
	ASSERT_WITH_MSG(pts_src1.size() == pts_src2.size(), "The size of 2d points from two view should be equal while doing two view triangulation.");
	std::vector<pts_3d_conf> pts_tmp;

    std::vector<pts_2d_conf> pts_src1_converted = cv2conf_vec_pts2d(pts_src1);
    std::vector<pts_2d_conf> pts_src2_converted = cv2conf_vec_pts2d(pts_src2);
	triangulation_from_two_views(pts_src1_converted, pts_src2_converted, camera1, camera2, pts_tmp, consider_dist);
	pts_dst = conf2cv_vec_pts3d(pts_tmp);
}

void triangulation_from_two_views(std::vector<pts_2d_conf>& pts_src1, std::vector<pts_2d_conf>& pts_src2, mycamera& camera1, \
mycamera& camera2, std::vector<pts_3d_conf>& pts_dst, const bool consider_dist) {
	ASSERT_WITH_MSG(pts_src1.size() > 0, "The size of 2d points from two view should be larger than 0 while doing two view triangulation.");
	ASSERT_WITH_MSG(pts_src1.size() == pts_src2.size(), "The size of 2d points from two view should be equal while doing two view triangulation.");

	// undistortion and intrinsic inversion
	std::vector<pts_2d_conf> pts_src_undistorted1;
	std::vector<pts_2d_conf> pts_src_undistorted2;
	undistort_single_point(pts_src1, camera1, pts_src_undistorted1, consider_dist);
	undistort_single_point(pts_src2, camera2, pts_src_undistorted2, consider_dist);

	// triangulation to solve depth ambiguality using two cameras
	cv::Mat pts_world = cv::Mat(1, pts_src1.size(), CV_64FC4);	// 4 channels input required by the opencv2.4.13
	cv::triangulatePoints(camera1.extrinsic, camera2.extrinsic, cv::Mat(conf2cv_vec_pts2d(pts_src_undistorted1)), cv::Mat(conf2cv_vec_pts2d(pts_src_undistorted2)), pts_world);
	pts_world = pts_world.t();

	// normalize the 4D world points to 3D
	size_t length = pts_src1.size();		// number of points
	for (size_t i = 0; i < length; i++) {
		pts_world.row(i) = pts_world.row(i) / pts_world.at<double>(i, 3);
	}
	for (size_t i = 0; i < length; ++i) {
		cv::Mat pts_3d_mat_tmp = pts_world.row(i).colRange(0, 3);
        std::vector<double> pts_3d_vec_tmp = mat2vec(pts_3d_mat_tmp);
        cv::Point3d pts_3d_tmp = vec2cv_pts3d(pts_3d_vec_tmp);
		pts_dst.push_back(pts_3d_conf(pts_3d_tmp, (pts_src1[i].conf + pts_src2[i].conf)/2));
	}
}

void multi_view_projection(std::vector<cv::Point3d>& pts_src, std::vector<mycamera>& camera_cluster, \
std::map<std::string, std::vector<cv::Point2d>>& pts_dst, const bool consider_dist) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The number of input 3d points is 0 while doing multiview projection for multiple points.");
	std::map<std::string, cv::Point2d> pts_tmp;
	for (int i = 0; i < pts_src.size(); i++) {
		pts_tmp.clear();
		multi_view_projection(pts_src[i], camera_cluster, pts_tmp, consider_dist);		// project single 2d points for multiview
		ASSERT_WITH_MSG(camera_cluster.size() == pts_tmp.size(), "The size of camera cluster and 2d points provided should be equal \
			while doing multiview projecction for multiple points.");

		// convert temporart 2d points to output type
		for (std::map<std::string, cv::Point2d>::iterator pts_2d = pts_tmp.begin(); pts_2d != pts_tmp.end(); pts_2d++) {
			pts_dst[pts_2d->first].push_back(pts_2d->second);
		}
	}
}

void multi_view_projection(cv::Point3d& pts_src, std::vector<mycamera>& camera_cluster, \
std::map<std::string, cv::Point2d>& pts_dst, const bool consider_dist, const bool debug_mode) {
	cv::Point2d pts_camera;
	mycamera camera_temp;
	cv::Mat pts_src_mat = cv::Mat(1, 3, CV_64FC1);
	pts_src_mat.at<double>(0, 0) = pts_src.x;
	pts_src_mat.at<double>(0, 1) = pts_src.y;
	pts_src_mat.at<double>(0, 2) = pts_src.z;

	for (int i = 0; i < camera_cluster.size(); i++) {
		cv::Mat pts_temp = cv::Mat(1, 2, CV_64FC1);
		camera_temp = camera_cluster[i];
		cv::Mat distCoeffs;
		if (debug_mode)
			std::cout << "Warning: The distortion is not considered. \
				Please make sure the image is already undistorted while projecting from 3d to 2d." << std::endl << std::endl;

		if (consider_dist) {
			ASSERT_WITH_MSG(!camera_cluster[i].distCoeffs.empty(), "The distortion coefficient is null \
				while doing multiview projection for single point. Please check if distortion is considered.");
			distCoeffs = camera_temp.distCoeffs;
		}

		// reproject 3D to 2D for a single camera
		cv::projectPoints(pts_src_mat, camera_temp.getRotationVector(), camera_temp.getTranslationVector(), \
			camera_temp.intrinsic, distCoeffs, pts_temp);		
		ASSERT_WITH_MSG(!pts_temp.empty(), "The projection failed. No 2d points is obtained while doing multiview projection \
			for single point.");
		pts_temp = pts_temp.reshape(1);

		pts_camera = cv::Point2d(pts_temp.row(0));
		pts_dst[camera_temp.name] = pts_camera;  // save all ooints for current camera to the output
	}
}


// TODO: test for correctness
//double calculate_projection_error(std::vector<cv::Point3d>& pts_src, std::vector<mycamera>& camera_cluster, \
//std::map<std::string, std::vector<pts_2d_conf>>& pts_2d_given, const bool consider_dist) {
//	ASSERT_WITH_MSG(pts_src.size() > 0, "The size of input points should be larger than 0 while calculating projection error.");
//	ASSERT_WITH_MSG(camera_cluster.size() > 0, "The number of camera should be larger than 0 while calculating projection error.");
//	ASSERT_WITH_MSG(pts_2d_given.size() == camera_cluster.size(), "The number of camera should be equal to the size of given 2d points \
//		while calculating projection error.");
//	ASSERT_WITH_MSG(pts_src.size() > 0, "The size of input points should be larger than 0 while calculating projection error.");
//
//	std::map<std::string, std::vector<cv::Point2d>> pts_dst;
//	multi_view_projection(pts_src, camera_cluster, pts_dst, consider_dist);	// get 2d points for all views
//
//	double error_total, error_tmp;
//	error_total = 0;
//	for (std::map<std::string, std::vector<pts_2d_conf>>::iterator it = pts_2d_given.begin(); it != pts_2d_given.end(); it++) {
//		ASSERT_WITH_MSG(pts_src.size() == it->second.size(), "The size of points to evaluate should be equal from prediction and groundtruth \
//			while calculating projection error.");
//		error_tmp = 0;		// error for each view
//		std::cout << "processing camera " + it->first;
//		for (int i = 0; i < pts_dst[it->first].size(); i++) {
//			error_tmp += it->second[i].conf * sqrt((pts_dst[it->first][i].x - it->second[i].x) * (pts_dst[it->first][i].x - it->second[i].x) \
//					+ (pts_dst[it->first][i].y - it->second[i].y) * (pts_dst[it->first][i].y - it->second[i].y));	// weighted error based on confidence
//		}
//		error_total += error_tmp;
//	}
//
//	return error_total;
//}



// Naive version now, only sort error and take the first 5 error, no confidence is used
// TODO: 1. weighted error based on confidence (-logx - 1)
// TODO: 2. use threshold instead of number of inlier
// TODO: 3. based on 2, recalculating the model using multiview triangulation
double calculate_projection_error(cv::Point3d& pts_src, std::vector<mycamera>& camera_cluster, \
std::map<std::string, pts_2d_conf>& pts_2d_given, const bool consider_dist) {
	ASSERT_WITH_MSG(camera_cluster.size() > 0, "The number of camera should be larger than 0 while calculating projection error.");
	ASSERT_WITH_MSG(camera_cluster.size() == pts_2d_given.size(), "The size of camera cluster and 2d points provided should be equal \
		while calculating projection error.");

	std::map<std::string, cv::Point2d> pts_dst;
	multi_view_projection(pts_src, camera_cluster, pts_dst, consider_dist);	// get 2d points for all views

	std::vector<double> error_list;
	int num_inlier = default_num_inliers;
	double error_total, error_tmp;
	error_total = 0;
	int count = 0;
	for (std::map<std::string, pts_2d_conf>::iterator it = pts_2d_given.begin(); it != pts_2d_given.end(); it++, count++) {
		if (it->second.conf < 0.5)	// if the condidence of the original 2d points is too low, we don't trust the error in this view for this point
			continue;
		error_tmp = sqrt((pts_dst[it->first].x - it->second.x) * (pts_dst[it->first].x - it->second.x) \
				+ (pts_dst[it->first].y - it->second.y) * (pts_dst[it->first].y - it->second.y));	// weighted error based on confidence
		error_list.push_back(error_tmp);
	}
	//print_vec(error_list);
	sort(error_list.begin(), error_list.end());
	//print_vec(error_list);
	for (int i = 0; i < num_inlier; i++) {
		if (i >= error_list.size())
			break;
		error_total += error_list[i];
	}
	//print_sca(error_total);
	return error_total;
}



void multiview_ransac_single_point(std::map<std::string, pts_2d_conf>& pts_2d_given, std::vector<mycamera>& camera_cluster, \
cv::Point3d& pts_dst, const bool consider_dist, const int num_ransac) {
	ASSERT_WITH_MSG(camera_cluster.size() == pts_2d_given.size(), "The size of camera cluster and 2d points provided should be equal \
		while doing multiview RANSAC for single point.");
	int count = 0;	// count the number of randomization executed
	std::vector<int> view_selected(2);		// id for selected two view
	std::vector<pts_2d_conf> pts_tmp1;
	std::vector<pts_2d_conf> pts_tmp2;
	double min_error = DBL_MAX;
	double error_tmp;

	mycamera camera_tmp2;
	std::srand(SEED);
	while (count < num_ransac) {
		get_two_random_camera_id(pts_2d_given, view_selected, std::rand());		// randomly select two views
		mycamera camera_tmp1 = camera_cluster[pts_2d_given.size() - 1 - view_selected[0]];
		mycamera camera_tmp2 = camera_cluster[pts_2d_given.size() - 1 - view_selected[1]];

		std::map<std::string, pts_2d_conf>::iterator it = pts_2d_given.begin();
		std::advance(it, view_selected[0]);
		//std::cout << it->first << std::endl;
		//std::cout << camera_tmp1.name << std::endl;
		ASSERT_WITH_MSG(camera_tmp1.name.compare(it->first) == 0, "The camera from points and camera file are not corresponding whlie RANSAC!");
		it = pts_2d_given.begin();
		std::advance(it, view_selected[1]);
		//std::cout << it->first << std::endl;
		//std::cout << camera_tmp2.name << std::endl;
		ASSERT_WITH_MSG(camera_tmp2.name.compare(it->first) == 0, "The camera from points and camera file are not corresponding whlie RANSAC!");

		pts_tmp1.clear();
		pts_tmp2.clear();
		pts_tmp1.push_back(pts_2d_given[camera_tmp1.name]);
		pts_tmp2.push_back(pts_2d_given[camera_tmp2.name]);

		std::vector<pts_3d_conf> pts_src;
		triangulation_from_two_views(pts_tmp1, pts_tmp2, camera_tmp1, camera_tmp2, pts_src, consider_dist);		// two view triangluation for single point
		ASSERT_WITH_MSG(pts_src.size() == 1, "The size of 3d point should be only 1 while doing multiview RANSAC for single point.");

        cv::Point3d pts_3d_tmp = pts_src[0].convert_to_point3d();
        error_tmp = calculate_projection_error(pts_3d_tmp, camera_cluster, pts_2d_given, consider_dist);
		//print_sca(error_tmp);

		//print_sca(view_selected[0]);
		//print_sca(view_selected[1]);
		//std::cout << camera_tmp1.name << std::endl;
		//std::cout << camera_tmp2.name << std::endl;
		if (error_tmp < min_error) {
			min_error = error_tmp;
			//print_sca(min_error);
			//print_vec(view_selected);
			pts_dst = pts_src[0].convert_to_point3d();
		}
		count++;
	}
	ASSERT_WITH_MSG(!cv::Mat(pts_dst).empty(), "The output 3d point is empty while doing multiview RANSAC for single point.");
}

// TODO: test for correctness
void multiview_ransac_multiple_points(std::map<std::string, std::vector<pts_2d_conf>>& pts_2d_given, std::vector<mycamera>& camera_cluster, \
std::vector<cv::Point3d>& pts_dst, const bool consider_dist, const int num_ransac) {
	ASSERT_WITH_MSG(camera_cluster.size() == pts_2d_given.size(), "The size of camera cluster and 2d points provided should be equal \
		while doing multiview RANSAC for multiple points.");
	int num_pts = pts_2d_given.begin()->second.size();
	std::map<std::string, pts_2d_conf> pts_2d_tmp;
	cv::Point3d pts_3d_tmp;
	for (int i = 0; i < num_pts; i++) {
		for (std::map<std::string, std::vector<pts_2d_conf>>::iterator it = pts_2d_given.begin(); it != pts_2d_given.end(); it++) {
			ASSERT_WITH_MSG(num_pts == it->second.size(), "The 2d points given should be equal for all views \
				while doing multiview RANSAC for multiple points.");
			pts_2d_tmp[it->first] = it->second[i];
		}

		multiview_ransac_single_point(pts_2d_tmp, camera_cluster, pts_3d_tmp, consider_dist, num_ransac);
		pts_dst.push_back(pts_3d_tmp);
	}
}


// TODO: think about the confidence from 2d to 3d and then back to 2d
void multiview_optimization_single_point(std::map<std::string, pts_2d_conf>& pts_src, std::vector<mycamera>& camera_cluster, \
std::map<std::string, pts_2d_conf>& pts_2d, const bool consider_dist, const int num_ransac) {
	ASSERT_WITH_MSG(camera_cluster.size() == pts_src.size(), "The size of camera cluster and 2d points provided should be equal \
		while doing multiview optimization for single point.");
	cv::Point3d pts_3d_best;

	// using RANSAC	to find the best 3d point
	multiview_ransac_single_point(pts_src, camera_cluster, pts_3d_best, consider_dist, num_ransac);
	ASSERT_WITH_MSG(!cv::Mat(pts_3d_best).empty(), "The output 3d point is empty while doing multiview optimization for single point.");
	std::map<std::string, cv::Point2d> pts_dst;
	multi_view_projection(pts_3d_best, camera_cluster, pts_dst, consider_dist);		// project 3d point to 2d for all views
	for (int i = 0; i < camera_cluster.size(); i++) {
		pts_2d[camera_cluster[i].name] = pts_2d_conf(pts_dst[camera_cluster[i].name], pts_src[camera_cluster[i].name].conf);
	}
}


void multiview_optimization_multiple_points(std::map<std::string, std::vector<pts_2d_conf>>& pts_src, std::vector<mycamera>& camera_cluster, \
std::map<std::string, std::vector<pts_2d_conf>>& pts_2d, const bool consider_dist, const int num_ransac) {
	ASSERT_WITH_MSG(camera_cluster.size() == pts_src.size(), "The size of camera cluster and 2d points provided should be equal \
		while doing multiview optimization for multiple points.");
	int num_pts = pts_src.begin()->second.size();

	std::map<std::string, pts_2d_conf> pts_src_tmp;
	std::map<std::string, pts_2d_conf> pts_2d_tmp;
	for (int i = 0; i < num_pts; i++) {
		// extract a single point for all views
		for (std::map<std::string, std::vector<pts_2d_conf>>::iterator it = pts_src.begin(); it != pts_src.end(); it++) {
			ASSERT_WITH_MSG(num_pts == it->second.size(), "The 2d points given should be equal for all views \
				while doing multiview optimization for multiple points.");
			pts_src_tmp[it->first] = it->second[i];
		}

		multiview_optimization_single_point(pts_src_tmp, camera_cluster, pts_2d_tmp, consider_dist, num_ransac);
		
		// save the current point to points vector
		for (std::map<std::string, pts_2d_conf>::iterator it = pts_2d_tmp.begin(); it != pts_2d_tmp.end(); it++) {
			pts_2d[it->first].push_back(it->second);
		}
	}
}


/****************************************************************************************************** miscellaneous ***********************************************************************************************************/
void get_3d_ray(pts_2d_conf& pts_2d, mycamera& mycamera, cv::Point3d& C, std::vector<double>& ray, const bool consider_dist) {
//	std::cout << "getting 3d ray...." << std::endl;
	pts_2d_conf pts_undist = pts_2d;
	if (consider_dist)
		ASSERT_WITH_MSG(!mycamera.distCoeffs.empty(), "The size of distortion coefficients in the camera should not be 0 \
			while getting 3d ray under the condition of considering distortion.");
	else
		ASSERT_WITH_MSG(mycamera.distCoeffs.empty(), "The size of distortion coefficients in the camera should be 0 \
			while getting 3d ray under the condition of not considering distortion.");

    std::cout << "distorted 2d point is:  ";
    pts_2d.print();
    undistort_single_point(pts_2d, mycamera, pts_undist, consider_dist);
    std::cout << "undistorted 2d point is:  ";
    pts_undist.print();

	double x = pts_undist.x;
	double y = pts_undist.y;
	double conf = pts_undist.conf;
	cv::Mat M_mat = mycamera.extrinsic;

	std::vector<std::vector<double>> M(3, std::vector<double>(4, 0));
	ASSERT_WITH_MSG(M_mat.type() == CV_64F, "The type of input projection matrix should be CV_64F.");
	cv::Mat A(3, 3, CV_64F);
	cv::Mat b(3, 1, CV_64F);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			M[i][j] = M_mat.at<double>(i, j);
			if (j < 3) {
				A.at<double>(i, j) = M[i][j];
			}
			else {
				b.at<double>(i, 0) = M[i][3];
			}
		}
	}
	cv::Mat center = -A.inv() * b;		// camera center coordinate

	std::vector<double> C_tmp;
	for (int i = 0; i < 3; i++) {
		C_tmp.push_back(center.at<double>(i, 0));
	}

	// expand projection equation to find two planes
	for (int i = 0; i < 3; i++) {
		M[0][i] -= x * M[2][i];
		M[1][i] -= y * M[2][i];
	}

	// extract the first three variable from plane vector as the direction vector
	std::vector<double> plane1, plane2;
	for (int i = 0; i < 3; i++) {
		plane1.push_back(M[0][i]);
		plane2.push_back(M[1][i]);
	}

	ray = cross(plane1, plane2);

	double acc = 0;
	for (int i = 0; i < 3; i++) {
		acc += ray[i] * ray[i];
	}
	acc = sqrt(acc);
	cv::Mat test_point(4, 1, CV_64F);
	for (int i = 0; i < 3; i++) {
		test_point.at<double>(i, 0) = C_tmp[i] + ray[i];
	}
	test_point.at<double>(3, 0) = 1;
	test_point = M_mat * test_point;
	if (test_point.at<double>(2, 0) < 0) {		//behind camera
		acc *= -1;
	}

	for (int i = 0; i < 3; i++) {
		ray[i] /= acc;
	}

	C = vec2cv_pts3d(C_tmp);
}

// TODO: test camera with high probability
void get_two_random_camera_id(std::map<std::string, pts_2d_conf>& pts_2d_given, std::vector<int>& view_selected, int seed) {
	ASSERT_WITH_MSG(view_selected.size() == 2, "The size of input vector for saving random selected view \
		should be 2 while doing two view triangulation.");

	// extract confidence from pts 2d given
	std::vector<double> weights;
	for (std::map<std::string, pts_2d_conf>::iterator it = pts_2d_given.begin(); it != pts_2d_given.end(); it++) {
		//std::cout << it->first << std::endl;
		weights.push_back(it->second.conf);
	}

	 /*random weighted sample, which could handle more than 1 point corrected or no points corrected except for the coincidence, \
	while both two selected indexes are the same*/
	std::srand(seed);
	generate_weighted_randomization(weights, view_selected, std::rand());		
	//print_vec(view_selected);

	// two view picked are the same. It might because only 1 weight are not zero or just because of coincidence
	if (view_selected[0] == view_selected[1]) {
		std::vector<int> view_selected_retry(1);
		weights.erase(weights.begin() + view_selected[0]);	// remove the already selected index from weights
		generate_weighted_randomization(weights, view_selected_retry, std::rand());	// only need one to avoid coincidence
		int offset = 0;
		if (view_selected_retry[0] >= view_selected[0])
			offset = 1;
		view_selected[1] = view_selected_retry[0] + offset;
		//print_vec(view_selected);
	}
}


// TODO: test for correctness
void epipolar_2d_lines_from_anchor_point(pts_2d_conf& pts_anchor, mycamera& camera_anchor, std::vector<mycamera>& camera_cluster, \
std::map<std::string, std::vector<double>>& lines, const bool consider_dist) {
	ASSERT_WITH_MSG(camera_cluster.size() > 0, "The size of camera loaded is zero while calculating epipolar line.");
	cv::Point3d camera_center;
	std::vector<double> ray;
	get_3d_ray(pts_anchor, camera_anchor, camera_center, ray, consider_dist);
	ASSERT_WITH_MSG(ray.size() == 3, "The size of 3d ray should be 3 while calculating epipolar line.");

	// calculate two 3d points
	cv::Point3d pts_3d_tmp1;
	cv::Point3d pts_3d_tmp2;
	const double t1 = -10000;			// final 3d point is (t * ray + camera_center)
	const double t2 = 10000;			// final 3d point is (t * ray + camera_center)
	pts_3d_tmp1.x = t1 * ray[0] + camera_center.x;
	pts_3d_tmp1.y = t1 * ray[1] + camera_center.y;
	pts_3d_tmp1.z = t1 * ray[2] + camera_center.z;
	pts_3d_tmp2.x = t2 * ray[0] + camera_center.x;
	pts_3d_tmp2.y = t2 * ray[1] + camera_center.y;
	pts_3d_tmp2.z = t2 * ray[2] + camera_center.z;

	// project two 3d points back to 2d space
	std::map<std::string, cv::Point2d> pts_dst1;
	std::map<std::string, cv::Point2d> pts_dst2;
	multi_view_projection(pts_3d_tmp1, camera_cluster, pts_dst1, consider_dist);
	multi_view_projection(pts_3d_tmp2, camera_cluster, pts_dst2, consider_dist);

	ASSERT_WITH_MSG(pts_dst1.size() > 0, "The size of output points should be larger than 0 while calculating epipolar lines.");
	ASSERT_WITH_MSG(pts_dst1.size() == pts_dst2.size(), "The size of output should be equal while calculating epipolar lines.");

	// convert to line as output
	for (std::map<std::string, cv::Point2d>::iterator it = pts_dst1.begin(); it != pts_dst1.end(); it++) 
		get_2d_line(it->second, pts_dst2[it->first], lines[it->first]);
	ASSERT_WITH_MSG(lines.size() == camera_cluster.size(), "The size of output lines should be should be equal \
		with the size of camera while calculating epipolar lines.");
}

void undistort_single_point(pts_2d_conf& pts_src, mycamera& camera, pts_2d_conf& pts_dst, const bool consider_dist) {
	std::vector<pts_2d_conf> pts_src_tmp, pts_dst_tmp;
	pts_src_tmp.push_back(pts_src);
	undistort_single_point(pts_src_tmp, camera, pts_dst_tmp, consider_dist);
	pts_dst = pts_dst_tmp[0];
}

void undistort_single_point(std::vector<pts_2d_conf>& pts_src, mycamera& camera, std::vector<pts_2d_conf>& pts_dst, const bool consider_dist) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The size of input points should be larger than 0 while undistorting.");
	cv::Mat distCoeffs;
	cv::Mat pts_src_undistorted;
	if (consider_dist)
		distCoeffs = camera.distCoeffs;

	
	cv::undistortPoints(cv::Mat(conf2cv_vec_pts2d(pts_src)), pts_src_undistorted, camera.intrinsic, distCoeffs);
	pts_src_undistorted = pts_src_undistorted.t();
	ASSERT_WITH_MSG(pts_src_undistorted.cols == pts_src.size(), "The size of output undistorted points should be equal to inputs while undistorting.");
	ASSERT_WITH_MSG(pts_src_undistorted.rows == 1, "The dimension of the undistorted points is not correct while undistorting.");

	pts_2d_conf pts_tmp;
	for (int i = 0; i < pts_src_undistorted.cols; i++) {
		pts_tmp.conf = pts_src[i].conf;
		pts_tmp.x = pts_src_undistorted.at<cv::Vec2d>(0, i)[0];
		pts_tmp.y = pts_src_undistorted.at<cv::Vec2d>(0, i)[1];
		pts_dst.push_back(pts_tmp);
	}
}
