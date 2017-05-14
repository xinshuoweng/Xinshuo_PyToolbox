// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu

#include "type_conversion.h"
#include "pts_2d_conf.h"
#include "pts_2d_tool.h"
#include "pts_3d_conf.h"

/************************************************** intra-class conversion ************************************************/
// TODO: test for correctness
std::vector<cv::Point2d> float2double_vec_pts2d(std::vector<cv::Point2f>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<cv::Point2d> pts_dst;
	for (int i = 0; i < pts_src.size(); i++) {
		pts_dst.push_back(float2double_pts2d(pts_src[i]));
	}
	return pts_dst;
}
// TODO: test for correctness
std::vector<cv::Point2f> double2float_vec_pts2d(std::vector<cv::Point2d>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<cv::Point2f> pts_dst;
	for (int i = 0; i < pts_src.size(); i++) {
		pts_dst.push_back(double2float_pts2d(pts_src[i]));
	}
	return pts_dst;
}
// TODO: test for correctness
std::vector<cv::Point2d> int2double_vec_pts2d(std::vector<cv::Point2i>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<cv::Point2d> pts_dst;
	for (int i = 0; i < pts_src.size(); i++) {
		pts_dst.push_back(int2double_pts2d(pts_src[i]));
	}
	return pts_dst;
}
// TODO: test for correctness
std::vector<cv::Point2i> double2int_vec_pts2d(std::vector<cv::Point2d>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<cv::Point2i> pts_dst;
	for (int i = 0; i < pts_src.size(); i++) {
		pts_dst.push_back(double2int_pts2d(pts_src[i]));
	}
	return pts_dst;
}


// TODO: test for correctness
std::vector<cv::Point3d> float2double_vec_pts3d(std::vector<cv::Point3f>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<cv::Point3d> pts_dst;
	for (int i = 0; i < pts_src.size(); i++) {
		pts_dst.push_back(float2double_pts3d(pts_src[i]));
	}
	return pts_dst;
}
// TODO: test for correctness
std::vector<cv::Point3f> double2float_vec_pts3d(std::vector<cv::Point3d>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<cv::Point3f> pts_dst;
	for (int i = 0; i < pts_src.size(); i++) {
		pts_dst.push_back(double2float_pts3d(pts_src[i]));
	}
	return pts_dst;
}
// TODO: test for correctness
std::vector<cv::Point3d> int2double_vec_pts3d(std::vector<cv::Point3i>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<cv::Point3d> pts_dst;
	for (int i = 0; i < pts_src.size(); i++) {
		pts_dst.push_back(int2double_pts3d(pts_src[i]));
	}
	return pts_dst;
}
// TODO: test for correctness
std::vector<cv::Point3i> double2int_vec_pts3d(std::vector<cv::Point3d>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<cv::Point3i> pts_dst;
	for (int i = 0; i < pts_src.size(); i++) {
		pts_dst.push_back(double2int_pts3d(pts_src[i]));
	}
	return pts_dst;
}


// TODO: test for correctness
cv::Point2d float2double_pts2d(cv::Point2f& pts_src) {
	return cv::Point2d(double(pts_src.x), double(pts_src.y));
}
// TODO: test for correctness
cv::Point2f double2float_pts2d(cv::Point2d& pts_src) {
	return cv::Point2f(float(pts_src.x), float(pts_src.y));
}
// TODO: test for correctness
cv::Point2d int2double_pts2d(cv::Point2i& pts_src) {
	return cv::Point2d(double(pts_src.x), double(pts_src.y));
}
// TODO: test for correctness
cv::Point2i double2int_pts2d(cv::Point2d& pts_src) {
	return cv::Point2i(int(pts_src.x), int(pts_src.y));
}


// TODO: test for correctness
cv::Point3d float2double_pts3d(cv::Point3f& pts_src) {
	return cv::Point3d(double(pts_src.x), double(pts_src.y), double(pts_src.z));
}
// TODO: test for correctness
cv::Point3f double2float_pts3d(cv::Point3d& pts_src) {
	return cv::Point3f(float(pts_src.x), float(pts_src.y), float(pts_src.z));
}
// TODO: test for correctness
cv::Point3d int2double_pts3d(cv::Point3i& pts_src) {
	return cv::Point3d(double(pts_src.x), double(pts_src.y), double(pts_src.z));
}
// TODO: test for correctness
cv::Point3i double2int_pts3d(cv::Point3d& pts_src) {
	return cv::Point3i(int(pts_src.x), int(pts_src.y), int(pts_src.z));
}




std::vector<double> float2double_vec(std::vector<float>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<double> pts_dst(pts_src.begin(), pts_src.end());
	return pts_dst;
}

std::vector<float> double2float_vec(std::vector<double>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<float> pts_dst(pts_src.begin(), pts_src.end());
	return pts_dst;
}

std::vector<double> int2double_vec(std::vector<int>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<double> pts_dst(pts_src.begin(), pts_src.end());
	return pts_dst;
}
// TODO: test for correctness
std::vector<int> double2int_vec(std::vector<double>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<int> pts_dst(pts_src.begin(), pts_src.end());
	return pts_dst;
}


/************************************************** inter-class conversion ************************************************/
pcl::PointXY cv2pcl_pts2d(cv::Point2d& pts_src) {
	pcl::PointXY pts_dst;
	pts_dst.x = (float)pts_src.x;
	pts_dst.y = (float)pts_src.y;
	return pts_dst;
}
// TODO: test for correctness
pcl::PointXYZ cv2pcl_pts3d(cv::Point3d& pts_src) {
	return pcl::PointXYZ(pts_src.x, pts_src.y, pts_src.z);
}

cv::Point2d pcl2cv_pts2d(pcl::PointXY& pts_src) {
	return cv::Point2d(pts_src.x, pts_src.y);
}

cv::Point3d pcl2cv_pts3d(pcl::PointXYZ& pts_src) {
	return cv::Point3d(pts_src.x, pts_src.y, pts_src.z);
}



std::vector<cv::Point2d> conf2cv_vec_pts2d(std::vector<pts_2d_conf>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<cv::Point2d> pts_dst;
	for (int i = 0; i < pts_src.size(); i++)
		pts_dst.push_back(pts_src[i].convert_to_point2d());
	return pts_dst;
}
std::vector<cv::Point3d> conf2cv_vec_pts3d(std::vector<pts_3d_conf>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<cv::Point3d> pts_dst;
	for (int i = 0; i < pts_src.size(); i++)
		pts_dst.push_back(pts_src[i].convert_to_point3d());
	return pts_dst;
}
std::vector<pts_2d_conf> cv2conf_vec_pts2d(std::vector<cv::Point2d>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<pts_2d_conf> pts_dst;
	for (int i = 0; i < pts_src.size(); i++)
		pts_dst.push_back(pts_2d_conf(pts_src[i]));
	return pts_dst;
}
// TODO: test for correctness
std::vector<pts_3d_conf> cv2conf_vec_pts3d(std::vector<cv::Point3d>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The input vector is empty while converting the type.");
	std::vector<pts_3d_conf> pts_dst;
	for (int i = 0; i < pts_src.size(); i++)
		pts_dst.push_back(pts_3d_conf(pts_src[i]));
	return pts_dst;
}


std::vector<double> cv2vec_pts2d(cv::Point2d& pts_src) {
	std::vector<double> pts_dst;
	pts_dst.push_back(pts_src.x);
	pts_dst.push_back(pts_src.y);
	return pts_dst;
}
std::vector<double> cv2vec_pts3d(cv::Point3d& pts_src) {
	std::vector<double> pts_dst;
	pts_dst.push_back(pts_src.x);
	pts_dst.push_back(pts_src.y);
	pts_dst.push_back(pts_src.z);
	return pts_dst;
}
cv::Point2d vec2cv_pts2d(std::vector<double>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() == 2, "The input vector is empty while converting the type.");
	return cv::Point2d(pts_src[0], pts_src[1]);
}
cv::Point3d vec2cv_pts3d(std::vector<double>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() == 3, "The input vector is empty while converting the type.");
	return cv::Point3d(pts_src[0], pts_src[1], pts_src[2]);
}

std::vector<double> mat2vec(cv::Mat pts_src) {
	ASSERT_WITH_MSG(pts_src.type() == CV_32F || pts_src.type() == CV_64F, "Only CV_32F and CV_64F \
		are supported right now while converting the type.");
	ASSERT_WITH_MSG(!pts_src.empty(), "The input cv::Mat is empty while converting the type.");
	ASSERT_WITH_MSG((pts_src.channels() == 1 && pts_src.rows == 1 && pts_src.cols >= 1) || \
		(pts_src.channels() == 1 && pts_src.rows >= 1 && pts_src.cols == 1), \
		"The input cv::Mat is not a one dimensional vector while converting the type.");
	std::vector<double> vec_dst;
	if (pts_src.channels() == 1 && pts_src.rows == 1 && pts_src.cols >= 1) {
		for (size_t i = 0; i < pts_src.cols; i++) {
			if (pts_src.type() == CV_32F)
				vec_dst.push_back(pts_src.at<float>(0, i));
			else if (pts_src.type() == CV_64F)
				vec_dst.push_back(pts_src.at<double>(0, i));
			else
				ASSERT_WITH_MSG(1 == 0, "Unknown error while converting the type.");
		}
	}
	else if (pts_src.channels() == 1 && pts_src.rows >= 1 && pts_src.cols == 1) {
		for (size_t i = 0; i < pts_src.rows; i++) {
			if (pts_src.type() == CV_32F)
				vec_dst.push_back(pts_src.at<float>(i, 0));
			else if (pts_src.type() == CV_64F)
				vec_dst.push_back(pts_src.at<double>(i, 0));
			else
				ASSERT_WITH_MSG(1 == 0, "Unknown error while converting the type.");
		}
	}
	else
		ASSERT_WITH_MSG(1 == 0, "Unknown error while converting the type.");
	return vec_dst;
}