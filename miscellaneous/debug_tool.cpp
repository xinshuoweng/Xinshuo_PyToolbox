// Author: Xinshuo Weng
// Email: xinshuow@andrew.cmu.edu


#include "debug_tool.h"
#include "type_conversion.h"
#include "pts_2d_conf.h"
#include "pts_3d_conf.h"


/************************************** print function ********************************************/
// TODO: support multichannel mat
void print_mat(cv::Mat& mat, int prec)
{
	ASSERT_WITH_MSG(mat.type() == CV_32F || mat.type() == CV_64F, "Current print function only support CV_32F and CV_64F. Please check the type.");
	for (int i = 0; i < mat.size().height; i++)
	{
		std::cout << "[";
		for (int j = 0; j < mat.size().width; j++)
		{
			if (mat.type() == CV_32F)	//
				std::cout << std::setprecision(prec) << mat.at<float>(i, j);
			else if (mat.type() == CV_64F)
				std::cout << std::setprecision(prec) << mat.at<double>(i, j);
			
			if (j != mat.size().width - 1)
				std::cout << ", ";
			else
				std::cout << "]" << std::endl;
		}
	}
}

void print_mat_info(cv::Mat& test_mat) {
	std::cout << "rows: " << test_mat.rows << std::endl;
	std::cout << "cols: " << test_mat.cols << std::endl;
	std::cout << "dims: " << test_mat.dims << std::endl;
	std::cout << "depth: " << test_mat.depth() << std::endl;
	std::cout << "channels: " << test_mat.channels() << std::endl;
	std::cout << "type: " << test_mat.type() << std::endl;
	std::cout << "total: " << test_mat.total() << std::endl;
	std::cout << "size: " << test_mat.size() << std::endl;
	std::cout << "size.width: " << test_mat.size().width << std::endl;
	std::cout << "size.height: " << test_mat.size().height << std::endl;
}

void print_vec_pts3d(std::vector<cv::Point3d>& v) {
	ASSERT_WITH_MSG(v.size() > 0, "The input vector is empty while printing the vector of point3d.");
	for (int i = 0; i < v.size(); i++) {
		print_pts3d(v[i]);
	}
	std::cout << std::endl;
}


void print_sca(double sca) {
	std::cout << sca << std::endl;
}

void print_sca(float sca) {
	std::cout << sca << std::endl;
}

void print_sca(int sca) {
	std::cout << sca << std::endl;
}

// TODO: test for correctness
void print_vec_pts3d(std::vector<cv::Point3f>& v) {
	ASSERT_WITH_MSG(v.size() > 0, "The input vector is empty while printing the vector of point3f.");

	std::vector<cv::Point3d> pts_3d_vec = float2double_vec_pts3d(v);
	print_vec_pts3d(pts_3d_vec);
}
// TODO: test for correctness
void print_vec_pts3d(std::vector<cv::Point3i>& v) {
	ASSERT_WITH_MSG(v.size() > 0, "The input vector is empty while printing the vector of point3i.");

	std::vector<cv::Point3d> pts_3d_vec = int2double_vec_pts3d(v);
	print_vec_pts3d(pts_3d_vec);
}

void print_vec_pts2d(std::vector<cv::Point2d>& v) {
	ASSERT_WITH_MSG(v.size() > 0, "The input vector is empty while printing the vector of point2d.");
	for (int i = 0; i < v.size(); i++) {
		print_pts2d(v[i]);
	}
	std::cout << std::endl;
}
// TODO: test for correctness
void print_vec_pts2d(std::vector<cv::Point2f>& v) {
	ASSERT_WITH_MSG(v.size() > 0, "The input vector is empty while printing the vector of point2f.");

	std::vector<cv::Point2d> pts_2d_vec = float2double_vec_pts2d(v);
	print_vec_pts2d(pts_2d_vec);
}
// TODO: test for correctness
void print_vec_pts2d(std::vector<cv::Point2i>& v) {
	ASSERT_WITH_MSG(v.size() > 0, "The input vector is empty while printing the vector of point2i.");

	std::vector<cv::Point2d> pts_2d_vec = int2double_vec_pts2d(v);
	print_vec_pts2d(pts_2d_vec);
}



void print_vec(std::vector<double>& v, int prec) {
	ASSERT_WITH_MSG(v.size() > 0, "The input vector is empty while printing the vector of double.");
	std::cout << "[";
	for (int i = 0; i < v.size() - 1; i++) {
		std::cout << std::setprecision(prec) << v[i] << ", ";
	}
	std::cout << std::setprecision(prec) << v[v.size() - 1] << "]" << std::endl;
}

void print_vec(std::vector<float>& v, int prec) {
	ASSERT_WITH_MSG(v.size() > 0, "The input vector is empty while printing the vector of float.");

	std::vector<double> double_vec = float2double_vec(v);
	print_vec(double_vec, prec);
}

void print_vec(std::vector<int>& v, int prec) {
	ASSERT_WITH_MSG(v.size() > 0, "The input vector is empty while printing the vector of int.");

	std::vector<double> double_vec = int2double_vec(v);
	print_vec(double_vec, prec);
}



void print_pts2d(cv::Point2d& pts) {
	std::cout << pts << std::endl;
}
// TODO: test for correctness
void print_pts2d(cv::Point2f& pts) {
	std::cout << pts << std::endl;
}
// TODO: test for correctness
void print_pts2d(cv::Point2i& pts) {
	std::cout << pts << std::endl;
}
void print_pts3d(cv::Point3d& pts) {
	std::cout << pts << std::endl;
}
// TODO: test for correctness
void print_pts3d(cv::Point3f& pts) {
	std::cout << pts << std::endl;
}
// TODO: test for correctness
void print_pts3d(cv::Point3i& pts) {
	std::cout << pts << std::endl;
}


void print_vec_pts_2d_conf(std::vector<pts_2d_conf> pts) {
	ASSERT_WITH_MSG(pts.size() > 0, "The size of input vector should not be empty while printing.");
	for (int i = 0; i < pts.size(); i++)
		pts[i].print();
}

void print_vec_pts_3d_conf(std::vector<pts_3d_conf> pts) {
	ASSERT_WITH_MSG(pts.size() > 0, "The size of input vector should not be empty while printing.");
	for (int i = 0; i < pts.size(); i++)
		pts[i].print();
}


/************************************** check function ********************************************/
bool CHECK_VEC_EQ(std::vector<double>& vec1, std::vector<double>& vec2) {
	ASSERT_WITH_MSG(vec1.size() > 0, "The input vector is empty while checking the equality.");
	ASSERT_WITH_MSG(vec1.size() == vec2.size(), "The input vector is empty while checking the equality.");
	for (int i = 0; i < vec1.size(); i++) {
		if (CHECK_SCALAR_EQ(vec1[i], vec2[i]))
			continue;
		else
			return false;
	}
	return true;
}

bool CHECK_VEC_EQ(std::vector<float>& vec1, std::vector<float>& vec2) {
	std::vector<double> double_vec_tmp1 = float2double_vec(vec1);
	std::vector<double> double_vec_tmp2 = float2double_vec(vec2);

	return CHECK_VEC_EQ(double_vec_tmp1, double_vec_tmp2);
}
// TODO: test for correctness
bool CHECK_VEC_EQ(std::vector<int>& vec1, std::vector<int>& vec2) {
	std::vector<double> double_vec_tmp1 = int2double_vec(vec1);
	std::vector<double> double_vec_tmp2 = int2double_vec(vec2);

	return CHECK_VEC_EQ(double_vec_tmp1, double_vec_tmp2);
}

bool CHECK_MAT_EQ(cv::Mat& mat1, cv::Mat& mat2) {
	ASSERT_WITH_MSG(!mat1.empty(), "The input cv::mat is empty while checking the equality.");
	ASSERT_WITH_MSG(mat1.total() == mat2.total(), "The input size of two cv::mat is not equal while checking the equality.");
	ASSERT_WITH_MSG(mat1.channels() == mat2.channels(), "The input size of two cv::mat is not equal while checking the equality.");
	ASSERT_WITH_MSG(mat1.dims == mat2.dims, "The input dimension of two cv::mat is not equal while checking the equality.");
	ASSERT_WITH_MSG(mat1.cols == mat2.cols, "The input columns of two cv::mat is not equal while checking the equality.");
	ASSERT_WITH_MSG(mat1.rows == mat2.rows, "The input rows of two cv::mat is not equal while checking the equality.");
	ASSERT_WITH_MSG(mat1.type() == CV_32F || mat1.type() == CV_64F, "Only CV_32F or CV_64F is supported for cv::mat now while checking the equality.");
	ASSERT_WITH_MSG(mat2.type() == CV_32F || mat2.type() == CV_64F, "Only CV_32F or CV_64F is supported for cv::mat now while checking the equality.");
	print_mat(mat1);
	print_mat(mat2);
	double value1, value2;
	for (int i = 0; i < mat1.rows; i++) {
		for (int j = 0; j < mat1.cols; j++) {
			if (mat1.type() == CV_32F)
				value1 = mat1.at<float>(i, j);
			else
				value1 = mat1.at<double>(i, j);
			if (mat2.type() == CV_32F)
				value2 = mat2.at<float>(i, j);
			else
				value2 = mat2.at<double>(i, j);
			if (CHECK_SCALAR_EQ(value1, value2))
				continue;
			else
				return false;
		}
	}
	return true;
}



bool CHECK_CV_PTS_EQ(cv::Point3d& pts1, cv::Point3d& pts2) {
	if (!CHECK_SCALAR_EQ(pts1.x, pts2.x))
		return false;
	if (!CHECK_SCALAR_EQ(pts1.y, pts2.y))
		return false;
	if (!CHECK_SCALAR_EQ(pts1.z, pts2.z))
		return false;
	return true;
}
// TODO: test for correctness
bool CHECK_CV_PTS_EQ(cv::Point3f& pts1, cv::Point3f& pts2) {
	cv::Point3d pts_3d_tmp1 = float2double_pts3d(pts1);
	cv::Point3d pts_3d_tmp2 = float2double_pts3d(pts2);
	return CHECK_CV_PTS_EQ(pts_3d_tmp1, pts_3d_tmp2);
}
// TODO: test for correctness
bool CHECK_CV_PTS_EQ(cv::Point3i& pts1, cv::Point3i& pts2) {
	cv::Point3d pts_3d_tmp1 = int2double_pts3d(pts1);
	cv::Point3d pts_3d_tmp2 = int2double_pts3d(pts2);
	return CHECK_CV_PTS_EQ(pts_3d_tmp1, pts_3d_tmp2);
}

bool CHECK_CV_PTS_EQ(cv::Point2d& pts1, cv::Point2d& pts2) {
	cv::Point3d pts_3d_tmp1 = cv::Point3d(pts1.x, pts1.y, 0);
	cv::Point3d pts_3d_tmp2 = cv::Point3d(pts2.x, pts2.y, 0);
	return CHECK_CV_PTS_EQ(pts_3d_tmp1, pts_3d_tmp2);
}
// TODO: test for correctness
bool CHECK_CV_PTS_EQ(cv::Point2f& pts1, cv::Point2f& pts2) {
	cv::Point2d pts_2d_tmp1 = float2double_pts2d(pts1);
	cv::Point2d pts_2d_tmp2 = float2double_pts2d(pts2);
	return CHECK_CV_PTS_EQ(pts_2d_tmp1, pts_2d_tmp2);
}
// TODO: test for correctness
bool CHECK_CV_PTS_EQ(cv::Point2i& pts1, cv::Point2i& pts2) {
	cv::Point2d pts_2d_tmp1 = int2double_pts2d(pts1);
	cv::Point2d pts_2d_tmp2 = int2double_pts2d(pts2);
	return CHECK_CV_PTS_EQ(pts_2d_tmp1, pts_2d_tmp2);
}


bool CHECK_SCALAR_EQ(double sca1, double sca2) {
	if (std::abs(sca1 - sca2) > EPS_SMALL)
		return false;
	else
		return true;
}
// TODO: test for correctness
bool CHECK_SCALAR_EQ(float sca1, float sca2) {
	return CHECK_SCALAR_EQ(double(sca1), double(sca2));
}
// TODO: test for correctness
bool CHECK_SCALAR_EQ(int sca1, int sca2) {
	return CHECK_SCALAR_EQ(double(sca1), double(sca2));
}

// TODO: test for correctness
bool CHECK_SCALAR_EQ(int sca1, double sca2) {
	return CHECK_SCALAR_EQ(double(sca1), sca2);
}
// TODO: test for correctness
bool CHECK_SCALAR_EQ(double sca1, int sca2) {
	return CHECK_SCALAR_EQ(sca1, double(sca2));
}



// TODO: test for correctness
bool CHECK_VEC_CV_PTS_EQ(std::vector<cv::Point2d>& pts1, std::vector<cv::Point2d>& pts2) {
	ASSERT_WITH_MSG(pts1.size() > 0, "The size of input vector should not be empty while checking the equality.");
	ASSERT_WITH_MSG(pts1.size() == pts2.size(), "The size of two input vectors should be equal while checking the equality.");
	for (int i = 0; i < pts1.size(); i++) {
		if (CHECK_CV_PTS_EQ(pts1[i], pts2[i]))
			continue;
		else
			return false;
	}
	return true;
}
// TODO: test for correctness
bool CHECK_VEC_CV_PTS_EQ(std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2) {
	ASSERT_WITH_MSG(pts1.size() > 0, "The size of input vector should not be empty while checking the equality.");
	ASSERT_WITH_MSG(pts1.size() == pts2.size(), "The size of two input vectors should be equal while checking the equality.");
	for (int i = 0; i < pts1.size(); i++) {
		if (CHECK_CV_PTS_EQ(pts1[i], pts2[i]))
			continue;
		else
			return false;
	}
	return true;
}
// TODO: test for correctness
bool CHECK_VEC_CV_PTS_EQ(std::vector<cv::Point2i>& pts1, std::vector<cv::Point2i>& pts2) {
	ASSERT_WITH_MSG(pts1.size() > 0, "The size of input vector should not be empty while checking the equality.");
	ASSERT_WITH_MSG(pts1.size() == pts2.size(), "The size of two input vectors should be equal while checking the equality.");
	for (int i = 0; i < pts1.size(); i++) {
		if (CHECK_CV_PTS_EQ(pts1[i], pts2[i]))
			continue;
		else
			return false;
	}
	return true;
}
// TODO: test for correctness
bool CHECK_VEC_CV_PTS_EQ(std::vector<cv::Point3d>& pts1, std::vector<cv::Point3d>& pts2) {
	ASSERT_WITH_MSG(pts1.size() > 0, "The size of input vector should not be empty while checking the equality.");
	ASSERT_WITH_MSG(pts1.size() == pts2.size(), "The size of two input vectors should be equal while checking the equality.");
	for (int i = 0; i < pts1.size(); i++) {
		if (CHECK_CV_PTS_EQ(pts1[i], pts2[i]))
			continue;
		else
			return false;
	}
	return true;
}
// TODO: test for correctness
bool CHECK_VEC_CV_PTS_EQ(std::vector<cv::Point3f>& pts1, std::vector<cv::Point3f>& pts2) {
	ASSERT_WITH_MSG(pts1.size() > 0, "The size of input vector should not be empty while checking the equality.");
	ASSERT_WITH_MSG(pts1.size() == pts2.size(), "The size of two input vectors should be equal while checking the equality.");
	for (int i = 0; i < pts1.size(); i++) {
		if (CHECK_CV_PTS_EQ(pts1[i], pts2[i]))
			continue;
		else
			return false;
	}
	return true;
}
// TODO: test for correctness
bool CHECK_VEC_CV_PTS_EQ(std::vector<cv::Point3i>& pts1, std::vector<cv::Point3i>& pts2) {
	ASSERT_WITH_MSG(pts1.size() > 0, "The size of input vector should not be empty while checking the equality.");
	ASSERT_WITH_MSG(pts1.size() == pts2.size(), "The size of two input vectors should be equal while checking the equality.");
	for (int i = 0; i < pts1.size(); i++) {
		if (CHECK_CV_PTS_EQ(pts1[i], pts2[i]))
			continue;
		else
			return false;
	}
	return true;
}