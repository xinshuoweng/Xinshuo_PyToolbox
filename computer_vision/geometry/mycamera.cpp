// Author: Xinshuo Weng
// Email: xinshuow@andrew.cmu.edu


#include "mycamera.h"
#include <opencv2/calib3d/calib3d.hpp>
#include "debug_tool.h"

mycamera::mycamera() {
	//this->intrinsic = cv::Mat(3, 3, CV_64FC1);
	//this->extrinsic = cv::Mat(3, 4, CV_64FC1);
	//this->distCoeffs = cv::Mat(1, 5, CV_64FC1);
}

mycamera::mycamera(cv::Mat& myintrinsic) {
	this->intrinsic = myintrinsic;
	//this->extrinsic = cv::Mat(3, 4, CV_64FC1);
	//this->distCoeffs = cv::Mat(1, 5, CV_64FC1);
}

mycamera::mycamera(cv::Mat& myintrinsic, cv::Mat& myextrinsic) {
	this->intrinsic = myintrinsic;
	this->extrinsic = myextrinsic;
	//this->distCoeffs = cv::Mat(1, 5, CV_64FC1);
}

mycamera::mycamera(cv::Mat& myintrinsic, cv::Mat& myextrinsic, cv::Mat& mydistCoeffs) {
	this->intrinsic = myintrinsic;
	this->extrinsic = myextrinsic;
	this->distCoeffs = mydistCoeffs;
}

mycamera::mycamera(cv::Mat& myintrinsic, cv::Mat& myextrinsic, cv::Mat& mydistCoeffs, std::string myname) {
	this->intrinsic = myintrinsic;
	this->extrinsic = myextrinsic;
	this->distCoeffs = mydistCoeffs;
	this->name = myname;
}

mycamera::mycamera(cv::Mat& myintrinsic, cv::Mat& myextrinsic, std::string myname) {
	this->intrinsic = myintrinsic;
	this->extrinsic = myextrinsic;
	//this->distCoeffs = cv::Mat(1, 5, CV_64FC1);
	this->name = myname;
}


// return a 3x3 cv::Mat for rotation matrix
cv::Mat mycamera::getRotationMatrix() {
	ASSERT_WITH_MSG(this->extrinsic.empty() == 0, "The extrinsic matrix is null!");
	return extrinsic.colRange(0, 3);
}

// return a 1x3 cv::Mat for rotation matrix
cv::Mat mycamera::getRotationVector() {
	cv::Mat rvec;
	cv::Rodrigues(getRotationMatrix(), rvec);
	return rvec;
}

// return a 1x3 cv::Mat for translation matrix
cv::Mat mycamera::getTranslationVector() {
	ASSERT_WITH_MSG(this->extrinsic.empty() == 0, "The extrinsic matrix is null!");
	return extrinsic.colRange(3, 4);
}

// return a 3x4 cv::Mat for projection matrix
cv::Mat mycamera::getProjectionMatrix() {
	ASSERT_WITH_MSG(this->extrinsic.empty() == 0, "The extrinsic matrix is null!");
	ASSERT_WITH_MSG(this->intrinsic.empty() == 0, "The intrinsic matrix is null!");
	cv::Mat projection(3, 4, CV_64FC1);
	projection = this->intrinsic * this->extrinsic;
	return projection;
}


// print info
void mycamera::print() {
	ASSERT_WITH_MSG(!this->intrinsic.empty(), "The camera intrinsic matrix is empty while printing info.");
	std::cout << "Intrinsic matrix: ";
	print_mat(this->intrinsic);
	ASSERT_WITH_MSG(!this->extrinsic.empty(), "The camera extrinsic matrix is empty while printing info.");
	std::cout << "Extrinsix matrix: ";
	print_mat(this->extrinsic);
	ASSERT_WITH_MSG(!this->distCoeffs.empty(), "The camera extrinsic matrix is empty while printing info.");
	std::cout << "Distortion coefficient: ";
	print_mat(this->distCoeffs);
}


//void mycamera::empty() {
//}

mycamera::~mycamera() {
}
