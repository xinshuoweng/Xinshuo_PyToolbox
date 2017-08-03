// Author: Xinshuo Weng
// Email: xinshuow@andrew.cmu.edu
#pragma once


#ifndef __MYCAEMRA_H_INCLUDED__
#define __MYCAMERA_H_INCLUDED__

#include <myheader.h>

class mycamera
{
public:
	cv::Mat intrinsic;	// 3x3 cv::Matrix for intrinsic parameter
	cv::Mat distCoeffs; // 1x5 (3,5,6) cv::Matrix for distortion coffecient, generally only read 5 parameters
	cv::Mat extrinsic; // 3x4 cv::Matrix for extrinsic parameter
	std::string name;

public:
	mycamera();
	mycamera(cv::Mat& myintrinsic);
	mycamera(cv::Mat& myintrinsic, cv::Mat& myextrinsic);
	mycamera(cv::Mat& myintrinsic, cv::Mat& myextrinsic, cv::Mat& mydistCoeffs);
	mycamera(cv::Mat& myintrinsic, cv::Mat& myextrinsic, std::string myname);
	mycamera(cv::Mat& myintrinsic, cv::Mat& myextrinsic, cv::Mat& mydistCoeffs, std::string name);
	~mycamera();

	cv::Mat getRotationMatrix();		// return a 3x3 cv::Mat for rotation Matrix
	cv::Mat getRotationVector();		// return a 1x3 cv::Mat for rotation Matrix
	cv::Mat getTranslationVector();		// return a 1x3 cv::Mat for translation Matrix
	cv::Mat getProjectionMatrix();		// return a 3x4 cv::Mat for projection matrix
	void print();						// print info
};

#endif
