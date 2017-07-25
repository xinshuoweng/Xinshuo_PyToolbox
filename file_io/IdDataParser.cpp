// Author: Takaaki
// Modified by: Xinshuo
// Email: xinshuow@andrew.cmu.edu

//#include <iterator> // for ostream_iterator
#include "IdDataParser.h"
#include "mycamera.h"
#include "debug_tool.h"


//using namespace std;

bool LoadIdCalibration(const std::string& calib_fpath, std::vector<mycamera>& out_cameras, const bool consider_skew, const bool consider_dist) {
	std::ifstream calib_content(calib_fpath.c_str());
	if (calib_content.fail())
	{
		std::cout << "Error: fail to open calibration file" << std::endl;
		std::cout << "\t input file path: " << calib_fpath << std::endl;

		out_cameras.clear();

		return false;
	}

	out_cameras.clear();

	std::string curr_line_buf;
	std::string separater_buf;
	while (1)
	{

		std::getline(calib_content, curr_line_buf);
		// for first camera
		if (!curr_line_buf.empty())
		{
			std::string camera_name = curr_line_buf;
//			std::cout << "camera ID:" << camera_name << std::endl;
//			std::cout << std::endl;

			// intrinsics
			cv::Mat new_intrinsics = cv::Mat(3, 3, CV_64FC1);
			calib_content
				>> new_intrinsics.at<double>(0, 0) >> new_intrinsics.at<double>(0, 1) >> new_intrinsics.at<double>(0, 2)
				>> new_intrinsics.at<double>(1, 0) >> new_intrinsics.at<double>(1, 1) >> new_intrinsics.at<double>(1, 2)
				>> new_intrinsics.at<double>(2, 0) >> new_intrinsics.at<double>(2, 1) >> new_intrinsics.at<double>(2, 2);
//			std::cout << "intrinsics:" << std::endl;
//			print_mat(new_intrinsics, 12);

			if (!consider_skew)
			{
				//new_intrinsics(0, 1) = 0;
				new_intrinsics.at<double>(0, 1) = 0;
			}

			// distortion params
			cv::Mat new_dist = cv::Mat(1, 5, CV_64FC1);
			calib_content
				>> new_dist.at<double>(0, 0) >> new_dist.at<double>(0, 1) >> new_dist.at<double>(0, 2) >> new_dist.at<double>(0, 3) >> new_dist.at<double>(0, 4);
//			std::cout << "distortion parameters:" << std::endl;
//			print_mat(new_dist, 12);

			// extrinsics
			cv::Mat new_extrinsics = cv::Mat(3, 4, CV_64FC1);
			calib_content
				>> new_extrinsics.at<double>(0, 0) >> new_extrinsics.at<double>(0, 1) >> new_extrinsics.at<double>(0, 2) >> new_extrinsics.at<double>(0, 3)
				>> new_extrinsics.at<double>(1, 0) >> new_extrinsics.at<double>(1, 1) >> new_extrinsics.at<double>(1, 2) >> new_extrinsics.at<double>(1, 3)
				>> new_extrinsics.at<double>(2, 0) >> new_extrinsics.at<double>(2, 1) >> new_extrinsics.at<double>(2, 2) >> new_extrinsics.at<double>(2, 3);
//			std::cout << "extrinsics:" << std::endl;
//			print_mat(new_extrinsics, 12);
//			std::cout << "***********************************************************************" << std::endl << std::endl;

			// set camera model
			mycamera new_color_camera;
			if (consider_dist == false) {
				new_color_camera = mycamera(new_intrinsics, new_extrinsics, camera_name);
				//std::cout << "Warning: The distortion coefficient is not considered. Please check." << std::endl;
			}
			else
				new_color_camera = mycamera(new_intrinsics, new_extrinsics, new_dist, camera_name);
			out_cameras.push_back(new_color_camera);

			std::getline(calib_content, separater_buf); // \n of the last line of extrinsics
			std::getline(calib_content, separater_buf); // space after extrinsics

		}
		else // two consecutive lines are empty
			break;
	}

	std::cout << "# total cameras from parser: " << out_cameras.size() << std::endl;
	return true;
}