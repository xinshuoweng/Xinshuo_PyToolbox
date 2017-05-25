// Author: Xinshuo Weng
// Email: xinshuow@andrew.cmu.edu


#include "myheader.h"
#include "type_conversion.h"
#include "debug_tool.h"

#include <opencv2/highgui/highgui.hpp>


int main(int argc, char** argv)
{
	// test vec2cv_pts2d and cv2vec_pts2d
	std::cout << std::endl;
	std::cout << "Testing vec2cv_pts2d and cv2vec_pts2d............." << std::endl;
	cv::Point2d pts_test1(1, 2);
	std::vector<double> pts_test2 = {1, 2};
	print_pts2d(pts_test1);
	print_vec(cv2vec_pts2d(pts_test1));
	print_vec(pts_test2);
	print_pts2d(vec2cv_pts2d(pts_test2));
	ASSERT_WITH_MSG(CHECK_VEC_EQ(pts_test2, cv2vec_pts2d(pts_test1)), "Something wrong about cv2vec_pts2d.");
	ASSERT_WITH_MSG(CHECK_CV_PTS_EQ(vec2cv_pts2d(pts_test2), pts_test1), "Something wrong about cv2vec_pts2d.");

	// test vec2cv_pts3d and cv2vec_pts3d
	std::cout << std::endl;
	std::cout << "Testing vec2cv_pts3d and cv2vec_pts3d............." << std::endl;
	cv::Point3d pts_test1_3d(1, 2, 3);
	std::vector<double> pts_test2_3d = { 1, 2, 3 };
	print_pts3d(pts_test1_3d);
	print_vec(cv2vec_pts3d(pts_test1_3d));
	print_vec(pts_test2_3d);
	print_pts3d(vec2cv_pts3d(pts_test2_3d));
	ASSERT_WITH_MSG(CHECK_VEC_EQ(pts_test2_3d, cv2vec_pts3d(pts_test1_3d)), "Something wrong about cv2vec_pts3d.");
	ASSERT_WITH_MSG(CHECK_CV_PTS_EQ(vec2cv_pts3d(pts_test2_3d), pts_test1_3d), "Something wrong about cv2vec_pts3d.");


	std::cout << std::endl;
	std::cout << "Testing done! Everything is good if you think.............." << std::endl;
	system("pause");
}

