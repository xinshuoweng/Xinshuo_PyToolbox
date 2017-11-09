// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu

#include <xinshuo_vision/geometry/pts_2d_conf.h>

cv::Point2d pts_2d_conf::convert_to_point2d() {
	return cv::Point2d(this->x, this->y);
}

std::vector<double> pts_2d_conf::convert_to_pts_vec() {
	std::vector<double> pts_vec;
	pts_vec.push_back(this->x);
	pts_vec.push_back(this->y);
	return pts_vec;
}


std::vector<double> pts_2d_conf::convert_to_pts_vec_conf() {
	std::vector<double> pts_vec_conf;
	pts_vec_conf.push_back(this->x);
	pts_vec_conf.push_back(this->y);
	pts_vec_conf.push_back(this->conf);
	return pts_vec_conf;
}

void pts_2d_conf::print(int prec) {
	std::cout << "[";
	std::cout << std::setprecision(prec) << this->x << ", ";
	std::cout << std::setprecision(prec) << this->y << "] ";
	std::cout << "conf: " << std::setprecision(prec) << this->conf << std::endl;
}
