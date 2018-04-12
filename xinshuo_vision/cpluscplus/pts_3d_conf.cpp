// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu

#include <xinshuo_vision/geometry/pts_3d_conf.h>

// TODO: test for correctness
cv::Point3d pts_3d_conf::convert_to_point3d() {
	return cv::Point3d(this->x, this->y, this->z);
}

// TODO: test for correctness
std::vector<double> pts_3d_conf::convert_to_pts_vec() {
	std::vector<double> pts_vec;
	pts_vec.push_back(this->x);
	pts_vec.push_back(this->y);
	pts_vec.push_back(this->z);
	return pts_vec;
}

// TODO: test for correctness
std::vector<double> pts_3d_conf::convert_to_pts_vec_conf() {
	std::vector<double> pts_vec_conf;
	pts_vec_conf.push_back(this->x);
	pts_vec_conf.push_back(this->y);
	pts_vec_conf.push_back(this->z);
	pts_vec_conf.push_back(this->conf);
	return pts_vec_conf;
}

void pts_3d_conf::print(int prec) {
	std::cout << "[";
	std::cout << std::setprecision(prec) << this->x << ", ";
	std::cout << std::setprecision(prec) << this->y << ", ";
	std::cout << std::setprecision(prec) << this->z << "] ";
	std::cout << "conf: " << std::setprecision(prec) << this->conf << std::endl;
}
