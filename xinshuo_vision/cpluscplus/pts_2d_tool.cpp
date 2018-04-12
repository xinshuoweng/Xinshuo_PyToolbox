// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu

#include <xinshuo_vision/geometry/pts_2d_tool.h>

pts_2d_conf pts_2d_tool::convert_to_pts_2d_conf() {
	return pts_2d_conf(this->x, this->y, this->conf);
}

void pts_2d_tool::print(int prec) {
	std::cout << "[";
	std::cout << std::setprecision(prec) << this->x << ", ";
	std::cout << std::setprecision(prec) << this->y << "] ";
	std::cout << "conf: " << std::setprecision(prec) << this->conf << ", ";
	std::cout << "anchor: " << this->anchor << std::endl;
}
