// Author: Xinshuo Weng
// Email: xinshuow@andrew.cmu.edu

#include <file_io/io_point.h>
#include <computer_vision/geometry/pts_2d_conf.h>
#include <computer_vision/geometry/mycamera.h>

void fileparts(std::string str, std::string separator, std::string *path, std::string *filename, std::string *extension)
{
	std::string spath, sfile, sext;
	std::string tmp;

	spath = str.substr(0, str.find_last_of(separator));
	tmp = str.substr(spath.size()+separator.size(), str.length());
	sfile = tmp.substr(0, tmp.find_last_of("."));
	sext = tmp.substr(sfile.size(), tmp.size());

	*path = spath;
	*filename = sfile;
	*extension = sext;
}

// TODO: test for correctness
void save_point(const char* path, std::vector<cv::Point2d>& pts_src) {
	FILE *out = fopen(path, "w");
	ASSERT_WITH_MSG(out != NULL, "The path for saving point is not valid.");
	for (int i = 0; i < pts_src.size(); i++) {
		fprintf(out, "%g %g\n", pts_src[i].x, pts_src[i].y);
	}
	fclose(out);
}

// TODO: add fileparts
void save_points_with_conf(const char* folder_path, const char* file_name, std::vector<pts_2d_conf>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The size of input points to save should not be empty while saving points with confidence.");
	char file_tmp[1024];
	std::sprintf(file_tmp, "%s\\%s", folder_path, file_name);
	std::cout << "saving 2d data to " << file_tmp << std::endl;
	FILE* out = fopen(file_tmp, "w");
	if (out == NULL) {
		int ret;					// store the return from running command line
		char cmd[1024];				// store the command line
		std::sprintf(cmd, "if not exist %s mkdir %s", folder_path, folder_path);		// create folder 
		ret = system(cmd);
		ASSERT_WITH_MSG(ret == 0, "create folder failed while saving points with confidence.");
		out = fopen(file_tmp, "w");
		ASSERT_WITH_MSG(out != NULL, "The path for saving point is not valid.");
	}
	for (int i = 0; i < pts_src.size(); i++) {
		fprintf(out, "%g %g %g\n", pts_src[i].x, pts_src[i].y, pts_src[i].conf);
	}
	fclose(out);
}


void save_points_with_conf_multiview(std::string path, int frame, std::map<std::string, std::vector<pts_2d_conf>>& pts_src) {
	ASSERT_WITH_MSG(pts_src.size() > 0, "The size of input points to save should not be empty.");
	std::cout << "saving 2d data for multiple view to folder: " << path << std::endl;
	char folder_path[1024];
	char file_name[1024];
	for (std::map<std::string, std::vector<pts_2d_conf>>::iterator it = pts_src.begin(); it != pts_src.end(); it++) {
		std::sprintf(folder_path, "%s\\cam%s", path.c_str(), it->first.c_str());
		std::sprintf(file_name, "%05d.pose", frame);
		save_points_with_conf(folder_path, file_name, it->second);
	}
}

// TODO: test for correctness
void load_point(const char* path, std::vector<cv::Point2d>& pts_dst, std::vector<double>& conf_vec) {
	FILE* in = fopen(path, "r");
	ASSERT_WITH_MSG(in != NULL, "The path for loading point is not valid.");
	double x, y, conf;
	while (fscanf(in, "%lf %lf %lf\n", &x, &y, &conf) > 0) {	// read all keypoint coordinates and confidence
		pts_dst.push_back(cv::Point2d(x, y));
		conf_vec.push_back(conf);
	}
	std::fclose(in);
}

void load_points_with_conf(const char* path, std::vector<pts_2d_conf>& pts_dst, double resize_factor, int number_pts_load) {
	std::cout << "loading 2d data from " << path << std::endl;
	FILE* in = fopen(path, "r");
	ASSERT_WITH_MSG(in != NULL, "The path to load points is null.");
	double x, y, conf;
	int count = 0;
	while (fscanf(in, "%lf %lf %lf\n", &x, &y, &conf) > 0) {	// read all keypoint coordinates and confidence
		pts_dst.push_back(pts_2d_conf(resize_factor * x, resize_factor * y, conf));
		count += 1;
		if (count >= number_pts_load)
			break;
	}
	std::fclose(in);
}

void load_points_with_conf_multiview(std::string path, std::vector<mycamera> camera_cluster, int frame, std::map<std::string, \
std::vector<pts_2d_conf>>& pts_dst, double resize_factor, int number_pts_load) {
	ASSERT_WITH_MSG(camera_cluster.size() > 0, "The size of subfolder for camera view should not be empty.");
	std::cout << "loading 2d data for multiple view from folder: " << path << std::endl;
	char file_tmp[1024];
	std::vector<pts_2d_conf> pts_tmp;
	for (int i = 0; i < camera_cluster.size(); i++) {
		pts_tmp.clear();
		std::sprintf(file_tmp, "%s\\cam%s\\%05d.pose", path.c_str(), camera_cluster[i].name.c_str(), frame);
		load_points_with_conf(file_tmp, pts_tmp, resize_factor, number_pts_load);
		pts_dst[camera_cluster[i].name] = pts_tmp;
	}
}
