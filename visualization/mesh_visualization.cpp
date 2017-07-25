
// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu


#include <visualization/mesh_visualization.h>
#include <math/math_functions.h>


// visualize keypoints and mesh for debuging reprojection from 2d to 3d
boost::shared_ptr<pcl::visualization::PCLVisualizer> keypoint_mesh_visualization(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr keypoints, pcl::PointCloud<pcl::PointXYZ>::ConstPtr mesh) {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(keypoints);
	viewer->addPointCloud<pcl::PointXYZRGB>(keypoints, rgb, "3d keypoints");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "3d keypoints");
	
	viewer->addPointCloud<pcl::PointXYZ>(mesh, "reconstructed mesh");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0.2, "reconstructed mesh");

	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return (viewer);
}

// visualize a stack of rgb cloud. The cloud input is a map. Key is the string id and value is the cloud
boost::shared_ptr<pcl::visualization::PCLVisualizer> keypoint_line_mesh_visualization(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr keypoints, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr line, pcl::PointCloud<pcl::PointXYZ>::ConstPtr mesh) {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_keypoints(keypoints);
	viewer->addPointCloud<pcl::PointXYZRGB>(keypoints, rgb_keypoints, "3d keypoints");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "3d keypoints");

	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_line(line);
	viewer->addPointCloud<pcl::PointXYZRGB>(line, rgb_line, "3d line");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "3d line");

	viewer->addPointCloud<pcl::PointXYZ>(mesh, "reconstructed mesh");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0.05, "reconstructed mesh");

	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return (viewer);
}

// visualize a stack of rgb cloud. The cloud input is a map. Key is the string id and value is the cloud
boost::shared_ptr<pcl::visualization::PCLVisualizer> rgb_cloud_visualization(std::map<std::string, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> cloud_map) {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);

	for (std::map<std::string, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr>::iterator i = cloud_map.begin(); i != cloud_map.end(); i++) {
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(i->second);
		viewer->addPointCloud<pcl::PointXYZRGB>(i->second, rgb, i->first);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, i->first);
	}

	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return (viewer);
}

// get rgb point cloud for a set of points. The default color is red
void get_cloud_from_points(std::vector<std::vector<double>> points, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
	ASSERT_WITH_MSG(points.size() > 0, "The size of points for converting to cloud is zero. Please check the input points");
	uint8_t r(255), g(15), b(15);
	uint32_t rgb_display = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
	
	pcl::PointXYZRGB point;
	for (int i = 0; i < points.size(); i++) {
		ASSERT_WITH_MSG(points[i].size() == 3, "The input set of points is not 3d points. Please check!");
		point.x = points[i][0];
		point.y = points[i][1];
		point.z = points[i][2];

		point.rgb = *reinterpret_cast<float*>(&rgb_display);
		cloud->points.push_back(point);
	}
	
	cloud->width = (int)cloud->points.size();
	cloud->height = 1;
}

// get rgb point cloud for a set of points. The default color is red
void get_cloud_from_points(std::vector<std::vector<float>> points, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
	ASSERT_WITH_MSG(points.size() > 0, "The size of points for converting to cloud is zero. Please check the input points");
	uint8_t r(255), g(15), b(15);
	uint32_t rgb_display = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));

	pcl::PointXYZRGB point;
	for (int i = 0; i < points.size(); i++) {
		ASSERT_WITH_MSG(points[i].size() == 3, "The input set of points is not 3d points. Please check!");
		point.x = points[i][0];
		point.y = points[i][1];
		point.z = points[i][2];

		point.rgb = *reinterpret_cast<float*>(&rgb_display);
		cloud->points.push_back(point);
	}

	cloud->width = (int)cloud->points.size();
	cloud->height = 1;
}



// get rgb point cloud for a single ray. The default color is blue
// range should an integer bigger than 0
void get_cloud_from_line(std::vector<double> line, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointXYZ pts_start, uint32_t range, double interval) {
	ASSERT_WITH_MSG(line.size() == 4, "The size of ray shoule be 4 representing the direction and offset. Please check the input ray");
	uint8_t r(15), g(15), b(255);
	uint32_t rgb_display = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));

	pcl::PointXYZRGB point;
	pcl::PointXYZ direction;
	double norm = sqrt(line[0] * line[0] + line[1] * line[1] + line[2] * line[2] + line[3] * line[3]);
	direction.x = line[0] / norm;
	direction.y = line[1] / norm;
	direction.z = line[2] / norm;

	// go through the line
	for (int i = 0; i < range; i++) {
		// find the projection from [1,1,1] to the direction of provided line
		pcl::PointXYZ ref(i * interval, i * interval, i * interval);
		double inner_product = std::abs(ref.x * direction.x + ref.y * direction.y + ref.z * direction.z);
		point.x = pts_start.x + inner_product * direction.x;
		point.y = pts_start.y + inner_product * direction.y;
		point.z = pts_start.z + inner_product * direction.z;

		point.rgb = *reinterpret_cast<float*>(&rgb_display);
		cloud->points.push_back(point);
	}

	cloud->width = (int)cloud->points.size();
	cloud->height = 1;
}

// get rgb point cloud for a single ray. The default color is blue
// range should an integer bigger than 0
void get_cloud_from_line(std::vector<float> line, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointXYZ pts_start, uint32_t range, double interval) {
	ASSERT_WITH_MSG(line.size() == 4, "The size of ray shoule be 4 representing the direction and offset. Please check the input ray");
	uint8_t r(15), g(15), b(255);
	uint32_t rgb_display = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));

	pcl::PointXYZRGB point;
	pcl::PointXYZ direction;
	double norm = sqrt(line[0] * line[0] + line[1] * line[1] + line[2] * line[2] + line[3] * line[3]);
	direction.x = line[0] / norm;
	direction.y = line[1] / norm;
	direction.z = line[2] / norm;

	// go through the line
	for (int i = 0; i < range; i++) {
		// find the projection from [1,1,1] to the direction of provided line
		pcl::PointXYZ ref(i * interval, i * interval, i * interval);
		double inner_product = std::abs(ref.x * direction.x + ref.y * direction.y + ref.z * direction.z);
		point.x = pts_start.x + inner_product * direction.x;
		point.y = pts_start.y + inner_product * direction.y;
		point.z = pts_start.z + inner_product * direction.z;

		point.rgb = *reinterpret_cast<float*>(&rgb_display);
		cloud->points.push_back(point);
	}

	cloud->width = (int)cloud->points.size();
	cloud->height = 1;
}



// get rgb point cloud for a set of rays. The default color is blue
// range should an integer bigger than 0, which represent the length of the line to visualize
void get_cloud_from_lines(std::vector<std::vector<double>> lines, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::vector<pcl::PointXYZ> pts_start, std::vector<uint32_t> range, double interval) {
	ASSERT_WITH_MSG(lines.size() == pts_start.size(), "The size of input lines should be equal to size of starting points");
	ASSERT_WITH_MSG(range.size() == pts_start.size(), "The size of input lines should be equal to size of length of input lines");
	//ASSERT_WITH_MSG(range.size() == clouds.size(), "The size of length of input lines should be equal to size of clouds to save");

	// define the color of clouds
	uint8_t r(15), g(15), b(255);
	uint32_t rgb_display = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));

	int number_lines = lines.size();
	//int count = 0;

	// define temporary variable
	pcl::PointXYZRGB point;
	point.rgb = *reinterpret_cast<float*>(&rgb_display);
	pcl::PointXYZ direction;
	double norm;
	pcl::PointXYZ pts_start_tmp;

	uint32_t range_tmp;
	std::vector<double> line_tmp;
	for (int count = 0; count < number_lines; count++) {
		line_tmp = lines[count];
		ASSERT_WITH_MSG(line_tmp.size() == 4, "The size of ray shoule be 4 representing the direction and offset. Please check the input ray");
		range_tmp = range[count];
		pts_start_tmp = pts_start[count];
		//pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_tmp = i->second;		// reference

		norm = l2_norm(line_tmp);
		direction.x = line_tmp[0] / norm;
		direction.y = line_tmp[1] / norm;
		direction.z = line_tmp[2] / norm;

//        std::cout << "direction is " << direction.x << " " << direction.y << " " << direction.z << " " << std::endl;

		// go through all the lines
		for (int i = 0; i < range_tmp; i++) {
			// find the projection from [1,1,1] to the direction of provided line
			pcl::PointXYZ ref(i * interval, i * interval, i * interval);
			double inner_product = std::abs(ref.x * direction.x + ref.y * direction.y + ref.z * direction.z);
			point.x = pts_start_tmp.x + inner_product * direction.x;
			point.y = pts_start_tmp.y + inner_product * direction.y;
			point.z = pts_start_tmp.z + inner_product * direction.z;
//            std::cout << "visualized line points are " << point.x << " " << point.y << " " << point.z << std::endl;

			cloud->points.push_back(point);
		}
	}

	// organize the cloud
	cloud->width = (int)cloud->points.size();
	cloud->height = 1;
}


// get rgb point cloud for a set of rays. The default color is blue
// range should an integer bigger than 0, which represent the length of the line to visualize
void get_cloud_from_lines(std::vector<std::vector<float>> lines, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::vector<pcl::PointXYZ> pts_start, std::vector<uint32_t> range, double interval) {
	ASSERT_WITH_MSG(lines.size() == pts_start.size(), "The size of input lines should be equal to size of starting points");
	ASSERT_WITH_MSG(range.size() == pts_start.size(), "The size of input lines should be equal to size of length of input lines");
	//ASSERT_WITH_MSG(range.size() == clouds.size(), "The size of length of input lines should be equal to size of clouds to save");

	// define the color of clouds
	uint8_t r(15), g(15), b(255);
	uint32_t rgb_display = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));

	int number_lines = lines.size();
	//int count = 0;

	// define temporary variable
	pcl::PointXYZRGB point;
	point.rgb = *reinterpret_cast<float*>(&rgb_display);
	pcl::PointXYZ direction;
	double norm;
	pcl::PointXYZ pts_start_tmp;

	uint32_t range_tmp;
	std::vector<float> line_tmp;
	for (int count = 0; count < number_lines; count++) {
		line_tmp = lines[count];
		ASSERT_WITH_MSG(line_tmp.size() == 4, "The size of ray shoule be 4 representing the direction and offset. Please check the input ray");
		range_tmp = range[count];
		pts_start_tmp = pts_start[count];
		//pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_tmp = i->second;		// reference

		norm = l2_norm(line_tmp);
		direction.x = line_tmp[0] / norm;
		direction.y = line_tmp[1] / norm;
		direction.z = line_tmp[2] / norm;

		// go through all the lines
		for (int i = 0; i < range_tmp; i++) {
			// find the projection from [1,1,1] to the direction of provided line
			pcl::PointXYZ ref(i * interval, i * interval, i * interval);
			double inner_product = std::abs(ref.x * direction.x + ref.y * direction.y + ref.z * direction.z);
			point.x = pts_start_tmp.x + inner_product * direction.x;
			point.y = pts_start_tmp.y + inner_product * direction.y;
			point.z = pts_start_tmp.z + inner_product * direction.z;

			cloud->points.push_back(point);
		}
	}

	// organize the cloud
	cloud->width = (int)cloud->points.size();
	cloud->height = 1;
}


