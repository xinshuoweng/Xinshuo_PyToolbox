#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <iostream>
#include <opencv/cv.h>
#include <pcl/io/auto_io.h>
#include <pcl/registration/icp.h>
#include <unordered_set>

// self-contained library
#include "myheader.h"
#include "math_functions.h"
#include "MyMesh.h"
#include "pts_2d_conf.h"
#include "pts_3d_conf.h"
#include "pts_on_mesh.h"
#include "mycamera.h"
#include "camera_geometry.h"
#include "IdDataParser.h"
#include "io_point.h"
#include "debug_tool.h"
//#include "mesh_visualization.h"
#include "type_conversion.h"


#define consider_dist_test		false
#define consider_skew_test		false

int main(int argc, char* argv[]){
    // read arguments
    if( argc != 10 ){
        fprintf(stderr, "calibration pts_src pts_dst mesh camera_src camera_dst num_pts resize_factor tmp_dir\n");
        return -1;
    }


    // read num_pts, resize_factor
    int num_pts;
    double resize_factor;
    sscanf(argv[7], "%d", &num_pts);	        // 51 keypoints
    sscanf(argv[8], "%lf", &resize_factor);		// downsample scalar
    std::cout << "number of points is " << num_pts << std::endl;
    std::cout << "resize factor is " << resize_factor << std::endl;
    std::cout << "calibration file is " << argv[1] << std::endl;
    std::cout << "source file to process is " << argv[2] << std::endl;
    std::cout << "destination file is " << argv[3] << std::endl;
    std::cout << "mesh file to load is " << argv[4] << std::endl;
//    std::cout << "source camera is " << argv[5] << std::endl;
//    std::cout << "destination camera is " << argv[6] << std::endl;

    //
    std::string spath, filename, ext;
    fileparts(argv[2], "/", &spath, &filename, &ext);
//    std::cout << spath << " " << filename << " " << ext << std::endl;

    // read camera parameters
    std::vector<mycamera> camera_cluster;
    camera_cluster.clear();
    LoadIdCalibration(argv[1], camera_cluster, consider_skew_test, consider_dist_test);		// consider distortion

    // read source and destination camera
    char camera_src_char[10], camera_dst_char[10];
    sprintf(camera_src_char, "%s", argv[5]);	    // source camera
    sprintf(camera_dst_char, "%s", argv[6]);	    // destination camera
    std::string camera_src_str = std::string(camera_src_char);
    std::string camera_dst_str = std::string(camera_dst_char);
    bool check_camera_src = false;
    bool check_camera_dst = false;
    mycamera camera_src, camera_dst;
    for (int camera_index = 0; camera_index < camera_cluster.size(); camera_index++) {
        mycamera camera_tmp = camera_cluster[camera_index];
        if (camera_src_str.compare(camera_tmp.name) == 0) {
            check_camera_src = true;
            camera_src = camera_tmp;
//            std::cout << camera_src.name << std::endl;
        }
        if (camera_dst_str.compare(camera_tmp.name) == 0) {
            check_camera_dst = true;
            camera_dst = camera_tmp;

        }
    }
    std::cout << "source camera is " << camera_src.name << std::endl;
    std::cout << "destination camera is " << camera_dst.name << std::endl;
    ASSERT_WITH_MSG(check_camera_dst && check_camera_src, "source camera or destination camera not found in calibration file!");

    // read mesh
    std::cout << "reading mesh file" << std::endl;
    const clock_t begin_time = clock();
    MyMesh mesh(argv[4], 1);			        // input reconstructed mesh
    std::cout << "It spend " << float(clock() - begin_time) / CLOCKS_PER_SEC << " second to read the mesh file" << std::endl;

    // open destination pts file
    char cmd[1024];
    sprintf(cmd, "%s/%s.pose3d", argv[9], filename.c_str());
    FILE *out = fopen(cmd, "w");
//    int ret = system(cmd);
//    ASSERT_WITH_MSG(ret == 0, "open 3d point file failed!\n");

    // read source pts file
    fprintf(stderr, "processing, the 2d keypoint file is in the %s\n", argv[2]);
    std::cout << "reading 2d keypoint file" << std::endl;
    FILE* in = fopen(argv[2], "r");
    if( in == NULL ){
        fprintf(stderr, "No 2d points have been read from %s\n", argv[2]);
        return -1;
    }

    // for visualization
//    std::vector<std::vector<double>> pts_visualization;
//    std::vector<std::vector<double>> rays;
//    std::vector<double> ray_tmp;
//    cv::Point3d camera_center;

    // projecting to mesh
    double x, y, conf;
    int pts_index = 0;
    std::vector<pts_3d_conf> pts_3d_all(num_pts, pts_3d_conf());    // view, num_pts, pts_3d_conf
    while(fscanf(in, "%lf %lf %lf\n", &x, &y, &conf) > 0){
        fprintf(stderr, "pts %d\n", pts_index);
        x *= resize_factor;
        y *= resize_factor;
        std::cout << "2D coordinate in the original image is (" << x << ", " << y << ")" << std::endl;

        pts_2d_conf pts_2d_tmp(x, y, conf);
        pts_on_mesh* pom = mesh.pts_back_projection_single_view(pts_2d_tmp, camera_src, consider_dist_test);
        pts_3d_conf pts_3d_tmp = pom->convert_to_pts_3d_conf();
        pts_3d_all[pts_index] = pts_3d_tmp;
        pts_index++;

        // for visualization
//        get_3d_ray(pts_2d_tmp, camera_src, camera_center, ray_tmp, consider_dist_test);
//        ASSERT_WITH_MSG(ray_tmp.size() == 4, "The size of the output ray is not correct. Please check!");
//        std::cout << "start point is " << std::endl;
//        print_pts3d(camera_center);
//        std::cout << "ray is " << std::endl;
//        print_vec(ray_tmp);
//        std::cout << "3d point is " << std::endl;
//        pts_3d_tmp.print();

//        pts_visualization.push_back(pts_3d_tmp.convert_to_pts_vec());
//        ray_tmp.push_back(-1);
//        rays.push_back(ray_tmp);

        fprintf(out, "%g %g %g %g\n", pts_3d_tmp.x, pts_3d_tmp.y, pts_3d_tmp.z, pts_3d_tmp.conf);
    }
    fclose(in);
    fclose(out);

//    // for visualization in 3d
//    pts_visualization.push_back(cv2vec_pts3d(camera_center));
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr line_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
//    get_cloud_from_points(pts_visualization, keypoints_cloud_ptr);
//    pcl::PointXYZ pts_start_tmp(camera_center.x, camera_center.y, camera_center.z);
//    std::vector<pcl::PointXYZ> pts_start;
//    std::vector<uint32_t> range;
//    for (int i = 0; i < num_pts; i++) {
//        pts_start.push_back(pts_start_tmp);
//        range.push_back(10000);
//    }
//    get_cloud_from_lines(rays, line_cloud_ptr, pts_start, range);
//    viewer = keypoint_line_mesh_visualization(keypoints_cloud_ptr, line_cloud_ptr, mesh.cloud);
//    while (!viewer->wasStopped())
//    {
//        viewer->spinOnce(10000);
//        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//    }

    // project to 2d on destination camera point of view
    std::cout << "project 3d point back to 2d from the destination camera viewpoint";
    sprintf(cmd, "%s", argv[3]);
    out = fopen(cmd, "w");
//    ASSERT_WITH_MSG(ret == 0, "open the projected 2d point location file failed!");
    for(int pts_index = 0; pts_index < num_pts; pts_index++){
        cv::Mat P(4, 1, CV_64FC1);
        cv::Mat results(3, 1, CV_64FC1);
        P.at<double>(0, 0) = pts_3d_all[pts_index].x;
        P.at<double>(1, 0) = pts_3d_all[pts_index].y;
        P.at<double>(2, 0) = pts_3d_all[pts_index].z;
        P.at<double>(3, 0) = 1;
        results = camera_dst.getProjectionMatrix() * P;
        fprintf(out, "%g %g %g\n", results.at<double>(0, 0) / results.at<double>(2, 0) / resize_factor, results.at<double>(1, 0) / results.at<double>(2, 0) / resize_factor, pts_3d_all[pts_index].conf);
    }
    fclose(out);

    return 0;
}