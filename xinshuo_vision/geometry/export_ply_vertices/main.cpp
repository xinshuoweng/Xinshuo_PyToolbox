#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <opencv/cv.h>
#include <pcl/io/auto_io.h>
#include <pcl/registration/icp.h>
#include <unordered_set>

// self-contained library
#include <myheader.h>
#include <computer_vision/geometry/MyMesh.h>
#include <computer_vision/geometry/mycamera.h>
#include <computer_vision/geometry/camera_geometry.h>
#include <file_io/camera_io.h>
#include <visualization/mesh_visualization.h>
#include <miscellaneous/type_conversion.h>

#define consider_dist_test		false
#define consider_skew_test		false

int main(int argc, char* argv[]){
    if( argc != 6 ){
        fprintf(stderr, "calibration mesh mask camera_src output_file\n");
        return -1;
    }

    char cmd[1024];
    int polygon_id;
    std::vector<int> polygon_array;
    char str_tmp[1024];
    double conf_threshold = 0.3;

    // read camera
    std::vector<mycamera> camera_cluster;
    camera_cluster.clear();
    LoadIdCalibration(argv[1], camera_cluster, consider_skew_test, consider_dist_test);		// consider distortion

    // read mask file, only export vertices not in the mask list
    sprintf(str_tmp, "%s", argv[3]);
    fprintf(stderr, "reading mask file %s\n", str_tmp);
    FILE* mask_file = fopen(str_tmp, "r");
    if( mask_file == NULL ){
        fprintf(stderr, "no mask file has been read!!!! error!!!!!!!!!!!!!!!!!!!!\n");
        return -1;
    }
    fscanf(mask_file, "[\n");
    std::cout << "reading polygon id...." << std::endl;
    while(fscanf(mask_file, "    %d,\n", &polygon_id) > 0) {
        std::cout << polygon_id << ", ";

        polygon_array.push_back(polygon_id);
    }
    int size_requested = polygon_array.size();
    std::cout << "size of polygon to export is " << size_requested << std::endl;
    fclose(mask_file);

    // read ply mesh file
    std::cout << "reading mesh file" << std::endl;
    const clock_t begin_time = clock();
    MyMesh mesh(argv[2], 1);			// input reconstructed mesh
    std::cout << "It spend " << float(clock() - begin_time) / CLOCKS_PER_SEC << " second to read the mesh file" << std::endl;

    // read source and reference camera
    char camera_src_char[10];
    sprintf(camera_src_char, "%s", argv[4]);	    // source camera
    std::string camera_src_str = std::string(camera_src_char);
    bool check_camera_src = false;
    mycamera camera_src;
    for (int camera_index = 0; camera_index < camera_cluster.size(); camera_index++) {
        mycamera camera_tmp = camera_cluster[camera_index];
        if (camera_src_str.compare(camera_tmp.name) == 0) {
            check_camera_src = true;
            camera_src = camera_tmp;
        }
    }
    std::cout << "source camera is " << camera_src.name << std::endl;
    ASSERT_WITH_MSG(check_camera_src, "source camera not found in calibration file!\n");

    // for visualization
//    std::vector<std::vector<double>> pts_vis;

    // open the output file
    sprintf(cmd, "%s", argv[5]);
    FILE *out = fopen(cmd, "w");
    int polygon_index = 0;
    std::vector<mycamera> camera_array;
    camera_array.push_back(camera_src);
    for (int polygon_index = 0; polygon_index < size_requested; polygon_index++) {
        fprintf(stderr, "polygon index %d/%d, ", polygon_index, size_requested);
        polygon_id = polygon_array[polygon_index];
        std::vector<int> pts_id_array = mesh.plane_pts_idx[polygon_id];
        for (int vertice_index = 0; vertice_index < pts_id_array.size(); vertice_index++) {
            int vertice_id = pts_id_array[vertice_index];

            pcl::PointXYZ pts_tmp_pcl = mesh.cloud->points[vertice_id];

            // get 3d point
            std::vector<double> pts_tmp;
            pts_tmp.push_back(pts_tmp_pcl.x);
            pts_tmp.push_back(pts_tmp_pcl.y);
            pts_tmp.push_back(pts_tmp_pcl.z);
            fprintf(stderr, "(%g, %g, %g) ", pts_tmp[0], pts_tmp[1], pts_tmp[2]);

            // get projected 2d point
            std::map<std::string, cv::Point2d> pts_dst;
            cv::Point3d pts_src = vec2cv_pts3d(pts_tmp);
            multi_view_projection(pts_src, camera_array, pts_dst, consider_dist_test);

            // write to file
            fprintf(out, "[%g, %g]", pts_dst[camera_src.name].x, pts_dst[camera_src.name].y);

//             for visualization
//            pts_vis.push_back(pts_tmp);
        }
        fprintf(stderr, "\n");
        fprintf(out, "\n");
    }
    fclose(out);

//
//    // for visualization in 3d
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
//    get_cloud_from_points(pts_vis, keypoints_cloud_ptr);
//
//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
//    viewer = keypoint_mesh_visualization(keypoints_cloud_ptr, mesh.cloud);
//    while (!viewer->wasStopped())
//    {
//        viewer->spinOnce(10000);
//        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//    }

    return 0;
}