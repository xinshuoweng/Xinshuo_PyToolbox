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
#include <computer_vision/geometry/pts_2d_conf.h>
#include <computer_vision/geometry/pts_3d_conf.h>
#include <computer_vision/geometry/pts_on_mesh.h>
#include <computer_vision/geometry/mycamera.h>
#include <computer_vision/geometry/camera_geometry.h>
#include <file_io/camera_io.h>
#include <visualization/mesh_visualization.h>
#include <miscellaneous/type_conversion.h>

#define consider_dist_test		false
#define consider_skew_test		false

int main(int argc, char* argv[]){
    if( argc != 6 ){
        fprintf(stderr, "calibration pose_3d_file num_pts frame pose2d_out_dir\n");
        return -1;
    }
    int frame;
    sscanf(argv[4], "%d", &frame);		// frame number
    char cmd[1024];
    int num_pts;
    char str_tmp[1024];
    sscanf(argv[3], "%d", &num_pts);	// 51 keypoints

    std::vector<mycamera> camera_cluster;
    camera_cluster.clear();
    LoadIdCalibration(argv[1], camera_cluster, consider_skew_test, consider_dist_test);		// consider distortion

    int ret = system(cmd);
    std::vector<pts_3d_conf> pts_3d_allviews(std::vector<pts_3d_conf>(num_pts, pts_3d_conf()));    // view, num_pts, pts_3d_conf
    sprintf(str_tmp, "%s", argv[2]);
    FILE* in = fopen(str_tmp, "r");
    if( in == NULL ){
        fprintf(stderr, "No 3d points have been read\n");
        return -1;
    }

    double x, y, z, conf, id1, id2;
    int pts_index = 0;
    while(fscanf(in, "%lf %lf %lf %lf %lf %lf\n", &x, &y, &z, &conf, &id1, &id2) > 0){
        fprintf(stderr, "point %d\n", pts_index);
        pts_3d_allviews[pts_index] = pts_3d_conf(x, y, z, conf);
        pts_index++;
        if (pts_index == num_pts)
            break;
    }
    fclose(in);


    std::cout << "back projecting 3d point to 2d." << std::endl;
    for (int camera_index = 0; camera_index < camera_cluster.size(); camera_index++) {
        mycamera camera_tmp = camera_cluster[camera_index];

        sprintf(cmd, "mkdir -p %s/%s", argv[5], camera_tmp.name.c_str());
        ret = system(cmd);
        ASSERT_WITH_MSG(ret == 0, "creating folder for back projecting 2d locations on camera failed!");

        // write output to file
        sprintf(cmd, "%s/%s/%05d.pose", argv[5], camera_tmp.name.c_str(), frame);
        FILE *out = fopen(cmd, "w");
        ASSERT_WITH_MSG(ret == 0, "opening folder for back projecting 2d locations failed");
        for(int pts_index = 0; pts_index < num_pts; pts_index++){
            cv::Mat P(4, 1, CV_64FC1);
            P.at<double>(0, 0) = pts_3d_allviews[pts_index].x;
            P.at<double>(1, 0) = pts_3d_allviews[pts_index].y;
            P.at<double>(2, 0) = pts_3d_allviews[pts_index].z;
            P.at<double>(3, 0) = 1;
            P = camera_tmp.getProjectionMatrix() * P;       // projection 3d to 2d
            fprintf(out, "%g %g %g\n", P.at<double>(0, 0) / P.at<double>(2, 0), P.at<double>(1, 0) / P.at<double>(2, 0), pts_3d_allviews[pts_index].conf);
        }
        fclose(out);
    }
    return 0;
}