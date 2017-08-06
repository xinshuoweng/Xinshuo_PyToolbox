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
    if( argc != 11 ){
        fprintf(stderr, "calibration pose_dir mesh num_pts camera_src frame pose3dout_dir resize_factor pose_ref_dir camera_ref\n");
        return -1;
    }

    char cmd[1024];
    int num_pts;
    int frame;
    double resize_factor;
    char str_tmp[1024];
    double conf_threshold = 0.3;
    sscanf(argv[4], "%d", &num_pts);	// 51 keypoints
    sscanf(argv[6], "%d", &frame);		// frame number
    sscanf(argv[8], "%lf", &resize_factor);		// downsample scalar

    // read camera
    std::vector<mycamera> camera_cluster;
    camera_cluster.clear();
    LoadIdCalibration(argv[1], camera_cluster, consider_skew_test, consider_dist_test);		// consider distortion

    // read ply mesh file
    std::cout << "reading mesh file" << std::endl;
    const clock_t begin_time = clock();
    MyMesh mesh(argv[3], 1);			// input reconstructed mesh
    std::cout << "It spend " << float(clock() - begin_time) / CLOCKS_PER_SEC << " second to read the mesh file" << std::endl;

    // save converted obj file
    sprintf(cmd, "mkdir -p %s/obj", argv[7]);
    int ret = system(cmd);
    ASSERT_WITH_MSG(ret == 0, "creating folder for converted obj file failed!");
    sprintf(str_tmp, "%s/obj/%05d.obj", argv[7], frame);
    const std::string savepath = str_tmp;
    std::cout << "Save mesh as obj to " << savepath << std::endl;
    pcl::io::save(savepath, *(mesh.poly_ptr));

    // read source and reference camera
    char camera_src_char[10], camera_ref_char[10];
    sprintf(camera_src_char, "%s", argv[5]);	    // source camera
    sprintf(camera_ref_char, "%s", argv[10]);	    // destination camera
    std::string camera_src_str = std::string(camera_src_char);
    std::string camera_ref_str = std::string(camera_ref_char);
    bool check_camera_src = false;
    bool check_camera_ref = false;
    mycamera camera_src, camera_ref;
    for (int camera_index = 0; camera_index < camera_cluster.size(); camera_index++) {
        mycamera camera_tmp = camera_cluster[camera_index];
        if (camera_src_str.compare(camera_tmp.name) == 0) {
            check_camera_src = true;
            camera_src = camera_tmp;
        }
        if (camera_ref_str.compare(camera_tmp.name) == 0) {
            check_camera_ref = true;
            camera_ref = camera_tmp;
        }
    }
    std::cout << "source camera is " << camera_src.name << std::endl;
    std::cout << "reference camera is " << camera_ref.name << std::endl;
    ASSERT_WITH_MSG(check_camera_ref && check_camera_src, "source camera or reference camera not found in calibration file!");




    std::map<std::string, std::vector<pcl::PointXY>> pts_2d_allviews;
    std::vector<pts_3d_conf> pts_3d_allviews(num_pts, pts_3d_conf());    // view, num_pts, pts_3d_conf


//     for visualization
//    std::vector<std::vector<double>> pts_visualization;
//    std::vector<std::vector<double>> rays;
//    std::vector<double> ray_tmp;
//    cv::Point3d camera_center;

    // create 3d point folder and open the file to write
    sprintf(cmd, "mkdir -p %s/pose3d/cam%s", argv[7], argv[5]);
    ret = system(cmd);
    ASSERT_WITH_MSG(ret == 0, "creating folder for 3d locations failed!");
    sprintf(cmd, "%s/pose3d/cam%s/%05d.pose", argv[7], argv[5], frame);
    FILE *out = fopen(cmd, "w");
    ASSERT_WITH_MSG(ret == 0, "open 3d point file failed!");

    // read 2d point file
    sprintf(str_tmp, "%s/%05d.pose", argv[2], frame);
    fprintf(stderr, "reading 2d keypoint file %s\n", str_tmp);
    FILE* in = fopen(str_tmp, "r");
    if( in == NULL ){
        fprintf(stderr, "No 2d points have been read, skipping %s\n", camera_src.name.c_str());
        return -1;
    }
    sprintf(str_tmp, "%s/%05d.pose", argv[9], frame);
    fprintf(stderr, "reading 2d keypoint reference file %s\n", str_tmp);
    FILE* in_ref = fopen(str_tmp, "r");
    if( in_ref == NULL ){
        fprintf(stderr, "No 2d reference points have been read, skipping %s\n", camera_ref.name.c_str());
        return -1;
    }

    double x, y, conf, x_ref, y_ref, conf_ref;
    int pts_index = 0;

    std::vector<pcl::PointXY> parts2d;
    while(fscanf(in, "%lf %lf %lf\n", &x, &y, &conf) > 0){
        if (pts_index >= num_pts) {
            break;
        }

        fprintf(stderr, "point %d\n", pts_index);

        fscanf(in_ref, "%lf %lf %lf\n", &x_ref, &y_ref, &conf_ref);

        // rescale the point location
        x *= resize_factor;
        y *= resize_factor;
        x_ref *= resize_factor;
        y_ref *= resize_factor;
        pcl::PointXY p;
        p.x = x;
        p.y = y;

        std::cout << "2D coordinate in the original image is (" << x << ", " << y << ", " << conf << ")" << std::endl;
        std::cout << "2D coordinate in the reference image is (" << x_ref << ", " << y_ref << ", " << conf_ref << ")" << std::endl;
        parts2d.push_back(p);
        pts_2d_allviews[camera_src.name] = parts2d;

        pts_2d_conf pts_2d_tmp(x, y, conf);
        pts_2d_conf pts_2d_ref(x_ref, y_ref, conf_ref);
        pts_on_mesh* pom;
        if (conf <= conf_threshold || conf_ref <= conf_threshold) {
            pom = new pts_on_mesh(-1, -1, 0, 0, 0, 0);
        }
        else {
            pom = mesh.pts_back_projection_single_view(pts_2d_tmp, pts_2d_ref, camera_src, camera_ref, consider_dist_test);
        }

        pts_3d_conf pts_3d_tmp = pom->convert_to_pts_3d_conf();
        pts_3d_allviews[pts_index] = pts_3d_tmp;
        fprintf(out, "%g %g %g %g %d %d\n", pts_3d_tmp.x, pts_3d_tmp.y, pts_3d_tmp.z, pts_3d_tmp.conf, pom->vertice_id, pom->triangle_id);
        pts_index++;


        // for visualization
//            get_3d_ray(pts_2d_tmp, camera_src, camera_center, ray_tmp, consider_dist_test);
//            ASSERT_WITH_MSG(ray_tmp.size() == 4, "The size of the output ray is not correct. Please check!");
//            std::cout << "start point is " << std::endl;
//            print_pts3d(camera_center);
//            std::cout << "ray is " << std::endl;
//            print_vec(ray_tmp);
//            std::cout << "3d point is " << std::endl;
//            pts_3d_tmp.print();

//            pts_visualization.push_back(pts_3d_tmp.convert_to_pts_vec());
//            ray_tmp.push_back(-1);
//            rays.push_back(ray_tmp);

    }

    fclose(in);
    fclose(in_ref);
    fclose(out);


    // for visualization in 3d
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
//

//
//    Eigen::MatrixXf mymatrix(svd_m.size(), 4);
//    for(int i = 0; i < svd_m.size(); i++){
//        mymatrix(i, 0) = svd_m[i].x;
//        mymatrix(i, 1) = svd_m[i].y;
//        mymatrix(i, 2) = svd_m[i].z;
//        mymatrix(i, 3) = 1;
//    }
//    Eigen::JacobiSVD<Eigen::MatrixXf> svd(mymatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
//    vector<float> cutting_plane;
//    auto matV = svd.matrixV();
//    for(int i = 0; i < 4; i++){
//        cutting_plane.push_back(matV(i, 3)); //last column
//    }
//
//    float acc = 0;
//    acc += svd_m[0].x * cutting_plane[0]; //svd_m[0] is left eyebrow
//    acc += svd_m[0].y * cutting_plane[1]; //svd_m[0] is left eyebrow
//    acc += svd_m[0].z * cutting_plane[2]; //svd_m[0] is left eyebrow
//    acc += cutting_plane[3];
//    cutting_plane[3] -= acc; //if the plane was distance d away, now make it 0 away
//    acc = 0;
//    acc += svd_m[13].x * cutting_plane[0]; //svd_m[13] is tip of nose
//    acc += svd_m[13].y * cutting_plane[1];
//    acc += svd_m[13].z * cutting_plane[2];
//    acc += cutting_plane[3];
//
//    sprintf(cmd, "%s/del/%05d.obj", argv[6], frame);
//    out = fopen(cmd, "w");
//    assert(out != NULL);
//    unordered_set<int> faces_to_delete;
//    int pts = mesh.cloud->size();
//    map<int, int> remap_vertices;
//    for(int i = 0; i < pts; i++){
//        float acc2 = 0;
//        acc2 += cutting_plane[0] * mesh.cloud->points[i].x;
//        acc2 += cutting_plane[1] * mesh.cloud->points[i].y;
//        acc2 += cutting_plane[2] * mesh.cloud->points[i].z;
//        acc2 += cutting_plane[3];
//        if( acc * acc2 < 0 || inside_mouth_eye(Ms[330030] ,mesh.cloud->points[i], parts2d_all_cams[330030]) ){ // not on same side as the detected points, throw away
//            for(int j = 0; j < mesh.planes_of_vertices[i].size(); j++){
//                faces_to_delete.insert(mesh.planes_of_vertices[i][j]);
//            }
//        }else{
//            remap_vertices[ i ] = remap_vertices.size();
//            fprintf(out, "v %g %g %g\n", mesh.cloud->points[i].x, mesh.cloud->points[i].y, mesh.cloud->points[i].z);
//        }
//    }
//
//    for(int i = 0; i < mesh.plane_pts_idx.size(); i++){
//        if( faces_to_delete.find(i) == faces_to_delete.end() ){
//            int ok = 1;
//            for(int j = 0; j < 3; j++){
//                if( remap_vertices.find( mesh.plane_pts_idx[i][j] ) == remap_vertices.end() ){
//                    ok = 0;
//                    break;
//                }
//            }
//            if( ok == 1 )
//                fprintf(out, "f %d %d %d\n", remap_vertices[ mesh.plane_pts_idx[i][0] ], remap_vertices[ mesh.plane_pts_idx[i][1] ], remap_vertices[ mesh.plane_pts_idx[i][2] ]);
//        }
//    }
//    fclose(out);

    std::cout << "back projecting 3d point to 2d." << std::endl;
    for (int camera_index = 0; camera_index < camera_cluster.size(); camera_index++) {
        mycamera camera_tmp = camera_cluster[camera_index];

        // create folder
        sprintf(cmd, "mkdir -p %s/back2d/%s", argv[7], camera_tmp.name.c_str());
        ret = system(cmd);
        ASSERT_WITH_MSG(ret == 0, "creating folder for back projecting 2d locations on camera failed!");

        // write output to file
        sprintf(cmd, "%s/back2d/%s/%05d.pose", argv[7], camera_tmp.name.c_str(), frame);
        FILE *out = fopen(cmd, "w");
        ASSERT_WITH_MSG(ret == 0, "opening folder for back projecting 2d locations failed");
        for(int pts_index = 0; pts_index < num_pts; pts_index++){
//            if( closest[i] == -1 ){
//                fprintf(out, "-1 -1 -1\n");
//                continue;
//            }
            cv::Mat P(4, 1, CV_64FC1);
            P.at<double>(0, 0) = pts_3d_allviews[pts_index].x;
            P.at<double>(1, 0) = pts_3d_allviews[pts_index].y;
            P.at<double>(2, 0) = pts_3d_allviews[pts_index].z;
            P.at<double>(3, 0) = 1;
            P = camera_tmp.getProjectionMatrix() * P;       // projection 3d to 2d
            fprintf(out, "%g %g %g\n", P.at<double>(0, 0) / P.at<double>(2, 0) / resize_factor, P.at<double>(1, 0) / P.at<double>(2, 0) / resize_factor, pts_3d_allviews[pts_index].conf);
        }
        fclose(out);
    }
    return 0;
}