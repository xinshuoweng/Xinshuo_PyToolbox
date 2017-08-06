#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
//#include <cmath>
#include <iostream>
//#include <opencv/cv.h>
#include <pcl/io/auto_io.h>
//#include <pcl/registration/icp.h>
//#include <unordered_set>

// self-contained library
//#include <myheader.h>
//#include <computer_vision/geometry/MyMesh.h>
//#include <computer_vision/geometry/pts_2d_conf.h>
//#include <computer_vision/geometry/pts_3d_conf.h>
//#include <computer_vision/geometry/pts_on_mesh.h>
//#include <computer_vision/geometry/mycamera.h>
//#include <computer_vision/geometry/camera_geometry.h>
//#include <file_io/camera_io.h>
//#include <visualization/mesh_visualization.h>
//#include <miscellaneous/type_conversion.h>
//#include <miscellaneous/debug_tool.h>

int main(int argc, char* argv[]){
    if( argc != 3 ){
        fprintf(stderr, "obj ply\n");
        return -1;
    }

//    char cmd[1024];
    char str_tmp[1024];

    // read obj mesh file
    std::cout << "reading obj mesh file" << std::endl;
    const clock_t begin_time = clock();

    pcl::PolygonMesh::Ptr poly_ptr_tmp(new pcl::PolygonMesh);
    pcl::io::load(argv[1], *poly_ptr_tmp.get());

//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;			// 3D coordinate indexed by point id
//    cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
//    pcl::fromPCLPointCloud2(poly_ptr_tmp->cloud, *cloud.get());
    std::cout << "It spend " << float(clock() - begin_time) / CLOCKS_PER_SEC << " second to read the mesh file" << std::endl;

    // save converted ply file
    sprintf(str_tmp, "%s", argv[2]);
    const std::string savepath = str_tmp;
    std::cout << "Save mesh as ply to " << savepath << std::endl;
    pcl::io::save(savepath, *(poly_ptr_tmp));

    return 0;
}