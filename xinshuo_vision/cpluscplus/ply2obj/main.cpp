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
#include <xinshuo_vision/geometry/MyMesh.h>
#include <xinshuo_vision/geometry/pts_2d_conf.h>
#include <xinshuo_vision/geometry/pts_3d_conf.h>
#include <xinshuo_vision/geometry/pts_on_mesh.h>
#include <xinshuo_vision/geometry/mycamera.h>
#include <xinshuo_vision/geometry/camera_geometry.h>
#include <xinshuo_io/camera_io.h>
#include <xinshuo_visualization/mesh_visualization.h>
#include <xinshuo_miscellaneous/type_conversion.h>

#define consider_dist_test		false
#define consider_skew_test		false

int main(int argc, char* argv[]){
    if( argc != 3 ){
        fprintf(stderr, "ply obj\n");
        return -1;
    }

    char str_tmp[1024];
 
    // read ply mesh file
    std::cout << "reading mesh file" << std::endl;
    const clock_t begin_time = clock();
    MyMesh mesh(argv[1], 1);			// input reconstructed mesh
    std::cout << "It spend " << float(clock() - begin_time) / CLOCKS_PER_SEC << " second to read the mesh file" << std::endl;

    // save converted obj file
    // sprintf(cmd, "mkdir -p %s/obj", argv[2]);
    sprintf(str_tmp, "%s", argv[2]);
    // int ret = system(cmd);
    // ASSERT_WITH_MSG(ret == 0, "creating folder for converted obj file failed!");
    // sprintf(str_tmp, "%s/obj/%05d.obj", argv[7], frame);
    const std::string savepath = str_tmp;
    std::cout << "Save mesh as obj to " << savepath << std::endl;
    pcl::io::save(savepath, *(mesh.poly_ptr));

    return 0;
}