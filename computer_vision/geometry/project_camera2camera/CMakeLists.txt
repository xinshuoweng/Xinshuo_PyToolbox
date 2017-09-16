cmake_minimum_required(VERSION 3.7)
project(project_camera2camera)

include_directories(/home/xinshuo/include/pcl-1.8)
include_directories(/home/xinshuo/include/eigen3)
include_directories(/home/xinshuo/include)
include_directories(/opt/VTK-7.0.0/include/vtk-7.0)
include_directories(/usr/include)

link_directories(/usr/local/lib)
link_directories(/usr/lib/x86_64-linux-gnu)
link_directories(/opt/VTK-7.0.0/lib)
link_directories(/home/xinshuo/Qt5.7.0/5.7/gcc_64/lib)

set(CMAKE_CXX_STANDARD 11)

# with visualization
#set(SOURCE_FILES projection_camera2camera.cpp mesh_visualization.cpp io_point.cpp camera_geometry.cpp IdDataParser.cpp mycamera.cpp MyMesh.cpp pts_2d_conf.cpp pts_3d_conf.cpp pts_on_mesh.cpp type_conversion.cpp math_functions.cpp debug_tool.cpp)

# without visualization
set(SOURCE_FILES projection_camera2camera.cpp io_point.cpp camera_geometry.cpp IdDataParser.cpp mycamera.cpp MyMesh.cpp pts_2d_conf.cpp pts_3d_conf.cpp pts_on_mesh.cpp type_conversion.cpp math_functions.cpp debug_tool.cpp)
add_executable(project_camera2camera ${SOURCE_FILES})
#set(SOURCE_FILES main.cpp camera_geometry.cpp IdDataParser.cpp mycamera.cpp MyMesh.cpp pts_2d_conf.cpp pts_3d_conf.cpp pts_on_mesh.cpp type_conversion.cpp pts_2d_tool.cpp math_functions.cpp debug_tool.cpp)
#set(SOURCE_FILES test_mesh_visualization.cpp mesh_visualization.cpp camera_geometry.cpp IdDataParser.cpp mycamera.cpp MyMesh.cpp pts_2d_conf.cpp pts_3d_conf.cpp pts_on_mesh.cpp type_conversion.cpp pts_2d_tool.cpp math_functions.cpp debug_tool.cpp)
#add_executable(project3d ${SOURCE_FILES})

# with visualization
#target_link_libraries(project_camera2camera libvtkCommonCore-7.0.so libQt5Gui.so.5 libvtkCommonMath-7.0.so libvtkRenderingCore-7.0.so libvtkCommonDataModel-7.0.so libpcl_visualization.so libboost_system.so libboost_thread.so libopencv_core.so libopencv_calib3d.so libopencv_imgproc.so libpcl_kdtree.so libpcl_common.so libpcl_io.so)

# without visualization
target_link_libraries(project_camera2camera libboost_system.so libboost_thread.so libopencv_core.so libopencv_calib3d.so libopencv_imgproc.so libpcl_kdtree.so libpcl_common.so libpcl_io.so)
