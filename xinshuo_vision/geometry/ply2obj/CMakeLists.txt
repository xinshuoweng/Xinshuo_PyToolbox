cmake_minimum_required(VERSION 3.7)
project(ply2obj)

message(STATUS "start running cmake...")

#find_package(PCL 1.8 REQUIRED)
#
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

include_directories(/home/xinshuo/include/pcl-1.8)
include_directories(/home/xinshuo/include/eigen3)
include_directories(/home/xinshuo/include)
include_directories(/opt/VTK-7.0.0/include/vtk-7.0)
include_directories(/usr/include)
include_directories(/home/xinshuo/lib/xinshuo_toolbox)

link_directories(/usr/local/lib)
link_directories(/usr/lib/x86_64-linux-gnu)
link_directories(/opt/VTK-7.0.0/lib)
link_directories(/home/xinshuo/Qt5.7.0/5.7/gcc_64/lib)

# without visualization
file(GLOB LIBS "/home/xinshuo/lib/xinshuo_toolbox/xinshuo_vision/geometry/*.cpp" "/home/xinshuo/lib/xinshuo_toolbox/xinshuo_io/*.cpp" "/home/xinshuo/lib/xinshuo_toolbox/xinshuo_miscellaneous/type_conversion.cpp" "/home/xinshuo/lib/xinshuo_toolbox/xinshuo_miscellaneous/debug_tool.cpp" "/home/xinshuo/lib/xinshuo_toolbox/xinshuo_math/*.cpp")

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp)
add_executable(ply2obj ${SOURCE_FILES} ${LIBS})

# without visualization
target_link_libraries(ply2obj libboost_system.so libboost_thread.so libopencv_core.so libopencv_calib3d.so libopencv_imgproc.so libpcl_kdtree.so libpcl_common.so libpcl_io.so)
