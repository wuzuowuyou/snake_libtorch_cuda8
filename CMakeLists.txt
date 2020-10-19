cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(snake-refinenet)


SET(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/3rdparty/libtorch)
find_package(Torch REQUIRED)

set(CMAKE_PREFIX_PATH "/home/yhl/software_install/opencv3.2")
find_package(OpenCV REQUIRED)


link_directories(${CMAKE_SOURCE_DIR}/3rdparty/dcn_cpp_plugin/lib)
#link_libraries(dcn_v2_cuda_forward_v2)


include_directories(/usr/include/python3.5m) ##报错找不到 Python.h 加的


include_directories(${CMAKE_SOURCE_DIR}/3rdparty/extreme_utils/include)
link_directories(${CMAKE_SOURCE_DIR}/3rdparty/extreme_utils/lib)
#link_libraries(extreme_utils)


add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_link_libraries(example-app ${OpenCV_LIBS})
target_link_libraries(example-app dcn_v2_cuda_forward_v2 dl extreme_utils)
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
