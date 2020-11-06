cmake_minimum_required(VERSION 2.8.3)
project(cuda_dasp)

add_compile_options(--std=c++11)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    cv_bridge
    image_transport
    sensor_msgs
    nodelet
)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV 3 REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(CUDA REQUIRED)
include("${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake")
list(APPEND CUDA_NVCC_FLAGS "--std=c++11 --expt-relaxed-constexpr")

#generate_messages(
#    DEPENDENCIES
#    geometry_msgs
#    )
    
#generate_dynamic_reconfigure_options(
#)

catkin_package(
    CATKIN_DEPENDS roscpp cv_bridge image_transport sensor_msgs message_filters
    )

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

cuda_include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

cuda_add_library(dasp
    src/dasp.cu
    src/dasp_kernels.cu
)
target_link_libraries(dasp
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    )

add_executable(dasp_node
    src/main.cpp 
    src/dasp_node.cpp
    )
target_link_libraries(dasp_node
    dasp
    )