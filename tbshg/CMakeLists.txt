cmake_minimum_required(VERSION 3.2)

project(shg)
set(project_path "/home/zln/shg")
# set(CMAKE_MODULE_PATH "${project_path}/eigen/cmake/;${CMAKE_MODULE_PATH}")

message("${CMAKE_MODULE_PATH}")
# find_package(OpenMP)
# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()
# find_package(Eigen3 3.3 REQUIRED)
set(EIGEN3_INCLUDE_DIRS "./lib")
message("${EIGEN3_INCLUDE_DIRS}")
include_directories(${EIGEN3_INCLUDE_DIRS} ./src /home/zln/anaconda3/include/python3.9/) # /usr/include/python3.8)
add_subdirectory(pybind11)
# link_libraries(Eigen3::Eigen)
pybind11_add_module(tbshg src/tbshg.cpp)

install(
    TARGETS
        tbshg
    LIBRARY
        DESTINATION tbshg
    )

