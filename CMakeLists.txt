cmake_minimum_required(VERSION 2.8.1)
project(testQR)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -g")

find_package(Eigen3)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(testQR testQR.cpp)
