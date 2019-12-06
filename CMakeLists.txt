cmake_minimum_required(VERSION 2.8.12)
project(pykds)



if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


add_subdirectory(pybind11)
pybind11_add_module(pykds pykds.cpp kds/kds.hpp kds/LogWeightProb.hpp)

target_link_libraries(pykds PRIVATE pybind11::module) 


add_executable(example example.cpp)
add_executable(test_case test.cpp)
