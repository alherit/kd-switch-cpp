cmake_minimum_required(VERSION 2.8.12)
project(pykds)

set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest /W4")
    # Default debug flags are OK
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -fPIC")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -D_DEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

message(STATUS ${CMAKE_BUILD_TYPE} )


add_subdirectory(pybind11)
pybind11_add_module(pykds pykds.cpp kds/kds.hpp kds/LogWeightProb.hpp)

target_link_libraries(pykds PRIVATE pybind11::module) 


add_executable(example example.cpp)
add_executable(test_case test.cpp)
