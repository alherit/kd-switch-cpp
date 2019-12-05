cmake_minimum_required(VERSION 2.8.12)
project(pykds)

add_subdirectory(pybind11)
pybind11_add_module(pykds pykds.cpp kds/kds.hpp kds/LogWeightProb.h)

target_link_libraries(pykds PRIVATE pybind11::module) 
