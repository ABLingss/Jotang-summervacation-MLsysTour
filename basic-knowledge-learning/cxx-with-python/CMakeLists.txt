cmake_minimum_required(VERSION 3.14)
project(example)

# 寻找 Python 和 pybind11
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

# 添加模块
pybind11_add_module(example example.cpp)