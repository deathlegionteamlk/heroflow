cmake_minimum_required(VERSION 3.10)
project(heroflow VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(heroflow/core)

# Core library
add_library(heroflow_core STATIC
    heroflow/core/tensor.cpp
    heroflow/core/ops.cpp
    heroflow/core/autograd.cpp
)

# Python module
pybind11_add_module(heroflow_python heroflow/python/bindings.cpp)
target_link_libraries(heroflow_python PRIVATE heroflow_core)
set_target_properties(heroflow_python PROPERTIES OUTPUT_NAME heroflow_internal)

# Executables for C++ tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/tensor_test.cpp")
    add_executable(tensor_test tests/cpp/tensor_test.cpp)
    target_link_libraries(tensor_test heroflow_core)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/autograd_test.cpp")
    add_executable(autograd_test tests/cpp/autograd_test.cpp)
    target_link_libraries(autograd_test heroflow_core)
endif()
