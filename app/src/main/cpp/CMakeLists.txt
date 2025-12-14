# CMakeLists.txt for QNN Model Runner JNI

cmake_minimum_required(VERSION 3.18.1)

project("qnn_model_runner")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QNN SDK headers are located in include/QNN/
# We need both paths:
# - include/ for #include "QNN/QnnInterface.h" style
# - include/QNN/ for internal SDK includes like #include "System/QnnSystemCommon.h"
set(QNN_INCLUDE_BASE "${CMAKE_SOURCE_DIR}/include")
set(QNN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include/QNN")

message(STATUS "QNN Include Base: ${QNN_INCLUDE_BASE}")
message(STATUS "QNN Include Dir: ${QNN_INCLUDE_DIR}")

# Include directories - both paths needed for proper header resolution
include_directories(
    ${QNN_INCLUDE_BASE}
    ${QNN_INCLUDE_DIR}
)

# Create the shared library
add_library(
    qnn_model_runner
    SHARED
    qnn_model_runner.cpp
)

# Find required libraries
find_library(log-lib log)
find_library(dl-lib dl)

# Link libraries
target_link_libraries(
    qnn_model_runner
    ${log-lib}
    ${dl-lib}
)

# Set compile options
target_compile_options(qnn_model_runner PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
    -O2
    -fPIC
)

# No need to define QNN_API_VERSION macros - they come from the SDK headers
