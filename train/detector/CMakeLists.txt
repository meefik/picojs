cmake_minimum_required(VERSION 3.6)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH "../opencv_build")

project(detector)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(detector ${OpenCV_LIBS})
