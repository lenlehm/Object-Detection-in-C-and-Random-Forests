cmake_minimum_required(VERSION 3.13)
project(tdcv)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 14)
add_executable(tdcv main.cpp hog_visualization.h ClassifierOperations.h BoundingBoxOperations.h)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(tdcv ${OpenCV_LIBS})