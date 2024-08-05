cmake_minimum_required(VERSION 3.0)
project(LaneDetMobilenetSsdGstreamer)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Look for pkg-config, which is used to get the compile and link flags
find_package(PkgConfig REQUIRED)
pkg_check_modules(CAIRO REQUIRED cairo)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

# Include directories and link libraries
include_directories(${CAIRO_INCLUDE_DIRS} ${GSTREAMER_INCLUDE_DIRS})
link_directories(${CAIRO_LIBRARY_DIRS} ${GSTREAMER_LIBRARY_DIRS})

# Add source files
add_executable(lane_detection lane_detection.cpp)

# Link OpenCV library with the library found by pkg-config
target_link_libraries(lane_detection
  ${CAIRO_LIBRARIES}
  ${GSTREAMER_LIBRARIES}
  opencv_objdetect
  opencv_highgui
  opencv_imgproc
  opencv_core
  opencv_videoio
  opencv_imgcodecs
  opencv_dnn
  opencv_dnn_objdetect
)
