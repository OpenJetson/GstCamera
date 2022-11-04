cmake_minimum_required(VERSION 2.6)
project(gst_camera)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/include/aarch64-linux-gnu/)
include_directories(/usr/include/gstreamer-1.0)
include_directories(/usr/lib/aarch64-linux-gnu/glib-2.0/include)
include_directories(/usr/include/glib-2.0)

link_directories(/usr/lib/aarch64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wno-char-subscripts -Wno-unused-parameter -Wno-format -fpermissive")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gst_camera ${PROJECT_SOURCE_DIR}/src/main.cpp ${PROJECT_SOURCE_DIR}/src/http_stream.cpp)
target_link_libraries(gst_camera ${OpenCV_LIBS})
target_link_libraries(gst_camera gstreamer-1.0 gstrtspserver-1.0 glib-2.0 gobject-2.0 gmodule-2.0)
target_link_libraries(gst_camera -lpthread -lm)
