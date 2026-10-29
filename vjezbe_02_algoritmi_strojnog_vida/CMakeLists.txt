cmake_minimum_required(VERSION 3.5)
project(OpenCV)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(Cunjevi src/test_color_cunjevi.cpp)
add_executable(Znakovi src/test_shape_znakovi.cpp)
add_executable(Auto src/test_track_autonomous_vehicles.cpp)

add_executable(OpenCV src/test.cpp)


target_link_libraries(Cunjevi ${OpenCV_LIBS} )
target_link_libraries(Znakovi ${OpenCV_LIBS} )
target_link_libraries(Auto ${OpenCV_LIBS} )
target_link_libraries(OpenCV ${OpenCV_LIBS} )
