cmake_minimum_required(VERSION 3.30)
project(Hough_transform)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_testing()

find_package(OpenCV REQUIRED)

include(FetchContent)
include(cmake/GetOpenCL.cmake)
include(cmake/GetDependency.cmake)

add_subdirectory(data)
add_subdirectory(src)
add_subdirectory(tests)

add_executable(Hough_transform main.cpp)
target_link_libraries(Hough_transform PUBLIC
        lib_hough_transform
        ${OpenCV_LIBS}
        DataFilesCopy
)
