cmake_minimum_required(VERSION 3.20)
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

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if (MSVC)
        target_compile_options(Hough_transform PRIVATE /Ox)
    endif()
endif()

# Copy config file
set(CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/config.toml")
add_custom_target("copy_config"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CONFIG_FILE}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/)
add_dependencies(Hough_transform "copy_config")
