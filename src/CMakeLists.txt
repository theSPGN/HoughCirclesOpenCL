
set(SOURCE_FILES
    KernelUtils.cpp
)

file(GLOB OPENCL_KERNELS *.cl)

add_library(lib_hough_transform STATIC ${SOURCE_FILES})
target_link_libraries(lib_hough_transform PUBLIC  ${OPENCL_LIBRARIES} tomlplusplus::tomlplusplus)
target_include_directories(lib_hough_transform PUBLIC ${OPENCL_INCLUDE_DIRS} $ENV{AMDAPPSDKROOT}/include/ .)
set_target_properties(lib_hough_transform PROPERTIES LINKER_LANGUAGE CXX)


# Copy OpenCL kernels to build folder
foreach(kernel IN LISTS OPENCL_KERNELS)
    cmake_path(GET kernel FILENAME FileName)
    add_custom_target("copy_kernel_${FileName}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${kernel}
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cl/${FileName})
    add_dependencies(lib_hough_transform "copy_kernel_${FileName}")
endforeach()

