set(TEST_FILES
    SampleTest.cpp
)

add_executable(tests ${TEST_FILES})
target_link_libraries(tests PRIVATE lib_hough_transform ${OPENCL_LIBRARIES} Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(tests)