
add_library(DataFilesCopy INTERFACE)


file(GLOB DATA_FILES *.bmp *.png *.jpg)

foreach(data IN LISTS DATA_FILES)
    cmake_path(GET data FILENAME FileName)
    add_custom_target("copy_data_${FileName}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${data}
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data/${FileName}
    )
    add_dependencies(DataFilesCopy "copy_data_${FileName}")
endforeach()

