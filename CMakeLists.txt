cmake_minimum_required(VERSION 3.20)
project(OnnxDeployLab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ORT_ROOT "D:/Project/OpticalSelectionMachine/dll/onnxruntime-win-x64-gpu-1.18.0"
    CACHE PATH "ONNX Runtime package root")
set(OpenCV_DIR "D:/Project/OpticalSelectionMachine/dll/opencv-4.8.1/build"
    CACHE PATH "Directory containing OpenCVConfig.cmake")
find_package(OpenCV REQUIRED)

add_executable(model_inspect
    src/main.cpp
    src/Classification.cpp
)
target_include_directories(model_inspect PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${ORT_ROOT}/include"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(model_inspect PRIVATE
    "${ORT_ROOT}/lib/onnxruntime.lib"
    ${OpenCV_LIBS}
)
if(WIN32)
    add_custom_command(TARGET model_inspect POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_ROOT}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:model_inspect>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_DIR}/x64/vc16/bin/opencv_world481$<$<CONFIG:Debug>:d>.dll"
            "$<TARGET_FILE_DIR:model_inspect>"
        VERBATIM
    )
endif()

add_executable(inspect_seg src/inspect_seg.cpp)
target_include_directories(inspect_seg PRIVATE
    "${ORT_ROOT}/include"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(inspect_seg PRIVATE
    "${ORT_ROOT}/lib/onnxruntime.lib"
    ${OpenCV_LIBS}
)
if(WIN32)
    add_custom_command(TARGET inspect_seg POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_ROOT}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:inspect_seg>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OpenCV_DIR}/x64/vc16/bin/opencv_world481$<$<CONFIG:Debug>:d>.dll"
            "$<TARGET_FILE_DIR:inspect_seg>"
        VERBATIM
    )
endif()
