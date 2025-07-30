cmake_minimum_required(VERSION 3.10)
project(DocumentDetection LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)

add_executable(DocumentDetection main.cpp)

find_package(OpenCV REQUIRED)

set(BUILD_SHARED_LIBS OFF)
target_include_directories(DocumentDetection PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(DocumentDetection PRIVATE ${OpenCV_LIBS})

# target_include_directories(Document_recognition PRIVATE
#     "${OpenCV_DIR}/include/opencv2"
# )

# find_library(OPENCV_LIB NAMES opencv_world4110.lib)
# if(NOT OPENCV_LIB)
#     message(FATAL_ERROR "OpenCV library not found!")
# endif()

# target_link_libraries(Document_recognition PRIVATE
#     ${OPENCV_LIB}
# )

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(DocumentDetection PRIVATE 
        -Wno-deprecated-declarations
)
endif()