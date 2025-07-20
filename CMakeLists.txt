cmake_minimum_required(VERSION 3.10)
project(Document_recognition LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_executable(Document_recognition main.cpp)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(Document_recognition ${OpenCV_LIBS})

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
    target_compile_options(Document_recognition PRIVATE 
        -Wno-deprecated-declarations
)
endif()
