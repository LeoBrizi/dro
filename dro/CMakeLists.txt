cmake_minimum_required(VERSION 3.14)
project(dro LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

set(Torch_DIR ${PROJECT_SOURCE_DIR}/libtorch)

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

add_compile_options(${TORCH_CXX_FLAGS})

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/app)
