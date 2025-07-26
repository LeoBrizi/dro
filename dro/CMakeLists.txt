cmake_minimum_required(VERSION 3.14)
project(dro LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -DTORCH_USE_CUDA_DSA -g")

# set(CUDAToolkit_ROOT "${CMAKE_SOURCE_DIR}/local/cuda-12.9")
# set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_SOURCE_DIR}/local/cuda-12.9")
# set(CUDA_BIN_PATH "${CMAKE_SOURCE_DIR}/local/cuda-12.9/bin")
set(CMAKE_CUDA_ARCHITECTURES 89)
find_library(CUDA_CUDART_LIBRARY
    NAMES libcudart.so libcudart
    PATH_SUFFIXES lib
    HINTS /usr/local/cuda-12.8/targets/x86_64-linux
)

# Force the correct CUDART path before Torch overrides it
set(CUDA_CUDART_LIBRARY "/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so" CACHE FILEPATH "CUDA CUDART" FORCE)

if(NOT CUDA_CUDART_LIBRARY)
    message(FATAL_ERROR "Could not find libcudart.so")
endif()

message(STATUS "Found CUDA CUDART library: ${CUDA_CUDART_LIBRARY}") 

# find_library(NVTOOLSEXT_LIBRARY
#     NAMES nvToolsExt
#     PATHS /usr/lib/x86_64-linux-gnu
#     REQUIRED
# )

# add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
# set_target_properties(CUDA::nvToolsExt PROPERTIESs
#     IMPORTED_LOCATION "${NVTOOLSEXT_LIBRARY}"
#     INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
# )

set(Torch_DIR ${PROJECT_SOURCE_DIR}/libtorch)
set(CMAKE_PREFIX_PATH "${Torch_DIR}/share/cmake/Torch")
message(${Torch_DIR})

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

find_package(Eigen3 REQUIRED)

message(STATUS "Found CUDA CUDART library: ${CUDA_CUDART_LIBRARY}") 

add_compile_options(${TORCH_CXX_FLAGS})

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/app)
