find_package(yaml-cpp)

if(NOT DEFINED yaml-cpp_FOUND)
  message(CHECK_START "Fetching yaml-cpp")
  list(APPEND CMAKE_MESSAGE_INDENT "  ")
  include(FetchContent)
  FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.8.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
  )
  FetchContent_GetProperties(yaml-cpp)

  if(NOT yaml-cpp_POPULATED)
    message(STATUS "Fetching yaml-cpp...")
    FetchContent_Populate(yaml-cpp)
    add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
  endif()

  message(CHECK_PASS "fetched")
endif()

add_executable(radar_gp_state_estimation radar_gp_state_estimation.cpp)
target_link_libraries(radar_gp_state_estimation PRIVATE
  libdro
  yaml-cpp
  ${TORCH_LIBRARIES}
  ${CUDA_CUDART_LIBRARY}
)