add_library(libdro STATIC
  gp_doppler.cpp
  motion_models.cpp
  utils.cpp
)

target_link_libraries(libdro PUBLIC
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
)