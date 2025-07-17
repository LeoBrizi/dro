#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// Represents a single radar frame, with paths to its data
struct RadarFrame {
    std::string sensor_root;  // e.g., "/path/to/session"
    std::string frame;        // e.g., "000000.png"
};

namespace utils {
    torch::Tensor dopplerUpDown(const RadarFrame& rf);
    bool checkChirp(const RadarFrame& rf);
    torch::Tensor applyGaussianBlur2D(const torch::Tensor& input, int kx, int ky, double sx, double sy);
}