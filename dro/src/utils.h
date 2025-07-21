#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <string>

// Represents a single radar frame, with paths to its data
struct RadarFrame {
    std::string sensor_root;  // e.g., "/path/to/session"
    std::string frame;        // e.g., "000000.png"
};

struct RadarData {
    torch::Tensor timestamps;  // float64 tensor of length H
    torch::Tensor azimuths;    // float32 tensor of length H
    torch::Tensor polar;       // float32 tensor of shape (H, W')
};

namespace utils {
    torch::Tensor dopplerUpDown(const RadarFrame& rf);
    bool checkChirp(const RadarFrame& rf);
    torch::Tensor applyGaussianBlur2D(const torch::Tensor& input, int kx, int ky, double sx, double sy);
    
    // Load a CSV with double values. Skips the first `skiprows` lines.
    // Assumes every data row has the same number of columns.
    Eigen::MatrixXd loadCsv(const std::string& filename,
                            char delimiter = ',',
                            int skiprows = 1);
    Eigen::Isometry3f loadIsometry3fFromFile(const std::string& filename);

    RadarData loadRadarData(
        const std::string &filename,
        int encoder_size = 16000,
        int min_id = 0);
}
}