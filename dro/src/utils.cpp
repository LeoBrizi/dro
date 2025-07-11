#include "utils.h"

namespace utils {

torch::Tensor dopplerUpDown(const RadarFrame& rf) {
    // Compute base path by removing last directory
    auto pos = rf.sensor_root.find_last_of('/');
    std::string base = (pos != std::string::npos)
        ? rf.sensor_root.substr(0, pos)
        : rf.sensor_root;
    std::string doppler_path = base + "/radar/" + rf.frame + ".png";

    cv::Mat img = cv::imread(doppler_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + doppler_path);
    }

    // Extract column 10 as 1D tensor
    cv::Mat col = img.col(10).clone();
    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    torch::Tensor t = torch::from_blob(
        col.data, {col.rows}, options
    ).to(torch::kInt32).clone();
    return t;
}

bool checkChirp(const RadarFrame& rf) {
    torch::Tensor up_chirps = dopplerUpDown(rf);
    // Check even indices == 255 and odd indices == 0
    auto evens = up_chirps.index({torch::arange(0, up_chirps.size(0), 2)});
    auto odds  = up_chirps.index({torch::arange(1, up_chirps.size(0), 2)});
    bool even_ok = evens.eq(255).all().item<bool>();
    bool odd_ok  = odds.eq(0).all().item<bool>();
    // If pattern invalid, return true to indicate chirp_up (needs flip)
    return !(even_ok && odd_ok);
}

} // namespace utils