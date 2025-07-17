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

torch::Tensor getGaussianKernel2D(int ksize_x, int ksize_y, double sigma_x, double sigma_y, torch::Device device) 
{
    int half_x = ksize_x / 2;
    int half_y = ksize_y / 2;

    auto x = torch::arange(-half_x, half_x + 1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto y = torch::arange(-half_y, half_y + 1, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    auto xx = x.pow(2).div(2 * sigma_x * sigma_x).unsqueeze(1);  // shape (kx, 1)
    auto yy = y.pow(2).div(2 * sigma_y * sigma_y).unsqueeze(0);  // shape (1, ky)

    auto kernel = torch::exp(-(xx + yy));  // shape (kx, ky)
    kernel /= kernel.sum();  // normalize

    return kernel;
}

torch::Tensor applyGaussianBlur2D(const torch::Tensor& input, int kx, int ky, double sx, double sy) 
{
    auto kernel = getGaussianKernel2D(kx, ky, sx, sy, input.device());
    kernel = kernel.view({1, 1, ky, kx});  // Conv2d expects [out_channels, in_channels, H, W]

    auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, {ky, kx}).padding({ky / 2, kx / 2}).bias(false));
    conv->weight.set_data(kernel.clone());  // clone to detach from computation graph
    conv->to(input.device());

    return conv->forward(input);
}


} // namespace utils