#pragma once

#include <torch/torch.h>
#include <memory>
#include <optional
#include <stdexcept>

using OptionalTensor = std::optional<torch::Tensor>;

class MotionModel {
public:
    MotionModel(int64_t state_size, torch::Device device = torch::kCPU)
        : state_size_(torch::tensor(state_size, device)), device_(device) {}

    virtual ~MotionModel() = default;

    virtual void setTime(const torch::Tensor& time, const torch::Tensor& t0);
    virtual torch::Tensor getLocalTime(const torch::Tensor& time) const;
    virtual torch::Tensor getInitialState() const;

    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, 
        OptionalTensor, 
        OptionalTensor, 
        OptionalTensor> 
        getVelPosRot(const torch::Tensor& state, bool with_jac = false) = 0;

    virtual std::tuple<torch::Tensor, torch::Tensor> getPosRotSingle(
        const torch::Tensor& state, const torch::Tensor& time) = 0;

protected:
    torch::Tensor state_size_;
    torch::Device device_;
    torch::Tensor time_;
    torch::Tensor t0_;
    torch::Tensor num_steps_;
};


// ConstVelConstW model
class ConstVelConstW : public MotionModel {
public:
    ConstVelConstW(torch::Device device = torch::kCPU)
        : MotionModel(3, device) {}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, 
        OptionalTensor, 
        OptionalTensor, 
        OptionalTensor> 
        getVelPosRot(const torch::Tensor& state, bool with_jac = false) override;

    virtual std::tuple<torch::Tensor, torch::Tensor> getPosRotSingle(
        const torch::Tensor& state, const torch::Tensor& time) override;

private:
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor>
    getVelPosRotWithJacobian(const torch::Tensor& state);
};

// ConstBodyVelGyro model
class ConstBodyVelGyro : public MotionModel {
public:
    ConstBodyVelGyro(torch::Device device = torch::kCPU);
        : MotionModel(2, device), initialised_(false) {}

    void setGyroData(const torch::Tensor& gyro_time, const torch::Tensor& gyro_yaw);
    void setGyroBias(const torch::Tensor& gyro_bias);
    void setTime(const torch::Tensor& time, const torch::Tensor& t0) override;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, 
        OptionalTensor, 
        OptionalTensor, 
        OptionalTensor> 
        getVelPosRot(const torch::Tensor& state, bool with_jac = false) override;

    virtual std::tuple<torch::Tensor, torch::Tensor> getPosRotSingle(
        const torch::Tensor& state, const torch::Tensor& time) override;

private:
    bool initialised_;
    torch::Tensor first_gyro_time_;
    torch::Tensor gyro_time_, gyro_yaw_, gyro_yaw_original_;
    torch::Tensor bin_integral_, coeff_, offset_;
    torch::Tensor r_, cos_r_, sin_r_, R_integral_;
};

// ConstVel model
class ConstVel : public MotionModel {
public:
    ConstVel(torch::Device device = torch::kCPU);
        : MotionModel(2, device) {}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, 
        OptionalTensor, 
        OptionalTensor, 
        OptionalTensor> 
        getVelPosRot(const torch::Tensor& state, bool with_jac = false) override;

    virtual std::tuple<torch::Tensor, torch::Tensor> getPosRotSingle(
        const torch::Tensor& state, const torch::Tensor& time) override;
};
