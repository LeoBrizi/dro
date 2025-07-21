#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "yaml-cpp/yaml.h"
#include "gp_doppler.h"
#include "utils.h"
#include "pyboreas.h"

namespace fs = std::filesystem;

int main() {
    YAML::Node config = YAML::LoadFile("config.yaml");

    // Data loading
    std::string data_path = config["data"]["data_path"].as<std::string>();

    // Logging and visualization
    bool visualise = config["log"]["display"].as<bool>();
    bool save_images = config["log"]["save_images"].as<bool>();
    bool verbose = config["log"]["verbose"].as<bool>();

    bool use_gyro = config["estimation"]["use_gyro"].as<bool>();
    bool doppler_radar = config["radar"]["doppler_enabled"].as<bool>();
    bool chirp_up = false;
    if (!doppler_radar) {
        chirp_up = config["radar"]["chirp_up"].as<bool>();
    }

    float gyro_bias_alpha = 0.01
    bool estimate_gyro_bias = false;
    bool pose_estimation = true;
    bool estimate_ang_vel = false;
    std::string motion_model = "";
    if (use_gyro) {
        motion_model = "const_body_vel_gyro";
    } else if (config["estimation"]["direct_cost"].as<bool>()){
        motion_model = "const_vel_const_w";
        estimate_ang_vel = true;
    } else if (config["estimation"]["doppler_cost"].as<bool>()){
        motion_model = "const_vel";
        pose_estimation = false;
    } else {
        std::cerr("Ambiguous configuration: no motion model selected");
        return -1;
    }

    float vy_bias_alpha = 0.01;
    float vy_bias = 0.0;
    bool estimate_vy_bias = false;
    if (doppler_radar) {
        estimate_vy_bias = config["estimation"]["estimate_doppler_vy_bias"].as<bool>();
        if (estimate_vy_bias) {
            const auto T_axle_radar_vec = config["estimation"]["T_axle_radar"].as<std::vector<std::vector<double>>>();
            const Eigen::Matrix3d T_axle_radar = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(T_axle_radar_vec.data());
        }
        if (config["estimation"]["vy_bias_alpha"].IsDefined()) {
            vy_bias_alpha = config["estimation"]["vy_bias_alpha"].as<float>();
        }
    }

    res = config["radar"]["resolution"].as<double>();
    auto state_estimator = GPStateEstimator(config, res);

    if (use_gyro) {
        float gyro_bias = 0.0;
        int gyro_bias_counter = 0;
        bool gyro_bias_initialised = false;
        bool previous_vel_null = false;
        estimate_gyro_bias = config["estimation"]["estimate_gyro_bias"].as<bool>();

            // Need to account for the IMU type
        std::vector<double> imu_time;
        std::vector<double> imu_yaw;
        std::string imu_type = config["imu"]["type"].as<std::string>();
        fs::path imu_path;
        Eigen::MatrixXd imu_data;

        if (imu_type == "applanix") {
            imu_path = data_path / "applanix" / "imu_raw.csv";
            imu_data = utils::loadCsv(imu_path.string(), /*delimiter=*/',', /*skiprows=*/1);
            imu_time.assign(imu_data.col(0).data(), imu_data.col(0).data() + imu_data.rows());
            // Stack columns 3,2,1 into a Nx3 matrix
            Eigen::MatrixXd gyro_raw(imu_data.rows(), 3);
            gyro_raw.col(0) = imu_data.col(3);
            gyro_raw.col(1) = imu_data.col(2);
            gyro_raw.col(2) = imu_data.col(1);
            // Transform to radar frame
            auto T_applanix_lidar = utils::loadIsometry3fFromFile(data_path / "applanix" / "T_applanix_lidar.txt");
            auto T_radar_lidar = utils::loadIsometry3fFromFile(data_path / "radar" / "T_radar_lidar.txt");
            auto T_applanix_radar = T_applanix_lidar * T_radar_lidar.inverse();
            auto imu_gyro = gyro_raw * T_applanix_radar.linear();
            imu_yaw.resize(imu_gyro.rows());
            for (int i = 0; i < imu_gyro.rows(); ++i) {
                imu_yaw[i] = -imu_gyro(i, 2);
            }
        }
        else if (imu_type == "dmu") {
            imu_path = data_path / "applanix" / "dmu_imu.csv";
            imu_data = utils::loadCsv(imu_path.string(), /*delimiter=*/',', /*skiprows=*/1);
            imu_time.resize(imu_data.rows());
            imu_yaw.resize(imu_data.rows());
            for (int i = 0; i < imu_data.rows(); ++i) {
                imu_time[i] = imu_data(i, 0) * 1e-9;
                imu_yaw[i]  = imu_data(i, 9);
            }
        }
        else {
            std::cerr << "Unknown IMU type: " << imu_type << std::endl;
            return;
        }

        // Pass IMU data to the state estimator
        state_estimator.setGyroData(imu_time, imu_yaw);

        // Compute minimum samples for gyro bias init
        std::vector<double> dt;
        dt.reserve(imu_time.size()-1);
        for (size_t i = 1; i < imu_time.size(); ++i) {
            dt.push_back(imu_time[i] - imu_time[i-1]);
        }
        double mean_dt = std::accumulate(dt.begin(), dt.end(), 0.0) / dt.size();
        int min_gyro_sample_bias = static_cast<int>(config["imu"]["min_time_bias_init"].as<double>() / mean_dt);

        if (estimate_vy_bias) {
            vy_bias = config["estimation"]["vy_bias_prior"].as<double>();
        }
        if (config["estimation"]["gyro_bias_prior"]) {
            gyro_bias = config["estimation"]["gyro_bias_prior"].as<double>();
            gyro_bias_initialised = true;
            gyro_bias_counter = min_gyro_sample_bias + 1;
        }
    }

    std::string seq_id = data_path.filename().string();

    // Output directories
    fs::path seq_output_path = fs::path("output") / seq_id;
    fs::create_directories(seq_output_path);
    fs::path odom_output_path = seq_output_path / "odometry_result" / (seq_id + ".txt");
    fs::create_directories(odom_output_path.parent_path());

    std::ofstream odom_output(odom_output_path);

    // Before the frame loop:
    double time_sum = 0.0;
    double opti_time_sum = 0.0;
    int time_counter = 0;
    // If no Doppler radar, initialize chirp_up once from config:
    if (!doppler_radar) {
        chirp_up = !config["radar"]["chirp_up"].as<bool>();
    }

    fs::path radar_dir = fs::path(data_path) / "radar";
    if (!fs::exists(radar_dir) || !fs::is_directory(radar_dir)) {
        throw std::runtime_error(radar_dir.string() + " is not a valid directory");
    }

    std::vector<fs::path> radar_frames;

    for (auto& entry : fs::directory_iterator(radar_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            radar_frames.push_back(entry.path());
        }
    }

    // 2) Sort them lexicographically by filename
    std::sort(radar_frames.begin(), radar_frames.end(),
            [](auto const& a, auto const& b){
                return a.filename().string() < b.filename().string();
            });

    // 3) Now you have both the count and the filenames
    int end_id = static_cast<int>(radar_frames.size());

    for (int i = 0; i < end_id; ++i) {
        // Start total-timer
        auto t_loop_start = std::chrono::steady_clock::now();

        // Load frame and record GT
        auto radar_frame = utils::loadRadarData(radar_frames[i].string());

        if (i == 0) {
            gt_first_T_inv = radar_frame.pose.inverse();
        }
        gt_xyz.emplace_back(
            (gt_first_T_inv * radar_frame.pose).block<3,1>(0,3)
        );

        // Update gyro bias if enabled
        if (use_gyro && estimate_gyro_bias && gyro_bias_initialised) {
            state_estimator.motion_model_->setGyroBias(gyro_bias);
        }

        // Recompute chirp_up each frame if using Doppler radar
        if (doppler_radar) {
            chirp_up = utils::checkChirp(radar_frame);
        }

        // Print progress
        if (time_counter == 0) {
            std::cout << "Frame " << (i+1) << " / " << end_id << "\r";
        } else {
            double avg_opti = opti_time_sum / time_counter;
            double time_left = (end_id - i) * time_sum / time_counter / 60.0;
            std::cout << "Frame " << (i+1) << " / " << end_id
                    << " - Avg. opti: " << avg_opti << "s, time left: "
                    << time_left << "min    \r";
        }

        // Start optimization-timer
        auto t_op_start = std::chrono::steady_clock::now();

        // Prepare polar image, accounting for offset
        auto polar_img = radar_frame.polar; // torch::Tensor
        double offset = config["radar"]["range_offset"].as<double>() 
                        / radar_frame.resolution;
        if (offset > 0) {
            auto pad = torch::zeros(
                {polar_img.size(0), int(std::round(offset))},
                torch::TensorOptions().dtype(torch::kFloat32).device(device_)
            );
            polar_img = torch::cat({pad, polar_img}, 1);
        } else if (offset < 0) {
            int ofs = int(std::round(-offset));
            polar_img = polar_img.index({Slice(), Slice(ofs, None)});
        }

        // Run odometry
        auto state = state_estimator.odometryStep(
            polar_img,
            radar_frame.azimuths,
            radar_frame.timestamps,
            chirp_up
        );

        // End optimization-timer
        auto t_op_end = std::chrono::steady_clock::now();
        double op_duration = std::chrono::duration<double>(t_op_end - t_op_start).count();
        if (time_counter == 5) {
            opti_time_sum = op_duration * 5;
        }
        opti_time_sum += op_duration;

        // Extract 2D velocity
        auto vel_tensor = state.index({Slice(None,2)}).to(torch::kCPU);
        Eigen::Vector2d velocity{
            vel_tensor[0].item<double>(),
            vel_tensor[1].item<double>()
        };

        // Apply body-acc model if needed
        if (config["estimation"]["motion_model"].as<std::string>() 
            == "const_body_acc_gyro")
        {
            double w = state[2].item<double>();
            velocity *= (1.0 + w*0.125);
        }

        // Doppler-based vy-bias update
        double vel_norm = velocity.norm();
        if (estimate_vy_bias && vel_norm > 3.0) {
            state_estimator.vy_bias_ = 0.0;
            auto doppler_vel = state_estimator.getDopplerVelocity(); // vector<double> size 2
            Eigen::Vector3d dv{doppler_vel[0], doppler_vel[1], 0.0};
            std::cout << "\nDoppler velocity: [" 
                    << dv.transpose() << "]\n";

            // Compute axle_vel and vy as in Python...
            // vy_bias = vy_bias_alpha*vy + (1-vy_bias_alpha)*vy_bias;
            state_estimator.vy_bias_ = vy_bias;
        }
        if (estimate_vy_bias) {
            std::cout << "\nVy bias: " << vy_bias << "\n";
        }

        // Verbose logging
        if (verbose) {
            std::cout << "\nVelocity: [" << velocity.transpose() << "]\n";
            Eigen::Vector2d gt_rate = radar_frame.body_rate.head<2>();
            std::cout << "Velocity GT: [" << gt_rate.transpose() << "]\n";
            double gt_norm = radar_frame.body_rate.head<3>().norm();
            std::cout << "Diff norm: " << (vel_norm - gt_norm) << "\n\n";
        }

        // Gyro‑bias estimation on near‑zero velocity
        if (estimate_gyro_bias && vel_norm < 0.05) {
            if (previous_vel_null) {
                // Extract imu_yaw between timestamps, update gyro_bias
            }
            previous_vel_null = true;
        }

        // End total‑timer and update stats
        auto t_loop_end = std::chrono::steady_clock::now();
        double loop_duration = std::chrono::duration<double>(t_loop_end - t_loop_start).count();
        if (time_counter == 1) {
            time_sum = loop_duration;
        }
        time_sum += loop_duration;
        time_counter++;
    }

    odom_output.close();

    return 0;
}
