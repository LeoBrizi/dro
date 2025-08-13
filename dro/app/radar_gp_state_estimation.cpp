#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "yaml-cpp/yaml.h"
#include "gp_doppler.h"
#include "utils.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>" << std::endl;
        return -1;
    }
    YAML::Node config = YAML::LoadFile(argv[1]);

    // Data loading
    fs::path data_path = config["data"]["data_path"].as<std::string>();

    // Logging and visualization
    bool visualise = config["log"]["display"].as<bool>();
    bool save_images = config["log"]["save_images"].as<bool>();
    bool verbose = config["log"]["verbose"].as<bool>();
    int start_idx = config["estimation"]["start_idx"].as<int>();

    bool use_gyro = config["estimation"]["use_gyro"].as<bool>();
    bool doppler_radar = config["radar"]["doppler_enabled"].as<bool>();
    bool chirp_up = false;
    if (!doppler_radar) {
        chirp_up = config["radar"]["chirp_up"].as<bool>();
    }

    float gyro_bias_alpha = 0.01;
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
        std::cerr << "Ambiguous configuration: no motion model selected" << std::endl;
        return -1;
    }
    config["estimation"]["motion_model"] = motion_model;

    float vy_bias_alpha = 0.01;
    float vy_bias = 0.0;
    bool estimate_vy_bias = false;
    Eigen::Matrix4d T_axle_radar;
    if (doppler_radar) {
        estimate_vy_bias = config["estimation"]["estimate_doppler_vy_bias"].as<bool>();
        if (estimate_vy_bias) {
            const auto T_axle_radar_vec = config["estimation"]["T_axle_radar"].as<std::vector<std::vector<double>>>();
            std::vector<double> flat_matrix;
            flat_matrix.reserve(16);
            for (const auto& row : T_axle_radar_vec) {
                flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
            }

            // Map it into an Eigen matrix
            T_axle_radar = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(flat_matrix.data());
        }
        if (config["estimation"]["vy_bias_alpha"].IsDefined()) {
            vy_bias_alpha = config["estimation"]["vy_bias_alpha"].as<float>();
        }
    }

    double res = config["radar"]["resolution"].as<double>();
    auto state_estimator = GPStateEstimator(config, res);

    torch::Device device = state_estimator.getDevice();

    float gyro_bias = 0.0;
    int gyro_bias_counter = 0;
    bool gyro_bias_initialised = false;
    bool previous_vel_null = false;
    std::vector<double> imu_time;
    std::vector<double> imu_yaw;
    std::string imu_type = config["imu"]["type"].as<std::string>();
    fs::path imu_path;
    Eigen::MatrixXd imu_data;
    int min_gyro_sample_bias = 0;

    if (use_gyro) {
        estimate_gyro_bias = config["estimation"]["estimate_gyro_bias"].as<bool>();
        if (imu_type == "applanix") {
            imu_path = data_path / "applanix" / "imu_raw.csv";
            imu_data = utils::loadCsv(imu_path.string(), ',', 1);
            imu_time.assign(imu_data.col(0).data(), imu_data.col(0).data() + imu_data.rows());
            // Stack columns 3,2,1 into a Nx3 matrix
            Eigen::MatrixXd gyro_raw(imu_data.rows(), 3);
            gyro_raw.col(0) = imu_data.col(3);
            gyro_raw.col(1) = imu_data.col(2);
            gyro_raw.col(2) = imu_data.col(1);
            // Transform to radar frame
            // fs::path calib_path = data_path / "calib";
            // // I need to change the path 
            // auto T_applanix_lidar = utils::loadIsometry3dFromFile(calib_path / "T_applanix_lidar.txt");
            // auto T_radar_lidar = utils::loadIsometry3dFromFile(calib_path / "T_radar_lidar.txt");
            // auto T_applanix_radar = T_applanix_lidar * T_radar_lidar.inverse();

            // then I will read it directly from a file that has T_imu_radar
            auto T_imu_radar = utils::loadIsometry3dFromFile(data_path / "T_imu_radar.txt"); // Sam: could be the inverse here

            auto imu_gyro = (gyro_raw * T_imu_radar.linear());
            imu_yaw.resize(imu_gyro.rows());
            for (int i = 0; i < imu_gyro.rows(); ++i) {
                imu_yaw[i] = -imu_gyro(i, 2);
            }
        }
        else if (imu_type == "dmu") {
            imu_path = data_path / "applanix" / "dmu_imu.csv";
            imu_data = utils::loadCsv(imu_path.string(), ',', 1);
            imu_time.resize(imu_data.rows());
            imu_yaw.resize(imu_data.rows());
            for (int i = 0; i < imu_data.rows(); ++i) {
                imu_time[i] = imu_data(i, 0) * 1e-9;
                imu_yaw[i]  = imu_data(i, 9);
            }
        }
        else {
            std::cerr << "Unknown IMU type: " << imu_type << std::endl;
            return -1;
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
        min_gyro_sample_bias = static_cast<int>(config["imu"]["min_time_bias_init"].as<double>() / mean_dt);

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
    std::ofstream x_y_odom_output(seq_output_path / "x_y_odometry.txt");

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

    std::sort(radar_frames.begin(), radar_frames.end(),
            [](auto const& a, auto const& b){
                return a.filename().string() < b.filename().string();
            });

    int end_id = static_cast<int>(radar_frames.size());

    for (int i = start_idx; i < end_id; ++i) {
        auto t_loop_start = std::chrono::steady_clock::now();

        RadarFrame radar_frame;
        radar_frame.sensor_root = radar_dir.string();
        radar_frame.frame = radar_frames[i].filename().string();

        // Load the radar frame
        int min_id = static_cast<int>(std::round(2.5f / res));
        auto radar_data = utils::loadRadarData(radar_frames[i].string(), 5600, min_id);
        std::cerr << "PROCESSING THE RADAR: " << radar_frames[i].string() << std::endl;

        // Update gyro bias if enabled
        if (use_gyro && estimate_gyro_bias && gyro_bias_initialised) {
            state_estimator.setGyroBias(gyro_bias);
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
        auto polar_img = radar_data.polar; // torch::Tensor
        double offset = config["radar"]["range_offset"].as<double>() 
                        / res;
        if (offset > 0) {
            auto pad = torch::zeros(
                {polar_img.size(0), int(std::round(offset))},
                torch::TensorOptions().dtype(torch::kFloat32).device(device)
            );
            polar_img = torch::cat({pad, polar_img}, 1);
        } else if (offset < 0) {
            int ofs = int(std::round(-offset));
            polar_img = polar_img.index({torch::indexing::Slice(), torch::indexing::Slice(ofs, torch::indexing::None)});
        }

        // Run odometry
        auto state = state_estimator.odometryStep(
            polar_img,
            radar_data.azimuths,
            radar_data.timestamps,
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
        auto vel_tensor = state.index({torch::indexing::Slice(torch::indexing::None,2)}).to(torch::kCPU);
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
            state_estimator.setVyBias(0.0);
            Eigen::Vector3d dv = Eigen::Vector3d::Zero();

            if (config["estimation"]["doppler_cost"].as<bool>()) {
                auto doppler_vel = state_estimator.getDopplerVelocity(); // vector<double> size 2
                dv[0] = doppler_vel[0].item<double>();
                dv[1] = doppler_vel[1].item<double>();
            }

            std::cout << "\nDoppler velocity: [" 
                    << dv.transpose() << "]\n";
            Eigen::Vector3d axle_vel;

            if (use_gyro) {
                // Get first and last timestamps from radar data
                auto radar_timestamps_cpu = radar_data.timestamps.to(torch::kCPU).contiguous();
                double t0 = radar_timestamps_cpu[0].item<double>() * 1e-6;
                double t1 = radar_timestamps_cpu[-1].item<double>() * 1e-6;

                // Select IMU yaw samples between t0 and t1
                std::vector<double> selected_gyro_yaw;
                for (size_t k = 0; k < imu_time.size(); ++k) {
                    if (imu_time[k] >= t0 && imu_time[k] <= t1) {
                        selected_gyro_yaw.push_back(imu_yaw[k]);
                    }
                }

                Eigen::Vector3d gyro_data;
                if (selected_gyro_yaw.empty()) {
                    gyro_data = Eigen::Vector3d::Zero();
                } else {
                    double gyro_mean_yaw = std::accumulate(selected_gyro_yaw.begin(), selected_gyro_yaw.end(), 0.0) / selected_gyro_yaw.size();
                    gyro_data = T_axle_radar.block<3, 3>(0, 0) * Eigen::Vector3d(0.0, 0.0, gyro_mean_yaw);
                }

                Eigen::Vector3d translation_axle_radar = T_axle_radar.block<3, 1>(0, 3);
                axle_vel = T_axle_radar.block<3, 3>(0, 0) * dv + gyro_data.cross(translation_axle_radar);
            } else {
                axle_vel = T_axle_radar.block<3,3>(0,0) * dv;
            }

            Eigen::Vector3d lateral_velocity(0.0, axle_vel.y(), 0.0);
            double vy = (T_axle_radar.block<3,3>(0,0).transpose() * lateral_velocity).y();
            vy_bias = vy_bias_alpha * vy + (1.0 - vy_bias_alpha) * vy_bias;
            state_estimator.setVyBias(vy_bias);
        }

        if (estimate_vy_bias) {
            std::cout << "\nVy bias: " << vy_bias << "\n";
        }

        // Verbose logging
        // if (verbose) {
        //     std::cout << "\nVelocity: [" << velocity.transpose() << "]\n";
        //     Eigen::Vector2d gt_rate = radar_frame.body_rate.head<2>();
        //     std::cout << "Velocity GT: [" << gt_rate.transpose() << "]\n";
        //     double gt_norm = radar_frame.body_rate.head<3>().norm();
        //     std::cout << "Diff norm: " << (vel_norm - gt_norm) << "\n\n";
        // }

        // Gyro‑bias estimation on near‑zero velocity
        if (estimate_gyro_bias && vel_norm < 0.05) {
            if (previous_vel_null) {
                // Get gyro measurements between the first and last azimuth
                std::vector<double> gyro_data;
                for (size_t idx = 0; idx < imu_time.size(); ++idx) {
                    double t = imu_time[idx];
                    if (t >= radar_data.timestamps[0].item<double>() * 1e-6 &&
                        t <= radar_data.timestamps[-1].item<double>() * 1e-6) {
                        gyro_data.push_back(imu_yaw[idx]);
                    }
                }

                bool invalid = false;
                if (gyro_bias_counter != 0 && !gyro_data.empty()) {
                    double gyro_mean = std::accumulate(gyro_data.begin(), gyro_data.end(), 0.0) / gyro_data.size();
                    if (std::abs(gyro_mean - gyro_bias) > 2 * std::abs(gyro_bias)) {
                        invalid = true;
                    }
                }

                if (!invalid && !gyro_data.empty()) {
                    double gyro_sum = std::accumulate(gyro_data.begin(), gyro_data.end(), 0.0);
                    if (!gyro_bias_initialised) {
                        gyro_bias += gyro_sum;
                        gyro_bias_counter += gyro_data.size();
                        if (gyro_bias_counter > min_gyro_sample_bias) {
                            gyro_bias /= static_cast<double>(gyro_bias_counter);
                            gyro_bias_initialised = true;
                        }
                    } else {
                        double gyro_mean = gyro_sum / gyro_data.size();
                        gyro_bias = gyro_bias_alpha * gyro_mean + (1 - gyro_bias_alpha) * gyro_bias;
                    }
                }
            }
            previous_vel_null = true;
            } else {
                previous_vel_null = false;
        }

        if (estimate_gyro_bias && verbose){
            if(gyro_bias_initialised) {
                std::cout << "Gyro bias: " << gyro_bias << "\n";
            } else {
                std::cout << "Gyro bias not initialised yet\n";
            }
        }

        auto [pos, rot] = state_estimator.getAzPosRot();

        if ( pos.has_value() && rot.has_value()) {
            auto current_pos = pos.value();
            auto current_rot = rot.value();

            // Eigen::MatrixXd current_pos_eig = Eigen::Map<Eigen::MatrixXd>(current_pos.squeeze().to(torch::kCPU).data_ptr<double>(), current_pos.size(0), current_pos.size(1));
            // Eigen::VectorXd current_rot_eig = Eigen::Map<Eigen::VectorXd>(current_rot.squeeze().to(torch::kCPU).data_ptr<double>(), current_rot.size(0));

            double radar_timestamp_sec = radar_data.timestamp;
            double min_diff = std::numeric_limits<double>::max();
            int mid_id = 0;
            for (int idx = 0; idx < radar_data.timestamps.size(0); ++idx) {
                double diff = std::abs(radar_data.timestamps[idx].item<double>() * 1e-6 - radar_timestamp_sec);
                if (diff < min_diff) {
                    min_diff = diff;
                    mid_id = idx;
                }
            }

            // Create transformation matrix
            Eigen::Matrix4d trans_mat = Eigen::Matrix4d::Identity();
            double cos_theta = std::cos(current_rot[mid_id].item<double>());
            double sin_theta = std::sin(current_rot[mid_id].item<double>());
            trans_mat(0, 0) = cos_theta;
            trans_mat(0, 1) = -sin_theta;
            trans_mat(1, 0) = sin_theta;
            trans_mat(1, 1) = cos_theta;
            trans_mat(0, 3) = current_pos[mid_id][0].item<double>();
            trans_mat(1, 3) = current_pos[mid_id][1].item<double>();

            x_y_odom_output << trans_mat(0, 3) << " " << trans_mat(1, 3) << std::endl;

            Eigen::Matrix4d trans_mat_inv = trans_mat.inverse();

            odom_output << static_cast<int64_t>(radar_data.timestamps[mid_id].item<double>()) << " ";
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 4; ++c) {
                    odom_output << trans_mat_inv(r, c);
                    if (!(r == 2 && c == 3)) odom_output << " ";
                }
            }
            odom_output << std::endl;
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
    x_y_odom_output.close();

    return 0;
}
