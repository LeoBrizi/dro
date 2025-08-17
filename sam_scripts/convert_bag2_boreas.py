import os
from pathlib import Path
from turtle import width
import numpy as np
import math
import cv2
import argparse
import rclpy
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores, get_types_from_msg
from rosbags.serde import deserialize_cdr
import tqdm

# import audio_utils as autils

# Define custom message types
RADAR_SCAN_MSG = """
# A ROS message carrying a B Scan and its associated metadata (e.g. timestamps, encoder IDs)
sensor_msgs/Image b_scan_img
uint16[] encoder_values
uint64[] timestamps
"""

RADAR_FFT_MSG = """
# A ROS message based on an FFT data message from a radar
std_msgs/Header header
uint8[] angle
uint8[] azimuth
uint8[] sweep_counter
uint8[] ntp_seconds
uint8[] ntp_split_seconds
uint8[] data
uint8[] data_length
"""
RADAR_CONFIG_MSG = """
# A ROS message based on a configuration data message from a radar, with corrected types

# add a header message to hold message timestamp
std_msgs/Header header

# azimuth_samples (uint16)
uint16 azimuth_samples

# encoder_size (uint16)
uint16 encoder_size

# user-provided azimuth offset (uint16)
uint16 azimuth_offset

# bin_size (float64)
float32 bin_size

# range_in_bins (uint16)
uint16 range_in_bins

# expected_rotation_rate (uint16)
uint16 expected_rotation_rate

# range_gain (float32)
float32 range_gain

# range_offset (float32)
float32 range_offset
"""
SYSNERGIE_MSG = """
# A ROS message for Sysnergie batteries
std_msgs/Header header

float32 pack_voltage
float32 pack_current

int8 state
int8 main_fault_reg
int8 soc
int64 remaining_capacity

float32 min_temp
float32 max_temp
float32 min_cell_volt
float32 max_cell_volt

float32 brick_volt_avg
float32 cell_volt_avg
float32 pack_temp_avg
int64 pack_imbalance
"""
CLEARPATH_POWER_MSG = """
# A ROS message for Clearpath power
# Robot Power readings

std_msgs/Header header

# AC Power
int8 NOT_APPLICABLE=-1

int8 shore_power_connected  # Indicates if AC power is connected.
int8 battery_connected      # Indicates if battery is connected.
int8 power_12v_user_nominal # Indicates if the 12V user power is good.
int8 charger_connected      # Indicates if a charger is connected.
int8 charging_complete      # Indicates if charging is complete.

# Voltage rails, in volts
# Averaged over the message period

# Jackal (J100)
uint8 J100_MEASURED_BATTERY=0
uint8 J100_MEASURED_5V=1
uint8 J100_MEASURED_12V=2

# Dingo 1.0 (D100)
uint8 D100_MEASURED_BATTERY=0
uint8 D100_MEASURED_5V=1
uint8 D100_MEASURED_12V=2

# Dingo 1.5 (D150)
uint8 D150_MEASURED_BATTERY=0
uint8 D150_MEASURED_5V=1
uint8 D150_MEASURED_12V=2

# Warthog (W200)
uint8 W200_MEASURED_BATTERY=0
uint8 W200_MEASURED_12V=1
uint8 W200_MEASURED_24V=2
uint8 W200_MEASURED_48V=3

# Ridgeback (R100)
uint8 R100_MEASURED_BATTERY=0
uint8 R100_MEASURED_5V=1
uint8 R100_MEASURED_12V=2
uint8 R100_MEASURED_INVERTER=3
uint8 R100_MEASURED_FRONT_AXLE=4
uint8 R100_MEASURED_REAR_AXLE=5
uint8 R100_MEASURED_LIGHT=6

# Husky (A200)
uint8 A200_BATTERY_VOLTAGE=0
uint8 A200_LEFT_DRIVER_VOLTAGE=1
uint8 A200_RIGHT_DRIVER_VOLTAGE=2
uint8 A200_VOLTAGES_SIZE=3

float32[] measured_voltages

# Current senses available on platform, in amps.
# Averaged over the message period

# Jackal (J100)
uint8 J100_TOTAL_CURRENT=0
uint8 J100_COMPUTER_CURRENT=1
uint8 J100_DRIVE_CURRENT=2
uint8 J100_USER_CURRENT=3

# Dingo 1.0 (D100)
uint8 D100_TOTAL_CURRENT=0
uint8 D100_COMPUTER_CURRENT=1

# Dingo 1.5 (D150)
uint8 D150_TOTAL_CURRENT=0
uint8 D150_COMPUTER_CURRENT=1

# Warthog (W200)
uint8 W200_TOTAL_CURRENT=0
uint8 W200_COMPUTER_CURRENT=1
uint8 W200_12V_CURRENT=2
uint8 W200_24V_CURRENT=3

# Ridgeback (R100)
uint8 R100_TOTAL_CURRENT=0

# Husky (A200)
uint8 A200_MCU_AND_USER_PORT_CURRENT=0
uint8 A200_LEFT_DRIVER_CURRENT=1
uint8 A200_RIGHT_DRIVER_CURRENT=2
uint8 A200_CURRENTS_SIZE=3

float32[] measured_currents
"""
CLEARPATH_STATUS_MSG = """
# This message represents lower-frequency status updates
# Default publish frequency is 1Hz.

std_msgs/Header header

# Robot Hardware ID
string hardware_id

# Firmware version
string firmware_version

# Times since MCU power-on.
builtin_interfaces/Duration mcu_uptime
builtin_interfaces/Duration connection_uptime

# Temperature of MCU's PCB in Celsius.
float32 pcb_temperature
float32 mcu_temperature
"""
CLEARPATH_LIGHTS_MSG = """
# Represents a command for the pairs of RGB body lights on a CPR robot.

# Dingo 1.0 (D100)
uint8 D100_LIGHTS_REAR_LEFT=0
uint8 D100_LIGHTS_FRONT_LEFT=1
uint8 D100_LIGHTS_FRONT_RIGHT=2
uint8 D100_LIGHTS_REAR_RIGHT=3

# Dingo 1.5 (D150)
uint8 D150_LIGHTS_REAR_LEFT=0
uint8 D150_LIGHTS_FRONT_LEFT=1
uint8 D150_LIGHTS_FRONT_RIGHT=2
uint8 D150_LIGHTS_REAR_RIGHT=3

# Ridgeback (R100)
uint8 R100_LIGHTS_FRONT_PORT_UPPER=0
uint8 R100_LIGHTS_FRONT_PORT_LOWER=1
uint8 R100_LIGHTS_FRONT_STARBOARD_UPPER=2
uint8 R100_LIGHTS_FRONT_STARBOARD_LOWER=3
uint8 R100_LIGHTS_REAR_PORT_UPPER=4
uint8 R100_LIGHTS_REAR_PORT_LOWER=5
uint8 R100_LIGHTS_REAR_STARBOARD_UPPER=6
uint8 R100_LIGHTS_REAR_STARBOARD_LOWER=7

# Warthog (W200)
uint8 W200_LIGHTS_FRONT_LEFT=0
uint8 W200_LIGHTS_FRONT_RIGHT=1
uint8 W200_LIGHTS_REAR_LEFT=2
uint8 W200_LIGHTS_REAR_RIGHT=3

RGB[] lights
"""
CLEARPATH_RGB_MSG = """
# Represents the intensity of a single RGB LED, either reported or commanded.
# Each channel is limited to a range of [0, 255]

uint8 red
uint8 green
uint8 blue
"""

BASLER_CURRENT_PARAMS_MSG = """
uint32 offset_x # -20000 = Error
uint32 offset_y # -20000 = Error
bool reverse_x
bool reverse_y

int32 black_level # -10000 = error/not available

int32 pgi_mode # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Off, 1 = On
int32 demosaicing_mode # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Simple, 1 = BaslerPGI
float32 noise_reduction # -20000.0 = Error, -10000.0 = Not available
float32 sharpness_enhancement # -20000.0 = Error, -10000.0 = Not available
int32 light_source_preset # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Off, 1 = Daylight5000K, 2 = Daylight6500K, 3 = Tungsten2800K
int32 balance_white_auto # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Off, 1 = Once, 2 = Continuous

int32 sensor_readout_mode # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Normal, 1 = Fast
int32 acquisition_frame_count # -20000 = Error, -10000 = Not available
int32 trigger_selector # -3 = Unknown, -2 = Error, -1 = Not available, 0 = FrameStart, 1 = FrameBurstStart(USB)/AcquisitionStart(GigE)
int32 trigger_mode # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Off, 1 = On
int32 trigger_source # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Software, 1 = Line1, 2 = Line3, 3 = Line4, 4 = Action1(Selected Gige)
int32 trigger_activation # -3 = Unknown, -2 = Error, -1 = Not available, 0 = RisingEdge, 1 = FallingEdge
float32 trigger_delay # -20000.0 = Error, -10000.0 = Not available

int32 user_set_selector # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Default, 1 = UserSet1, 2 = UserSet2, 3 = UserSet3, 4 = HighGain, 5 = AutoFunctions, 6 = ColorRaw
int32 user_set_default_selector # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Default, 1 = UserSet1, 2 = UserSet2, 3 = UserSet3, 4 = HighGain, 5 = AutoFunctions, 6 = ColorRaw

bool is_sleeping
float32 brightness
float32 exposure
float32 gain
float32 gamma
uint32 binning_x
uint32 binning_y
float32 temperature # Shows the camera temperature. If not available, then 0.0. USB uses DeviceTemperature and GigE TemperatureAbs parameters.
int32 max_num_buffer		# -2 = Error, -1 = Not available
sensor_msgs/RegionOfInterest roi

string[] available_image_encoding
string current_image_encoding
string current_image_ros_encoding

bool success
string message

string ptp_status           # latched state of the PTP clock, see https://ja.docs.baslerweb.com/pylonapi/net/T_Basler_Pylon_PLCamera_PtpStatusEnum
string ptp_servo_status     # latched state of the clock servo, see https://docs.baslerweb.com/pylonapi/net/T_Basler_Pylon_PLCamera_PtpServoStatusEnum
int64 ptp_offset    # ptp offset from master in ticks [ns]
"""
RTF_MSG = """
std_msgs/Header header

float64 c0
float64 c1
float64 c00
float64 c01
float64 c10
float64 c11
float64 c20
float64 c21
float64 c30

float64 raw_pressure
float64 raw_temperature
float64 scale_pressure
float64 scale_temperature
float64 pressure
float64 temperature
"""

AUDIO_INFO_MSG = """
# This message contains the audio meta data

# Number of channels
uint8 channels
# Sampling rate [Hz]
uint32 sample_rate
# Audio format (e.g. S16LE)
string sample_format
# Amount of audio data per second [bits/s]
uint32 bitrate
# Audio coding format (e.g. WAVE, MP3)
string coding_format
"""

AUDIO_DATA_MSG = """
uint8[] data
"""

AUDIO_DATA_STAMPED_MSG = """
std_msgs/Header header
audio_common_msgs/AudioData audio
"""
POINT_CLOUD2_MSG= """
std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense
"""

DATA_TYPES = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def get_fomo_typestore():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(
        get_types_from_msg(RADAR_FFT_MSG, "nav_messages/msg/RadarFftDataMsg")
    )
    typestore.register(
        get_types_from_msg(RADAR_SCAN_MSG, "navtech_msgs/msg/RadarBScanMsg")
    )
    typestore.register(
        get_types_from_msg(RADAR_CONFIG_MSG, "navtech_msgs/msg/RadarConfigurationMsg")
    )
    typestore.register(
        get_types_from_msg(CLEARPATH_POWER_MSG, "clearpath_platform_msgs/msg/Power")
    )
    typestore.register(
        get_types_from_msg(SYSNERGIE_MSG, "sysnergie_msgs/msg/BatteryLog")
    )
    typestore.register(
        get_types_from_msg(CLEARPATH_STATUS_MSG, "clearpath_platform_msgs/msg/Status")
    )
    typestore.register(
        get_types_from_msg(CLEARPATH_LIGHTS_MSG, "clearpath_platform_msgs/msg/Lights")
    )
    typestore.register(
        get_types_from_msg(CLEARPATH_RGB_MSG, "clearpath_platform_msgs/msg/RGB")
    )
    typestore.register(
        get_types_from_msg(
            BASLER_CURRENT_PARAMS_MSG, "pylon_ros2_camera_interfaces/msg/CurrentParams"
        )
    )
    typestore.register(
        get_types_from_msg(RTF_MSG, "rtf_sensors_msgs/msg/CustomPressureTemperature")
    )

    typestore.register(
        get_types_from_msg(AUDIO_INFO_MSG, "audio_common_msgs/msg/AudioInfo")
    )
    typestore.register(
        get_types_from_msg(AUDIO_DATA_MSG, "audio_common_msgs/msg/AudioData")
    )
    typestore.register(
        get_types_from_msg(
            AUDIO_DATA_STAMPED_MSG, "audio_common_msgs/msg/AudioDataStamped"
        )
    )
    return typestore

# rotation matrix helper
def rot_x(a): 
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=np.float64)

def rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=np.float64)

def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=np.float64)


class BagToDir():
    def __init__(self, bag_file, output_dir):
        self.bag_file = bag_file
        self.output_dir = output_dir
        
        # Create output directories
        self.radar_image_dir = output_dir

    def read_bag(self):
        bag_path = Path(self.bag_file)
        typestore = get_fomo_typestore()

        with Reader(bag_path) as reader:
            reader.typestore = typestore
            connections = list(reader.connections)
                 
            # loop through all the connections
            for connection, timestamp, rawdata in tqdm.tqdm(reader.messages(), total=reader.message_count, desc="Processing data"):
                try:
                    topic_name = connection.topic
                    
                    # radar navtech topic
                    if topic_name.startswith('/radar_data/b_scan_msg'):
                        if not os.path.exists(self.radar_image_dir):
                            self.radar_image_dir = os.path.join(self.output_dir, 'navtech')
                            os.makedirs(self.radar_image_dir, exist_ok=True)
                        self.save_radar_image(connection, rawdata, typestore)


                except Exception as e:
                    print(f'Error processing message: {str(e)}')


    def save_radar_image(self, msg, rawdata, typestore):
        msg = deserialize_cdr(rawdata, msg.msgtype, typestore=typestore)
        try:
            img_msg = msg.b_scan_img
            timestamp_row = np.array(msg.timestamps, dtype=np.uint64) # possibly in nano-secs

            # put it in microsecs
            timestamp_row = np.floor(timestamp_row / 1000).astype(np.uint64)  # convert to microseconds
            # print("Timestamp row:", timestamp_row)
            encoder_values = np.array(msg.encoder_values, dtype=np.uint16)

            radar_data = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width)
            
            timestamp = img_msg.header.stamp
            nano_sec = img_msg.header.stamp.nanosec
            stamp_in_micro = timestamp.sec * 1_000_000 + (nano_sec // 1_000)

            # floor the stamp in micro
            stamp_in_micro = math.floor(stamp_in_micro)
            image_filename = os.path.join(self.radar_image_dir, f'{str(stamp_in_micro)}.png')

            timestamp_bytes = np.frombuffer(timestamp_row.tobytes(), dtype=np.uint8).reshape(-1, 8)
            encoder_bytes = np.frombuffer(encoder_values.tobytes(), dtype=np.uint8).reshape(-1, 2)

            final_data = np.zeros((radar_data.shape[0], radar_data.shape[1] + 11), dtype=np.uint8)
            final_data[:, :8] = timestamp_bytes
            final_data[:, 8:10] = encoder_bytes
            final_data[:, 11:] = radar_data

            cv2.imwrite(image_filename, final_data)
            print(f'Saved image: {image_filename}')

        except Exception as e:
            print(f'Error saving radar image: {str(e)}')


def main():
    parser = argparse.ArgumentParser(description='Convert ROS2 bag to sensor data files.')
    parser.add_argument('--input', type=str, default='/home/samqiao/ASRL/vtr3_data/new_rss_routes/grassy/grassy2', help='Path to input bag file')
    parser.add_argument('--output', type=str, default='/home/samqiao/ASRL/dro/data/grassy_t2/radar', help='Output directory')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output directory')
    args = parser.parse_args()

    if os.path.exists(args.output):
        if args.overwrite:
            import shutil
            shutil.rmtree(args.output)
        else:
            raise FileExistsError(f"Output directory {args.output} already exists. Use --overwrite to replace.")
    
    os.makedirs(args.output, exist_ok=True)
    
    converter = BagToDir(args.input, args.output)
    converter.read_bag()
    print("Conversion completed successfully!")

if __name__ == '__main__':
    main()