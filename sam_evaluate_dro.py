import numpy as np
import matplotlib.pyplot as plt

# path_to_odometry_result = "/home/samqiao/ASRL/dro/dro/build/app/output/boreas-2020-11-26-13-58/odometry_result/boreas-2020-11-26-13-58.txt"
# # Read data from file
# data = []
# with open(path_to_odometry_result, 'r') as file:  # Replace with your filename
#     for line in file:
#         parts = line.strip().split()
#         # Convert all parts to floats (ignore timestamp for plotting)
#         row = list(map(float, parts))
#         data.append(row)

# # Extract translation vectors (last column of each 3x4 matrix)
# translations = []
# for row in data:
#     # The matrix elements are indices 1 to 12 (after timestamp)
#     matrix_flat = row[1:13]
#     # Extract translation components: indices 3, 7, 11 in the flattened array
#     x = matrix_flat[3]
#     y = matrix_flat[7]
#     z = matrix_flat[11]
#     translations.append([x, y, z])

# translations = np.array(translations)

# # Create 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot trajectory
# ax.plot(translations[:, 0], translations[:, 1], translations[:, 2], 
#         'b-', linewidth=2, label='Trajectory')
# ax.scatter(translations[0, 0], translations[0, 1], translations[0, 2], 
#            c='green', s=100, label='Start')
# ax.scatter(translations[-1, 0], translations[-1, 1], translations[-1, 2], 
#            c='red', s=100, label='End')

# # Labels and title
# ax.set_xlabel('X Position', fontsize=12)
# ax.set_ylabel('Y Position', fontsize=12)
# ax.set_zlabel('Z Position', fontsize=12)
# ax.set_title('DRO Odometry Trajectory', fontsize=16)
# ax.legend(fontsize=10)
# ax.grid(True)

# # Equal aspect ratio for better visualization
# max_range = np.array([
#     translations[:,0].max()-translations[:,0].min(), 
#     translations[:,1].max()-translations[:,1].min(), 
#     translations[:,2].max()-translations[:,2].min()
# ]).max() * 0.5
# mid_x = (translations[:,0].max()+translations[:,0].min()) * 0.5
# mid_y = (translations[:,1].max()+translations[:,1].min()) * 0.5
# mid_z = (translations[:,2].max()+translations[:,2].min()) * 0.5
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)

# plt.tight_layout()
# plt.show()


# we need to align the trajectory with the ground truth and they project them into gps frame and then calculate the error
# also there is another file where there is only two columns, one is x and the other is y
x_y_txt_path = "/home/samqiao/ASRL/dro/dro/build/app/output/boreas-2020-11-26-13-58/x_y_odometry.txt"

# Read x and y data
x_y_data = []
with open(x_y_txt_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        # Convert all parts to floats
        row = list(map(float, parts))
        x_y_data.append(row)

x_y_data = np.array(x_y_data)   
# Extract x and y coordinates
x_coords = x_y_data[:, 0]
y_coords = x_y_data[:, 1]

# Create 2D plot for x and y coordinates
plt.figure(figsize=(10, 8))
plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='Trajectory')
plt.scatter(x_coords[0], y_coords[0], c='green', s=100
              , label='Start')  
plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, label='End')
plt.xlabel('X Position', fontsize=12)
plt.ylabel('Y Position', fontsize=12)
plt.title('DRO CPP Odometry Trajectory (2D)', fontsize=16)
plt.legend(fontsize=10)
plt.grid(True)
plt.axis('equal')  # Equal scaling for x and y axes
plt.tight_layout()
plt.show()
