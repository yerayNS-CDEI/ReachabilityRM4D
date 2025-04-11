##############################################################################
### CODE FOR TRANSFORMING COMPLETE REACHABILITY MAPS IN .NPY TO .CSV
##############################################################################

# import os
# import numpy as np
# # import matplotlib
# # matplotlib.use('TkAgg')  # or 'Agg' for non-GUI environments
# import matplotlib.pyplot as plt
# from exp_utils import robot_types
# from rm4d.robots import Simulator

# # Ensure that the loaded file is a dictionary
# robot_name = 'ur10e'
# filename = "reachability_map_27_fused"
# fn_npy = f"{filename}.npy"
# reachability_map_fn = os.path.join('data',f'eval_poses_{robot_name}',fn_npy)
# reachability_map = np.load(reachability_map_fn, allow_pickle=True).item()
# csv_map = []

# for z_value, reachability_slice in reachability_map.items():
#     # Plotting the reachability map
#     plt.figure()
#     plt.imshow(reachability_slice, cmap='hot', interpolation='nearest')
#     plt.colorbar(label='Reachability')
#     plt.title(f'Reachability Map at Z = {z_value}')
#     plt.xlabel('X Index')
#     plt.ylabel('Y Index')
#     plt.show(block=False)
    
#     # Extract grid size and resolution
#     parts = filename.split('_')
#     grid_size = int(parts[2])
    
#     # Simulate the robot setup
#     sim = Simulator(with_gui=False)
#     robot = robot_types[robot_name](sim)
#     radius = robot.range_radius
#     x_min = -radius
#     x_max = radius
#     y_min = -radius
#     y_max = radius
    
#     # Calculate resolution and x, y values
#     resolution = (x_max - x_min) / grid_size
#     x_vals = np.linspace(x_min + (resolution / 2), x_max - (resolution / 2), grid_size)
#     y_vals = np.linspace(y_min + (resolution / 2), y_max - (resolution / 2), grid_size)

#     # Loop over the reachability slice and store (x, y, z_value, reachability)
#     for i, x in enumerate(x_vals):
#         for j, y in enumerate(y_vals):
#             reachability_value = reachability_slice[i, j]  # The reachability value at (x, y)
#             csv_map.append([x, y, z_value, reachability_value])

# # Write the data to a CSV file
# fn_csv = os.path.join('data',f'eval_poses_{robot_name}',f"{filename}.csv")
# np.savetxt(fn_csv, csv_map, delimiter=",")
# print(f"Reachability maps saved to {fn_csv}.")


# # Wait for user interaction before closing all plots
# input("Press Enter to close all plots...")

##############################################################################
##############################################################################
##############################################################################

##############################################################################
### CODE FOR VISUALIZING COMPLETE REACHABILITY MAPS IN .NPY FILES - by slices
##############################################################################

# import numpy as np
# # import matplotlib
# # matplotlib.use('TkAgg')  # or 'Agg' for non-GUI environments
# import matplotlib.pyplot as plt

# # Ensure that the loaded file is a dictionary
# reachability_map = np.load("data/eval_poses_ur10e/reachability_map_27.npy", allow_pickle=True).item()

# for z_value, reachability_slice in reachability_map.items():
#     # Plotting the reachability map
#     plt.figure()
#     plt.imshow(reachability_slice, cmap='hot', interpolation='nearest')
#     plt.colorbar(label='Reachability')
#     plt.title(f'Reachability Map at Z = {z_value}')
#     plt.xlabel('X Index')
#     plt.ylabel('Y Index')
#     plt.show(block=False)

# # Wait for user interaction before closing all plots
# input("Press Enter to close all plots...")

##############################################################################
##############################################################################
##############################################################################

##############################################################################
### CODE FOR VISUALIZING COMPLETE REACHABILITY MAPS IN .NPY FILES - in 3D
##############################################################################

import os
import numpy as np
from exp_utils import robot_types
from rm4d.robots import Simulator
import matplotlib.pyplot as plt

# Ensure that the loaded file is a dictionary
robot_name = 'ur10e'
filename = "reachability_map_27_fused"
fn_npy = f"{filename}.npy"
reachability_map_fn = os.path.join('data',f'eval_poses_{robot_name}',fn_npy)
reachability_map = np.load(reachability_map_fn, allow_pickle=True).item()

# Extract grid size and resolution
parts = filename.split('_')
grid_size = int(parts[2])

# Simulate the robot setup
sim = Simulator(with_gui=False)
robot = robot_types[robot_name](sim)
radius = robot.range_radius
x_min = -radius
x_max = radius
y_min = -radius
y_max = radius

# Calculate resolution and x, y values
resolution = (x_max - x_min) / grid_size
x_vals = np.linspace(x_min + (resolution / 2), x_max - (resolution / 2), grid_size)
y_vals = np.linspace(y_min + (resolution / 2), y_max - (resolution / 2), grid_size)

reach_data = []
for z_value, reachability_slice in reachability_map.items():
    # Loop over the reachability slice and store (x, y, z_value, reachability)
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            reachability_value = reachability_slice[i, j]  # The reachability value at (x, y)
            reach_data.append([x, y, z_value, reachability_value])

reach_data = np.array(reach_data)  # shape: (N, 4) where columns are x, y, z, reachability

nonzero_mask = reach_data[:, 3] != 0
filtered_map = reach_data[nonzero_mask]

reach_x = filtered_map[:, 0]
reach_y = filtered_map[:, 1]
reach_z = filtered_map[:, 2]
reachability = filtered_map[:, 3]

# Plotting the reachability map
fig = plt.figure()
ax = plt.axes(projection='3d')
scatter = ax.scatter(reach_y, reach_x, reach_z, c=reachability, cmap='viridis', marker='o', s=50)
# plt.figure()
# plt.imshow(reachability_slice, cmap='hot', interpolation='nearest')
# plt.colorbar(label='Reachability')
ax.set_title("Reachability Map")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
fig.colorbar(scatter, ax=ax, label='Reachability')
plt.show(block=False)

# # Wait for user interaction before closing all plots
# input("Press Enter to close all plots...")

##############################################################################
##############################################################################
##############################################################################

##############################################################################
### CODE FOR FUSING REACHABILITY MAPS IN .NPY FILES
##############################################################################

# import numpy as np
# import matplotlib.pyplot as plt

# # File paths
# file1 = "data/eval_poses_ur10e/reachability_map_27_pre.npy"
# file2 = "data/eval_poses_ur10e/reachability_map_27.npy"
# output_file = "data/eval_poses_ur10e/reachability_map_27_fused.npy"

# # Load the maps
# map1 = np.load(file1, allow_pickle=True).item()
# map2 = np.load(file2, allow_pickle=True).item()

# # Check for overlapping z-values
# overlapping_keys = set(map1.keys()) & set(map2.keys())
# if overlapping_keys:
#     raise ValueError(f"Overlap detected in Z-values: {sorted(overlapping_keys)}. Aborting fusion.")

# # Merge the two maps
# fused_map = {**map1, **map2}  # since keys are disjoint

# # Save the fused result
# np.save(output_file, fused_map)
# print(f"Fused reachability map saved to: {output_file}")

# # Optional visualization
# for z_value in sorted(fused_map.keys()):
#     reachability_slice = fused_map[z_value]
#     plt.figure()
#     plt.imshow(reachability_slice, cmap='hot', interpolation='nearest')
#     plt.colorbar(label='Reachability')
#     plt.title(f'Fused Reachability Map at Z = {z_value}')
#     plt.xlabel('X Index')
#     plt.ylabel('Y Index')
#     plt.show(block=False)

# input("Press Enter to close all plots...")

##############################################################################
##############################################################################
##############################################################################

##############################################################################
### CODE FOR VISUALIZING REACHABILITY SLICES IN .NPY FILES
##############################################################################

# import numpy as np
# import glob
# import os
# import matplotlib.pyplot as plt

# # Specify the folder containing the slice files
# folder_path = grids_dir = os.path.join('data','eval_poses_ur10e')

# # Use glob to find all files matching the pattern "reachability_slice_XX.npy"
# file_pattern = os.path.join(folder_path, 'reachability_slice_*.npy')
# file_list = glob.glob(file_pattern)

# # Specify the range of numbers you want to filter
# min_number = 20  # Lower bound for the slice number
# max_number = 30  # Upper bound for the slice number

# # Filter files that contain a number within the specified range
# selected_files = []
# for file in file_list:
#     # Get the filename without extension
#     filename = os.path.splitext(os.path.basename(file))[0]
    
#     # Split the filename by underscores and check the grid size (second element)
#     parts = filename.split('_')

#     # Ensure the second part (grid size) can be converted to an integer
#     try:
#         grid_size = int(parts[2])
#         if min_number <= grid_size <= max_number:
#             selected_files.append(file)
#     except ValueError:
#         # If there's an error, skip this file
#         print(f"Filename {file} not correctly detected.")
#         continue
# selected_files.sort()

# # Loop over each file
# for file in selected_files:
#     # Load the reachability map slice from the file
#     reachability_slice = np.load(file, allow_pickle=True)
    
#     # Extract the Z value from the file name (the last part after the second underscore)
#     z_value = os.path.splitext(os.path.basename(file))[0].split('_')[-1]

#     # Plotting the reachability map
#     plt.figure()
#     plt.imshow(reachability_slice, cmap='hot', interpolation='nearest')
#     plt.colorbar(label='Reachability')
#     plt.title(f'Reachability Map at Z = {z_value}')
#     plt.xlabel('X Index')
#     plt.ylabel('Y Index')
#     plt.show(block=False)

##############################################################################
##############################################################################
##############################################################################

# #############################################################################
# ## CODE FOR VISUALIZING EVENLY SAMPLED ORIENTATIONS
# #############################################################################

# # import matplotlib
# # matplotlib.use('TkAgg')  # or 'Agg' for non-GUI environments
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial.transform import Rotation
# import numpy as np

# def fibonacci_sphere(samples=100):
#     """
#     Generate points uniformly distributed on the surface of a sphere using Fibonacci sampling.
    
#     :param samples: Number of points to sample
#     :returns: (N, 3) array of points on the unit sphere
#     """
#     x = []
#     y = []
#     z = []

#     phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
#     for i in range(samples):
#         y.append(1 - (i / float(samples - 1)) * 2)  # y goes from 1 to -1
#         radius = np.sqrt(1 - y[i] * y[i])  # radius at y
#         x.append(np.cos(phi * i) * radius)  # x = cos(phi) * radius
#         z.append(np.sin(phi * i) * radius)  # z = sin(phi) * radius

#     return np.array(list(zip(x, y, z)))

# # Generate 20 evenly distributed points on the unit sphere
# samples = 20
# points = fibonacci_sphere(samples)

# # The original vector is the X-axis (1, 0, 0)
# origin_vector = np.array([0, 0, 1])

# # We need to create rotations to align `origin_vector` with each of the sampled points
# rots = []
# rot_matrices = []
# for point in points:
#     # The rotation matrix that aligns origin_vector to the point on the sphere
#     # The point is a unit vector, so we can directly treat it as the desired direction
#     rotation = Rotation.align_vectors([point], [origin_vector])[0]  # Align origin_vector to the point
#     rots.append(rotation)
#     rot_matrices.append(rotation.as_matrix())

# rot_matrices = np.array(rot_matrices)

# # Apply the rotations to the origin vector (1, 0, 0)
# rotated_vectors = np.array([rot.apply(origin_vector) for rot in rots])

# # Plot the result
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the rotated vectors
# ax.scatter(rotated_vectors[:, 0], rotated_vectors[:, 1], rotated_vectors[:, 2], s=50)
# ax.quiver(0, 0, 0, rotated_vectors[:, 0], rotated_vectors[:, 1], rotated_vectors[:, 2], color='b', length=0.8, normalize=True)

# ax.set_title("Evenly Distributed Rotated Orientations")
# plt.show(block=False)

# # Wait for user interaction before closing all plots
# input("Press Enter to close all plots...")

##############################################################################
##############################################################################
##############################################################################