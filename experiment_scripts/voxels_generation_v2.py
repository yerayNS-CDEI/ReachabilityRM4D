##############################################################################
### CODE TO GENERATE GRIDS FOR THE REACHABILITY MAP
##############################################################################

import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')  # or 'Agg' for non-GUI environments
import matplotlib.pyplot as plt
import math

from exp_utils import robot_types
from rm4d.robots import Simulator
from scipy.spatial.transform import Rotation
import os
import pathlib


def create_2d_grid(x_min, x_max, y_min, y_max, resolution):
    # This function gets the center of the N voxels (number_divisions)
    x_vals = np.linspace(x_min+(resolution/2), x_max-(resolution/2), int((x_max - x_min) / resolution))
    y_vals = np.linspace(y_min+(resolution/2), y_max-(resolution/2), int((y_max - y_min) / resolution))
    
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')  # shape: (764, 764)
    grid = np.stack((xx, yy), axis=-1)  # shape: (764, 764, 2)

    return grid, x_vals, y_vals


def filter_circle_area(grid, radius):
    # rows, cols = grid.shape[:2]

    # Compute center coordinates
    # center = grid[rows // 2, cols // 2]
    center = [0, 0]

    # Vectorized distance check
    dx = grid[:, :, 0] - center[0]
    dy = grid[:, :, 1] - center[1]
    distance_squared = dx**2 + dy**2

    mask = distance_squared <= radius**2
    filtered_coords = grid[mask]  # shape: (N, 2)

    return filtered_coords, mask


def world_to_grid(x, y, x_vals, y_vals):
    """
    Converts real-world coordinates (like x = 0.35, y = -0.62) into 
    their corresponding grid indices (i, j) in the 2D array.
    x,y: real-world coordinates
    x_vals,y_vals: 1D arrays of grid coordinates along x and y (create_2d_grid)
    """
    i = np.argmin(np.abs(x_vals - x))   # find closest x in the grid
    j = np.argmin(np.abs(y_vals - y))   # find closest y in the grid
    return i, j

# i, j = world_to_grid(0.0, -0.5, x_vals, y_vals)
# print(i, j)  # might print something like (381, 208)
# print(grid[381][208])

def generate_random_orientations(n_orientations, seed):
    rng = np.random.default_rng(seed)
    rotations = Rotation.random(n_orientations, random_state=rng)
    return rotations.as_matrix()  # shape: (n_orientations, 3, 3)

def fibonacci_sphere(samples=100):
    """
    Generate points uniformly distributed on the surface of a sphere using Fibonacci sampling.
    
    :param samples: Number of points to sample
    :returns: (N, 3) array of points on the unit sphere
    """
    x = []
    y = []
    z = []

    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y.append(1 - (i / float(samples - 1)) * 2)  # y goes from 1 to -1
        radius = np.sqrt(1 - y[i] * y[i])  # radius at y
        x.append(np.cos(phi * i) * radius)  # x = cos(phi) * radius
        z.append(np.sin(phi * i) * radius)  # z = sin(phi) * radius

    return np.array(list(zip(x, y, z)))


def get_poses_from_positions_and_orientations(valid_positions, z_value, orientations):
    """
    Generates poses by combining each (x, y) with all fixed orientations.

    :param valid_positions: np.ndarray (P, 2)
    :param z_value: float
    :param orientations: np.ndarray (N, 3, 3), fixed orientation matrices
    :return: np.ndarray (P * N, 4, 4)
    """
    n_positions = valid_positions.shape[0]
    n_orientations = orientations.shape[0]
    total = n_positions * n_orientations

    tfs_ee = np.full((total, 4, 4), np.eye(4))

    # Repeat positions and tile orientations
    x_pos = np.repeat(valid_positions[:, 0], n_orientations)
    y_pos = np.repeat(valid_positions[:, 1], n_orientations)
    z_pos = np.repeat(z_value, total)
    tfs_ee[:, 0, 3] = x_pos
    tfs_ee[:, 1, 3] = y_pos
    tfs_ee[:, 2, 3] = z_pos

    # Tile orientation matrices across positions
    tiled_rotations = np.tile(orientations, (n_positions, 1, 1))
    tfs_ee[:, :3, :3] = tiled_rotations

    return tfs_ee


sim = Simulator(with_gui=True)
robot = robot_types['ur10e'](sim)    # base_pos=[0, 0, 0.8]

robot_name = 'ur10e'
radius = robot.range_radius
z_max = robot.range_z

print("Robot name: ",robot_name)
print("Robot radius: ", radius)
print("Robot altitude: ", z_max)

x_min = -radius
x_max = radius
y_min = -radius
y_max = radius
print("Value ranges: [",x_min,",",x_max,"] and [",y_min,",",y_max,"]")

# Set a desired grid resolution (number of grid cells per unit distance)
number_divisions = 9   # Number of voxels in x and y directions
voxel_distance = (x_max-x_min)/number_divisions # In the axis directions
max_error = math.sqrt(3)*voxel_distance*1000
print(f"Max error: {max_error:.2f} mm")
print(f"Number of divisions: {number_divisions}")
print(f"Grid resolution / Voxel distance: {voxel_distance} mm")

# Calculate the z values to use on the computation
z_initial = 0
z_values = np.arange(z_initial, z_max, voxel_distance)
print(f"Z coordinates of the grid planes: {z_values}.")
print("Number of Z planes: ",z_values.shape[0])

# Calculate the grid size based on the actual range and grid resolution
grid_size_x = number_divisions
grid_size_y = number_divisions
print("Grid size:", grid_size_x, " x ", grid_size_y)

# Example usage
resolution = (x_max - x_min) / (grid_size_x)    # Same as voxel distance
grid, x_vals, y_vals = create_2d_grid(x_min, x_max, y_min, y_max, resolution)
filtered_coords, mask = filter_circle_area(grid, radius)
print(filtered_coords[0:10])
print("x_vals: ",x_vals)
print("y_vals: ",y_vals)
print("grid shape: ",grid.shape[1])

# Save the reachability maps to a file for future access
grids_dir = os.path.join('data','grids')
pathlib.Path(grids_dir).mkdir(parents=True, exist_ok=True)
grid_npy_fn = os.path.join(grids_dir,f"{robot_name}_grid2D_{grid_size_x}.npy")
grid_csv_fn = os.path.join(grids_dir,f"{robot_name}_grid2D_{grid_size_x}.csv")

# Check if the files already exists and warn before overwriting
if os.path.exists(grid_npy_fn) or os.path.exists(grid_csv_fn):
    response = input(f"Warning: {grid_npy_fn} or {grid_csv_fn} already exists. Do you want to overwrite them? (y/n): ")
    if response.lower() != 'y':
        print("Files not overwritten.")
    else:
        np.save(grid_npy_fn, filtered_coords)
        np.savetxt(grid_csv_fn, filtered_coords, delimiter=",")
        print(f"Reachability maps saved to {grid_npy_fn} and {grid_csv_fn}.")
else:
    np.save(grid_npy_fn, filtered_coords)
    np.savetxt(grid_csv_fn, filtered_coords, delimiter=",")
    print(f"Reachability maps saved to {grid_npy_fn} and {grid_csv_fn}.")

occupancy_map = np.zeros((grid_size_x, grid_size_y), dtype=np.uint8)
occupancy_map[mask] = 1  # only mark valid cells

print("Total points in full grid:", grid.shape[0] * grid.shape[1])
print("Points inside circle:", filtered_coords.shape[0])

plt.figure()
plt.imshow(mask.T, origin='lower', cmap='gray')
plt.title("Circular Area Mask")
plt.xlabel("Y index")
plt.ylabel("X index")
plt.show(block=False)


##############################################################################
### CODE TO GENERATE ORIENTATIONS
##############################################################################
# # orientations = generate_random_orientations(n_orientations=20, seed=42)

# # Generate 20 evenly distributed points on the unit sphere
# samples = 20
# points = fibonacci_sphere(samples)

# # The original vector is the Z-axis (0, 0, 1)
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

# # rots, rot_matrices = fibonacci_rotations(samples)

# # Apply the rotations to the origin vector (0, 0, 1)
# rotated_vectors = np.array([rot.apply(origin_vector) for rot in rots])
# print(rotated_vectors[0])
# print(rot_matrices[0])

# # Plot the result
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the rotated vectors
# ax.scatter(rotated_vectors[:, 0], rotated_vectors[:, 1], rotated_vectors[:, 2], s=50)
# ax.quiver(0, 0, 0, rotated_vectors[:, 0], rotated_vectors[:, 1], rotated_vectors[:, 2], color='b', length=0.8, normalize=True)

# ax.set_title("Evenly Distributed Rotated Orientations")
# plt.show(block=False)

# # z_value = 0.3
# # poses = get_poses_from_positions_and_orientations(filtered_coords, z_value, rot_matrices)

# # sim = Simulator(with_gui=True)

# # for i in range(10):
# #     print([poses[i*20+1,0,3],poses[i*20+1,1,3],z_value])
# #     sim.add_sphere([poses[i*20+1,0,3],poses[i*20+1,1,3],z_value], 0.005, [255,0,0,1.0])
# #     # Wait for user interaction
# #     input("Press Enter to close all plots...")
# # sim.add_sphere([-1.097116644823067, -0.027391874180864972, 0.3], 0.005, [255,0,0,1.0])

# # Wait for user interaction
# input("Press Enter to close all plots...")

##############################################################################
##############################################################################
##############################################################################