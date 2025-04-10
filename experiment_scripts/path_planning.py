import os
import numpy as np
from exp_utils import robot_types
from rm4d.robots import Simulator
import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation

def dilate_obstacles(occupancy_grid, dilation_distance, grid_size):
    # Create a structuring element for dilation (3D cube of size dilation_distance)
    dilation_size = int(np.ceil(dilation_distance / (x_vals[1] - x_vals[0])))  # Convert distance to grid units
    struct_element = np.ones((dilation_size, dilation_size, dilation_size), dtype=np.uint8)
    
    # Perform 3D dilation
    dilated_grid = binary_dilation(occupancy_grid, structure=struct_element).astype(np.uint8)
    return dilated_grid

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

# # Plotting the reachability map
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# scatter = ax.scatter(reach_y, reach_x, reach_z, c=reachability, cmap='viridis', marker='o', s=50)
# # plt.figure()
# # plt.imshow(reachability_slice, cmap='hot', interpolation='nearest')
# # plt.colorbar(label='Reachability')
# ax.set_title("Reachability Map")
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# fig.colorbar(scatter, ax=ax, label='Reachability')
# plt.show(block=False)

# # Wait for user interaction before closing all plots
# input("Press Enter to close all plots...")

###########################################################################
# Defining Obstacles & Defining grid (fusion reachability map + obstacles)
###########################################################################

# Get Z levels and determine shape of the occupancy map
z_levels = sorted(reachability_map.keys())
grid_shape = (grid_size, grid_size, len(z_levels))
occupancy_grid = np.zeros(grid_shape, dtype=np.uint8)

# Get the z-values similar to x_vals and y_vals
z_min = min(z_levels)
z_max = max(z_levels)
z_vals = np.linspace(z_min, z_max, len(z_levels))  # assumes uniform spacing

# 3D meshgrid of coordinates
X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')  # shape: (grid_size, grid_size, num_z)

# Define cube bounds (in world coordinates)
cube_center = (0.2, 0.2, 1)  # (x, y, z)
cube_size = 0.4  # length of each side

# Compute bounds
x0, y0, z0 = cube_center
half = cube_size / 2

cube_mask = (
    (X >= x0 - half) & (X <= x0 + half) &
    (Y >= y0 - half) & (Y <= y0 + half) &
    (Z >= z0 - half) & (Z <= z0 + half)
)
occupancy_grid[cube_mask] = 1

# Define sphere parameters
sphere_center = [(-0.3, 0.0, 0.4), (0.3, 0.5, 0.4)]
sphere_radius = [0.3, 0.3]

# Create mask for sphere
for i in range(len(sphere_radius)):
    dist_squared = (X - sphere_center[i][0])**2 + (Y - sphere_center[i][1])**2 + (Z - sphere_center[i][2])**2
    sphere_mask = dist_squared <= sphere_radius[i]**2
    occupancy_grid[sphere_mask] = 1

# Plotting the reachability map
fig = plt.figure()
ax = plt.axes(projection='3d')

# Reachability points (non-zero)
scatter = ax.scatter(reach_y, reach_x, reach_z, c=reachability, cmap='viridis', marker='o', s=10, alpha=0.2, label='Reachability')

# Obstacle points
# Get voxel coordinates where occupancy = 1
obstacle_indices = np.argwhere(occupancy_grid == 1)
obs_x = x_vals[obstacle_indices[:, 0]]
obs_y = y_vals[obstacle_indices[:, 1]]
obs_z = z_vals[obstacle_indices[:, 2]]

# Plot obstacles in red
ax.scatter(obs_y, obs_x, obs_z, c='red', marker='s', s=20, alpha=0.9, label='Obstacle')

# Labels and aesthetics
ax.set_title("Reachability Map with Obstacles")
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z')
fig.colorbar(scatter, ax=ax, label='Reachability')
ax.legend()
plt.show(block=False)

# Wait for user interaction before closing all plots
input("Press Enter to close all plots...")

# Define the dilation distance (in meters, for example)
dilation_distance = 0.3  # Enlarge obstacles by 0.1 meters in all directions

# Apply dilation
occupancy_grid_dilated = dilate_obstacles(occupancy_grid, dilation_distance, grid_size)

# Plotting the reachability map
fig = plt.figure()
ax = plt.axes(projection='3d')

# Reachability points (non-zero)
scatter = ax.scatter(reach_y, reach_x, reach_z, c=reachability, cmap='viridis', marker='o', s=10, alpha=0.2, label='Reachability')

# Obstacle points
# Get voxel coordinates where occupancy = 1 in the dilated occupancy grid
obstacle_indices = np.argwhere(occupancy_grid_dilated == 1)
obs_x = x_vals[obstacle_indices[:, 0]]
obs_y = y_vals[obstacle_indices[:, 1]]
obs_z = z_vals[obstacle_indices[:, 2]]

# Plot obstacles in red
ax.scatter(obs_y, obs_x, obs_z, c='red', marker='s', s=20, alpha=0.9, label='Enlarged Obstacle')

# Labels and aesthetics
ax.set_title("Reachability Map with Enlarged Obstacles")
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z')
fig.colorbar(scatter, ax=ax, label='Reachability')
ax.legend()
plt.show(block=False)

# Wait for user interaction before closing all plots
input("Press Enter to close all plots...")

# Defining start and goal positions (and orientations)
# We need to define an "ideal" position as a home for starting the algorithms
start_position = [0, 0, 0]  # [x,y,z]
goal_position = [0, 0, 0]  # [x,y,z]

# Computing EE path (A* Algorithm)



# Computing interpolated orientations



# Computing robot's joint values (Closed form algorithms)



# Presenting the results in the simulated environement



# Creo que se deberian añadir tambien medidas de seguridad para que el 
# robot se detenga en caso de que aparezcan obstaculos en el camino.
# El entorno debe ser dinamico para detectar estos cambios.
# Una buena manera seria recalcular los puntos en que ahora el robot colisone
# o modificar el path cada vez que aparezcan nuevos obstaculos en el 
# reachability map calculado.

# Tareas posteriores: añadirlo en nodo de ros para que se mueva el robot simulado
