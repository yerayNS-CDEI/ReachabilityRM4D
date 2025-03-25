import numpy as np
import matplotlib.pyplot as plt

from exp_utils import robot_types
from rm4d.robots import Simulator


def create_2d_grid(x_min, x_max, y_min, y_max, resolution):
    x_vals = np.linspace(x_min, x_max, int((x_max - x_min) / resolution) + 1)
    y_vals = np.linspace(y_min, y_max, int((y_max - y_min) / resolution) + 1)

    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')  # shape: (764, 764)
    grid = np.stack((xx, yy), axis=-1)  # shape: (764, 764, 2)

    return grid, x_vals, y_vals


def filter_circle_area(grid, radius):
    rows, cols = grid.shape[:2]

    # Compute center coordinates
    center = grid[rows // 2, cols // 2]

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


sim = Simulator(with_gui=False)
robot = robot_types['ur5e'](sim, base_pos=[0, 0, 0.8])

radius = robot.range_radius
z_max = robot.range_z

print("Robot name: ur5e")
print("Robot radius: ", radius)
print("Robot altitude: ", z_max)

x_min = -radius
x_max = radius
y_min = -radius
y_max = radius
print("Value ranges: [",x_min,",",x_max,"] and [",y_min,",",y_max,"]")

# Set a desired grid resolution (number of grid cells per unit distance)
max_error_distance = 0.002886 # Distance from voxel to voxel (maximum distnace in the diagonal to obtain a 5mm error - KPI)
grid_resolution = 1/max_error_distance  # This is the number of cells per unit distance (you can adjust this as needed)

# Calculate the grid size based on the actual range and grid resolution
grid_size_x = int(np.ceil((x_max - x_min) * grid_resolution))  # Grid size in x-direction
grid_size_y = int(np.ceil((y_max - y_min) * grid_resolution))  # Grid size in y-direction
print("Grid size:", grid_size_x, " x ", grid_size_y)

# Example usage
resolution = (x_max - x_min) / 763  # ensures 764 steps
grid, x_vals, y_vals = create_2d_grid(x_min, x_max, y_min, y_max, resolution)
filtered_coords, mask = filter_circle_area(grid, radius)

occupancy_map = np.zeros((764, 764), dtype=np.uint8)
occupancy_map[mask] = 1  # only mark valid cells

print("Total points in full grid:", grid.shape[0] * grid.shape[1])
print("Points inside circle:", filtered_coords.shape[0])

plt.imshow(mask.T, origin='lower', cmap='gray')
plt.title("Circular Area Mask")
plt.xlabel("Y index")
plt.ylabel("X index")
plt.show()

# # Wait for user interaction
# input("Press Enter to close all plots...")

# i, j = world_to_grid(0.0, -0.5, x_vals, y_vals)
# print(i, j)  # might print something like (381, 208)
# print(grid[381][208])