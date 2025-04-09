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

# Wait for user interaction before closing all plots
input("Press Enter to close all plots...")

# Defining Obstacles



# Defining start and goal positions (and orientations)
# We need to define an "ideal" positin as a home for starting the algorithms



# Defining grid (fusion reachability map + obstacles)



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
