# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')  # or 'Agg' for non-GUI environments
# import matplotlib.pyplot as plt

# # Ensure that the loaded file is a dictionary
# reachability_maps = np.load('reachability_maps.npy', allow_pickle=True).item()

# for z_value, reachability_slice in reachability_maps.items():
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

import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for non-GUI environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import numpy as np

# # rots = Rotation.random(20)
# # vectors = rots.apply([1, 0, 0])  # rotate Z-axis

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], s=1)
# # ax.quiver(0, 0, 0, vectors[:, 0], vectors[:, 1], vectors[:, 2], color='b', length=0.8, normalize=True)
# # ax.set_title("Distribution of rotated Z-axes")
# # plt.show()


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

# Generate 20 evenly distributed points on the unit sphere
samples = 20
points = fibonacci_sphere(samples)

# The original vector is the X-axis (1, 0, 0)
origin_vector = np.array([0, 0, 1])

# We need to create rotations to align `origin_vector` with each of the sampled points
rots = []
rot_matrices = []
for point in points:
    # The rotation matrix that aligns origin_vector to the point on the sphere
    # The point is a unit vector, so we can directly treat it as the desired direction
    rotation = Rotation.align_vectors([point], [origin_vector])[0]  # Align origin_vector to the point
    rots.append(rotation)
    rot_matrices.append(rotation.as_matrix())

rot_matrices = np.array(rot_matrices)

# Apply the rotations to the origin vector (1, 0, 0)
rotated_vectors = np.array([rot.apply(origin_vector) for rot in rots])

# Plot the result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the rotated vectors
ax.scatter(rotated_vectors[:, 0], rotated_vectors[:, 1], rotated_vectors[:, 2], s=50)
ax.quiver(0, 0, 0, rotated_vectors[:, 0], rotated_vectors[:, 1], rotated_vectors[:, 2], color='b', length=0.8, normalize=True)

ax.set_title("Evenly Distributed Rotated Orientations")
plt.show()