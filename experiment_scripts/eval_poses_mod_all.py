import os
import pathlib
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import matplotlib.pyplot as plt

from rm4d.robots import Simulator
from rm4d.robots.assets import FRANKA_150_URDF, FRANKA_160_URDF, FRANKA_URDF
from exp_utils import robot_types, franka_versions

import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--robot_type', choices=list(robot_types.keys()), help='robot type.', default='franka')
    parser.add_argument('-n', '--num_samples', type=int, help='maximum number of samples', default=int(1e5))
    parser.add_argument('-t', '--threshold', type=int, help='mm/deg aberration accepted for IK', default=25)
    parser.add_argument('-i', '--iterations', type=int, help='number of trials for IK', default=100)
    parser.add_argument('-d', '--degrees', choices=list(franka_versions.keys()), type=int,
                        help='range for franka joint 1 and 7, only applies to franka. default 166.', default=166)
    parser.add_argument('-s', '--seed', type=int, default=27)
    parser.add_argument('-z', '--z_values', type=float, nargs='+', help='Array of z-values for each iteration', default=[0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3])  # Added argument for constant z-plane value
    
    return parser.parse_args()


def get_sim_and_robot(robot_type, degrees):
    sim = Simulator(with_gui=False)
    if robot_type == 'franka':
        urdf_fn = franka_versions[degrees]
        robot = robot_types[robot_type](sim, urdf_fn=urdf_fn)
    else:
        robot = robot_types[robot_type](sim) # base_pos=[0, 0, 0.8]
    return sim, robot


def get_evaluation_poses(max_radius, z_value, n_samples, seed):
    """
    We uniformly sample positions from a cylinder fitted into the space covered by the reachability map. Note that this
    excludes the corners of the map, but they are trivially not reachable by the robot.
    The orientation of the end-effector is randomly uniformly sampled.

    :param max_radius: radius of the cylinder from which to sample
    :param max_z: height of the cylinder form which to sample
    :param n_samples: int, how many poses to sample
    :param seed: int, random seed
    :returns: np.ndarray (n_samples, 4, 4), EE-poses
    """
    rng = np.random.default_rng(seed)

    # start from identity matrices
    tfs_ee = np.full((n_samples, 4, 4), np.eye(4))

    # first, uniformly sample an orientation (as per scipy documentation)
    tfs_ee[:, :3, :3] = Rotation.random(num=n_samples, random_state=rng).as_matrix()

    # now, we want to uniformly sample the xy position from within a circle, see:
    # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/50746409#50746409
    radii = max_radius * np.sqrt(rng.uniform(0, 1, n_samples))
    angles = 2 * np.pi * rng.uniform(0, 1, n_samples)
    x_pos = radii * np.cos(angles)
    y_pos = radii * np.sin(angles)
    tfs_ee[:, 0, 3] = x_pos
    tfs_ee[:, 1, 3] = y_pos
    # we uniformly sample z, so the 3d position will be uniformly within a cylinder.
    # tfs_ee[:, 2, 3] = rng.uniform(0.0, max_z, n_samples)
    tfs_ee[:, 2, 3] = z_value

    return tfs_ee

# def get_full_map_poses_from_grid(valid_positions, z_value, samples_per_point, seed):
#     """
#     Sample poses at every (x, y) point in the grid, limited to the circular mask area.
#     Each point gets N random end-effector orientations.

#     :param grid: np.ndarray of shape (H, W, 2), with (x, y) coordinates
#     :param mask: np.ndarray of shape (H, W), bool mask for points inside the circle
#     :param z_value: float, z position for all poses
#     :param samples_per_point: int, number of poses to generate per (x, y) point
#     :param seed: int, random seed for reproducibility
#     :returns: np.ndarray of shape (N, 4, 4), with N = num_masked_points * samples_per_point
#     """
#     rng = np.random.default_rng(seed)

#     n_positions = valid_positions.shape[0]
#     total_samples = n_positions * samples_per_point

#     # Create (total_samples, 4, 4) identity matrices
#     tfs_ee = np.full((total_samples, 4, 4), np.eye(4))

#     # Random orientations for each sample
#     tfs_ee[:, :3, :3] = Rotation.random(num=total_samples, random_state=rng).as_matrix()

#     # Assign positions
#     # Repeat each position samples_per_point times
#     x_pos = np.repeat(valid_positions[:, 0], samples_per_point)
#     y_pos = np.repeat(valid_positions[:, 1], samples_per_point)

#     tfs_ee[:, 0, 3] = x_pos
#     tfs_ee[:, 1, 3] = y_pos
#     tfs_ee[:, 2, 3] = z_value

#     return tfs_ee

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

# def generate_fixed_orientations(n_orientations, seed):
#     rng = np.random.default_rng(seed)
#     rotations = Rotation.random(n_orientations, random_state=rng)
#     return rotations.as_matrix()  # shape: (n_orientations, 3, 3)

# # Precompute once
# orientations = generate_fixed_orientations(n_orientations=20, seed=42)
# valid_positions = grid[mask]

# # Use in main
# tfs = get_poses_from_positions_and_orientations(valid_positions, z_value=1.25, orientations=orientations)
# print("Total poses:", tfs.shape[0])  # should be 457265 * 20

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# rots = Rotation.random(1000)
# vectors = rots.apply([0, 0, 1])  # rotate Z-axis

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], s=1)
# ax.set_title("Distribution of rotated Z-axes")
# plt.show()



def evaluate_ik(tfs_ee, sim, robot, threshold, iterations, seed):
    """
    Evaluates whether the end-effector poses are reachable by the robot type using Inverse Kinematics.
    :param tfs_ee: ndarray (n, 4, 4), TCP poses
    :param sim: Simulator instance
    :param robot: RobotBase instance
    :param threshold: int, mm/deg threshold for IK
    :param iterations: int, number of trials for IK
    :param seed: int, random seed used for IK
    :return: bool ndarray (n,), true if reachable, false if not
    """
    reachable_by_ik = np.zeros(len(tfs_ee), dtype=bool)

    for i in tqdm(range(len(tfs_ee))):
        pos, quat = sim.tf_to_pos_quat(tfs_ee[i])
        ik_sln = robot.inverse_kinematics(pos, quat, threshold=threshold, trials=iterations, seed=seed)
        reachable_by_ik[i] = ik_sln is not None

        # if ik_sln is not None:
        #     input("Press Enter to continue...")

    return reachable_by_ik

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

def fibonacci_rotations(samples=20):
    """ 
    Generates evenly distributed orientations. 
    :param samples: number of orientations for each point
    :return: vector of rotations and rotation matrices
    """
    # Generate 20 evenly distributed points on the unit sphere
    samples = 20
    points = fibonacci_sphere(samples)

    # The original vector is the Z-axis (0, 0, 1)
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

    return rots, rot_matrices

def main(args):
    robot_type = args.robot_type
    degrees = args.degrees
    # num_samples = args.num_samples
    threshold = args.threshold
    iterations = args.iterations
    seed = args.seed
    z_values = args.z_values  # Get the constant z value

    robot_name = robot_type
    if robot_type == 'franka':
        robot_name += str(degrees)
    # data_dir = os.path.join('data', f'eval_poses_{robot_name}_n{num_samples}_t{threshold}_i{iterations}')
    # pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    sim, robot = get_sim_and_robot(robot_type, degrees)
    z_max = robot.range_z
    radius = robot.range_radius

    print("Robot name: ", robot_name)
    print("Robot radius: ", radius)
    print("Robot altitude: ", z_max)

    x_min = -radius
    x_max = radius
    y_min = -radius
    y_max = radius
    print("Value ranges: [",x_min,",",x_max,"] and [",y_min,",",y_max,"]")

    voxel_distance = 0.2    # IMPORTANT TO BE CHANGED (SAME AS USED IN GRID CREATION)
    max_error = math.sqrt(3)*voxel_distance*1000   # Position error in mm
    grid_resolution = 1/voxel_distance
    print(f"Grid resolution: {grid_resolution:.2f}")
    print(f"Max error: {max_error:.2f} mm")

    grids_dir = os.path.join('data','grids')

    # Get a list of .npy files in the directory
    npy_files = [f for f in os.listdir(grids_dir) if f.endswith('.npy')]

    # Check if there are any .npy files in the directory
    if not npy_files:
        print("No .npy files found in the directory.")
    else:
        # Display the list of .npy files to the user
        print("Available .npy files:")
        for i, filename in enumerate(npy_files, start=1):
            print(f"{i}. {filename}")
        
        # Ask the user to choose a file
        choice = input("Enter the number of the file you want to load: ")
        
        # Ensure the input is a valid number and within the available range
        try:
            choice = int(choice)
            if 1 <= choice <= len(npy_files):
                selected_file = npy_files[choice - 1]
                grid_filename = os.path.join(grids_dir, selected_file)
                grid = np.load(grid_filename, allow_pickle=True)
                print(f"Loaded {grid_filename}")
            else:
                print("Invalid choice. No file loaded.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Calculate the grid size based on the actual range and grid resolution
    grid_size_x = int(np.ceil((x_max - x_min) * grid_resolution))  # Grid size in x-direction
    grid_size_y = int(np.ceil((y_max - y_min) * grid_resolution))  # Grid size in y-direction
    print("Grid size:", grid_size_x, " x ", grid_size_y)

    rots, rot_matrices = fibonacci_rotations(samples=20)

    num_samples = grid.shape[0]
    print("Number of samples: ",num_samples)

    input("Press Enter to start the Reachability Map computation...")

    # Store reachability maps for each z-level in a dictionary
    reachability_map = {}

    # Iterate through the z-values
    for z_value in z_values:
        print(f"Evaluating for z-value: {z_value}")

        # Get poses for the current z value
        poses = get_poses_from_positions_and_orientations(grid, z_value, rot_matrices)
        print("Number of total poses: ",poses.shape[0])

        # Create a map of reachable positions in the x-y plane
        reachability_slice = np.zeros((grid_size_x, grid_size_y))

        # Evaluate reachability using IK
        reachable_by_ik = evaluate_ik(poses, sim, robot, threshold, iterations, seed)
    
        # Map the x-y positions of the poses to the grid
        for i in range(len(poses)):
            # Scale x and y positions to the grid
            x_idx = int((poses[i, 0, 3] - x_min) * grid_resolution)  # Scale to grid resolution
            y_idx = int((poses[i, 1, 3] - y_min) * grid_resolution)  # Scale to grid resolution

            # Ensure indices are within bounds
            x_idx = np.clip(x_idx, 0, grid_size_x - 1)
            y_idx = np.clip(y_idx, 0, grid_size_y - 1)    
            
            # Set the corresponding grid position to True if reachable
            if reachable_by_ik[i]:
                reachability_slice[x_idx, y_idx] += 1 

            # Save the reachability map for the current z-value
            reachability_map[z_value] = reachability_slice

        # Plotting the reachability map
        plt.figure()
        plt.imshow(reachability_slice, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Reachability')
        plt.title('Reachability Map')
        plt.xlabel('X Index')
        plt.ylabel('Y Index')
        plt.show(block=False)

        print('completed.')
        # print(f'{num_samples} poses sampled and stored to {poses_fn}')
        # print(f'reachability evaluated and stored to {reachable_fn}')
        print(f'{100.0*reachable_by_ik.sum()/poses.shape[0]}% determined reachable.')
        print(f'{100.0*(poses.shape[0]-reachable_by_ik.sum())/poses.shape[0]}% determined not reachable.')
        print(f'{100.0*reachable_by_ik.sum()/num_samples}% determined reachable.')
        print(f'{100.0*(num_samples-reachable_by_ik.sum())/num_samples}% determined not reachable.')

        # Save the reachability slices to a file for future access
        data_dir = os.path.join('data', f'eval_poses_{robot_name}')
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
        reachability_slices_fn = os.path.join(data_dir,f"reachability_slice_{grid_size_x}_{z_value}.npy")
        
        # Check if the file already exists and warn before overwriting
        if os.path.exists(reachability_slices_fn):
            response = input(f"Warning: {reachability_slices_fn} already exists. Do you want to overwrite it? (y/n): ")
            if response.lower() != 'y':
                print("File not overwritten.")
            else:
                np.save(reachability_slices_fn, reachability_map)
                print(f"Reachability maps saved to {reachability_slices_fn}.")
        else:
            np.save(reachability_slices_fn, reachability_map)
            print(f"Reachability maps saved to {reachability_slices_fn}.")

    sim.disconnect()

    # Save the reachability map to a file for future access
    reachability_map_fn = os.path.join(data_dir,f"reachability_map_{grid_size_x}.npy")

    # Check if the file already exists and warn before overwriting
    if os.path.exists(reachability_map_fn):
        response = input(f"Warning: {reachability_map_fn} already exists. Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("File not overwritten.")
        else:
            np.save(reachability_map_fn, reachability_map)
            print(f"Reachability maps saved to {reachability_map_fn}.")
    else:
        np.save(reachability_map_fn, reachability_map)
        print(f"Reachability maps saved to {reachability_map_fn}.")

    # Wait for user interaction before closing all plots
    input("Press Enter to close all plots...")

if __name__ == '__main__':
    main(parse_args())
