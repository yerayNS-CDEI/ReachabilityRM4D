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
    sim = Simulator(with_gui=True)
    if robot_type == 'franka':
        urdf_fn = franka_versions[degrees]
        robot = robot_types[robot_type](sim, urdf_fn=urdf_fn)
    else:
        robot = robot_types[robot_type](sim, base_pos=[0, 0, 0.8])
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


def main(args):
    robot_type = args.robot_type
    degrees = args.degrees
    num_samples = args.num_samples
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

    # Determine the range of x and y positions
    # x_min, x_max = np.min(poses[:, 0, 3]), np.max(poses[:, 0, 3])
    # y_min, y_max = np.min(poses[:, 1, 3]), np.max(poses[:, 1, 3])
    x_min = -radius
    x_max = radius
    y_min = -radius
    y_max = radius    
    print("Value ranges: [",x_min,",",x_max,"] and [",y_min,",",y_max,"]")

    # Set a desired grid resolution (number of grid cells per unit distance)
    max_error_distance = 2.886 # Distance from voxel to voxel (maximum distnace in the diagonal to obtain a 5mm error - KPI)
    grid_resolution = 1000/max_error_distance  # This is the number of cells per unit distance (you can adjust this as needed)

    # Calculate the grid size based on the actual range and grid resolution
    grid_size_x = int(np.ceil((x_max - x_min) * grid_resolution))  # Grid size in x-direction
    grid_size_y = int(np.ceil((y_max - y_min) * grid_resolution))  # Grid size in y-direction
    print("Grid size:", grid_size_x, " x ", grid_size_y)

    input("Press Enter to start the Reachability Map computation...")

    # Store reachability maps for each z-level in a dictionary
    reachability_maps = {}

    # Iterate through the z-values
    for z_value in z_values:
        print(f"Evaluating for z-value: {z_value}")

        # Get poses for the current z value
        poses = get_evaluation_poses(radius, z_value, num_samples, seed)
        # poses_fn = os.path.join(data_dir, 'poses.npy')
        # np.save(poses_fn, poses)

        # Print the sampled poses
        # print("Selected poses (in x-y, constant z-plane):")
        # print(poses[:, 0, 3], poses[:, 1, 3], f"z={z_value}")  # x, y, and constant z

        # Create a map of reachable positions in the x-y plane
        reachability_map = np.zeros((grid_size_x, grid_size_y))

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
                reachability_map[x_idx, y_idx] += 1 

        # Save the reachability map for the current z-value
            reachability_maps[z_value] = reachability_map

        # Plotting the reachability map
        plt.figure()
        plt.imshow(reachability_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Reachability')
        plt.title('Reachability Map')
        plt.xlabel('X Index')
        plt.ylabel('Y Index')
        plt.show(block=False)

        print('completed.')
        # print(f'{num_samples} poses sampled and stored to {poses_fn}')
        # print(f'reachability evaluated and stored to {reachable_fn}')
        print(f'{100.0*reachable_by_ik.sum()/num_samples}% determined reachable.')
        print(f'{100.0*(num_samples-reachable_by_ik.sum())/num_samples}% determined not reachable.')


    # Save the reachability maps to a file for future access
    reachability_maps_fn = "reachability_maps.npy"
    np.save(reachability_maps_fn, reachability_maps)
    print(f"Reachability maps saved to {reachability_maps_fn}")

    # reachable_by_ik = evaluate_ik(poses, sim, robot, threshold, iterations, seed)
    sim.disconnect()
    # reachable_fn = os.path.join(data_dir, 'reachable_by_ik.npy')
    # np.save(reachable_fn, reachable_by_ik)

    # Wait for user interaction before closing all plots
    input("Press Enter to close all plots...")

if __name__ == '__main__':
    main(parse_args())

# np.load('reachability_maps.npy', allow_pickle=True)