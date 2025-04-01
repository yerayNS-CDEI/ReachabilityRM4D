import os
import pathlib
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import math

from rm4d.robots import Simulator
from rm4d.robots.assets import FRANKA_150_URDF, FRANKA_160_URDF, FRANKA_URDF
from exp_utils import robot_types, franka_versions

import inspect
# from .assets import UR5E_URDF
# from rm4d.robots import Franka, UR5E

# print("CLASS DEFINITION FILE:", inspect.getfile(UR5E_URDF))
# print("CLASS DEFINITION FILE:", inspect.getfile(UR5E))

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
        robot = robot_types[robot_type](sim)
    return sim, robot


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


def get_evaluation_pose(max_radius, z_value, n_samples, seed):
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

    # Generate 20 evenly distributed points on the unit sphere
    samples = n_samples
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
    
    # first, uniformly sample an orientation (as per scipy documentation)
    # tfs_ee[:, :3, :3] = Rotation.random(num=n_samples, random_state=rng).as_matrix()
    tfs_ee[:, :3, :3] = rot_matrices

    # now, we want to uniformly sample the xy position from within a circle, see:
    # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/50746409#50746409
    # radii = max_radius * np.sqrt(rng.uniform(0, 1, n_samples))
    # angles = 2 * np.pi * rng.uniform(0, 1, n_samples)
    # x_pos = radii * np.cos(angles)
    # y_pos = radii * np.sin(angles)
    x_pos = np.ones(n_samples)*1.5
    y_pos = np.ones(n_samples)*0.0
    tfs_ee[:, 0, 3] = x_pos
    tfs_ee[:, 1, 3] = y_pos
    # we uniformly sample z, so the 3d position will be uniformly within a cylinder.
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
        # if i%20 == 0:
        #     input("Press Enter to continue...")
        pos, quat = sim.tf_to_pos_quat(tfs_ee[i])

        # ik_sln = robot.inverse_kinematics(pos, quat, threshold=threshold, trials=iterations, seed=seed)
        try:
            ik_sln = robot.inverse_kinematics(pos, quat, threshold=threshold, trials=iterations, seed=seed)
        except Exception as e:
            print("IK call failed:", e)
            continue

        reachable_by_ik[i] = ik_sln is not None
        # if ik_sln is not None:
        #     input("Press Enter to continue...")

        if False:
            input("Press Enter to continue...")

    return reachable_by_ik

def main(args):
    robot_type = args.robot_type
    degrees = args.degrees
    num_samples = args.num_samples
    threshold = args.threshold
    iterations = args.iterations
    seed = args.seed

    robot_name = robot_type
    if robot_type == 'franka':
        robot_name += str(degrees)

    sim, robot = get_sim_and_robot(robot_type, degrees)
    z_max = robot.range_z
    radius = robot.range_radius

    print(f"Robot name: {robot_name}.")
    print(f"Radius: {radius}.")
    num_samples = 1000
    z_value = 0.2
    poses = get_evaluation_pose(radius, z_value, num_samples, seed)

    print("Joint indices:", robot._arm_joint_ids)
    # print("All joints:", robot.joint_names)
    print("TCP:", robot._end_effector_link_id)

    reachable_by_ik = evaluate_ik(poses, sim, robot, threshold, iterations, seed)
    sim.disconnect()
    
    print("Reachability: ", reachable_by_ik)
    reachability = 0
    for i in range(num_samples):
        if reachable_by_ik[i]:
                reachability += 1
    print("Reachable orientations sampled: ", reachability)

    print('Process Completed.')
    print(f'{num_samples} poses sampled')
    print(f'{100.0*reachable_by_ik.sum()/num_samples}% determined reachable.')
    print(f'{100.0*(num_samples-reachable_by_ik.sum())/num_samples}% determined not reachable.')

if __name__ == '__main__':
    main(parse_args())
