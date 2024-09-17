import os
import numpy as np
import burg_toolkit as burg
import matplotlib.pyplot as plt

from rm4d import ReachabilityMap4D
from rm4d.base_pos_grid import BasePosGrid
from rm4d.robots import Simulator, Franka

from exp_utils import Timer
from eval_poses import evaluate_ik
from calculate_accuracy import print_confusion_matrix


map_fn = 'data/rm4d_franka_joint_42/10000000/rmap.npy'
scene_dir = 'assets/scene01/'


def load_scene_in_sim(sim: Simulator, scene):
    color_map = plt.get_cmap('tab20')
    for color_idx, object_instance in enumerate(scene.objects):
        if object_instance.object_type.urdf_fn is None:
            raise ValueError(f'object instance of type {object_instance.object_type.identifier} has no urdf_fn.')
        if not os.path.exists(object_instance.object_type.urdf_fn):
            raise ValueError(f'could not find urdf file for object type {object_instance.object_type.identifier}.' +
                             f'expected it at {object_instance.object_type.urdf_fn}.')

        # pybullet uses center of mass as reference for the transforms in BasePositionAndOrientation
        # except in loadURDF - i couldn't figure out which reference system is used in loadURDF
        # because just putting the pose of the instance (i.e. the mesh's frame?) is not (always) working
        # workaround:
        #   all our visual/collision models have the same orientation, i.e. it is only the offset to COM
        #   add obj w/o pose, get COM, compute the transform burg2py manually and resetBasePositionAndOrientation
        object_id = sim.bullet_client.loadURDF(object_instance.object_type.urdf_fn)
        if object_id < 0:
            raise ValueError(f'could not add object {object_instance.object_type.identifier}. returned id is negative.')

        com = np.array(sim.bullet_client.getDynamicsInfo(object_id, -1)[3])
        tf_burg2py = np.eye(4)
        tf_burg2py[0:3, 3] = com
        start_pose = object_instance.pose @ tf_burg2py
        pos, quat = sim.tf_to_pos_quat(start_pose)
        sim.bullet_client.resetBasePositionAndOrientation(object_id, pos, quat)

        if sim.verbose:
            sim.bullet_client.changeVisualShape(object_id, -1, rgbaColor=color_map(0))


def load_grasps():
    # BURG toolkit uses different convention for TCP frame than Franka Panda for its gripper, so we need to transform
    tf_grasp2franka = np.asarray([
        0, 1, 0, 0,
        1, 0, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1
    ]).reshape(4, 4)

    grasps = {}
    for f in os.listdir(scene_dir):
        if not f.startswith('grasps'):
            continue

        grasp_file = os.path.join(scene_dir, f)
        g = np.load(grasp_file, allow_pickle=True)
        g = g @ tf_grasp2franka
        grasps[f] = g

    return grasps


def load_scene():
    scene_fn = os.path.join(scene_dir, 'scene.yaml')
    scene, lib, _ = burg.Scene.from_yaml(scene_fn)

    return scene


def main():
    rmap = ReachabilityMap4D.from_file(map_fn)
    scene = load_scene()
    grasps = load_grasps()
    sim = Simulator(with_gui=True)
    load_scene_in_sim(sim, scene)
    input('hit enter to proceed')

    grids = []
    timer = Timer()
    timer.start('inverse mapping')
    for key, grasps_per_object in grasps.items():
        base_grid = BasePosGrid(x_limits=[0, scene.ground_area[0]], y_limits=[0, scene.ground_area[1]],
                                n_bins_x=int(scene.ground_area[0]/0.05), n_bins_y=int(scene.ground_area[1]/0.05))

        # this could potentially be vectorized
        for g in grasps_per_object:
            base_grid.add_base_positions(rmap.get_base_positions(g))

        grids.append(base_grid)
        # base_grid.show_as_img()

    for i in range(1, len(grids)):
        grids[0].intersect(grids[i])
    timer.stop('inverse mapping')
    timer.print()

    x, y = grids[0].get_best_pos()
    tf_base = np.eye(4)
    tf_base[:2, 3] = x, y
    p, q = sim.tf_to_pos_quat(tf_base)
    sim.add_frame(p, q)

    sim_direct = Simulator(with_gui=False)
    robot = Franka(sim_direct)

    for key, grasps_per_object in grasps.items():
        grasps_per_object[:, 0, 3] -= x
        grasps_per_object[:, 1, 3] -= y
        reachable_by_ik = evaluate_ik(grasps_per_object, sim_direct, robot, 25, 100, 0)
        reachable_by_map = np.zeros(len(grasps_per_object), dtype=bool)
        timer.start('forward mapping')
        for i, g in enumerate(grasps_per_object):
            reachable_by_map[i] = rmap.is_reachable(rmap.get_indices_for_ee_pose(g))
        timer.stop('forward mapping')
        print(f'****************************')
        print(key)
        print_confusion_matrix(reachable_by_ik, reachable_by_map)

    timer.print()

    grids[0].visualize_in_sim(sim)
    input()



if __name__ == '__main__':
    main()
