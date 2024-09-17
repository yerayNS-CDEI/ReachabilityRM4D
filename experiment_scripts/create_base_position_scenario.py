import os
import numpy as np

import burg_toolkit as burg


lib_fn = 'assets/object_library/object_library.yaml'
scene_dir = 'assets/scene01/'
scene_fn = os.path.join(scene_dir, 'scene.yaml')


def create_scene():
    lib = burg.ObjectLibrary.from_yaml(lib_fn)
    area = (2, 2)

    box = lib['004_sugar_box']
    banana = lib['011_banana']
    screwdriver = lib['044_flat_screwdriver']
    ball = lib['056_tennis_ball']

    # get suitable stable poses
    box_pose = box.stable_poses[5][1]
    banana_pose = banana.stable_poses[0][1]
    screwdriver_pose = screwdriver.stable_poses[0][1]
    ball_pose = ball.stable_poses[0][1]

    # adjust position in the scene
    box_pose[:2, 3] = 0.5, 0.9
    banana_pose[:2, 3] = 0.6, 1.4
    screwdriver_pose[:2, 3] = 0.7, 0.6
    ball_pose[:2, 3] = 1.3, 0.8

    objects = [
        burg.ObjectInstance(box, box_pose),
        burg.ObjectInstance(banana, banana_pose),
        burg.ObjectInstance(screwdriver, screwdriver_pose),
        burg.ObjectInstance(ball, ball_pose)
    ]

    scene = burg.Scene(area, objects)
    burg.visualization.show_geometries([scene])
    scene.to_yaml(scene_fn, lib)


def create_grasps():
    scene, lib, _ = burg.Scene.from_yaml(scene_fn)
    for target_obj in scene.objects:
        save_fn = os.path.join(scene_dir, f'grasps_{target_obj.object_type.identifier}.npy')
        keep = 200  # max number of grasps to keep (random selection)
        gripper_width = 0.08  # suitable for franka panda gripper

        grasp_sampler = burg.sampling.AntipodalGraspSampler(
            n_orientations=7,
            only_grasp_from_above=True,
            no_contact_below_z=None
        )
        print('sampling...')
        grasp_candidates, contacts = grasp_sampler.sample(
            target_obj,
            n=2000,
            max_gripper_width=gripper_width,  # franka panda gripper width
        )
        print(f'sampled {len(grasp_candidates)} grasps.')
        print('checking collisions...')
        gripper = burg.gripper.TwoFingerGripperVisualisation(opening_width=gripper_width)
        collisions = grasp_sampler.check_collisions(
            grasp_candidates,
            scene,
            gripper.mesh,
            with_plane=True
        )
        grasps = grasp_candidates[collisions == 0]
        print(f'{len(grasps)} collision-free grasps remaining.')

        if len(grasps) > keep:
            print(f'sampling {keep} grasps from remaining set.')
            keep_indices = np.random.choice(len(grasps), keep, replace=False)
            grasps = grasps[keep_indices]

        print(f'saving to {save_fn}')
        np.save(save_fn, grasps.poses)
        # burg.visualization.show_grasp_set([target_obj], grasps, gripper)


if __name__ == '__main__':
    create_scene()
    create_grasps()
