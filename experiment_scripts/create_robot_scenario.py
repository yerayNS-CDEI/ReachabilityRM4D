import os
import numpy as np

import burg_toolkit as burg

import pybullet as p  # Import URDF handling library


lib_fn = 'assets/object_library/object_library.yaml'
scene_dir = 'assets/scene02/'
scene_fn = os.path.join(scene_dir, 'scene.yaml')

# Path to the URDF file for your robot
robot_urdf_fn = 'assets/robots/ur5e_2f85.urdf'

def load_robot_from_urdf(robot_urdf_fn):
    # Load the robot using pybullet to extract the URDF meshes
    p.connect(p.DIRECT)  # Use DIRECT mode to load the URDF without the GUI
    
    # Load the URDF and get the robot ID
    robot_id = p.loadURDF(robot_urdf_fn, basePosition=[1.0, 1.0, 0.0])  # Adjust as needed
    
    # Extract the number of joints in the robot
    num_joints = p.getNumJoints(robot_id)
    print(f"Number of joints in the robot: {num_joints}")
    
    robot_links = []
    
    for joint_index in range(num_joints):
        # Get the visual shape information for each link (mesh)
        link_info = p.getVisualShapeData(robot_id, joint_index)
        
        # Extract the geometry (assuming the URDF uses mesh files)
        for link in link_info:
            mesh_file = link[5]  # The mesh file (if available)
            if mesh_file:
                print(f"Link {joint_index} mesh: {mesh_file}")
                
                # Load the mesh for this link into burg_toolkit as an object instance
                # Convert the mesh into a burg object instance here (using some placeholder or your own logic)
                mesh_object = burg.ObjectInstance.from_mesh(mesh_file)
                robot_links.append(mesh_object)
    
    return robot_id, robot_links


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

    # Load the robot and extract meshes (in this case, just using pybullet to load and extract)
    robot_id, robot_links = load_robot_from_urdf(robot_urdf_fn)
    # Create placeholder poses (adjust based on your desired neutral pose)
    robot_pose = np.eye(4)
    robot_pose[:2, 3] = 1.0, 1.0  # Neutral pose for the robot

    # Create a scene with the robot links and adjust the robot pose
    scene_objects = [robot_link for robot_link in robot_links]
    
    # objects = [
    #     burg.ObjectInstance(box, box_pose),
    #     burg.ObjectInstance(banana, banana_pose),
    #     burg.ObjectInstance(screwdriver, screwdriver_pose),
    #     burg.ObjectInstance(ball, ball_pose),
    #     burg.ObjectInstance(mesh, robot_pose) for mesh in robot_meshes
    # ]

    scene_objects.append(burg.ObjectInstance(box, box_pose))

    scene = burg.Scene(area, scene_objects)
    burg.visualization.show_geometries([scene])
    scene.to_yaml(scene_fn, lib)

if __name__ == '__main__':
    create_scene()
