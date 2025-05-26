import os
import numpy as np
import burg_toolkit as burg
import matplotlib.pyplot as plt

from rm4d import ReachabilityMap4D
from rm4d.base_pos_grid import BasePosGrid
from rm4d.robots import Simulator, Franka, UR10E

from exp_utils import Timer
from eval_poses import evaluate_ik
from calculate_accuracy import print_confusion_matrix

import copy
from scipy.spatial.transform import Rotation as R
from closed_form_algorithm import closed_form_algorithm

def visualize_rmap_slice(rmap, z_idx, theta_idx):
    """
    Visualiza un slice del mapa de alcanzabilidad en 2D para un z_idx y theta_idx dados.

    :param rmap: instancia de ReachabilityMap4D
    :param z_idx: índice de altura
    :param theta_idx: índice de orientación
    """
    slice_2d = rmap.map[z_idx, theta_idx]
    title = f"Reachability Slice\nZ index: {z_idx} (z={rmap.z_limits[0] + z_idx * rmap.voxel_res:.2f}), " \
            f"Theta index: {theta_idx} (theta ≈ {theta_idx * rmap.theta_res * 180 / np.pi:.1f}°)"
    
    plt.figure(figsize=(6, 5))
    plt.imshow(slice_2d, cmap='hot', origin='lower')
    plt.title(title)
    plt.xlabel('Y index')
    plt.ylabel('X index')
    plt.colorbar(label='Reachable (True/False)')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def compute_coverage_per_pose(rmap, poses_ee_translated):
    """
    Devuelve un array booleano indicando si cada pose es alcanzable según el mapa.
    """
    reachable_by_map = np.zeros(len(poses_ee_translated), dtype=bool)
    for i, g in enumerate(poses_ee_translated):
        try:
            reachable_by_map[i] = rmap.is_reachable(rmap.get_indices_for_ee_pose(g))
        except IndexError:
            reachable_by_map[i] = False
    return reachable_by_map

def main():
    #########################################################
    ### In case the eval_poses_mod_all has been used to create the map
    #########################################################

    # # Loading file
    # path = 'data/eval_poses_ur10e/reachability_map_27_4d.npy'
    # map_dict = np.load(path, allow_pickle=True).item()

    # # Parameters of the map
    # z_vals = sorted(map_dict.keys())  # ordenados para garantizar consistencia en el eje z
    # n_bins_theta = len(map_dict[z_vals[0]])
    # n_bins_xy = list(map_dict[z_vals[0]].values())[0].shape[0]  # asumimos cuadrado

    # # Resolution estimation
    # voxel_res = 0.1
    # xy_limits = [-1.35, 1.35]
    # z_limits = [min(z_vals), max(z_vals)]

    # # Initiating map
    # rmap = ReachabilityMap4D(
    #     xy_limits=xy_limits,
    #     z_limits=z_limits,
    #     voxel_res=voxel_res,
    #     n_bins_theta=n_bins_theta,
    #     no_map=True
    # )

    # # Filling map from dictionary
    # n_bins_z = len(z_vals)
    # rmap.map = np.zeros((n_bins_z, n_bins_theta, n_bins_xy, n_bins_xy), dtype=bool)

    # for z_idx, z_val in enumerate(z_vals):
    #     for theta_idx, xy_slice in map_dict[z_val].items():
    #         rmap.map[z_idx, theta_idx] = xy_slice.astype(bool)

    # # Using the map
    # rmap.print_structure()
    # rmap.show_occupancy_per_dim()

    # # Saving the map
    # rmap.to_file('reachability_map_ur10e_final.npy')

    # visualize_rmap_slice(rmap, z_idx=0, theta_idx=0)
    # visualize_rmap_slice(rmap, z_idx=5, theta_idx=10)
    # visualize_rmap_slice(rmap, z_idx=2, theta_idx=19)

    # for z_idx in range(0, rmap.shape[0], 5):
    #     for theta_idx in [0, rmap.n_bins_theta // 4, rmap.n_bins_theta // 2]:
    #         visualize_rmap_slice(rmap, z_idx, theta_idx)

    # # rmap = ReachabilityMap4D(
    # #     xy_limits=[-1.3, 1.3],
    # #     z_limits=[0, 1.3],
    # #     voxel_res=0.1,
    # #     n_bins_theta=36,
    # #     no_map=True
    # # )
    # # rmap.map = my_array  # tu array booleano 4D

    #########################################################
    #########################################################
    #########################################################

    # === CARGAR MAPA DE ALCANZABILIDAD ===
    # map_fn = 'data/rm4d_franka_joint_42/10000000/rmap.npy'
    map_fn = 'data/rm4d_ur10e_joint_42/10000000/rmap.npy'
    rmap = ReachabilityMap4D.from_file(map_fn)
    print(rmap.map.shape)
    print(np.sum(rmap.map))  # Debería ser > 0 si hay celdas alcanzables
    # print(rmap)
    # print(rmap._get_xy_points())
    input('hit enter to proceed')

    poses_ee = None

    # # === OPCIÓN 1: Cargar desde archivo .npy ===
    # poses_path = "poses_ee_ur10e.npy"
    # if os.path.exists(poses_path):
    #     poses_ee = np.load(poses_path)
    #     print(f"[INFO] Cargadas {len(poses_ee)} poses desde '{poses_path}'")

    # === OPCIÓN 2: Definir manualmente algunas poses ===
    if poses_ee is None:
        pose1 = np.eye(4)
        pose1[:3, 3] = [1.7, 1.6, 0.8]  # posición x, y, z
        pose2 = np.eye(4)
        pose2[:3, 3] = [1.7, 2, 0.8]
        pose3 = np.eye(4)
        pose3[:3, 3] = [1.7, 1.6, 0.3]
        pose4 = np.eye(4)
        pose4[:3, 3] = [1.7, 2, 0.3]

        # Ángulos de rotación en grados
        roll = 0  # Rotación alrededor del eje X
        pitch = 90  # Rotación alrededor del eje Y
        yaw = 0  # Rotación alrededor del eje Z

        # Crear rotaciones usando los ángulos de Euler (roll, pitch, yaw)
        rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)

        # Obtener la matriz de rotación 3x3
        rotation_matrix = rotation.as_matrix()

        # Aplicar la rotación a las matrices de pose
        pose1[:3, :3] = rotation_matrix @ pose1[:3, :3]  # Aplicar rotación a la parte de rotación
        pose2[:3, :3] = rotation_matrix @ pose2[:3, :3]
        pose3[:3, :3] = rotation_matrix @ pose3[:3, :3]
        pose4[:3, :3] = rotation_matrix @ pose4[:3, :3]

        poses_ee = np.stack([pose1, pose2, pose3, pose4])
        print(f"[INFO] Usando {len(poses_ee)} poses definidas manualmente")

    # # === OPCIÓN 3: Generar aleatoriamente ===
    # if poses_ee is None:
    #     from eval_poses import get_evaluation_poses
    #     poses_ee = get_evaluation_poses(max_radius=1.3, max_z=1.5, n_samples=20, seed=27)
    #     print(f"[INFO] Generadas {len(poses_ee)} poses aleatorias")

    # === SIMULADOR PARA VISUALIZACIÓN ===
    sim = Simulator(with_gui=True)

    # === CREAR GRID DE POSICIONES BASE (área XY de -2 a 2 metros) ===
    timer = Timer()
    timer.start('inverse mapping')

    x_limits = [-4, 4]
    y_limits = [-4, 4]
    n_bins_x = int((x_limits[1]-x_limits[0])/0.1)   # resolution must be equal or multiple of map resolution (0.1)
    n_bins_y = int((y_limits[1]-y_limits[0])/0.1)   # 
    print("x_bins, y_bins: ", (n_bins_x, n_bins_y))

    grids = []
    for i, pose in enumerate(poses_ee):
        try:
            indv_grid = BasePosGrid(
                x_limits=x_limits,
                y_limits=y_limits,
                n_bins_x=n_bins_x,
                n_bins_y=n_bins_y
            )
            base_pos = rmap.get_base_positions(pose)
            # print('base_pos shape',np.shape(base_pos))
            indv_grid.add_base_positions(base_pos)
            grids.append(indv_grid)
            indv_grid.show_as_img()
            print(f"[DEBUG] Pose #{i} genera {len(base_pos)} posiciones base")
        except IndexError as e:
            print(f"[WARN] Pose #{i} fuera del mapa de alcanzabilidad: {e}")

    print('grids_shape',np.shape(grids))
    
    input('Individual grids already computed. Press Enter to continue.')

    grids_union = copy.deepcopy(grids)
    grids_intersect = copy.deepcopy(grids)

    for i in range(1, len(grids)):
        grids_union[0].union(grids_union[i])
        grids_intersect[0].intersect(grids_intersect[i])

    grids_intersect[0].show_as_img("grids_intersect")
    grids_union[0].show_as_img("grids_union")
    input('Press Enter to continue')

    timer.stop('inverse mapping')
    timer.print()

    tf_base = np.eye(4)
    p, q = sim.tf_to_pos_quat(tf_base)
    sim.add_frame(p, q)
    input('Origin')

    # === DEFINICION DE OBSTACULOS ===
    # Ejemplo de mapa con obstáculos (1: libre, 0: ocupado)
    occupancy_map = np.ones((n_bins_x, n_bins_y), dtype=np.uint8)
    obs_coord_x = [1.7, 3]
    obs_coord_y = [0, 3.5]
    obs_ctg = n_bins_x/((x_limits[1]-x_limits[0]))
    obs_ctg_cst = -n_bins_x*0.5
    # occupancy_map[int(obs_coord_x[0]*obs_coord_to_grid):int(obs_coord_x[1]*obs_coord_to_grid), int(obs_coord_y[0]*obs_coord_to_grid):int(obs_coord_y[1]*obs_coord_to_grid)] = 0  # Simulamos un obstáculo rectangular en el centro
    occupancy_map[int(obs_coord_x[0]*obs_ctg+obs_ctg_cst):int(obs_coord_x[1]*obs_ctg+obs_ctg_cst), int(obs_coord_y[0]*obs_ctg+obs_ctg_cst):int(obs_coord_y[1]*obs_ctg+obs_ctg_cst)] = 0  # Simulamos un obstáculo rectangular en el centro    
    # Asegúrate de que ambos arrays son del mismo tamaño
    assert occupancy_map.shape == grids_intersect[0].grid.shape, "Dimensiones incompatibles"
    # Opción 1: in-place usando indexación booleana
    grids_intersect[0].grid[occupancy_map == 0] = 0
    grids_union[0].grid[occupancy_map == 0] = 0
    grids_intersect[0].show_as_img("grids_intersect")
    grids_union[0].show_as_img("grids_union")
    if not np.any(grids_intersect[0].grid):
        print("[ERROR] No hay posiciones base válidas tras aplicar la máscara de obstáculos (intersección)")
    if not np.any(grids_union[0].grid):
        print("[ERROR] No hay posiciones base válidas tras aplicar la máscara de obstáculos (unión)")

    # === ENCONTRAR MEJOR POSICIÓN BASE ===
    print('Buscando mejor posicion ...')
    x_intersect, y_intersect = grids_intersect[0].get_best_pos()
    # x_intersect = 1.2
    # y_intersect = 1.8
    print('x_intersect, y_intersect: ', x_intersect, y_intersect)
    tf_base = np.eye(4)
    tf_base[:2, 3] = x_intersect, y_intersect
    p, q = sim.tf_to_pos_quat(tf_base)
    sim.add_frame(p, q)

    # x_union, y_union = grids_union[0].get_best_pos()
    # print('x_union, y_union: ', x_union, y_union)
    # tf_base = np.eye(4)
    # tf_base[:2, 3] = x_union, y_union
    # p, q = sim.tf_to_pos_quat(tf_base)
    # sim.add_frame(p, q)

    # === EVALUAR CON IK ===
    sim_direct = Simulator(with_gui=False)
    # robot = Franka(sim_direct)
    robot = UR10E(sim_direct)

    # === COMPARAR ALCANZABILIDAD DE LAS POSES DESDE LA BASE ELEGIDA ===
    poses_ee_translated = poses_ee.copy()
    poses_ee_translated[:, 0, 3] -= x_intersect
    poses_ee_translated[:, 1, 3] -= y_intersect
    timer.start('forward mapping')
    reachable_by_map = compute_coverage_per_pose(rmap, poses_ee_translated)
    timer.stop('forward mapping')

    # print(f"\n[RESULTADOS - {aggregation_mode.upper()}]")
    print(f"Base óptima encontrada en: x = {x_intersect:.3f}, y = {y_intersect:.3f}")
    print(f"Poses alcanzables según mapa desde esta base: {np.sum(reachable_by_map)} de {len(poses_ee)}")
    print(f"Porcentaje de cobertura: {100 * np.mean(reachable_by_map):.2f}%")

    reachable_by_ik = evaluate_ik(poses_ee_translated, sim_direct, robot, threshold=25, iterations=100, seed=0) 
    print_confusion_matrix(reachable_by_ik, reachable_by_map)
    timer.print()

    input()

    # === VISUALIZACIÓN ===
    # robot_vis = UR10E(sim)
    # grids_union[0].visualize_in_sim(sim)
    grids_intersect[0].visualize_in_sim(sim)
    for i, tf in enumerate(poses_ee):
        pos, quat = sim.tf_to_pos_quat(tf)
        # Añadir frame para mostrar orientación
        sim.add_frame(pos, quat)
        
        # Añadir esfera para marcar la posición
        sim.add_sphere(pos=pos, radius=0.03, color=[0.0, 0.5, 1.0])  # azul suave

        print(f"[INFO] Pose {i}: Posición {pos}, Quaternion {quat}")

    input("[INFO] Presiona Enter para continuar...")

    robot_vis = UR10E(sim, base_pos=[x_intersect, y_intersect, 0], base_orn=[0,0,1,0])  # Rotation of 180º of the robot base.
    home_position = np.array([0.0, -1.2, -2.3, -1.2, 1.57, 0.0])
    # Initial robot state
    for i in range(len(poses_ee)):
        modified_pose = poses_ee_translated[i, :, :].copy()
        q_current = closed_form_algorithm(modified_pose, np.array(home_position), type=0)
        robot_vis.reset_joint_pos(q_current)
        input(f"Step {i}: q = {np.round(q_current, 4)}")
        
    input("[INFO] Presiona Enter para salir...")  

if __name__ == '__main__':
    main()