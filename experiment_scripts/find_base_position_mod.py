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


# def load_scene_in_sim(sim: Simulator, scene):
#     color_map = plt.get_cmap('tab20')
#     for color_idx, object_instance in enumerate(scene.objects):
#         if object_instance.object_type.urdf_fn is None:
#             raise ValueError(f'object instance of type {object_instance.object_type.identifier} has no urdf_fn.')
#         if not os.path.exists(object_instance.object_type.urdf_fn):
#             raise ValueError(f'could not find urdf file for object type {object_instance.object_type.identifier}.' +
#                              f'expected it at {object_instance.object_type.urdf_fn}.')

#         # pybullet uses center of mass as reference for the transforms in BasePositionAndOrientation
#         # except in loadURDF - i couldn't figure out which reference system is used in loadURDF
#         # because just putting the pose of the instance (i.e. the mesh's frame?) is not (always) working
#         # workaround:
#         #   all our visual/collision models have the same orientation, i.e. it is only the offset to COM
#         #   add obj w/o pose, get COM, compute the transform burg2py manually and resetBasePositionAndOrientation
#         object_id = sim.bullet_client.loadURDF(object_instance.object_type.urdf_fn)
#         if object_id < 0:
#             raise ValueError(f'could not add object {object_instance.object_type.identifier}. returned id is negative.')

#         com = np.array(sim.bullet_client.getDynamicsInfo(object_id, -1)[3])
#         tf_burg2py = np.eye(4)
#         tf_burg2py[0:3, 3] = com
#         start_pose = object_instance.pose @ tf_burg2py
#         pos, quat = sim.tf_to_pos_quat(start_pose)
#         sim.bullet_client.resetBasePositionAndOrientation(object_id, pos, quat)

#         if sim.verbose:
#             sim.bullet_client.changeVisualShape(object_id, -1, rgbaColor=color_map(0))


# def load_grasps():
#     # BURG toolkit uses different convention for TCP frame than Franka Panda for its gripper, so we need to transform
#     tf_grasp2franka = np.asarray([
#         0, 1, 0, 0,
#         1, 0, 0, 0,
#         0, 0, -1, 0,
#         0, 0, 0, 1
#     ]).reshape(4, 4)

#     grasps = {}
#     for f in os.listdir(scene_dir):
#         if not f.startswith('grasps'):
#             continue

#         grasp_file = os.path.join(scene_dir, f)
#         g = np.load(grasp_file, allow_pickle=True)
#         g = g @ tf_grasp2franka
#         grasps[f] = g

#     return grasps


# def load_scene():
#     scene_fn = os.path.join(scene_dir, 'scene.yaml')
#     scene, lib, _ = burg.Scene.from_yaml(scene_fn)

#     return scene


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


# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial.transform import Rotation

# def visualize_fibonacci_orientations(n_orientations=20):
#     """
#     Visualiza los ejes Z (r_z) de las orientaciones generadas con fibonacci_rotations.

#     :param n_orientations: número de orientaciones a generar
#     """
#     def fibonacci_sphere(samples):
#         phi = np.pi * (3. - np.sqrt(5.))
#         y = np.linspace(1 - 1/samples, -1 + 1/samples, samples)
#         radius = np.sqrt(1 - y**2)
#         theta = phi * np.arange(samples)
#         x = np.cos(theta) * radius
#         z = np.sin(theta) * radius
#         return np.stack([x, y, z], axis=1)

#     vectors = fibonacci_sphere(n_orientations)

#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], s=50, c='r', label='r_z directions')
#     ax.quiver(np.zeros_like(vectors[:, 0]), np.zeros_like(vectors[:, 1]), np.zeros_like(vectors[:, 2]),
#               vectors[:, 0], vectors[:, 1], vectors[:, 2], length=1.0, normalize=True, color='blue', arrow_length_ratio=0.05)

#     ax.set_title(f'Distribución de ejes Z del TCP ({n_orientations} orientaciones)')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_box_aspect([1,1,1])
#     plt.tight_layout()
#     plt.legend()
#     plt.show()

# visualize_fibonacci_orientations(n_orientations=20)

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

def visualize_base_grid_heatmap(base_grid, title='Base Grid Heatmap'):
    """
    Visualiza un heatmap 2D del grid base con el número de poses alcanzables por celda.
    """
    heatmap = base_grid.grid.astype(int)
    plt.figure(figsize=(7, 6))
    plt.imshow(heatmap, cmap='viridis', origin='lower', extent=[
        base_grid.x_limits[0], base_grid.x_limits[1],
        base_grid.y_limits[0], base_grid.y_limits[1]
    ])
    plt.colorbar(label='Número de poses alcanzables')
    plt.title(title)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.tight_layout()
    plt.show(block=False)

def visualize_array_heatmap(array_grid, x_limits, y_limits, title='Base Grid Heatmap'):
    """
    Visualiza un heatmap 2D de un array ya acumulado (por ejemplo, int[][]).
    """
    plt.figure(figsize=(7, 6))
    plt.imshow(
        array_grid, cmap='plasma', origin='lower',
        extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]]
    )
    plt.colorbar(label='Número de poses alcanzables')
    plt.title(title)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.tight_layout()
    plt.show(block=False)

def visualize_array_heatmap_with_max(array_grid, x_limits, y_limits, title='Base Grid Heatmap'):
    """
    Visualiza un heatmap 2D con una marca en la celda de máxima cobertura.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        array_grid, cmap='plasma', origin='lower',
        extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]]
    )
    plt.colorbar(im, ax=ax, label='Número de poses alcanzables')
    ax.set_title(title)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    if np.any(array_grid > 0):
        max_idx = np.unravel_index(np.argmax(array_grid), array_grid.shape)
        x_bins = np.linspace(x_limits[0], x_limits[1], array_grid.shape[1])
        y_bins = np.linspace(y_limits[0], y_limits[1], array_grid.shape[0])
        max_x = x_bins[max_idx[1]]
        max_y = y_bins[max_idx[0]]
        ax.plot(max_x, max_y, 'ro', markersize=8, label='Máximo')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='white', bbox=dict(facecolor='red', alpha=0.5))

    plt.tight_layout()
    plt.show(block=False)


def visualize_in_matplotlib(grid: BasePosGrid, title='Base Grid Heatmap', mark_best=True, save_path=None):
    """
    Visualiza el grid base como una imagen usando Matplotlib.

    :param grid: instancia de BasePosGrid
    :param title: título del gráfico
    :param mark_best: si True, marca la celda con mayor valor
    :param save_path: si no es None, guarda la imagen en esta ruta
    """
    data = grid.grid.astype(int)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        data, cmap='plasma', origin='lower',
        extent=[grid.x_limits[0], grid.x_limits[1], grid.y_limits[0], grid.y_limits[1]]
    )
    plt.colorbar(im, ax=ax, label='Número de poses alcanzables')
    ax.set_title(title)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    if mark_best and np.any(data > 0):
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        x_bins = np.linspace(grid.x_limits[0], grid.x_limits[1], data.shape[1])
        y_bins = np.linspace(grid.y_limits[0], grid.y_limits[1], data.shape[0])
        max_x = x_bins[max_idx[1]]
        max_y = y_bins[max_idx[0]]
        ax.plot(max_x, max_y, 'ro', markersize=8, label='Máximo')
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Imagen guardada en: {save_path}")
    else:
        plt.show(block=False)


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

    # Modo de agregación: 'intersect' = intersección estricta, 'union' = sumar todo
    aggregation_mode = 'intersect'  # o 'union'

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
        pose1[:3, 3] = [0.2, 0.1, 1.0]  # posición x, y, z
        pose2 = np.eye(4)
        pose2[:3, 3] = [0.5, -0.2, 1.2]
        pose3 = np.eye(4)
        pose3[:3, 3] = [0.5, -0.2, 0.4]
        pose4 = np.eye(4)
        pose4[:3, 3] = [1, -0.4, 0.8]

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

    x_limits = [-2.0, 2.0]
    y_limits = [-2.0, 2.0]
    n_bins_x = 80
    n_bins_y = 80

    grids = []
    individual_grids = []  # Para guardar los grids individuales

    base_grid = BasePosGrid(
        x_limits=x_limits,
        y_limits=y_limits,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y
    )

    for i, pose in enumerate(poses_ee):
        try:
            base_pos = rmap.get_base_positions(pose)
            grid = BasePosGrid(
                x_limits=x_limits,
                y_limits=y_limits,
                n_bins_x=n_bins_x,
                n_bins_y=n_bins_y
            )
            grid.add_base_positions(base_pos)
            individual_grids.append((i, grid))
            base_grid.add_base_positions(base_pos)  # esto hace la unión de todas
            print(f"[DEBUG] Pose #{i} genera {len(base_pos)} posiciones base")
        except IndexError as e:
            print(f"[WARN] Pose #{i} fuera del mapa de alcanzabilidad: {e}")
        # grids.append(base_grid)
        # fig, ax = plt.subplots(figsize=(7, 6))
        # im = ax.imshow(
        #     individual_grids(i), cmap='plasma', origin='lower',
        #     extent=[grid.x_limits[0], grid.x_limits[1], grid.y_limits[0], grid.y_limits[1]]
        # )
        # plt.colorbar(im, ax=ax, label='Número de poses alcanzables')
        # ax.set_title('Hetmap en 2D')
        # ax.set_xlabel('x [m]')
        # ax.set_ylabel('y [m]')
    for idx, g in individual_grids:
        visualize_in_matplotlib(g, title=f"Heatmap individual para pose {idx}")
    input()
    
    print('grids: ',grids[0])
    print('individual_grids: ',individual_grids[0][1])
    visualize_in_matplotlib(base_grid, title="Heatmap en 2D")
    input('Matplotlib listo')
    grids[0].visualize_in_sim(sim)
    input('Simulation listo')

    # if aggregation_mode == 'intersect':
    #     grids = []
    #     for i, pose in enumerate(poses_ee):
    #         try:
    #             base_pos = rmap.get_base_positions(pose)
    #             grid = BasePosGrid(
    #                 x_limits=x_limits,
    #                 y_limits=y_limits,
    #                 n_bins_x=n_bins_x,
    #                 n_bins_y=n_bins_y
    #             )
    #             grid.add_base_positions(base_pos)
    #             grids.append(grid)
    #         except IndexError as e:
    #             print(f"[WARN] Pose #{i} fuera del mapa de alcanzabilidad: {e}")

    #     # Intersectar todos los grids
    #     if grids:
    #         base_grid = grids[0]
    #         for i in range(1, len(grids)):
    #             base_grid.intersect(grids[i])
    #     else:
    #         print("[ERROR] No se generó ninguna grilla válida.")
    #         return
        
    # else:  # 'union' o cualquier otro modo flexible
    #     for i, pose in enumerate(poses_ee):
    #         try:
    #             base_pos = rmap.get_base_positions(pose)
    #             base_grid.add_base_positions(base_pos)
    #         except IndexError as e:
    #             print(f"[WARN] Pose #{i} fuera del mapa de alcanzabilidad: {e}")

    timer.stop('inverse mapping')
    timer.print()

    # === VISUALIZACIÓN DE LOS HEATMAPS ===
    # Visualización individual por pose
    print(individual_grids)
    for idx, g in individual_grids:
        acc = g.grid.astype(int)
        print(acc)
        visualize_array_heatmap_with_max(acc, x_limits, y_limits, title=f'Heatmap de Base para Pose #{idx}')

    # # Visualización combinada (cuántas poses son alcanzables desde cada celda)
    # accumulated = accumulate_grids([g for (_, g) in individual_grids])
    # if accumulated is not None:
    #     visualize_array_heatmap_with_max(accumulated, x_limits, y_limits, title='Heatmap combinado: número de poses alcanzables')
    # else:
    #     print("[WARN] No hay grillas válidas para mostrar heatmap combinado.")
    # input()

    # === ENCONTRAR MEJOR POSICIÓN BASE ===
    x, y = base_grid.get_best_pos()
    tf_base = np.eye(4)
    tf_base[:2, 3] = x, y
    p, q = sim.tf_to_pos_quat(tf_base)
    sim.add_frame(p, q)

    # === EVALUAR CON IK ===
    sim_direct = Simulator(with_gui=False)
    # robot = Franka(sim_direct)
    robot = UR10E(sim_direct)

    # === COMPARAR ALCANZABILIDAD DE LAS POSES DESDE LA BASE ELEGIDA ===
    poses_ee_translated = poses_ee.copy()
    poses_ee_translated[:, 0, 3] -= x
    poses_ee_translated[:, 1, 3] -= y
    timer.start('forward mapping')
    reachable_by_map = compute_coverage_per_pose(rmap, poses_ee_translated)
    timer.stop('forward mapping')

    print(f"\n[RESULTADOS - {aggregation_mode.upper()}]")
    print(f"Base óptima encontrada en: x = {x:.3f}, y = {y:.3f}")
    print(f"Poses alcanzables según mapa desde esta base: {np.sum(reachable_by_map)} de {len(poses_ee)}")
    print(f"Porcentaje de cobertura: {100 * np.mean(reachable_by_map):.2f}%")

    reachable_by_ik = evaluate_ik(poses_ee_translated, sim_direct, robot, threshold=25, iterations=100, seed=0) 

    print_confusion_matrix(reachable_by_ik, reachable_by_map)
    timer.print()

    # === VISUALIZACIÓN ===
    # robot_vis = Franka(sim)
    robot_vis = UR10E(sim)
    base_grid.visualize_in_sim(sim)
    for i, tf in enumerate(poses_ee_translated):
        pos, quat = sim.tf_to_pos_quat(tf)
        
        # Añadir frame para mostrar orientación
        sim.add_frame(pos, quat)
        
        # Añadir esfera para marcar la posición
        sim.add_sphere(pos=pos, radius=0.03, color=[0.0, 0.5, 1.0])  # azul suave

        print(f"[INFO] Pose {i}: Posición {pos}, Quaternion {quat}")

    input("[INFO] Presiona Enter para continuar...")

    # robot_vis = Franka(sim, base_pos=[x, y, 0])
    robot_vis = UR10E(sim, base_pos=[x, y, 0])
    for i, tf in enumerate(poses_ee_translated):
        pos, quat = sim.tf_to_pos_quat(tf)
        ik_sol = robot_vis.inverse_kinematics(pos, quat, threshold=25, trials=100, seed=0)
        if ik_sol is not None:
            robot_vis.reset_joint_pos(ik_sol)
            print(f"[INFO] Pose {i} alcanzable. Presiona Enter para continuar...")
            input()
        else:
            print(f"[INFO] Pose {i} no alcanzable. Presiona Enter para continuar...")
            input()

    input("[INFO] Presiona Enter para salir...")

    # rmap.show_occupancy_per_dim()
    

if __name__ == '__main__':
    main()