# # client = MongoClient("mongodb+srv://yeraynavarro:LzXKhwG6QWadXY4X@ur10e-database-005.l56hsfs.mongodb.net/")
# # db = client["UR10e-database-005"]
# # collection = db["voxels"]

#                 ### METODOS DE MONGODB PARA BASE DE DATOS (Despues de calcular rel_pos)

#                 # collection.find({"coords":str(tuple(rel_pos[0],rel_pos[1], rel_pos[2]))})
                
#                 ##########################################

# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as R


# def load_database(db_path, orientations_path):
#     with open(db_path, 'rb') as f:
#         db = pickle.load(f)
#         print('[INFO] Database loaded.')
#     all_voxels = list(db.keys())
#     min_idx = np.min(all_voxels, axis=0)
#     max_idx = np.max(all_voxels, axis=0)
#     print(f"[DEBUG] DB index range: X:{min_idx[0]}-{max_idx[0]}, Y:{min_idx[1]}-{max_idx[1]}, Z:{min_idx[2]}-{max_idx[2]}")
#     with open(orientations_path, 'rb') as f:
#         orientations = pickle.load(f)
#         print('[INFO] Orientations loaded.')
#     return db, orientations


# def compute_relative_pose(base_pos, ee_pos):
#     return ee_pos[:3] - base_pos


# def find_closest_voxel(point, cart_step, cart_min):
#     return tuple(np.round((point - (cart_min + cart_step / 2)) / cart_step).astype(int))


# def orientation_similarity(o1, o2):
#     z1 = R.from_matrix(o1).apply([0, 0, 1])
#     z2 = R.from_matrix(o2).apply([0, 0, 1])
#     angle = np.arccos(np.clip(np.dot(z1, z2), -1.0, 1.0))
#     return angle


# def find_similar_orientations(required_rot, orientation_list, threshold_rad=np.deg2rad(30)):
#     similar_indices = []
#     for i, R_matrix in enumerate(orientation_list):
#         if orientation_similarity(required_rot, R_matrix) < threshold_rad:
#             similar_indices.append(i)
#     return similar_indices


# def define_global_grid(ee_targets, cart_step=0.05, global_size=4.0):
#     ee_positions = np.array([pos[:2] for pos, _ in ee_targets])
#     center = np.mean(ee_positions, axis=0)
#     half_size = global_size / 2
#     global_min = center - half_size
#     global_max = center + half_size
#     x_vals = np.arange(global_min[0], global_max[0] + cart_step, cart_step)
#     y_vals = np.arange(global_min[1], global_max[1] + cart_step, cart_step)
#     return x_vals, y_vals, global_min, global_max


# def evaluate_base_positions_global(ee_targets, db, orientations,
#                                    cart_step=0.05, area_size=1.6*2,
#                                    global_size=4.0, cart_min=-1.6):
#     x_vals, y_vals, global_min, global_max = define_global_grid(ee_targets, cart_step, global_size)
#     H, W = len(y_vals), len(x_vals)
#     union_map = np.zeros((H, W))
#     intersection_votes = np.zeros((H, W), dtype=int)

#     half_cells = int(area_size / (2 * cart_step))

#     for ee_pos, ee_rot in ee_targets:
#         local_votes = np.zeros((H, W), dtype=int)
#         cx = np.argmin(np.abs(x_vals - ee_pos[0]))
#         cy = np.argmin(np.abs(y_vals - ee_pos[1]))

#         for i in range(cy - half_cells, cy + half_cells):
#             for j in range(cx - half_cells, cx + half_cells):
#                 if 0 <= i < H and 0 <= j < W:
#                     base_pos = np.array([x_vals[j], y_vals[i], 0.0])
#                     rel_pos = compute_relative_pose(base_pos, ee_pos)
#                     voxel_idx = find_closest_voxel(rel_pos, cart_step, cart_min)

#                     # print(f"rel_pos: {rel_pos}, voxel_idx: {voxel_idx}")

#                     if all(v >= 0 for v in voxel_idx) and voxel_idx in db:
#                         similar_orients = find_similar_orientations(ee_rot, orientations)
#                         score = 0
#                         for idx in similar_orients:
#                             configs = db[voxel_idx].get(idx, [])
#                             # if configs:
#                             #     print(f"✓ Voxel {voxel_idx} → orient {idx} → {len(configs)} configs")
#                             score += len(configs)
#                         union_map[i, j] += score
#                         if score > 0:
#                             local_votes[i, j] = 1

#                     # if voxel_idx in db:
#                     #     print("✓ Voxel found!")


#         intersection_votes += local_votes

#     intersection_map = np.where(intersection_votes == len(ee_targets), union_map, 0)
#     return union_map, intersection_map, x_vals, y_vals


# def plot_score_map(score_map, x_vals, y_vals, title='Mapa', ee_targets=None, top_bases=None):
#     extent = [x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]]
#     plt.figure(figsize=(8, 6))
#     plt.imshow(score_map, origin='lower', extent=extent, cmap='viridis')
#     if ee_targets:
#         for i, (pos, _) in enumerate(ee_targets):
#             plt.plot(pos[0], pos[1], 'rx')
#             plt.text(pos[0] + 0.02, pos[1] + 0.02, f'P{i+1}', color='white')
#     if top_bases:
#         for i, (x, y) in enumerate(top_bases):
#             plt.plot(x, y, 'go')
#             plt.text(x + 0.02, y - 0.02, f'B{i+1}', color='lime')
#     plt.colorbar(label='Número total de soluciones')
#     plt.title(title)
#     plt.xlabel('X (m)')
#     plt.ylabel('Y (m)')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show(block=False)


# def get_top_bases(score_map, x_vals, y_vals, top_n=5):
#     flat = score_map.ravel()
#     indices = np.argsort(flat)[::-1]
#     coords = []
#     max_score = flat[indices[0]]
#     for idx in indices:
#         if len(coords) >= top_n:
#             break
#         i, j = np.unravel_index(idx, score_map.shape)
#         if score_map[i, j] == max_score:
#             coords.append((x_vals[j], y_vals[i]))
#     return coords, max_score


# def select_optimal_base(score_map, x_vals, y_vals, ee_targets, min_distance=0.3, perpendicular_tol=0.025):
#     """
#     Selecciona la base óptima entre celdas de máximo score que:
#     - Estén sobre la recta perpendicular al eje principal (vía PCA),
#     - Estén suficientemente lejos de los objetivos,
#     - Y tengan la menor distancia promedio a ellos.
#     """
#     max_score = np.max(score_map)
#     candidates = np.argwhere(score_map == max_score)

#     targets_xy = np.array([pos[:2] for pos, _ in ee_targets])
#     mean_targets = np.mean(targets_xy, axis=0)

#     # PCA para eje principal
#     centered = targets_xy - mean_targets
#     _, _, vh = np.linalg.svd(centered)
#     principal = vh[0]       # eje de los objetivos
#     perpendicular = np.array([-principal[1], principal[0]])  # eje ortogonal

#     best_score = np.inf
#     best_coord = None

#     for i, j in candidates:
#         x = x_vals[j]
#         y = y_vals[i]
#         base_xy = np.array([x, y])

#         # 1. Rechaza si está demasiado cerca de cualquier objetivo
#         dists = np.linalg.norm(targets_xy - base_xy, axis=1)
#         if np.any(dists < min_distance):
#             continue

#         # 2. Calcular proyección sobre la recta perpendicular
#         # Vector desde el centroide hasta la base
#         vec = base_xy - mean_targets
#         offset_along_perp = np.dot(vec, perpendicular)
#         offset_along_main = np.dot(vec, principal)

#         # 3. Acepta solo si la base está (casi) sobre la recta perpendicular
#         if abs(offset_along_main) > perpendicular_tol:
#             continue

#         # 4. Distancia promedio
#         dist_avg = np.mean(dists)

#         if dist_avg < best_score:
#             best_score = dist_avg
#             best_coord = (x, y)

#     return best_coord, best_score


# if __name__ == '__main__':
#     cart_step = 0.05
#     n_orientations = 20
#     area_size = 1.6*2
#     global_size = 4.0

#     db, orientations = load_database(
#         db_path=f"voxels_data_{cart_step}_step_{n_orientations}_orientations.pkl",
#         orientations_path="orientations.pkl"
#     )

#     example_targets = [
#         (np.array([1.7, 1.6, 0.8]), R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()),
#         (np.array([1.7, 2, 0.8]), R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()),
#         (np.array([1.7, 1.6, 0.3]), R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()),
#         (np.array([1.7, 2, 0.3]), R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix())
#     ]

#     union_map, intersection_map, x_vals, y_vals = evaluate_base_positions_global(
#         example_targets, db, orientations,
#         cart_step=cart_step, area_size=area_size, global_size=global_size
#     )

#     top_union, max_score_union = get_top_bases(union_map, x_vals, y_vals, top_n=2)

#     plot_score_map(union_map, x_vals, y_vals, title='Mapa de unión de todos los objetivos',
#                    ee_targets=example_targets, top_bases=top_union[:1])

#     plot_score_map(intersection_map, x_vals, y_vals, title='Mapa de intersección de todos los objetivos',
#                    ee_targets=example_targets, top_bases=top_union[1:2])

#     print(f"\nMejores bases encontradas (score máximo = {int(max_score_union)}):")
#     for i, (x, y) in enumerate(top_union[:2]):
#         print(f"B{i+1} → x: {x:.3f} m, y: {y:.3f} m")

#     optimal_base, optimal_score = select_optimal_base(intersection_map, x_vals, y_vals, example_targets, min_distance=0.3, perpendicular_tol=0.025)

#     if optimal_base:
#         print(f"\nBase óptima seleccionada:")
#         print(f"→ x: {optimal_base[0]:.3f} m, y: {optimal_base[1]:.3f} m")
#     else:
#         print("[ERROR] No se encontró ninguna base que cumpla el umbral mínimo de distancia.")
    
#     plot_score_map(intersection_map, x_vals, y_vals,
#                title='Mapa con base óptima personalizada',
#                ee_targets=example_targets,
#                top_bases=[optimal_base] if optimal_base else None)

#     input("Presiona ENTER para finalizar...")

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def load_database(db_path, orientations_path):
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
        print('[INFO] Database loaded.')
    all_voxels = list(db.keys())
    min_idx = np.min(all_voxels, axis=0)
    max_idx = np.max(all_voxels, axis=0)
    print(f"[DEBUG] DB index range: X:{min_idx[0]}-{max_idx[0]}, Y:{min_idx[1]}-{max_idx[1]}, Z:{min_idx[2]}-{max_idx[2]}")
    with open(orientations_path, 'rb') as f:
        orientations = pickle.load(f)
        print('[INFO] Orientations loaded.')
    return db, orientations


def compute_relative_pose(base_pos, ee_pos):
    return ee_pos[:3] - base_pos


def find_closest_voxel(point, cart_step, cart_min):
    return tuple(np.round((point - (cart_min + cart_step / 2)) / cart_step).astype(int))


def orientation_similarity(o1, o2):
    z1 = R.from_matrix(o1).apply([0, 0, 1])
    z2 = R.from_matrix(o2).apply([0, 0, 1])
    angle = np.arccos(np.clip(np.dot(z1, z2), -1.0, 1.0))
    return angle


def find_similar_orientations(required_rot, orientation_list, threshold_rad=np.deg2rad(30)):
    similar_indices = []
    for i, R_matrix in enumerate(orientation_list):
        if orientation_similarity(required_rot, R_matrix) < threshold_rad:
            similar_indices.append(i)
    return similar_indices


def define_global_grid(ee_targets, cart_step=0.05, global_size=4.0):
    ee_positions = np.array([pos[:2] for pos, _ in ee_targets])
    center = np.mean(ee_positions, axis=0)
    half_size = global_size / 2
    global_min = center - half_size
    global_max = center + half_size
    x_vals = np.arange(global_min[0], global_max[0] + cart_step, cart_step)
    y_vals = np.arange(global_min[1], global_max[1] + cart_step, cart_step)
    return x_vals, y_vals, global_min, global_max


# def evaluate_base_positions_global(ee_targets, db, orientations,
#                                    cart_step=0.05, area_size=1.6*2,
#                                    global_size=4.0, cart_min=-1.6):
#     x_vals, y_vals, global_min, global_max = define_global_grid(ee_targets, cart_step, global_size)
#     H, W = len(y_vals), len(x_vals)
#     print((H,W))
#     union_map = np.zeros((H, W))
#     intersection_votes = np.zeros((H, W), dtype=int)

#     half_cells = int(area_size / (2 * cart_step))

#     for ee_pos, ee_rot in ee_targets:
#         local_votes = np.zeros((H, W), dtype=int)
#         cx = np.argmin(np.abs(x_vals - ee_pos[0]))
#         cy = np.argmin(np.abs(y_vals - ee_pos[1]))

#         for i in range(cy - half_cells, cy + half_cells):
#             for j in range(cx - half_cells, cx + half_cells):
#                 if 0 <= i < H and 0 <= j < W:
#                     base_pos = np.array([x_vals[j], y_vals[i], 0.0])
#                     rel_pos = compute_relative_pose(base_pos, ee_pos)
#                     voxel_idx = find_closest_voxel(rel_pos, cart_step, cart_min)

#                     # print(f"rel_pos: {rel_pos}, voxel_idx: {voxel_idx}")

#                     if all(v >= 0 for v in voxel_idx) and voxel_idx in db:
#                         similar_orients = find_similar_orientations(ee_rot, orientations)
#                         score = 0
#                         if len(similar_orients) == 0:
#                             print(f"[WARN] Sin orientaciones similares para pose en {ee_pos[:2]}")
#                         for idx in similar_orients:
#                             configs = db[voxel_idx].get(idx, [])
#                             score += len(configs)
#                         union_map[i, j] += score
#                         if np.max(union_map) == 0:
#                             print("[ERROR] El mapa de unión está vacío. Verifica el umbral de orientación o la base de datos.")
#                         if score > 0:
#                             local_votes[i, j] = 1


#         intersection_votes += local_votes

#     intersection_map = np.where(intersection_votes == len(ee_targets), union_map, 0)
#     return union_map, intersection_map, x_vals, y_vals

def add_obstacle_by_coords(occupancy_map, x_vals, y_vals, x_min, y_min, x_max, y_max):
    """
    Marca como ocupadas (valor 0) las celdas del occupancy_map que caen dentro del
    rectángulo definido por las coordenadas reales (x_min, y_min) a (x_max, y_max).

    Parámetros:
        occupancy_map: matriz binaria (1 = libre, 0 = ocupado)
        x_vals, y_vals: arrays con los valores del grid global
        x_min, y_min, x_max, y_max: coordenadas reales del obstáculo

    Devuelve:
        occupancy_map modificado
    """
    # Convertir coordenadas reales a índices del grid
    j_min = np.searchsorted(x_vals, x_min, side='left')
    j_max = np.searchsorted(x_vals, x_max, side='right')
    i_min = np.searchsorted(y_vals, y_min, side='left')
    i_max = np.searchsorted(y_vals, y_max, side='right')

    # Asegurar que los índices están dentro de los límites
    i_min = max(0, i_min)
    i_max = min(len(y_vals), i_max)
    j_min = max(0, j_min)
    j_max = min(len(x_vals), j_max)

    # Marcar como ocupadas las celdas dentro del área
    occupancy_map[i_min:i_max, j_min:j_max] = 0
    return occupancy_map

def evaluate_base_positions_on_grid(
    ee_targets, db, orientations,
    x_vals, y_vals,
    cart_step=0.05, area_size=1.6 * 2,
    cart_min=-1.6, occupancy_map=None
):
    H, W = len(y_vals), len(x_vals)
    union_map = np.zeros((H, W))
    intersection_votes = np.zeros((H, W), dtype=int)
    half_cells = int(area_size / (2 * cart_step))

    for ee_pos, ee_rot in ee_targets:
        local_votes = np.zeros((H, W), dtype=int)
        cx = np.argmin(np.abs(x_vals - ee_pos[0]))
        cy = np.argmin(np.abs(y_vals - ee_pos[1]))

        for i in range(cy - half_cells, cy + half_cells):
            for j in range(cx - half_cells, cx + half_cells):
                if not (0 <= i < H and 0 <= j < W):
                    continue
                if occupancy_map is not None and occupancy_map[i, j] == 0:
                    continue

                base_pos = np.array([x_vals[j], y_vals[i], 0.0])
                rel_pos = compute_relative_pose(base_pos, ee_pos)
                voxel_idx = find_closest_voxel(rel_pos, cart_step, cart_min)

                if all(v >= 0 for v in voxel_idx) and voxel_idx in db:
                    similar_orients = find_similar_orientations(ee_rot, orientations, threshold_rad=np.deg2rad(30))
                    score = sum(len(db[voxel_idx].get(idx, [])) for idx in similar_orients)
                    union_map[i, j] += score
                    if score > 0:
                        local_votes[i, j] = 1

        intersection_votes += local_votes

    intersection_map = np.where(intersection_votes == len(ee_targets), union_map, 0)

    if np.max(union_map) == 0:
        print("[ERROR] El mapa de unión está vacío. Verifica el umbral de orientación o la base de datos.")
    if np.max(intersection_map) == 0:
        print("[ERROR] El mapa de intersección está vacío. Verifica el umbral de orientación o la base de datos.")
    return union_map, intersection_map


def plot_score_map(score_map, x_vals, y_vals, title='Mapa',
                   ee_targets=None, top_bases=None, occupancy_map=None):
    extent = [x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]]
    plt.figure(figsize=(8, 6))
    plt.imshow(score_map, origin='lower', extent=extent, cmap='viridis')

    if occupancy_map is not None:
        masked = np.ma.masked_where(occupancy_map == 1, occupancy_map)
        plt.imshow(masked, origin='lower', extent=extent, cmap='Greys', alpha=0.4)

    if ee_targets:
        for i, (pos, _) in enumerate(ee_targets):
            plt.plot(pos[0], pos[1], 'rx')
            plt.text(pos[0] + 0.02, pos[1] + 0.02, f'P{i+1}', color='white')

    if top_bases:
        for i, (x, y) in enumerate(top_bases):
            plt.plot(x, y, 'go')
            plt.text(x + 0.02, y - 0.02, f'B{i+1}', color='lime')

    plt.colorbar(label='Número total de soluciones')
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)


def get_top_bases(score_map, x_vals, y_vals, top_n=5):
    flat = score_map.ravel()
    indices = np.argsort(flat)[::-1]
    coords = []
    max_score = flat[indices[0]]
    for idx in indices:
        if len(coords) >= top_n:
            break
        i, j = np.unravel_index(idx, score_map.shape)
        if score_map[i, j] == max_score:
            coords.append((x_vals[j], y_vals[i]))
    return coords, max_score


def select_optimal_base(score_map, x_vals, y_vals, ee_targets, min_distance=0.3, perpendicular_tol=0.025):
    max_score = np.max(score_map)
    candidates = np.argwhere(score_map == max_score)

    targets_xy = np.array([pos[:2] for pos, _ in ee_targets])
    mean_targets = np.mean(targets_xy, axis=0)
    centered = targets_xy - mean_targets
    _, _, vh = np.linalg.svd(centered)
    principal = vh[0]
    perpendicular = np.array([-principal[1], principal[0]])

    best_score = np.inf
    best_coord = None

    for i, j in candidates:
        x = x_vals[j]
        y = y_vals[i]
        base_xy = np.array([x, y])

        dists = np.linalg.norm(targets_xy - base_xy, axis=1)
        if np.any(dists < min_distance):
            continue

        vec = base_xy - mean_targets
        offset_along_perp = np.dot(vec, perpendicular)
        offset_along_main = np.dot(vec, principal)

        if abs(offset_along_main) > perpendicular_tol:
            continue

        dist_avg = np.mean(dists)

        if dist_avg < best_score:
            best_score = dist_avg
            best_coord = (x, y)

    return best_coord, best_score

if __name__ == '__main__':
    cart_step = 0.05
    n_orientations = 20
    area_size = 1.6 * 2
    global_size = 4.0
    cart_min = -1.6

    # === 1. Cargar base de datos
    db, orientations = load_database(
        db_path=f"voxels_data_{cart_step}_step_{n_orientations}_orientations.pkl",
        orientations_path=f"orientations_{n_orientations}.pkl"
    )

    # === 2. Definir objetivos
    example_targets = [
        (np.array([1.7, 1.6, 0.8]), R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()),
        (np.array([1.7, 2.0, 0.8]), R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()),
        (np.array([1.7, 1.6, 0.3]), R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()),
        (np.array([1.7, 2.0, 0.3]), R.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix())
    ]

    # === 3. Definir grid global
    x_vals, y_vals, _, _ = define_global_grid(example_targets, cart_step=cart_step, global_size=global_size)
    H, W = len(y_vals), len(x_vals)

    # === 4. Crear máscara de obstáculos
    occupancy_map = np.ones((H, W), dtype=np.uint8)
    occupancy_map = add_obstacle_by_coords(occupancy_map, x_vals, y_vals,
                                       x_min=0.5, y_min=0.5,
                                       x_max=1.5, y_max=3.0)

    # Visualización rápida de la máscara
    plt.imshow(occupancy_map, cmap='gray')
    plt.title("Máscara de obstáculos (1=libre, 0=ocupado)")
    plt.xlabel('X (índices)')
    plt.ylabel('Y (índices)')
    plt.grid(True)
    plt.show(block=False)
    # input('Press Enter to continue...')

    # === 5. Evaluar posiciones base sobre el grid
    union_map, intersection_map = evaluate_base_positions_on_grid(
        example_targets, db, orientations,
        x_vals, y_vals,
        cart_step=cart_step,
        area_size=area_size,
        cart_min=cart_min,
        occupancy_map=occupancy_map
    )

    # === 6. Visualizar resultados
    plot_score_map(union_map, x_vals, y_vals, title='Mapa de unión con obstáculos',
                   ee_targets=example_targets, occupancy_map=occupancy_map)

    # plot_score_map(intersection_map, x_vals, y_vals, title='Mapa de intersección con obstáculos',
    #                ee_targets=example_targets, occupancy_map=occupancy_map)

    # === 7. Obtener mejores bases por score
    top_union, max_score_union = get_top_bases(union_map, x_vals, y_vals, top_n=2)

    print(f"\nMejores bases encontradas (score máximo = {int(max_score_union)}):")
    for i, (x, y) in enumerate(top_union[:2]):
        print(f"B{i+1} → x: {x:.3f} m, y: {y:.3f} m")

    # === 8. Base óptima por centrado y distancia
    optimal_base, optimal_score = select_optimal_base(intersection_map, x_vals, y_vals,
                                                      example_targets,
                                                      min_distance=0.5,
                                                      perpendicular_tol=0.1)

    if optimal_base:
        print(f"\nBase óptima seleccionada:")
        print(f"→ x: {optimal_base[0]:.3f} m, y: {optimal_base[1]:.3f} m")
    else:
        print("[ERROR] No se encontró ninguna base que cumpla el umbral mínimo de distancia.")

    # === 9. Visualizar base óptima
    plot_score_map(intersection_map, x_vals, y_vals,
                   title='Mapa de intersección con obstaculos y con base óptima personalizada',
                   ee_targets=example_targets,
                   top_bases=[optimal_base] if optimal_base else None,
                   occupancy_map=occupancy_map)

    input("Presiona ENTER para finalizar...")