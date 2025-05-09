import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pymongo import MongoClient

client = MongoClient("mongodb+srv://yeraynavarro:LzXKhwG6QWadXY4X@ur10e-database-005.l56hsfs.mongodb.net/")
db = client["UR10e-database-005"]
collection = db["voxels"]

def load_database(db_path, orientations_path):
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    with open(orientations_path, 'rb') as f:
        orientations = pickle.load(f)
    return db, orientations


def compute_relative_pose(base_pos, ee_pos):
    rel_pos = ee_pos[:3] - base_pos
    return rel_pos


def find_closest_voxel(point, cart_step):
    idx = tuple(np.round(point / cart_step).astype(int))
    return idx


def orientation_similarity(o1, o2):
    z1 = R.from_matrix(o1).apply([0, 0, 1])
    z2 = R.from_matrix(o2).apply([0, 0, 1])
    angle = np.arccos(np.clip(np.dot(z1, z2), -1.0, 1.0))
    return angle


def find_similar_orientations(required_rot, orientation_list, threshold_rad=np.deg2rad(15)):
    similar_indices = []
    for i, R_matrix in enumerate(orientation_list):
        if orientation_similarity(required_rot, R_matrix) < threshold_rad:
            similar_indices.append(i)
    return similar_indices


def evaluate_base_positions(ee_targets, db, orientations, cart_step=0.05, area_size=1.6*2):
    n_cells = int(area_size / cart_step)
    center_offset = n_cells // 2
    score_map = np.zeros((n_cells, n_cells))

    for ee_pos, ee_rot in ee_targets:
        for i in range(n_cells):
            for j in range(n_cells):
                dx = (i - center_offset) * cart_step + cart_step / 2
                dy = (j - center_offset) * cart_step + cart_step / 2
                base_pos = np.array([ee_pos[0] - dx, ee_pos[1] - dy, 0.0])
                print("base_pos",base_pos)
                rel_pos = compute_relative_pose(base_pos, ee_pos)

                ### METODOS DE MONGODB PARA BASE DE DATOS

                # collection.find({"coords":str(tuple(rel_pos[0],rel_pos[1], rel_pos[2]))})
                
                ##########################################

                print("rel_pos",rel_pos)
                voxel_idx = find_closest_voxel(rel_pos, cart_step)
                print(f"Posición relativa: {rel_pos}, Voxel calculado: {voxel_idx}")

                if voxel_idx in db:
                    print(f"Voxel {voxel_idx} encontrado")
                    similar_orient_idxs = find_similar_orientations(ee_rot, orientations)
                    for idx in similar_orient_idxs:
                        configs = db[voxel_idx].get(idx, [])
                        print(f"Configuraciones encontradas: {len(configs)}")
                        score_map[i, j] += len(configs)

    return score_map, center_offset, cart_step


def plot_score_map(score_map, cart_step=0.05):
    extent = [-score_map.shape[0] // 2 * cart_step, score_map.shape[0] // 2 * cart_step,
              -score_map.shape[1] // 2 * cart_step, score_map.shape[1] // 2 * cart_step]
    plt.figure(figsize=(8, 6))
    plt.imshow(score_map.T, origin='lower', extent=extent, cmap='viridis')
    plt.colorbar(label='Número total de soluciones')
    plt.title('Mapa de posiciones base factibles')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.show()


def print_top_base_positions(score_map, center_offset, cart_step, ee_targets, top_n=5):
    flat_indices = np.dstack(np.unravel_index(np.argsort(score_map.ravel())[::-1], score_map.shape))[0]
    max_score = score_map[flat_indices[0][0], flat_indices[0][1]]
    candidates = [idx for idx in flat_indices if score_map[idx[0], idx[1]] == max_score]

    print(f"\nTop {top_n} posiciones base más prometedoras (score máximo = {int(max_score)}):")
    for idx in candidates[:top_n]:
        i, j = idx
        dx = (i - center_offset) * cart_step + cart_step / 2
        dy = (j - center_offset) * cart_step + cart_step / 2
        print(f"Base en (x: {dx:.3f} m, y: {dy:.3f} m)")

    # Desempate por distancia al centro
    center_i, center_j = center_offset, center_offset
    def dist_to_center(i, j):
        return np.sqrt((i - center_i)**2 + (j - center_j)**2)

    best_center = min(candidates, key=lambda idx: dist_to_center(idx[0], idx[1]))
    i_c, j_c = best_center
    dx_c = (i_c - center_offset) * cart_step + cart_step / 2
    dy_c = (j_c - center_offset) * cart_step + cart_step / 2
    print(f"\n>> Mejor posición base centrada: (x: {dx_c:.3f} m, y: {dy_c:.3f} m)")

    # Desempate por distancia media a los targets
    def dist_to_targets(i, j):
        bx = (i - center_offset) * cart_step + cart_step / 2
        by = (j - center_offset) * cart_step + cart_step / 2
        base = np.array([bx, by])
        dists = [np.linalg.norm(base - ee[:2]) for ee, _ in ee_targets]
        return np.mean(dists)

    best_avg = min(candidates, key=lambda idx: dist_to_targets(idx[0], idx[1]))
    i_a, j_a = best_avg
    dx_a = (i_a - center_offset) * cart_step + cart_step / 2
    dy_a = (j_a - center_offset) * cart_step + cart_step / 2
    print(f"\n>> Mejor posición base por distancia promedio a los objetivos: (x: {dx_a:.3f} m, y: {dy_a:.3f} m)")


# === Ejemplo de uso ===
if __name__ == '__main__':
    cart_step = 0.05
    n_orientations = 20

    db, orientations = load_database(
        db_path=f"voxels_data_{cart_step}_step_{n_orientations}_orientations.pkl",
        orientations_path="orientations.pkl"
    )

    # Simulación de base de datos mínima (solo para testeo)
    # db = {
    #     (0, 0, 8): {0: [[0.1]*6], 1: [[0.2]*6]},
    #     (1, 1, 9): {5: [[0.3]*6]},
    #     (2, 2, 10): {10: [[0.4]*6, [0.5]*6]},
    # }
    # orientations = [np.eye(3) for _ in range(n_orientations)]  # identidades por simplicidad

    # Lista de objetivos: [(pos, rot)], donde rot es una matriz de rotación 3x3
    example_targets = [
        (np.array([0.0, 0.0, 0.4]), np.eye(3)),
        (np.array([0.3, -0.2, 0.6]), R.from_euler('xyz', [0, 0, 30], degrees=True).as_matrix()),
        (np.array([-0.4, 0.3, 0.5]), R.from_euler('xyz', [10, 20, 0], degrees=True).as_matrix()),
        (np.array([0.2, 0.2, 0.3]), R.from_euler('xyz', [0, 45, 0], degrees=True).as_matrix())
    ]

    score_map, center_offset, cart_step = evaluate_base_positions(example_targets, db, orientations, cart_step=cart_step)
    plot_score_map(score_map, cart_step=cart_step)
    print_top_base_positions(score_map, center_offset, cart_step, example_targets, top_n=5)
