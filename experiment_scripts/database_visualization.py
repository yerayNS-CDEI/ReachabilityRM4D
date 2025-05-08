# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict
# import os

# # ==== CONFIGURACIÓN ====
# cart_step = 0.05
# cart_min, cart_max = -1.6, 1.6
# n_orientations = 20
# filename = f"voxels_data_{cart_step}_step_{n_orientations}_orientations.pkl"

# # ==== CARGAR BASE DE DATOS ====
# if not os.path.exists(filename):
#     raise FileNotFoundError(f"No se encontró el archivo: {filename}")

# with open(filename, "rb") as f:
#     db = pickle.load(f)

# print(f"Archivo cargado con {len(db)} voxeles con al menos una orientación con soluciones.")

# # ==== INICIALIZACIÓN DE ESTRUCTURAS ====
# z_bins = int((cart_max - cart_min) / cart_step)
# z_distribution = np.zeros(z_bins, dtype=int)

# orientation_counts = np.zeros(n_orientations, dtype=int)
# total_solutions = 0

# # ==== RECORRIDO DE DATOS ====
# for (ix, iy, iz), orientations in db.items():
#     z_distribution[iz] += 1
#     for orientation_idx, q_solutions in orientations.items():
#         count = len(q_solutions)
#         orientation_counts[orientation_idx] += count
#         total_solutions += count

# print(f"Total de soluciones encontradas: {total_solutions}")

# # ==== GRAFICAR DISTRIBUCIÓN POR CAPA Z ====
# z_positions = [cart_min + cart_step/2 + i * cart_step for i in range(z_bins)]

# plt.figure(figsize=(10, 4))
# plt.bar(z_positions, z_distribution, width=cart_step * 0.9)
# plt.xlabel("Coordenada Z (m)")
# plt.ylabel("Voxeles con al menos una solución")
# plt.title("Distribución espacial por capa Z")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # ==== GRAFICAR DISTRIBUCIÓN POR ORIENTACIÓN ====
# plt.figure(figsize=(10, 4))
# plt.bar(range(n_orientations), orientation_counts)
# plt.xlabel("Índice de orientación")
# plt.ylabel("Número total de soluciones")
# plt.title("Distribución de soluciones por orientación")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

##################################################################
##################################################################
##################################################################

# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import os

# # === PARÁMETROS ===
# cart_step = 0.05
# cart_min, cart_max = -1.6, 1.6
# n_orientations = 20
# filename = f"voxels_data_{cart_step}_step_{n_orientations}_orientations.pkl"
# orientation_file = "orientations.pkl"

# # === CARGAR BASE DE DATOS ===
# with open(filename, "rb") as f:
#     db = pickle.load(f)

# with open(orientation_file, "rb") as f:
#     orientations = pickle.load(f)

# x_vals = np.arange(cart_min + cart_step/2, cart_max, cart_step)
# y_vals = np.arange(cart_min + cart_step/2, cart_max, cart_step)
# z_vals = np.arange(cart_min + cart_step/2, cart_max, cart_step)

# # === REPRESENTACIÓN 1: VOXELES COLOREADOS POR ORIENTACIONES CON SOLUCIONES ===
# print("Generando mapa 3D de voxeles con codificación por color...")

# vox_coords = []
# vox_colors = []
# max_sols = 0

# for (ix, iy, iz), orient_dict in db.items():
#     n_orients = len(orient_dict)
#     n_solutions = sum(len(qs) for qs in orient_dict.values())
#     if n_orients == 0:
#         continue
#     # Criterio de corte tipo "sandía abierta"
#     if ix < len(x_vals) // 2 or iy < len(y_vals) // 2:
#         continue
#     x, y, z = x_vals[ix], y_vals[iy], z_vals[iz]
#     vox_coords.append((x, y, z))
#     vox_colors.append(n_orients)
#     max_sols = max(max_sols, n_orients)

# vox_coords = np.array(vox_coords)
# vox_colors = np.array(vox_colors)

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# p = ax.scatter(vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2],
#                c=vox_colors, cmap='viridis', s=5, alpha=0.8)
# fig.colorbar(p, ax=ax, label='Orientaciones con soluciones')
# ax.set_title("Mapa 3D de voxeles (1/4 del espacio mostrado)")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.tight_layout()
# plt.show()

# # === REPRESENTACIÓN 2: CENTROS CON FLECHAS PARA ORIENTACIONES ===
# print("Generando flechas 3D de orientación...")

# import random
# from mpl_toolkits.mplot3d.art3d import Line3DCollection

# arrow_scale = 0.03
# n_voxels_to_plot = 5000  # puedes ajustar según lo que aguante tu PC

# # Elegir voxeles aleatorios del diccionario
# random_voxels = random.sample(list(db.items()), min(n_voxels_to_plot, len(db)))

# vectors = []
# origins = []

# for (ix, iy, iz), orient_dict in random_voxels:
#     x, y, z = x_vals[ix], y_vals[iy], z_vals[iz]
#     for idx, solutions in orient_dict.items():
#         if len(solutions) == 0:
#             continue
#         R = orientations[idx]
#         vec = R[:, 2] * arrow_scale  # eje z de la orientación
#         origins.append([x, y, z])
#         vectors.append([x + vec[0], y + vec[1], z + vec[2]])

# # Crear segmentos
# origins = np.array(origins)
# vectors = np.array(vectors)
# segments = [[o, v] for o, v in zip(origins, vectors)]

# # Dibujar
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# lc = Line3DCollection(segments, colors='red', linewidths=0.5)
# ax.add_collection3d(lc)
# ax.set_xlim(cart_min, cart_max)
# ax.set_ylim(cart_min, cart_max)
# ax.set_zlim(cart_min, cart_max)
# ax.set_title(f"Flechas 3D para {n_voxels_to_plot} voxeles aleatorios")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.tight_layout()
# plt.show()

##################################################################
##################################################################
##################################################################

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === CONFIGURACIÓN ===
cart_step = 0.05
n_orientations = 20
cart_min, cart_max = -1.6, 1.6
top_n = 20
export_csv = False
show_3d_plot = True
color_by = "orientations"  # también puede ser "solutions" o "orientations"

# Rango de zona espacial (puedes filtrar si quieres)
x_range = None  # ejemplo: (-0.5, 0.5)
y_range = None
z_range = None

db_file = f"voxels_data_{cart_step}_step_{n_orientations}_orientations.pkl"

# === CARGAR BASE DE DATOS ===
if not os.path.exists(db_file):
    raise FileNotFoundError(f"No se encontró la base de datos: {db_file}")

with open(db_file, "rb") as f:
    db = pickle.load(f)

print(f"Base de datos cargada con {len(db)} voxeles.")

with open("orientations.pkl", "rb") as f:
    orientations = pickle.load(f)
    
x_vals = np.arange(cart_min + cart_step/2, cart_max, cart_step)
y_vals = np.arange(cart_min + cart_step/2, cart_max, cart_step)
z_vals = np.arange(cart_min + cart_step/2, cart_max, cart_step)

# === ANALIZAR VOXELES ===
voxel_stats = []

for (ix, iy, iz), orient_dict in db.items():
    x, y, z = x_vals[ix], y_vals[iy], z_vals[iz]

    # FILTRO DE CUADRANTE (solo parte positiva del espacio)
    if x <= 0 or y <= 0 or z <= 0:
        continue

    if x_range and not (x_range[0] <= x <= x_range[1]):
        continue
    if y_range and not (y_range[0] <= y <= y_range[1]):
        continue
    if z_range and not (z_range[0] <= z <= z_range[1]):
        continue

    n_orients = len(orient_dict)
    total_solutions = sum(len(s) for s in orient_dict.values())

    voxel_stats.append({
        "voxel": (ix, iy, iz),
        "x": x, "y": y, "z": z,
        "orientations": n_orients,
        "solutions": total_solutions
    })

# === MOSTRAR TOP N ===
print(f"\n🔝 Voxeles con más orientaciones con soluciones:")
top_by_orient = sorted(voxel_stats, key=lambda v: v["orientations"], reverse=True)[:top_n]
for v in top_by_orient:
    print(f"{v['voxel']} | orientaciones: {v['orientations']} | soluciones: {v['solutions']}")

print(f"\n🔝 Voxeles con más soluciones totales:")
top_by_total = sorted(voxel_stats, key=lambda v: v["solutions"], reverse=True)[:top_n]
for v in top_by_total:
    print(f"{v['voxel']} | soluciones: {v['solutions']} | orientaciones: {v['orientations']}")

# === EXPORTACIÓN CSV ===
if export_csv:
    import csv
    with open("voxel_query_results.csv", "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["voxel", "x", "y", "z", "orientations", "solutions"])
        writer.writeheader()
        for v in voxel_stats:
            writer.writerow(v)
    print("CSV exportado: voxel_query_results.csv")

# === VISUALIZACIÓN 3D ===
if show_3d_plot and voxel_stats:
    print(f"\nGenerando visualización 3D coloreada por: {color_by}")

    coords = np.array([[v["x"], v["y"], v["z"]] for v in voxel_stats])
    if color_by == "orientations":
        colors = np.array([v["orientations"] for v in voxel_stats])
        color_label = "Orientaciones con soluciones"
    elif color_by == "solutions":
        colors = np.array([v["solutions"] for v in voxel_stats])
        color_label = "Soluciones totales"
    else:
        raise ValueError("color_by debe ser 'orientations' o 'solutions'")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=colors, cmap='viridis', s=6, alpha=0.8)

    cbar = fig.colorbar(sc, ax=ax, label=color_label)
    ax.set_title(f"Voxeles filtrados coloreados por {color_label}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    plt.show()
else:
    print("No hay voxeles para graficar.")



