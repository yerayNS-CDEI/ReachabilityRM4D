import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from matplotlib import animation

# Cargar archivo final de voxeles
angle_divisions = 20  # ajusta esto si cambiaste la resolución
filename = f"voxels_data_{angle_divisions}.pkl"

if not os.path.exists(filename):
    raise FileNotFoundError(f"No se encontró {filename}")

with open(filename, "rb") as f:
    voxels = pickle.load(f)

# Obtener estadísticas: número de configuraciones por voxel
voxel_counts = {k: len(v) for k, v in voxels.items()}

# Convertir a array para análisis
coords = np.array(list(voxel_counts.keys()))
counts = np.array(list(voxel_counts.values()))

# Visualización 3D de la densidad
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(121, projection='3d')
p = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=counts, cmap='viridis', s=20)
fig.colorbar(p, ax=ax, label='Configuraciones por voxel')
ax.set_title(f"Completo (div {angle_divisions})")
ax.set_xlabel("ix")
ax.set_ylabel("iy")
ax.set_zlabel("iz")

# Vista con "un cuarto eliminado" para exponer el centro
center_x, center_y, center_z = np.max(coords[:, 0]) // 2, np.max(coords[:, 1]) // 2, np.max(coords[:, 2]) // 2
mask_quarter = ~((coords[:, 0] > center_x) & (coords[:, 1] > center_y) & (coords[:, 2] > center_z))
exposed_coords = coords[mask_quarter]
exposed_counts = counts[mask_quarter]

ax2 = fig.add_subplot(122, projection='3d')
p2 = ax2.scatter(exposed_coords[:, 0], exposed_coords[:, 1], exposed_coords[:, 2], c=exposed_counts, cmap='plasma', s=20)
fig.colorbar(p2, ax=ax2, label='Configuraciones por voxel')
ax2.set_title("Centro expuesto (sin un cuarto)")
ax2.set_xlabel("ix")
ax2.set_ylabel("iy")
ax2.set_zlabel("iz")

plt.tight_layout()
plt.show()

# Animación por capas Z (ArtistAnimation)
fig2, ax2 = plt.subplots()
z_layers = sorted(set(coords[:, 2]))

ims = []
for z in z_layers:
    mask = coords[:, 2] == z
    layer_coords = coords[mask]
    layer_counts = counts[mask]
    if len(layer_coords) == 0:
        continue
    im = ax2.scatter(layer_coords[:, 0], layer_coords[:, 1], c=layer_counts, cmap='viridis', s=20)
    ax2.set_title(f"Capa Z = {z}")
    ims.append([im])

ani = animation.ArtistAnimation(fig2, ims, interval=500, blit=True)
plt.show()

# Mostrar estadística general
print("\nEstadísticas globales:")
print(f"Voxeles totales ocupados: {len(voxels)}")
print(f"Máximo por voxel: {np.max(counts)}")
print(f"Mínimo por voxel: {np.min(counts)}")
print(f"Promedio: {np.mean(counts):.2f}")
print(f"Mediana: {np.median(counts)}")
