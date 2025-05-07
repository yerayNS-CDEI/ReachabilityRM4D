################################################################
###### MODO SECUENCIAL ######
################################################################
#  import numpy as np
# from math import cos, sin, pi, floor
# from scipy.spatial.transform import Rotation as R
# from itertools import product
# from multiprocessing import Pool
# import os
# import pickle
# import gc
# from tqdm import tqdm

# # Parámetros DH UR10e
# d_vals = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]
# a_vals = [0, -0.6127, -0.57155, 0, 0, 0]
# alpha_vals = [pi/2, 0, 0, pi/2, -pi/2, 0]
# offset_vals = [0, 0, 0, 0, 0, 0]  # puedes poner -pi/2 donde haga falta

# # Espacio cartesiano discretizado
# cart_min, cart_max, cart_step = -1.5, 1.5, 0.1
# x_steps = y_steps = z_steps = int((cart_max - cart_min) / cart_step)

# # Rango y paso articular
# joint_min, joint_max, joint_step = -pi, pi, np.deg2rad(10)
# q_vals = np.arange(joint_min, joint_max + 0.001, joint_step)

# # Crear transformación DH
# def dh_matrix(theta, d, a, alpha):
#     return np.array([
#         [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
#         [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
#         [0,           sin(alpha),             cos(alpha),            d],
#         [0,           0,                      0,                     1]
#     ])

# # Cinemática directa completa
# def fk(q):
#     T = np.eye(4)
#     for i in range(6):
#         T = T @ dh_matrix(q[i] + offset_vals[i], d_vals[i], a_vals[i], alpha_vals[i])
#     return T

# # Probar FK con una configuración
# print("\nTest FK con configuración ejemplo:")
# test_q = [pi/4, pi/3, pi/2, pi/4, pi/3, pi/2]
# T_test = fk(test_q)
# print("Posición:", T_test[:3, 3])
# print("Orientación (matriz de rotación):\n", T_test[:3, :3])

# # Proceso por lote de configuraciones (usamos q6 = 0)
# def process_batch(batch):
#     batch_id = batch[0][0]  # índice único del batch
#     temp_dir = "voxel_temp"
#     os.makedirs(temp_dir, exist_ok=True)
#     batch_file = os.path.join(temp_dir, f"batch_{batch_id}.pkl")

#     # Saltar si el archivo ya existe y es válido
#     if os.path.exists(batch_file):
#         try:
#             with open(batch_file, "rb") as f:
#                 _ = pickle.load(f)
#             print(f"Saltando batch {batch_id} (ya procesado)")
#             return []
#         except Exception as e:
#             print(f"Archivo corrupto o vacío ({batch_file}), recalculando...", e)
#     results = []
#     for idx, config in batch:
#         q = list(config) + [0]  # q6 = 0
#         T = fk(q)
#         pos = T[:3, 3]
#         rot = T[:3, :3]

#         if not np.all((pos >= cart_min) & (pos <= cart_max)):
#             continue

#         ix = floor((pos[0] - cart_min) / cart_step)
#         iy = floor((pos[1] - cart_min) / cart_step)
#         iz = floor((pos[2] - cart_min) / cart_step)

#         quat = R.from_matrix(rot).as_quat()  # xyzw

#         results.append((ix, iy, iz, q, quat.tolist()))

#     temp_dir = "voxel_temp"
#     os.makedirs(temp_dir, exist_ok=True)
#     if results:
#         with open(batch_file, "wb") as f:
#             pickle.dump(results, f)

#     gc.collect()  # liberar memoria manualmente
#     import psutil
#     print(f"RAM usada: {psutil.virtual_memory().used / 1024**3:.2f} GB | Configuraciones válidas en este bloque: {len(results)}")
#     return []

# # Generar combinaciones de q1 a q5 sin cargarlas todas en memoria
# from itertools import islice

# def generate_batches(q_vals, block_size, max_batches):
#     counter = 0
#     batch = []
#     for idx, config in enumerate(product(q_vals, repeat=5)):
#         batch.append((idx, config))
#         if len(batch) == block_size:
#             yield batch
#             batch = []
#             counter += 1
#             if counter >= max_batches:
#                 break
#     if batch and counter < max_batches:
#         yield batch

# block_size = 100000  # más seguro para RAM limitada
# max_batches = 5  # puedes aumentarlo progresivamente (por ejemplo 10, 50, ...)
# max_batches = float('inf')
# batches = generate_batches(q_vals, block_size, max_batches)

# # Paralelizar con límite de procesos
# print("Ejecutando en modo totalmente secuencial...")
# for batch in tqdm(batches, total=max_batches):
#     process_batch(batch)

# print("Proceso completado. Archivos intermedios guardados en 'voxel_temp/'")

# # Fusión de archivos intermedios
# print("Fusionando archivos .pkl en voxels_data.pkl...")
# voxels = dict()
# for filename in tqdm(sorted(os.listdir("voxel_temp"))):
#     if filename.endswith(".pkl"):
#         with open(os.path.join("voxel_temp", filename), "rb") as f:
#             entries = pickle.load(f)
#         for ix, iy, iz, q_config, quat in entries:
#             key = (ix, iy, iz)
#             if key not in voxels:
#                 voxels[key] = []
#             voxels[key].append((q_config, quat))

# with open("voxels_data.pkl", "wb") as f:
#     pickle.dump(voxels, f)
# print("Fusión completada. voxels_data.pkl generado.")


################################################################
###### MODO MULTIPROCESSING ######
################################################################

# import numpy as np
# from math import cos, sin, pi, floor
# from scipy.spatial.transform import Rotation as R
# from itertools import product
# from multiprocessing import Pool
# import os
# import pickle
# import gc
# from tqdm import tqdm

# # Parámetros DH UR10e
# d_vals = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]
# a_vals = [0, -0.6127, -0.57155, 0, 0, 0]
# alpha_vals = [pi/2, 0, 0, pi/2, -pi/2, 0]
# offset_vals = [0, 0, 0, 0, 0, 0]  # puedes poner -pi/2 donde haga falta

# # Espacio cartesiano discretizado
# cart_min, cart_max, cart_step = -1.6, 1.6, 0.1
# x_steps = y_steps = z_steps = int((cart_max - cart_min) / cart_step)

# # Rango y paso articular
# joint_min, joint_max, joint_step = -pi, pi, np.deg2rad(10)
# angle_divisions = int((joint_max - joint_min) / joint_step)
# q_vals = np.arange(joint_min, joint_max + 0.001, joint_step)

# # Crear transformación DH
# def dh_matrix(theta, d, a, alpha):
#     return np.array([
#         [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
#         [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
#         [0,           sin(alpha),             cos(alpha),            d],
#         [0,           0,                      0,                     1]
#     ])

# # Cinemática directa completa
# def fk(q):
#     T = np.eye(4)
#     for i in range(6):
#         T = T @ dh_matrix(q[i] + offset_vals[i], d_vals[i], a_vals[i], alpha_vals[i])
#     return T

# # Probar FK con una configuración
# print("\nTest FK con configuración ejemplo:")
# test_q = [pi/4, pi/3, pi/2, pi/4, pi/3, pi/2]
# T_test = fk(test_q)
# print("Posición:", T_test[:3, 3])
# print("Orientación (matriz de rotación):\n", T_test[:3, :3])

# # Proceso por lote de configuraciones (usamos q6 = 0)
# def process_batch(batch):
#     batch_id = batch[0][0]  # índice único del batch
#     temp_dir = "voxel_temp"
#     os.makedirs(temp_dir, exist_ok=True)
    
#     batch_file = os.path.join(temp_dir, f"batch_{batch_id}_div{angle_divisions}.pkl")

#     # Saltar si el archivo ya existe y es válido
#     if os.path.exists(batch_file):
#         try:
#             with open(batch_file, "rb") as f:
#                 _ = pickle.load(f)
#             print(f"Saltando batch {batch_id} (ya procesado)")
#             return []
#         except Exception as e:
#             print(f"Archivo corrupto o vacío ({batch_file}), recalculando...", e)
#     results = []
#     for idx, config in batch:
#         q = list(config) + [0]  # q6 = 0
#         T = fk(q)
#         pos = T[:3, 3]
#         rot = T[:3, :3]

#         if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rot)):
#             print(f"[WARN] NaN o inf detectado en FK: q = {q}")
#             continue

#         if not np.all((pos >= cart_min) & (pos <= cart_max)):
#             print(f"[INFO] Posición fuera del volumen: {pos}")
#             continue

#         ix = floor((pos[0] - cart_min) / cart_step)
#         iy = floor((pos[1] - cart_min) / cart_step)
#         iz = floor((pos[2] - cart_min) / cart_step)

#         quat = R.from_matrix(rot).as_quat()  # xyzw

#         results.append((ix, iy, iz, q, quat.tolist()))

#     temp_dir = "voxel_temp"
#     os.makedirs(temp_dir, exist_ok=True)
#     if results:
#         with open(batch_file, "wb") as f:
#             pickle.dump(results, f)

#     gc.collect()  # liberar memoria manualmente
#     import psutil
#     print(f"RAM usada: {psutil.virtual_memory().used / 1024**3:.2f} GB | Configuraciones válidas en este bloque: {len(results)}")
#     return []

# # Generar combinaciones de q1 a q5 sin cargarlas todas en memoria
# from itertools import islice

# def generate_batches(q_vals, block_size, max_batches):
#     counter = 0
#     batch = []
#     for idx, config in enumerate(product(q_vals, repeat=5)):
#         batch.append((idx, config))
#         if len(batch) == block_size:
#             yield batch
#             batch = []
#             counter += 1
#             if counter >= max_batches:
#                 break
#     if batch and counter < max_batches:
#         yield batch

# block_size = 1000000  # más seguro para RAM limitada
# max_batches = 10  # puedes aumentarlo progresivamente (por ejemplo 10, 50, ...)
# max_batches = float('inf')
# batches = generate_batches(q_vals, block_size, max_batches)

# # Paralelizar con multiprocessing seguro
# print("Ejecutando en modo paralelo con multiprocessing...")
# from multiprocessing import cpu_count
# with Pool(processes=3) as pool:  # puedes ajustar a 2 o 4 según tu CPU
#     with tqdm(total=max_batches, desc="Procesando batches") as pbar:
#         for _ in pool.imap(process_batch, batches):
#             pbar.update()

# print("Proceso completado. Archivos intermedios guardados en 'voxel_temp/'")

# # Fusión de archivos intermedios por bloques
# print("Fusionando archivos .pkl en voxels_data.pkl...")
# voxels = dict()

# # Abrir el archivo final y escribir los datos bloque por bloque
# with open(f"voxels_data_{angle_divisions}.pkl", "wb") as f_out:
#     for filename in tqdm(sorted(os.listdir("voxel_temp"))):
#         if not filename.endswith(f"_div{angle_divisions}.pkl"):
#             continue
#         if filename.endswith(".pkl"):
#             with open(os.path.join("voxel_temp", filename), "rb") as f_in:
#                 entries = pickle.load(f_in)
            
#             # Fusionar datos de cada archivo
#             for ix, iy, iz, q_config, quat in entries:
#                 key = (ix, iy, iz)
#                 if key not in voxels:
#                     voxels[key] = []
#                 voxels[key].append((q_config, quat))
            
#             # Escribir los datos del bloque fusionado y limpiar memoria
#             pickle.dump(voxels, f_out)
#             voxels.clear()  # Limpiar datos temporales para evitar usar demasiada RAM

# print(f"Fusión completada. voxels_data_{angle_divisions}.pkl generado.")

# # # Guardar como JSON
# # import json
# # voxels_json = {str(k): v for k, v in voxels.items()}
# # with open(f"voxels_data_{angle_divisions}.json", "w") as f_json:
# #     json.dump(voxels_json, f_json)
# # print(f"Datos también guardados como voxels_data_{angle_divisions}.json")

# # # Guardar como CSV
# # import csv
# # with open(f"voxels_data_{angle_divisions}.csv", "w", newline="") as f_csv:
# #     writer = csv.writer(f_csv)
# #     writer.writerow(["ix", "iy", "iz", "q1", "q2", "q3", "q4", "q5", "q6", "qx", "qy", "qz", "qw"])
# #     for (ix, iy, iz), entries in voxels.items():
# #         for q_config, quat in entries:
# #             writer.writerow([ix, iy, iz] + q_config + quat)
# # print(f"Datos también guardados como voxels_data_{angle_divisions}.csv")

# # # Guardar como NPZ
# # np_voxels = {f"{ix}_{iy}_{iz}": np.array(entries, dtype=object) for (ix, iy, iz), entries in voxels.items()}
# # np.savez_compressed(f"voxels_data_{angle_divisions}.npz", **np_voxels)
# # print(f"Datos también guardados como voxels_data_{angle_divisions}.npz")

################################################################
###### MODO INVERSE KINEMATICS ######
################################################################

import numpy as np
from math import cos, sin, pi
from scipy.spatial.transform import Rotation
from itertools import product
from multiprocessing import Pool, cpu_count
import os
import pickle
import sqlite3
import gc
from tqdm import tqdm
from closed_form_algorithm import closed_form_algorithm_complete
from functools import partial

# Parámetros de discretización
cart_min, cart_max, cart_step = -1.6, 1.6, 0.05  # Discretización más fina de 0.05
x_steps = y_steps = z_steps = int((cart_max - cart_min) / cart_step)

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

# Función para generar orientaciones homogéneas (distribuidas uniformemente)
def generate_orientations(samples=20):
    points = fibonacci_sphere(samples)  # puntos distribuidos uniformemente sobre la esfera
    origin_vector = np.array([0, 0, 1])
    rots = []
    rot_matrices = []
    for point in points:
        rotation = Rotation.align_vectors([point], [origin_vector])[0]
        rots.append(rotation)
        rot_matrices.append(rotation.as_matrix())
    return rot_matrices

# Función para calcular IK en el centro de cada voxel
def process_voxel(voxel_idx, voxel_size=0.05, orientations=None, cart_step=0.05, result_file=None):
    voxel_data = []
    x_start, y_start, z_start = voxel_idx
    
    # El centro del voxel es simplemente el centro del intervalo
    pos = [x_start * voxel_size + voxel_size / 2, y_start * voxel_size + voxel_size / 2, z_start * voxel_size + voxel_size / 2]
    
    # Usamos las orientaciones precalculadas
    for rot in orientations:
        goal_matrix = np.eye(4)
        goal_matrix[:3, 3] = pos
        goal_matrix[:3, :3] = rot

        solutions = closed_form_algorithm_complete(goal_matrix, type=0)  # Obtener todas las soluciones de IK
        valid_solutions = [sol for sol in solutions if np.all(np.isfinite(sol))]  # Filtrar soluciones válidas (sin NaN)

        if valid_solutions:
            for sol in valid_solutions:
                voxel_data.append((pos, rot, sol))
                
                # Guardar en el archivo intermedio si está habilitado
                if result_file is not None:
                    with open(result_file, 'ab') as f:
                        pickle.dump((pos, rot, sol), f)

    return voxel_data

# Función que guarda los datos de la cola en la base de datos SQLite desde archivos
def save_to_db_from_files(result_dir, db_filename='reachability.db3', batch_size=100):
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reachability_data (
        x REAL,
        y REAL,
        z REAL,
        orientation TEXT,
        joint_values TEXT
    )
    ''')

    batch_data = []
    # Procesar los archivos en el directorio
    for filename in os.listdir(result_dir):
        if filename.endswith(".pkl"):
            with open(os.path.join(result_dir, filename), 'rb') as f:
                while True:
                    try:
                        data = pickle.load(f)
                        batch_data.append(data)

                        # Si hemos acumulado suficiente data, lo guardamos
                        if len(batch_data) >= batch_size:
                            for pos, rot, q in batch_data:
                                cursor.execute('''
                                INSERT INTO reachability_data (x, y, z, orientation, joint_values)
                                VALUES (?, ?, ?, ?, ?)
                                ''', (pos[0], pos[1], pos[2], str(rot.tolist()), str(q.tolist())))
                            conn.commit()
                            batch_data = []  # Limpiar el buffer después de guardarlo
                    except EOFError:
                        break  # Fin del archivo

    # Guardar cualquier dato restante que no haya sido guardado aún
    if batch_data:
        for pos, rot, q in batch_data:
            cursor.execute('''
            INSERT INTO reachability_data (x, y, z, orientation, joint_values)
            VALUES (?, ?, ?, ?, ?)
            ''', (pos[0], pos[1], pos[2], str(rot.tolist()), str(q.tolist())))
        conn.commit()

    conn.close()

# Función principal para ejecutar el proceso con multiprocessing
def main(cart_step=0.05, samples=20, batch_size=100, result_dir='result_files', group_size=1000):
    # Calcular el número total de operaciones (voxeles * orientaciones)
    total_operations = x_steps * y_steps * z_steps * samples
    # Calcular el máximo de datos que se pueden obtener (voxeles * orientaciones * 6 soluciones)
    max_data = total_operations * 6

    # Mostrar la cantidad de operaciones y el máximo de datos
    print(f"Total de operaciones: {total_operations}")
    print(f"Máximo de datos a generar (aproximadamente): {max_data} soluciones de IK")

    # Crear directorio para los resultados si no existe
    os.makedirs(result_dir, exist_ok=True)

    # Generar los voxeles (por ejemplo, usando las posiciones de (x, y, z))
    voxels = list(product(range(x_steps), range(y_steps), range(z_steps)))  # Convertimos a lista para poder medir el tamaño

    # Calcular las orientaciones solo una vez antes del procesamiento de los voxeles
    orientations = generate_orientations(samples)  # Esto se realiza una sola vez

    # Crear nombre dinámico para los archivos intermedios usando cart_step y samples
    result_file_template = os.path.join(result_dir, f"voxels_{cart_step}_step_{samples}_orientations_batch_{{}}.pkl")

    batch_counter = 1
    result_file = result_file_template.format(batch_counter)

    # Multiprocessing con barra de progreso
    with Pool(processes=cpu_count()) as pool:
        with tqdm(total=total_operations, desc="Procesando voxeles", unit="voxel") as pbar:
            # Asegúrate de pasar todos los parámetros correctamente en el partial
            process_voxel_partial = partial(process_voxel, result_file=result_file, cart_step=cart_step, orientations=orientations)

            for result in pool.imap_unordered(process_voxel_partial, voxels, chunksize=100):
                pbar.update(1)

                # Comprobar si hemos alcanzado el tamaño del grupo
                if os.path.getsize(result_file) >= group_size * 1024 * 1024:  # Cambiar el tamaño de archivo en bytes
                    batch_counter += 1
                    result_file = result_file_template.format(batch_counter)

    # Después de terminar, procesamos los archivos guardados para agregarlos a la base de datos
    save_to_db_from_files(result_dir, 'reachability.db3', batch_size)

if __name__ == '__main__':
    main(cart_step=0.05, samples=20, batch_size=1000)








