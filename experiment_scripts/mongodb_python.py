from pymongo import MongoClient
import pickle
import numpy as np
import json

client = MongoClient("mongodb+srv://yeraynavarro:LzXKhwG6QWadXY4X@ur10e-database-005.l56hsfs.mongodb.net/")
db = client["UR10e-database-005"]
collection = db["voxels"]

## NECESARIO REDONDEAR LOS INDICES DE LAS COORDENADAS

voxels = []

# === PARÁMETROS ===
cart_step = 0.05
cart_min, cart_max = -1.6, 1.6
n_orientations = 20
filename = f"voxels_data_{cart_step}_step_{n_orientations}_orientations.pkl"
orientation_file = "orientations.pkl"

x_vals = np.arange(cart_min + cart_step/2, cart_max, cart_step)
y_vals = np.arange(cart_min + cart_step/2, cart_max, cart_step)
z_vals = np.arange(cart_min + cart_step/2, cart_max, cart_step)

# === CARGAR BASE DE DATOS ===
with open(filename, "rb") as f:
    db = pickle.load(f)

for (ix, iy, iz), orient_dict in db.items():
    
    x, y, z = x_vals[ix], y_vals[iy], z_vals[iz]

    o_d = {}

    for orient, conf_list in orient_dict.items():
        o_d[str(orient)] = [conf for conf in conf_list]

    json_data = {"coords": (x, y, z),
                    "orientation":o_d}
    
    collection.insert_one(json_data)

