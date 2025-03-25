import numpy as np
import matplotlib.pyplot as plt

# Ensure that the loaded file is a dictionary
reachability_maps = np.load('reachability_maps.npy', allow_pickle=True).item()

for z_value, reachability_slice in reachability_maps.items():
    # Plotting the reachability map
    plt.figure()
    plt.imshow(reachability_slice, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Reachability')
    plt.title(f'Reachability Map at Z = {z_value}')
    plt.xlabel('X Index')
    plt.ylabel('Y Index')
    plt.show(block=False)

# Wait for user interaction before closing all plots
input("Press Enter to close all plots...")