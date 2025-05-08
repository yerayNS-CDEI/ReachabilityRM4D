# from rm4d import ReachabilityMap4D

# rmap = ReachabilityMap4D.from_file('data/rm4d_ur10e_joint_42/10000000/rmap.npy')

# from rm4d.robots import Simulator

# sim = Simulator(with_gui=True)

# import numpy as np

# # Identidad + una traslación para ubicar el EE
# tf_ee = np.eye(4)
# tf_ee[:3, 3] = [0.4, 0.5, 0.8]  # z=1.0 por ejemplo

# rmap.visualize_in_sim(sim, tf_ee)

# import time
# while True:
#     time.sleep(0.1)


from closed_form_algorithm import closed_form_algorithm_complete
import numpy as np

T = np.eye(4)
T[:3, 3] = [0.5, 0.2, 0.3]  # posición dentro del workspace del UR10e
solutions = closed_form_algorithm_complete(T, type=0)

print(f"Número de soluciones: {len(solutions)}")
for i, q in enumerate(solutions):
    print(f"Solución {i}: {q}")