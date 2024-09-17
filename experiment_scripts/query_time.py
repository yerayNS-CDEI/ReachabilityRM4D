import os
import numpy as np

from exp_utils import get_map_type_from_exp_dir, get_sample_points_from_exp_dir, Timer


experiments = {
            'Zacharias et al.': 'data/zach_franka_joint_42',
            'Zacharias et al. (5D)': 'data/za5d_franka_joint_42',
            'RM4D': 'data/rm4d_franka_joint_42',
        }

eval_data_dir = 'data/eval_poses_franka166_n1000000_t25_i100/'


def main():
    # get eval data
    tfs_ee = np.load(os.path.join(eval_data_dir, 'poses.npy'))
    n_samples = len(tfs_ee)
    timer = Timer()

    for key, exp_dir in experiments.items():
        # load the map
        # we are using the last sample point (although this should not matter)
        map_type = get_map_type_from_exp_dir(exp_dir)
        sample_point = get_sample_points_from_exp_dir(exp_dir)[-1]
        cur_dir = os.path.join(exp_dir, f'{sample_point}')
        rmap = map_type.from_file(os.path.join(cur_dir, 'rmap.npy'))

        timer.start(key)
        for j in range(n_samples):
            idcs = rmap.get_indices_for_ee_pose(tfs_ee[j])
            result = rmap.is_reachable(idcs)
        timer.stop(key)
    timer.print()


if __name__ == '__main__':
    main()
