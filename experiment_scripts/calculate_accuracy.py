import os
import argparse
from contextlib import redirect_stdout
import numpy as np
from tqdm import tqdm

from exp_utils import get_map_type_from_exp_dir, get_sample_points_from_exp_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='directory of experiment')
    parser.add_argument('eval_data_dir', type=str, help='directory with evaluation data for corresponding robot')

    return parser.parse_args()


def print_confusion_matrix(gt, pred, print_to=None):
    """
    assumes numpy arrays with binary values, prints on console.
    """
    tp = np.bitwise_and(gt == 1, pred == 1).sum() / len(gt)
    tn = np.bitwise_and(gt == 0, pred == 0).sum() / len(gt)
    fp = np.bitwise_and(gt == 0, pred == 1).sum() / len(gt)
    fn = np.bitwise_and(gt == 1, pred == 0).sum() / len(gt)

    acc = tp+tn
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    false_positive_rate = fp/(fp+tn)

    if print_to is not None:
        with open(print_to, 'w') as f:
            with redirect_stdout(f):
                print('confusion matrix:')
                print(f'        | gt 1  |  gt 0 |')
                print(f'--------|-------|-------|-------')
                print(f' pred 1 | {tp:.3f} | {fp:.3f} | {tp + fp:.3f}')
                print(f' pred 0 | {fn:.3f} | {tn:.3f} | {fn + tn:.3f}')
                print(f'--------|-------|-------|-------')
                print(f'        | {tp + fn:.3f} | {fp + tn:.3f} | {tp + tn + fp + fn:.3f}')

                print('metrics:')
                print(f'accuracy:\t {acc:.3f}')
                print(f'precision:\t {precision:.3f}')
                print(f'recall:\t {recall:.3f}')
                print(f'FPR:\t {false_positive_rate:.3f}')

    print('confusion matrix:')
    print(f'        | gt 1  |  gt 0 |')
    print(f'--------|-------|-------|-------')
    print(f' pred 1 | {tp:.3f} | {fp:.3f} | {tp+fp:.3f}')
    print(f' pred 0 | {fn:.3f} | {tn:.3f} | {fn+tn:.3f}')
    print(f'--------|-------|-------|-------')
    print(f'        | {tp+fn:.3f} | {fp+tn:.3f} | {tp+tn+fp+fn:.3f}')

    print('metrics:')
    print(f'accuracy:\t {acc:.3f}')
    print(f'precision:\t {precision:.3f}')
    print(f'recall:\t {recall:.3f}')
    print(f'FPR:\t {false_positive_rate:.3f}')

    return acc, precision, recall, false_positive_rate


def main(args):
    exp_dir = args.exp_dir
    eval_data_dir = args.eval_data_dir

    # get eval data
    tfs_ee = np.load(os.path.join(eval_data_dir, 'poses.npy'))
    reachable_by_ik = np.load(os.path.join(eval_data_dir, 'reachable_by_ik.npy'))
    n_samples = len(reachable_by_ik)

    # type of map
    map_type = get_map_type_from_exp_dir(exp_dir)

    # get individual sample points from experiment
    sample_points = get_sample_points_from_exp_dir(exp_dir)
    n_points = len(sample_points)

    accuracy = np.empty(n_points, dtype=float)
    precision = np.empty(n_points, dtype=float)
    recall = np.empty(n_points, dtype=float)
    false_positive_rate = np.empty(n_points, dtype=float)

    for i in range(n_points):
        cur_dir = os.path.join(exp_dir, f'{sample_points[i]}')

        # load the map
        rmap = map_type.from_file(os.path.join(cur_dir, 'rmap.npy'))
        reachable_by_map = np.zeros(n_samples, dtype=bool)

        for j in tqdm(range(n_samples)):
            try:
                idcs = rmap.get_indices_for_ee_pose(tfs_ee[j])
                reachable_by_map[j] = rmap.is_reachable(idcs)
            except IndexError:
                print(f'{j}: pose is not covered by map.')
                reachable_by_map[j] = 0

        fn = os.path.join(cur_dir, 'confusion_matrix.txt')
        acc, prec, rec, fpr = print_confusion_matrix(reachable_by_ik, reachable_by_map, print_to=fn)
        accuracy[i] = acc
        precision[i] = prec
        recall[i] = rec
        false_positive_rate[i] = fpr

    results = np.array([sample_points, accuracy, precision, recall, false_positive_rate])
    results_fn = os.path.join(exp_dir, 'accuracy_metrics.npy')
    np.save(results_fn, results)


if __name__ == '__main__':
    main(parse_args())
