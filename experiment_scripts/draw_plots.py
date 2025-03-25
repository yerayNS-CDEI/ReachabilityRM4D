import os
import numpy as np
import matplotlib.pyplot as plt

experiments = {
    'Franka':
        {
            # 'Zacharias et al.': 'data/zach_franka_joint_42',
            # 'Zacharias et al. (5D)': 'data/za5d_franka_joint_42',
            'RM4D': 'data/rm4d_franka_joint_42',
        },
    'UR5e':
        {
            # 'Zacharias et al.': 'data/zach_ur5e_joint_42',
            # 'Zacharias et al. (5D)': 'data/za5d_ur5e_joint_42',
            # 'RM4D': 'data/rm4d_ur5e_joint_42',
        }
}

franka_limited_exp = {
    'Franka': 'data/rm4d_franka_joint_42',
    # 'Franka 180': 'data/rm4d_franka180_joint_42',
    # 'Franka 160': 'data/rm4d_franka160_joint_42',
    # 'Franka 150': 'data/rm4d_franka150_joint_42',
}

accuracy_fn = 'accuracy_metrics.npy'
hit_stats_fn = 'hit_stats.npy'

line_styles = ['solid', 'dotted', 'dashed', 'dashdot']


def plot_accuracy():
    robots = list(experiments.keys())
    for robot in robots:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(robot)

        maps = list(experiments[robot].keys())
        ls = iter(line_styles)
        for map_key in maps:
            fn = os.path.join(experiments[robot][map_key], accuracy_fn)
            d = np.load(fn, allow_pickle=True)
            # sample_points, accuracy, precision, recall, fpr
            accuracy = d[1]
            precision = d[2]
            recall = d[3]
            fpr = d[4]
            f1_score = 2 * precision * recall / (precision + recall)
            highest_val = np.max(d[1])
            highest_idx = np.argmax(d[1])
            first_above_95 = None
            if highest_val >= 0.95:
                first_above_95 = d[0, next(idx for idx, value in enumerate(d[1]) if value >= 0.95)]
            print(f'{robot} -- {map_key}:')
            print(f'\taccuracy >= 95% for first time after {first_above_95} samples.')
            print(f'\tMaximum accuracy of {100*highest_val:.2f} after {d[0, highest_idx]} samples.')

            style = next(ls)
            axes[0].plot(d[0], accuracy, ls=style)
            axes[1].plot(d[0], recall, ls=style)
            axes[2].plot(d[0], fpr, ls=style)

        for i in range(3):
            axes[i].legend(maps)
            axes[i].set_xlabel('num samples')
            axes[i].set_xscale('log')
            axes[i].grid(True, which='both')

        axes[0].set_ylabel('Accuracy')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].yaxis.set_ticks(np.arange(0.0, 1.05, 0.1))
        axes[2].set_ylabel('False Positive Rate')
        fig.tight_layout()

        plt.show()

        print("Accuracy plotted.")


def plot_franka_limited_accuracy():
    robots = list(franka_limited_exp.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ls = iter(line_styles)
    for robot in robots:
        fn = os.path.join(franka_limited_exp[robot], accuracy_fn)
        d = np.load(fn, allow_pickle=True)
        # sample_points, accuracy, precision, recall, fpr
        accuracy = d[1]
        precision = d[2]
        recall = d[3]
        fpr = d[4]

        style = next(ls)
        axes[0].plot(d[0], accuracy, ls=style)
        axes[1].plot(d[0], recall, ls=style)
        axes[2].plot(d[0], fpr, ls=style)

    for i in range(3):
        axes[i].legend(robots)
        axes[i].set_xlabel('num samples')
        axes[i].set_xscale('log')
        axes[i].grid(True, which='both')

    axes[0].set_ylabel('Accuracy')
    axes[1].set_ylabel('True Positive Rate')
    axes[2].set_ylabel('False Positive Rate')
    fig.tight_layout()

    plt.show()

    print("Limited accuracy plotted.")


def plot_hit_stats():
    robots = list(experiments.keys())
    for robot in robots:
        fig, ax = plt.subplots(1, 1, figsize=(16/3, 4))
        fig.suptitle(robot)

        maps = list(experiments[robot].keys())
        ls = iter(line_styles)
        for map_key in maps:
            fn = os.path.join(experiments[robot][map_key], hit_stats_fn)
            d = np.load(fn, allow_pickle=True)
            num_samples = d[0]
            num_hits = d[1]
            new_voxels = np.zeros_like(num_hits)

            new_voxels[0] = num_hits[0]
            for i in range(1, len(num_hits)):
                new_voxels[i] = num_hits[i] - num_hits[i - 1]

            style = next(ls)
            ax.plot(num_samples, new_voxels, ls=style)

        ax.legend(maps)
        ax.set_xlabel('num samples')
        ax.set_ylabel('new map elements added')
        ax.set_yscale('log')
        ax.grid(True, which='both')
        plt.show()

        print("Hit status plotted.")


if __name__ == '__main__':
    plot_accuracy()
    plot_franka_limited_accuracy()
    plot_hit_stats()
