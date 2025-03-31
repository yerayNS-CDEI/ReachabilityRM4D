import os
import time

from rm4d import ReachabilityMap4D, ZachariasMap, JointSpaceConstructor, Zacharias5DMap
from rm4d.robots import Franka, UR5E, UR10E
from rm4d.robots.assets import FRANKA_150_URDF, FRANKA_160_URDF, FRANKA_URDF, FRANKA_180_URDF

robot_types = {
    'franka': Franka,
    'ur5e': UR5E,
    'ur10e': UR10E,
}

map_types = {
    'rm4d': ReachabilityMap4D,
    'zach': ZachariasMap,
    'za5d': Zacharias5DMap,
}

constructor_types = {
    'joint': JointSpaceConstructor,
}

franka_versions = {
    150: FRANKA_150_URDF,
    160: FRANKA_160_URDF,
    166: FRANKA_URDF,
    180: FRANKA_180_URDF,
}


def get_map_type_from_exp_dir(exp_dir):
    exp_id = os.path.basename(os.path.normpath(exp_dir))
    map_type = map_types[exp_id.split('_')[0]]
    return map_type


def get_robot_type_from_exp_dir(exp_dir):
    exp_id = os.path.basename(os.path.normpath(exp_dir))
    robot_type = robot_types[exp_id.split('_')[1]]
    return robot_type


def get_sample_points_from_exp_dir(exp_dir):
    # get individual sample points from experiment, i.e. after certain number of samples during creation of the map
    sub_dirs = [(os.path.join(exp_dir, d), d) for d in os.listdir(exp_dir)]
    samples = [int(d[1]) for d in sub_dirs if os.path.isdir(d[0])]
    return sorted(samples)


class Timer:
    """
    this is a class to conveniently measure timings and additional count occurrences
    once instantiated, you can use timer.start('key') and timer.stop('key') to measure time, if you do it repeatedly
    it will sum up the elapsed time between all start and stop calls.
    with timer.count('key') you can count occurrences.
    finally, timer.print() will provide a summary of all stats.
    """
    def __init__(self):
        self.timers = {}
        self.counters = {}

    def start(self, key):
        if key not in self.timers.keys():
            self.timers[key] = -time.time()
        else:
            self.timers[key] -= time.time()

    def stop(self, key):
        if key not in self.timers.keys():
            raise ValueError('attempting to stop timer that has not been started')
        self.timers[key] += time.time()

    def count(self, key):
        if key not in self.counters.keys():
            self.counters[key] = 1
        else:
            self.counters[key] += 1

    def print(self):
        print('************ TIMINGS ************')
        for key, val in self.timers.items():
            print(f'\t{key}:\t{val:.2f}s')
        print('*********** COUNTERS ************')
        for key, val in self.counters.items():
            print(f'\t{key}:\t{val}x')
        print('*********************************')


