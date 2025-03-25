import numpy as np

from .rmap import MapBase


class ZachariasMap(MapBase):
    """
    This is a re-implementation based on Zacharias et al.:
    The capability map: a tool to analyze robot arm workspaces, Int. Journal of Humanoid Robotics, 2013.

    It uses a 3d voxel grid for the position, using a voxel size of l_c = 50 mm (adjustable). Although they have a
    cubic map, we use the same limits as in our map, which makes it a cuboid. As they used in their paper, we use
    200 points on the inscribed sphere to describe rotation, along with in-plane rotations in discretized steps of
    30 degrees (i.e. 12 bins).
    """

    def __init__(self, xy_limits=None, z_limits=None, voxel_res=0.05, n_sphere_points=200,
                 n_bins_in_plane=12, no_map=False, sphere_point_tfs=None):
        if xy_limits is None:
            xy_limits = [-1.05, 1.05]
        if z_limits is None:
            z_limits = [0, 1.35]

        # [min, max]
        self.xy_limits = xy_limits
        self.z_limits = z_limits
        self.in_plane_limits = [0, 2*np.pi]

        # get dimensions
        self.voxel_res = voxel_res
        self.n_bins_xy = int(np.ceil((self.xy_limits[1] - self.xy_limits[0]) / voxel_res))
        self.n_bins_z = int(np.ceil((self.z_limits[1] - self.z_limits[0]) / voxel_res))
        self.n_sphere_points = n_sphere_points
        self.n_bins_in_plane = n_bins_in_plane

        # check achieved resolution
        assert np.isclose((self.xy_limits[1] - self.xy_limits[0]) / self.n_bins_xy, self.voxel_res)
        assert np.isclose((self.z_limits[1] - self.z_limits[0]) / self.n_bins_z, self.voxel_res)
        self.in_plane_res = (self.in_plane_limits[1] - self.in_plane_limits[0]) / self.n_bins_in_plane

        # generate sphere points for cache
        if sphere_point_tfs is None:
            self.sphere_point_tfs = self.get_sphere_point_tfs()
        else:
            assert len(sphere_point_tfs) == self.n_sphere_points
            self.sphere_point_tfs = sphere_point_tfs

        # create map
        if no_map:
            self.map = None
        else:
            self.map = np.zeros(
                shape=(self.n_bins_xy, self.n_bins_xy, self.n_bins_z, self.n_sphere_points, self.n_bins_in_plane),
                dtype=bool
            )

    def to_file(self, filename):
        save_dict = {
            'map': self.map,
            'xy_limits': self.xy_limits,
            'z_limits': self.z_limits,
            'voxel_res': self.voxel_res,
            'n_sphere_points': self.n_sphere_points,
            'n_bins_in_plane': self.n_bins_in_plane,
            # we are saving sphere tfs, as there is randomness in the in-pane rotation generation. this could be solved
            # differently to avoid using disk space, so we disregard the required memory in any measurements
            'sphere_point_tfs': self.sphere_point_tfs,
        }
        np.save(filename, save_dict)

    @classmethod
    def from_file(cls, filename):
        d = np.load(filename, allow_pickle=True).item()
        sphere_point_tfs = None
        if 'sphere_point_tfs' in d.keys():
            sphere_point_tfs = d['sphere_point_tfs']
        rm = cls(xy_limits=d['xy_limits'],
                 z_limits=d['z_limits'],
                 voxel_res=d['voxel_res'],
                 n_sphere_points=d['n_sphere_points'],
                 n_bins_in_plane=d['n_bins_in_plane'],
                 no_map=True,
                 sphere_point_tfs=sphere_point_tfs
                 )
        rm.map = d['map']

        print(f'{cls.__name__} loaded from {filename}')
        rm.print_structure()
        return rm

    def print_structure(self):
        print(f'structure of zacharias map:')
        print(f'\txy: {self.xy_limits}, {self.n_bins_xy} bins, {self.voxel_res} resolution')
        print(f'\t z: {self.z_limits}, {self.n_bins_z} bins, {self.voxel_res} resolution')
        print(f'\tns: {self.n_sphere_points} points on sphere')
        print(f'\t r: {self.in_plane_limits}, {self.n_bins_in_plane} bins, {np.rad2deg(self.in_plane_res)} deg resolution')
        print(f'total elements: {self.map.size}')
        print(f'memory required: {self.map.nbytes / 1024 / 1024:.2f}MB')

    def get_sphere_point_tfs(self, n_points=None, radius=None):
        """
        We uniformly sample points on a sphere (more or less), and then build a tf for each of them. The z-axis
        looks towards the sphere centre, x and y are chosen arbitrarily.
        Spiral point algorithm as described in Saff, Kuijlaars, "Distributing many points on a sphere", 1997.
        https://perswww.kuleuven.be/~u0017946/publications/Papers97/art97a-Saff-Kuijlaars-MI/Saff-Kuijlaars-MathIntel97.pdf
        """
        if n_points is None:
            n_points = self.n_sphere_points
        if radius is None:
            radius = self.voxel_res / 2

        theta = np.empty(n_points, dtype=float)  # theta: [0, pi]
        phi = np.empty(n_points, dtype=float)  # phi:  [0, 2pi]
        for i in range(n_points):
            h = -1.0 + 2.0 * i / (n_points - 1.0)
            theta[i] = np.arccos(h)
            if i in [0, n_points - 1]:
                phi[i] = 0
            else:
                phi[i] = (phi[i - 1] + 3.6 / np.sqrt(n_points) * 1.0 / np.sqrt(1 - h ** 2)) % (2 * np.pi)

        # position on surface of sphere
        tfs = np.full((n_points, 4, 4), np.eye(4))
        tfs[:, 0, 3] = radius * np.sin(theta) * np.cos(phi)
        tfs[:, 1, 3] = radius * np.sin(theta) * np.sin(phi)
        tfs[:, 2, 3] = radius * np.cos(theta)

        # z-axis orientated towards origin
        z_axes = -tfs[:, :3, 3]  # negative point coordinates
        z_axes = z_axes / np.linalg.norm(z_axes, axis=-1)[:, np.newaxis]

        # arbitrarily choose x and y axes to form coordinate system
        def get_random_unit_vec():
            vec = np.random.normal(size=3)
            mag = np.linalg.norm(vec)
            if mag <= 1e-5:
                return get_random_unit_vec()
            else:
                return vec / mag

        # find some y orthogonal to z by using cross product with a random unit vector
        tmp_vec = get_random_unit_vec()
        y_axes = np.cross(z_axes, tmp_vec)
        while (np.linalg.norm(y_axes, axis=-1) == 0).any():
            tmp_vec = get_random_unit_vec()
            y_axes = np.cross(z_axes, tmp_vec)
            print('note: repeatedly attempting cross product to get sphere points.')

        # ensure normalized
        y_axes = y_axes / np.linalg.norm(y_axes, axis=-1)[:, np.newaxis]

        # now get x-axis
        x_axes = np.cross(y_axes, z_axes)
        x_axes = x_axes / np.linalg.norm(x_axes, axis=-1)[:, np.newaxis]

        tfs[:, :3, 0] = x_axes
        tfs[:, :3, 1] = y_axes
        tfs[:, :3, 2] = z_axes

        return tfs

    def get_xy_index(self, xy):
        """
        Given EE x or y coordinate, gives the corresponding index in map.
        """
        x_idx = int((xy - self.xy_limits[0]) / self.voxel_res)
        if x_idx < 0:
            raise IndexError(f'xy_idx < 0 -- {xy}')
        if x_idx >= self.n_bins_xy:
            raise IndexError(f'xy idx too large -- {xy}')
        return x_idx

    def get_z_index(self, z):
        """
        Given EE z coordinate, gives the corresponding index in the map.
        """
        z_idx = int((z - self.z_limits[0]) / self.voxel_res)
        if z_idx < 0:
            raise IndexError(f'z idx < 0 -- {z}')
        if z_idx >= self.n_bins_z:
            raise IndexError(f'z idx too large -- {z}')
        return z_idx

    def get_sphere_point_index(self, tf_ee):
        """
        Finds the index of the sphere point with the most similar approach vector.
        I.e., the smallest angle between z-axis of the EE pose and z-axis of the sphere point frame.
        According to Eq. (34) in Zacharias et al.
        """
        query_r_z = tf_ee[:3, 2]
        spheres_r_z = self.sphere_point_tfs[:, :3, 2]

        idx = np.argmin(np.arccos(np.clip(np.dot(spheres_r_z, query_r_z), -1, 1)))
        return idx

    def get_in_plane_rotation_index(self, tf_ee, sphere_point_idx):
        """
        Finds the index for the in-plane rotation according to equations (35)-(38) in Zacharias et al., 2013
        """
        x_axis = tf_ee[:3, 0]
        rot = self.sphere_point_tfs[sphere_point_idx][:3, :3]
        p_xy = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        # project x-axis onto xy plane
        x_hat = rot @ p_xy @ np.linalg.inv(rot) @ x_axis
        beta = np.arccos(np.clip(np.dot(x_hat / np.linalg.norm(x_hat), rot[:3, 0]), -1, 1))
        idx = int(np.floor(beta / self.in_plane_res + 0.5))
        if idx < 0:
            raise IndexError(f'in-plane rot index < 0 -- {np.rad2deg(beta)} deg')
        if idx >= self.n_bins_in_plane:
            raise IndexError(f'in-plane rot index too big -- {np.rad2deg(beta)} deg')
        return idx

    def get_indices_for_ee_pose(self, tf_ee):
        """
        Gives the indices to the element of the reachability map that corresponds to the given end-effector pose.
        May throw an IndexError if the pose is not in the map.

        :param tf_ee: ndarray (4, 4), end-effector pose
        :returns: tuple, indices for the map
        """
        x_idx = self.get_xy_index(tf_ee[0, 3])
        y_idx = self.get_xy_index(tf_ee[1, 3])
        z_idx = self.get_z_index(tf_ee[2, 3])
        s_idx = self.get_sphere_point_index(tf_ee)
        r_idx = self.get_in_plane_rotation_index(tf_ee, s_idx)

        return x_idx, y_idx, z_idx, s_idx, r_idx

    @property
    def shape(self):
        return self.map.shape

    def mark_reachable(self, map_indices):
        self.map[map_indices] = 1

    def is_reachable(self, map_indices):
        return self.map[map_indices]
