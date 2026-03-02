import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from rm4d.robots import Simulator


class BasePosGrid:
    def __init__(self, x_limits=None, y_limits=None, n_bins_x=80, n_bins_y=80):
        if x_limits is None:
            x_limits = [-2.00, 2.00]
        if y_limits is None:
            y_limits = [-2.00, 2.00]

        self.x_limits = x_limits
        self.y_limits = y_limits

        self.n_bins_x = n_bins_x
        self.n_bins_y = n_bins_y

        self.x_res = (self.x_limits[1] - self.x_limits[0])/self.n_bins_x
        self.y_res = (self.y_limits[1] - self.y_limits[0])/self.n_bins_y

        self.grid = np.zeros(
            shape=(self.n_bins_x, self.n_bins_y),
            dtype=float  # due to interpolation we may not have uint anymore
        )

        self.empty = True

    def add_base_positions(self, base_pos):
        """
        Gets an array of points with corresponding reachability scores and fills them into the grid.

        :param base_pos: (n, 3), where [:, :2] are (x,y) coordinates and [:, 2] is the score
        """
        for point in base_pos:
            self._add_single_point(point)

        self.empty = False

    def _add_single_point(self, base_pos):
        x, y, score = base_pos
        if score == 0:
            return

        # calculate indices, minus 0.5 to align for bin centre (i.e., an idx of 1 is right at the border between bins
        # 0 and 1, we want that to be 50% in bin 0 and 50% in bin 1, so we call it a 0.5)
        x_idx = (x - self.x_limits[0]) / self.x_res - 0.5
        y_idx = (y - self.y_limits[0]) / self.y_res - 0.5

        x_idx_low, x_idx_high = int(np.floor(x_idx)), int(np.ceil(x_idx))
        y_idx_low, y_idx_high = int(np.floor(y_idx)), int(np.ceil(y_idx))

        # only add a point if all fields are within bounds
        if x_idx_low < 0 or y_idx_low < 0 or x_idx_high >= self.n_bins_x or y_idx_high >= self.n_bins_y:
            return

        # weights are inverse to the distance to index
        x_weight = 1 - (x_idx - x_idx_low)
        y_weight = 1 - (y_idx - y_idx_low)

        # add scores
        self.grid[x_idx_low, y_idx_low] += x_weight * y_weight * score
        self.grid[x_idx_low, y_idx_high] += x_weight * (1-y_weight) * score
        self.grid[x_idx_high, y_idx_low] += (1-x_weight) * y_weight * score
        self.grid[x_idx_high, y_idx_high] += (1-x_weight) * (1-y_weight) * score

    def intersect(self, other_grid):
        assert self.x_limits == other_grid.x_limits
        assert self.y_limits == other_grid.y_limits
        assert self.n_bins_x == other_grid.n_bins_x
        assert self.n_bins_y == other_grid.n_bins_y

        if self.empty or other_grid.empty:
            raise Warning('Intersect: One of the grids is empty, intersection will be zero.')

        self.grid = np.minimum(self.grid, other_grid.grid)

    def union(self, other_grid):
        assert self.x_limits == other_grid.x_limits
        assert self.y_limits == other_grid.y_limits
        assert self.n_bins_x == other_grid.n_bins_x
        assert self.n_bins_y == other_grid.n_bins_y

        self.grid += other_grid.grid

    def get_best_pos(self):
    #     x_idx, y_idx = np.unravel_index(np.argmax(self.grid, axis=None), self.grid.shape)
    #     x = self.x_limits[0] + (x_idx + 0.5) * self.x_res
    #     y = self.y_limits[0] + (y_idx + 0.5) * self.y_res
    #     return x, y
        best_list = self.ranked_positions(k=1, rel_tol=0.0)
        return best_list[0] if best_list else (None, None)

    def idx_to_xy(self, x_idx: int, y_idx: int):
        """Devuelve coordenadas físicas del centro de celda para (x_idx, y_idx)."""
        x = self.x_limits[0] + (x_idx + 0.5) * self.x_res
        y = self.y_limits[0] + (y_idx + 0.5) * self.y_res
        return x, y

    def ranked_positions(self, k=10, rel_tol=0.02, score_fn=None, alpha=0.7):
        """
        Devuelve hasta k posiciones (x,y) con mayor puntuación total.
        - rel_tol: acepta celdas con grid >= max*(1-rel_tol)
        - score_fn(x,y) -> [0..1] añade criterios (centrado, distancia, etc.)
        - alpha: peso de la grid frente al score_fn (0..1)
        """
        if np.all(self.grid == 0):
            return []
        max_val = float(np.max(self.grid))
        cand_idx = np.argwhere(self.grid >= max_val * (1.0 - float(rel_tol)))
        ranked = []
        for (i, j) in cand_idx:
            base = float(self.grid[i, j]) / max_val if max_val > 0 else 0.0
            x, y = self.idx_to_xy(int(i), int(j))
            extra = float(score_fn(x, y)) if score_fn is not None else 0.0
            total = alpha * base + (1.0 - alpha) * extra
            ranked.append((total, x, y))
        ranked.sort(key=lambda t: t[0], reverse=True)
        return [(x, y) for (_, x, y) in ranked[:k]]

    def show_as_img(self):
        img = self.grid / np.max(self.grid)
        plt.imshow(img)
        plt.show()

    def visualize_in_sim(self, sim: Simulator, skip_zero=True):
        max_val = np.max(self.grid)

        for i in range(self.n_bins_x):
            for j in range(self.n_bins_y):
                val = self.grid[i, j]
                if skip_zero and val == 0:
                    continue
                colour = [1.0 - val/max_val, val/max_val, 0.0, 1.0]
                # visual_shape = sim.bullet_client.createVisualShape(p.GEOM_SPHERE, radius=0.025, rgbaColor=colour)
                visual_shape = sim.bullet_client.createVisualShape(p.GEOM_BOX,
                                                                   halfExtents=[self.x_res/2, self.y_res/2, 0.005],
                                                                   rgbaColor=colour)
                x = self.x_limits[0] + (i + 0.5) * self.x_res
                y = self.y_limits[0] + (j + 0.5) * self.y_res
                sim.bullet_client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, basePosition=[x, y, 0])
