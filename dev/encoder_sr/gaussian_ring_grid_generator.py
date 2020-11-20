import math

import numpy as np
import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def GaussianRing(grid, radius, sigma):
    r = np.sqrt(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2)
    return 1./(sigma * np.sqrt(2 * math.pi)) * np.exp(-1/2 * ((r-radius)/sigma)**2)

class GaussianRingSpaceTimeGrid():
    def __init__(self, n_space_grid, n_time_grid,
                    space_min_x = -5, space_max_x = 5, time_min_t = 0.5, time_max_t = 2.5):

        self.N_SPACE_GRID = n_space_grid
        self.N_TIME_GRID = n_time_grid

        self.SPACE_MIN_X = space_min_x
        self.SPACE_MAX_X = space_max_x
        self.TIME_MIN_T = time_min_t
        self.TIME_MAX_T = time_max_t

        self.space_axis = torch.linspace(space_min_x, space_max_x, n_space_grid).to(DEVICE)
        self.time_axis = torch.linspace(time_min_t, time_max_t, n_time_grid).to(DEVICE)

        self.space_grid = torch.stack(
            torch.meshgrid(self.space_axis, self.space_axis, self.space_axis)
        ).to(DEVICE)
    
        self.space_time_grid = torch.stack(
            torch.meshgrid(self.time_axis, self.space_axis, self.space_axis, self.space_axis)
        ).to(DEVICE)

        self.values = torch.stack(
            [torch.tensor(GaussianRing(self.space_grid.cpu(), i, i/2)).unsqueeze(0) for i in self.time_axis]
        ).to(DEVICE)
