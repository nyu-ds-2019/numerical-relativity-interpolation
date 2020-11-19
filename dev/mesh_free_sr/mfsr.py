import time
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from .models.meshfreeSR import meshfreeSR
# from .models.meshfreeSR import UNet3D
# from .models.ndInterp import NDLinearInterpolation

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
LR = 1e-3
NUM_EPOCHS = 1000
SPACE_GRID_MIN = -5
SPACE_GRID_MAX = 5
TIME_GRID_MIN = 0.5
TIME_GRID_MAX = 2.5

T_SPACE_TIME_MIN = torch.tensor([TIME_GRID_MIN, SPACE_GRID_MIN, SPACE_GRID_MIN, SPACE_GRID_MIN]).float().to(DEVICE)
T_SPACE_TIME_MAX = torch.tensor([TIME_GRID_MAX, SPACE_GRID_MAX, SPACE_GRID_MAX, SPACE_GRID_MAX]).float().to(DEVICE)


def GaussianRing(grid, radius, sigma):
	r = np.sqrt(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2)
	return 1./(sigma * np.sqrt(2 * math.pi)) * np.exp(-1/2 * ((r-radius)/sigma)**2)


class SpaceTimeContext():
    def __init__(self, n_space_grid, n_time_grid, n_train_time_grid,
                    space_min_x = -5, space_max_x = 5, time_min_t = 0.5, time_max_t = 2.5):

        self.N_SPACE_GRID = n_space_grid
        self.N_TIME_GRID = n_time_grid
        self.N_TRAIN_TIME_GRID = n_train_time_grid

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

        self.train_context = torch.stack(
            [torch.tensor(GaussianRing(self.space_grid.cpu(), i, i/2)).unsqueeze(0) for i in self.time_axis]
        ).to(DEVICE)

        self.train_loc = self.generate_train_loc(self.N_TRAIN_TIME_GRID)
        self.train_value = self.generate_train_value(self.N_TRAIN_TIME_GRID)
    

    # returns space time grid locations of training points
    def generate_train_loc(self, n_train_time_grid = 20):
        train_loc = torch.cat(
            (torch.ones((self.N_SPACE_GRID ** 3, 1)).to(DEVICE) * self.TIME_MIN_T, self.space_grid.reshape(3,-1).T),
            axis = 1
        )

        # iterate over train times
        for i in np.linspace(self.TIME_MIN_T, self.TIME_MAX_T, num = n_train_time_grid):
            train_loc = torch.cat(
                (
                    train_loc, 
                    torch.cat(
                        (
                            torch.ones((self.N_SPACE_GRID ** 3, 1)).to(DEVICE) * i, 
                            self.space_grid.reshape(3, -1).T
                        ),
                        axis = 1
                    ).to(DEVICE)
                ),
                axis=0
            )
        
        train_loc *= 0.99
        return train_loc

    # returns space time grid values of the Gaussian Ring
    def generate_train_value(self, n_train_time_grid = 20):
        train_value = GaussianRing(
            self.space_grid.cpu(), 
            self.TIME_MIN_T, 
            self.TIME_MIN_T / 2
        ).reshape(-1).unsqueeze(-1).to(DEVICE)


        for i in np.linspace(self.TIME_MIN_T, self.TIME_MAX_T, num = n_train_time_grid):
            train_value = torch.cat(
                (
                    train_value,
                    GaussianRing(
                        self.space_grid.cpu(), 
                        i, 
                        i / 2
                    ).reshape(-1).unsqueeze(-1).to(DEVICE)
                ),
                axis = 0
            )
        
        train_value = train_value[:, 0]
        return train_value


class GaussianMFSR():
    def __init__(self, spaceTimeContext, in_channels = 1, out_channels = 16, n_layers = 3, 
                    n_dim = 4, linear_size = 256, num_epochs = 100, lr = 1e-3):
        self.spaceTimeContext = spaceTimeContext
        self.model = meshfreeSR(
            in_channels, 
            out_channels, 
            n_layers, 
            n_dim, 
            linear_size
        ).to(DEVICE)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.criterion = nn.MSELoss()
        self.train_loss_list = []
    
    # predict train_value from train_loc
    def train(self):
        self.model.train()

        for i in range(self.num_epochs):
            self.optimizer.zero_grad()
            
            output = self.model(
                self.spaceTimeContext.train_context, 
                self.spaceTimeContext.train_loc, 
                T_SPACE_TIME_MIN, 
                T_SPACE_TIME_MAX
            )
            loss = self.criterion(
                output.T[0],
                self.spaceTimeContext.train_value
            )

            print(f'epoch = {i} loss = {loss.item()}')

            self.train_loss_list.append(loss.item())
            loss.backward()
            self.optimizer.step()
    

    def test(self, n_test_space_grid, n_test_time_grid,
                space_min_x = -5, space_max_x = 5, time_min_t = 0.5, time_max_t = 2.5):
        self.model.eval()
        
        train_context = self.spaceTimeContext.train_context

        recon_axis = torch.linspace(space_min_x, space_max_x, n_test_space_grid).to(DEVICE)
        recon_time = torch.linspace(time_min_t, time_max_t, n_test_time_grid).to(DEVICE)
        recon_ngrid = recon_axis.shape[0]
        recon_step = recon_time.shape[0]
        
        recon_point = torch.stack(
            torch.meshgrid(recon_time, recon_axis, recon_axis, recon_axis)
        ).reshape(4, -1).T.to(DEVICE)

        recon_result = self.model(train_context, recon_point, T_SPACE_TIME_MIN, T_SPACE_TIME_MAX)
        true_grid = torch.stack(torch.meshgrid(recon_axis, recon_axis, recon_axis))

        start_timestamp = recon_time[0].item()
        true_result = GaussianRing(true_grid, start_timestamp, start_timestamp/2).to(DEVICE).reshape(1, -1).T
        for i in range(1, recon_step):
            true_result = torch.cat(
                (
                    true_result,
                    GaussianRing(true_grid, recon_time[i], recon_time[i]/2).to(DEVICE).reshape(1,-1).T
                ),
                axis = 0
            )
        
        print('Frobenius norm = ', torch.linalg.norm(recon_result - true_result).item())
        
        return recon_result, true_result
