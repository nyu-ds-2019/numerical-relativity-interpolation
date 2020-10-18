import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import h5py


class SingleChannelDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        file = h5py.File(self.file_path, 'r')

        self.input_np = np.array(file['Train']['input'])
        self.target_np = np.array(file['Train']['target'])
        
        self.x_mean = np.mean(self.input_np, axis=tuple(range(self.input_np.ndim-3)), keepdims=True)
        self.x_std = np.std(self.input_np, axis=tuple(range(self.input_np.ndim-3)), keepdims=True)
        
        self.scaled_input = np.divide(self.input_np - self.x_mean, self.x_std, out=np.zeros_like(self.input_np), where=self.x_std!=0)
        
        self.scaled_target = np.divide(self.target_np - self.x_mean, self.x_std, out=np.zeros_like(self.target_np), where=self.x_std!=0)

        assert self.input_np.shape[0] == self.target_np.shape[0]

    def __len__(self):
        return self.input_np.shape[0]

    def __getitem__(self, index):
        input_1 = self.scaled_input[index, 0, :, :, :]
        input_2 = self.scaled_input[index, 1, :, :, :]
        target = self.scaled_target[index, 0, :, :, :]

        return torch.tensor(input_1).unsqueeze(0).float(), torch.tensor(input_2).unsqueeze(0).float(), torch.tensor(target).unsqueeze(0).float()

