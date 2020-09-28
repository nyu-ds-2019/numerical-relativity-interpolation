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

        assert self.input_np.shape[0] == self.target_np.shape[0]

    def __len__(self):
        return self.input_np.shape[0]

    def __getitem__(self, index):
        input_1 = self.input_np[index, 0, :, :, :]
        input_2 = self.input_np[index, 1, :, :, :]
        target = self.target_np[index, 0, :, :, :]

        return torch.tensor(input_1).unsqueeze(0).float(), torch.tensor(input_2).unsqueeze(0).float(), torch.tensor(target).unsqueeze(0).float()

