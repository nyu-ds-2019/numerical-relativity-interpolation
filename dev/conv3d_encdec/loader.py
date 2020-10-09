import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import h5py


class SingleChannelDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __getitem__(self, index):
        frame_path = os.path.join(self.data_dir, f'frame_{index}', 'frame_data.hdf5')
        frame_file = h5py.File(frame_path)
        
        input1 = frame_file['input1']
        input2 = frame_file['input2']
        target = frame_file['target']

        return torch.tensor(input_1).unsqueeze(0).float(), torch.tensor(input_2).unsqueeze(0).float(), torch.tensor(target).unsqueeze(0).float()

