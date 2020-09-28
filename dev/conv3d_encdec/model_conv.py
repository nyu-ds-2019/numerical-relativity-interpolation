import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels = 1,
            out_channels = 8,
            kernel_size = (3, 3, 3),
            stride = (1, 1, 1),
            padding = (1, 1, 1)
        )
        
        self.maxpool1 = nn.MaxPool3d(
            kernel_size = (2, 2, 2),
            stride = (2, 2, 2)
        )
        
        self.conv2 = nn.Conv3d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = (3, 3, 3),
            stride = (1, 1, 1),
            padding = (1, 1, 1)
        )
        
        self.maxpool2 = nn.MaxPool3d(
            kernel_size = (3, 3, 3),
            stride = (2, 2, 2),
            padding = (1, 1, 1)
        )
        
        self.conv3 = nn.Conv3d(
            in_channels = 16,
            out_channels = 32,
            kernel_size = (3, 3, 3),
            stride = (1, 1, 1),
            padding = (1, 1, 1)
        )
        
        self.maxpool3 = nn.MaxPool3d(
            kernel_size = (2, 2, 2),
            stride = (2, 2, 2)
        )
        
        self.bn1 = nn.BatchNorm3d(8, affine = True)
        self.bn2 = nn.BatchNorm3d(16, affine = True)
        self.bn3 = nn.BatchNorm3d(32, affine = True)
        
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.maxpool3(x)
        
        return x



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.upsample3 = nn.Upsample(
            scale_factor = 2,
            mode = 'nearest'
        )
        
        self.conv3_r = nn.Conv3d(
            in_channels = 32,
            out_channels = 16,
            kernel_size = (3, 3, 3),
            stride = (1, 1, 1),
            padding = (1, 1, 1)
        )
        
        self.upsample2 = nn.Upsample(
            scale_factor = 2,
            mode = 'nearest'
        )
        
        self.conv2_r = nn.Conv3d(
            in_channels = 16,
            out_channels = 8,
            kernel_size = (3, 3, 3),
            stride = (1, 1, 1),
            padding = (1, 1, 1)
        )
        
        self.upsample1 = nn.Upsample(
            scale_factor = 2,
            mode = 'nearest'
        )
        
        self.conv1_r = nn.Conv3d(
            in_channels = 8,
            out_channels = 1,
            kernel_size = (3, 3, 3),
            stride = (1, 1, 1),
            padding = (1, 1, 1)
        )
        
        self.bn3_r = nn.BatchNorm3d(16, affine = True)
        self.bn2_r = nn.BatchNorm3d(8, affine = True)
        self.bn1_r = nn.BatchNorm3d(1, affine = True)
    
    def forward(self, x):
        x = self.upsample3(x)
        x = F.relu(self.conv3_r(x))
        x = self.bn3_r(x)
        x = self.upsample2(x)
        x = F.relu(self.conv2_r(x))
        x = self.bn2_r(x)
        x = self.upsample1(x)
        x = F.relu(self.conv1_r(x))
        x = self.bn1_r(x)
        
        return x