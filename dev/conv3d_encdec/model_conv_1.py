import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    
    def __init__(self, depth, non_linearity):
        super(ConvBlock, self).__init__()
        
        self.depth = depth
        self.non_linearity = non_linearity
        
        self.in_channels = 16 * (2 ** int((depth - 2) / 2)) if depth != 1 else 1
        self.out_channels = 16 * (2 ** int((depth - 1) / 2))
        
        # reduces input shape by 4
        self.conv = nn.Conv3d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size = (5, 5, 5),
            stride = (1, 1, 1),
            padding = (1, 1, 1)
        )
        
        self.bn = nn.BatchNorm3d(self.out_channels, affine = True)
        
        self.maxpool = nn.MaxPool3d(
            kernel_size = (3, 3, 3),
            stride = (1, 1, 1)
        )
    
    def forward(self, x):
        x = self.maxpool(
                self.non_linearity(
                    self.bn(
                        self.conv(x)
                    )
                )
            )
        
        return x








class ConvInverseBlock(nn.Module):
    
    def __init__(self, depth, num_layers, original_input_size, non_linearity):
        super(ConvInverseBlock, self).__init__()
        
        self.depth = depth
        self.non_linearity = non_linearity
        
        self.in_channels = 16 * (2 ** int((depth - 1) / 2))
        self.out_channels = 16 * (2 ** int((depth - 2) / 2)) if depth != 1 else 1
        
        upsample_size = original_input_size - 4 * (depth - 1) + 2
        
        self.upsample = nn.Upsample(
            size = upsample_size,
            mode = 'trilinear',
            align_corners = True
        )
        
        self.conv_r = nn.Conv3d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size = (5, 5, 5),
            stride = (1, 1, 1),
            padding = (1, 1, 1)
        )
        
        self.bn = nn.BatchNorm3d(self.out_channels, affine = True)

    
    def forward(self, x):
        x = self.non_linearity(
                self.bn(
                    self.conv_r(
                        self.upsample(x)
                    )
                )
            )
        
        return x







class Encoder(nn.Module):
    
    def __init__(self, num_layers, non_linearity):
        super(Encoder, self).__init__()
        
        self.non_linearity = non_linearity
        
        modules = []
        for i in range(1, num_layers + 1):
            modules.append(ConvBlock(i, self.non_linearity))
        
        self.conv = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.conv(x)
        
        return x






class Decoder(nn.Module):
    def __init__(self, num_layers, original_input_size, non_linearity):
        super(Decoder, self).__init__()
                
        modules = []
        for i in range(num_layers, 0, -1):
            modules.append(ConvInverseBlock(i, num_layers, original_input_size, non_linearity))
        
        self.conv_inv = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.conv_inv(x)
        
        return x