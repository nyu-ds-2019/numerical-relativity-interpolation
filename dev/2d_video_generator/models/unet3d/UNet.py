import torch
import torch.nn as nn
import numpy as np

from .swish import Swish
from .conv import ConvBlock1D,narrow_like,ConvBlock3D,narrow_like3D

class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_0l = ConvBlock1D(in_channels, 64, seq='CAC')
        self.down_0l = ConvBlock1D(64, seq='BADBA')
        self.conv_1l = ConvBlock1D(64, seq='CBAC')
        self.down_1l = ConvBlock1D(64, seq='BADBA')

        self.conv_2c = ConvBlock1D(64, seq='CBAC')

        self.up_1r = ConvBlock1D(64, seq='BAUBA')
        self.conv_1r = ConvBlock1D(128, 64, seq='CBAC')
        self.up_0r = ConvBlock1D(64, seq='BAUBA')
        self.conv_0r = ConvBlock1D(128, out_channels, seq='CAC')
        self.pad_size = self.check_pad_size()

    def forward(self, x,pad=True,pad_mode='reflective'):
        if pad==True:
            if pad_mode=='reflective': x = torch.cat((torch.flip(x[:,:,:self.pad_size],[2]),x,torch.flip(x[:,:,-self.pad_size:],[2])),2)
            if pad_mode=='periodic': x = torch.cat((x[:,:,-self.pad_size:],x,x[:,:,:self.pad_size]),2)
            if pad_mode=='zero':
                pad = x[:,:,:self.pad_size].zero_()
                x = torch.cat((pad,x,pad),2)
    
        y0 = self.conv_0l(x)
        x = self.down_0l(y0)
    
        y1 = self.conv_1l(x)
        x = self.down_1l(y1)
    
        x = self.conv_2c(x)
    
        x = self.up_1r(x)
        y1 = narrow_like(y1, x)
        x = torch.cat([y1, x], dim=1)
        del y1
        x = self.conv_1r(x)
    
        x = self.up_0r(x)
        y0 = narrow_like(y0, x) 
        x = torch.cat([y0, x], dim=1)
        del y0
        x = self.conv_0r(x)

        return x

    def check_pad_size(self):
        check_data = torch.from_numpy(np.random.uniform(size=(2,self.conv_0l.in_channels,1000))).float()
        output = self(check_data,pad=False)
        pad_size = (check_data.shape[2]-output.shape[2])/2.
        return int(pad_size)
        
class MLP(nn.Module):
    def __init__(self, input_size,output_size,hidden_size):
        super().__init__()
        self.output = nn.Sequential(
    nn.Linear(input_size, hidden_size), Swish(),
    nn.Linear(hidden_size, hidden_size), Swish(),
    nn.Linear(hidden_size, hidden_size), Swish(),
    nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.output(x)

class RUNet1D_cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.unet = UNet1D(2*in_channels, out_channels)
        self.hidden = UNet1D(2*in_channels, out_channels)

    def forward(self, x, hidden, pad=True): 
        x = torch.cat((x,hidden),1)
        output = self.unet(x,pad_mode='periodic')
        hidden = self.hidden(x,pad_mode='periodic')
        return output,hidden
    
    def initHidden(self,shape):
        return torch.zeros(shape,requires_grad=True)


class RUNet1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unet = UNet1D(in_channels, out_channels)
        self.hidden = UNet1D(in_channels, out_channels)

    def forward(self, x, hidden, pad=True): 
        output = self.unet(x+hidden,pad_mode='periodic')
        hidden = self.hidden(x+hidden,pad_mode='periodic')
        return output,hidden
    
    def initHidden(self,shape):
        return torch.zeros(shape,requires_grad=True)

class RUNet1D_MLP(nn.Module):
    def __init__(self, in_channels, out_channels,datasize):
        super().__init__()

        self.unet = UNet1D(in_channels, out_channels)
        self.hidden = MLP(datasize,datasize,datasize)

    def forward(self, x, hidden, pad=True):
        x = x + hidden.reshape(x.shape)
        output = self.unet(x)
        hidden = self.hidden(x.flatten(1))
        return output,hidden
    
    def initHidden(self,shape):
        return torch.zeros(shape,requires_grad=True)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_0l = ConvBlock3D(in_channels, 64, seq='CAC')
        self.down_0l = ConvBlock3D(64, seq='BADBA')
        self.conv_1l = ConvBlock3D(64, seq='CBAC')
        self.down_1l = ConvBlock3D(64, seq='BADBA')

        self.conv_2c = ConvBlock3D(64, seq='CBAC')

        self.up_1r = ConvBlock3D(64, seq='BAUBA')
        self.conv_1r = ConvBlock3D(128, 64, seq='CBAC')
        self.up_0r = ConvBlock3D(64, seq='BAUBA')
        self.conv_0r = ConvBlock3D(128, out_channels, seq='CAC')

    def forward(self, x):
        y0 = self.conv_0l(x)
        x = self.down_0l(y0)

        y1 = self.conv_1l(x)
        x = self.down_1l(y1)

        x = self.conv_2c(x)

        x = self.up_1r(x)
        y1 = narrow_like3D(y1, x)
        x = torch.cat([y1, x], dim=1)
        del y1
        x = self.conv_1r(x)

        x = self.up_0r(x)
        y0 = narrow_like3D(y0, x)   
        x = torch.cat([y0, x], dim=1)
        del y0
        x = self.conv_0r(x)

        return x

    def check_pad_size(self):
        check_data = torch.empty(2,self.conv_0l.in_channels,100,100,100)
        output = self(check_data)
        pad_size = (check_data.shape[2]-output.shape[2])/2.
        return int(pad_size)
    
