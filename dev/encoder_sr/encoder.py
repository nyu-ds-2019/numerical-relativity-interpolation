import torch
import torch.nn as nn
# from models.swish import Swish
# from models.ndInterp import NDLinearInterpolation

class LinearBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(LinearBlock,self).__init__()
        self.Linear = nn.Linear(in_channel,out_channel)
    #		self.Act = Swish()
        self.Act_3c = nn.ReLU()
    def forward(self,x):
        x = self.Linear(x)
        x = self.Act_3c(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,padding_mode='replicate'):
        super(ConvBlock,self).__init__()
        self.Conv = nn.Conv3d(in_channel,out_channel,kernel_size,padding=kernel_size//2,padding_mode=padding_mode)
        self.BatchNorm = nn.BatchNorm3d(out_channel)
    #		self.Act = Swish()
        self.Act_3c = nn.ReLU()

    def forward(self,x):
        x = self.Conv(x)
        x = self.BatchNorm(x)
        x = self.Act_3c(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,in_channel,out_channel,padding_mode='replicate'):
        super(ResBlock,self).__init__()
        self.shortcut = nn.Conv3d(in_channel,out_channel,1)
        self.Conv_1 = ConvBlock(in_channel,in_channel,1,padding_mode=padding_mode)
        self.Conv_2 = ConvBlock(in_channel,in_channel,3,padding_mode=padding_mode)
        self.Conv_3a = nn.Conv3d(in_channel,out_channel,1)
        self.BatchNorm_3b = nn.BatchNorm3d(out_channel)
    #		self.Act_3c = Swish()
        self.Act_3c = nn.ReLU()

    def forward(self,x):
        y = self.Conv_1(x)
        y = self.Conv_2(y)
        y = self.Conv_3a(y)
        y = self.BatchNorm_3b(y)
        y = self.shortcut(x)+y
        y = self.Act_3c(y)
        return y 

class SamplingBlock(nn.Module):	
    def __init__(self,in_channel,out_channel,mode,padding_mode='replicate'):
        super(SamplingBlock,self).__init__()	
        if mode == 'Up':
            self.conv = nn.Sequential(*[nn.Upsample(scale_factor=2),ResBlock(in_channel,out_channel)])
        if mode == 'Down':
            self.conv = nn.Sequential(*[ResBlock(in_channel,out_channel),nn.MaxPool3d(2)])

    def forward(self,x):
            return self.conv(x)

class Encoder3D(nn.Module):
    def __init__(self,in_channel,out_channel,n_pairs,padding_mode='replicate'):
        super(Encoder3D,self).__init__()	
        self.Res1 = ResBlock(in_channel,out_channel)
        self.n_pairs = n_pairs
        self.Down_array = nn.Sequential(*[SamplingBlock(2**i*out_channel,2**(i+1)*out_channel,'Down') for i in range(n_pairs)])

    def forward(self,x):
        y = self.Res1(x)
        for i in range(self.n_pairs):
            y = self.Down_array[i](y)
        y = y.permute(0, 2, 3, 4, 1).squeeze(1).squeeze(1).squeeze(1)
        return y

class SR(nn.Module):
    def __init__(self,in_channel,out_channel,n_layers):
        super(SR,self).__init__()
        self.contextEncoder = Encoder3D(in_channel,out_channel,n_layers)
        self.output_layer = nn.Sequential(nn.Linear(512*5 + 4*3,512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, context, vecs):
        outputs = []
        for i in range(context.shape[0]):
#             print(context[i].shape)
            outputs.append(self.contextEncoder(context[i].unsqueeze(0)))
#         print(outputs[0].shape)
        combine = torch.cat(outputs, dim=1)
        
        outs = []
        for i in range(vecs.shape[0]):
            vec = vecs[i]
            combine2 = torch.cat([combine, vec], dim=1)
            combine2 = torch.cat([combine2, vec], dim=1)
            combine2 = torch.cat([combine2, vec], dim=1)
            output = self.output_layer(combine2)
            outs.append(output)
        output = torch.cat(outs, dim=0)
        return output
