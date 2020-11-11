import torch
import torch.nn as nn
from models.swish import Swish
from models.ndInterp import NDLinearInterpolation

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

class UNet3D(nn.Module):
	def __init__(self,in_channel,out_channel,n_pairs,padding_mode='replicate'):
		super(UNet3D,self).__init__()	
		self.Res1 = ResBlock(in_channel,out_channel)
		self.n_pairs = n_pairs
		self.Down_array = nn.Sequential(*[SamplingBlock(2**i*out_channel,2**(i+1)*out_channel,'Down') for i in range(n_pairs)])
		self.Up_array = nn.Sequential(SamplingBlock(2**n_pairs*out_channel,2**(n_pairs-1)*out_channel,'Up'),*[SamplingBlock(2**i*out_channel,2**(i-2)*out_channel,'Up') for i in range(n_pairs,1,-1)])
		
	def forward(self,x):
		y = self.Res1(x)
		temp_array = []
		for i in range(self.n_pairs):
			temp_array.append(y)
			y = self.Down_array[i](y)
		temp_array.reverse()
		for i in range(self.n_pairs):
			y = self.Up_array[i](y)
			y = torch.cat((y,temp_array[i]),1)
		return y
			
class meshfreeSR(nn.Module):
	def __init__(self,in_channel,out_channel,n_layers,ndim,linear_size=32):
		super(meshfreeSR,self).__init__()
		self.contextUNet = UNet3D(in_channel,out_channel,n_layers)
		self.predict_linear1 = LinearBlock(2*out_channel+ndim,linear_size)
		self.output = nn.Linear(linear_size,in_channel)

	def forward(self,context,loc,xmin,xmax):
		context_grid = self.contextUNet(context).permute(1,0,2,3,4)
		context_vector =  NDLinearInterpolation(context_grid,loc,xmin,xmax)
		combine = torch.cat((context_vector,loc),axis=1)
		combine = self.predict_linear1(combine)
		combine = self.output(combine)
		return combine
