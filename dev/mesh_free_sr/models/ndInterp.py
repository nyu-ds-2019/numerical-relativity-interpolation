import numpy as np
import torch
from scipy.interpolate import interpn
from itertools import product

def Interp_context(grid,value,points):
	n_channel = value.shape[1]
	value = value.permute(1,0,2,3,4)
	output = torch.tensor(interpn(grid,value[0].numpy(),points)).unsqueeze(0)
	for i in range(1,n_channel):
		output = torch.cat((output,torch.tensor(interpn(grid,value[i].numpy(),points)).unsqueeze(0)),axis=0) 
	return output

def NDLinearInterpolation(data,points,xmin,xmax,device='cuda'):

	# data (channel,coord), points (npoint,coord)
	data = data.unsqueeze(-1)
	points = points.unsqueeze(0)
	points = torch.max(torch.min(points,xmax),xmin).to(device)
	ndim = torch.tensor(data.shape[1:-1]).to(device)
	npoint = points.shape[1]

	binary_table = torch.tensor(list(product([0,1],repeat=len(ndim)))).to(device)
	cubesize = (xmax-xmin)/(ndim-1)
	distance = ((points-xmin)/(xmax-xmin))
	lower_indices = torch.floor(distance*(ndim-1)).to(device)

	weight = distance*(ndim-1)-lower_indices
	weight = torch.cat((binary_table.unsqueeze(1).repeat_interleave(npoint,axis=1).float(),weight))
	weight = torch.abs((1-weight[:-1])-weight[-1])
	weight = weight.prod(2).unsqueeze(-1).to(device)

	corner_indices = torch.cat((binary_table.unsqueeze(1).repeat_interleave(npoint,axis=1),lower_indices.long()))
	corner_indices = corner_indices[:-1]+corner_indices[-1]	
	corner_value = data.permute(1,2,3,4,0,5)[corner_indices.permute(2,0,1).chunk(chunks=4,dim=0)].to(device)

	output = (corner_value*weight.unsqueeze(-1).unsqueeze(0)).sum(1)[0,:,:,0]

	return output
#	return corner_value,weight,corner_indices,output
