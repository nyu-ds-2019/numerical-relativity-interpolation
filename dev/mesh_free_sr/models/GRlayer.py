import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from .swish import Swish

torch_diff = lambda y, x: grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True,
                               allow_unused=True)[0]

def get_metric(ADM,coord):
# Dimension goes like (batch,channel). Channel order goes as (h11...h33,alpha,beta...)
	dADM = torch.stack([torch_diff(ADM[:,i],coord) for i in range(10)],-1) # (b,4,10)
	ddADM = torch.stack([torch_diff(dADM[:,j,i],coord) for j in range(4) for i in range(10)],-1).reshape(-1,4,4,10)
	
	metric = torch.zeros((ADM.shape[0],4,4)) # (b,u,v)
	dmetric = torch.zeros((ADM.shape[0],4,4,4)) # (b, x, u, v)
	ddmetric = torch.zeros((ADM.shape[0],4,4,4,4)) # (b, x, y, u, v)
	gamma_ij = torch.stack((ADM[:,[0,1,2]],ADM[:,[1,3,4]],ADM[:,[2,4,5]]),dim=-2)
	gamma_ij_inv = torch.inverse(gamma_ij)
	g00 = -ADM[:,6]**2+torch.einsum('bk,bkl,bl->b',ADM[:,7:],gamma_ij,ADM[:,7:])
	dg00 = torch_diff(g00,coord) # (b,4)
	ddg00 = torch.stack([torch_diff(dg00[:,i],coord) for i in range(4)],-1) # (b,4,4)
	
	metric[:,0,0] = g00
	metric[:,1:,1:] = gamma_ij
	metric[:,0,1:] = ADM[:,7:]
	metric[:,1:,0] = ADM[:,7:]
	
	dmetric[...,0,0] = dg00
	dmetric[...,1:,1:] = torch.stack((dADM[...,[0,1,2]],dADM[...,[1,3,4]],dADM[...,[2,4,5]]),dim=-2)
	dmetric[...,0,1:] = dADM[...,7:]
	dmetric[...,1:,0] = dADM[...,7:]
	
	ddmetric[...,0,0] = ddg00 
	ddmetric[...,1:,1:] = torch.stack((ddADM[...,[0,1,2]],ddADM[...,[1,3,4]],ddADM[...,[2,4,5]]),dim=-2)
	ddmetric[...,0,1:] = ddADM[...,7:]
	ddmetric[...,1:,0] = ddADM[...,7:]

	metric_inv = torch.inverse(metric)	
	dmetric_inv = torch.stack([-metric_inv@dmetric[:,i]@metric_inv for i in range(4)],dim=1)

	return metric,dmetric,ddmetric,metric_inv,dmetric_inv


def Christoffel_array(gmu,dgmu,gmu_inv):
# gmu (b,u,v)
# dgmu (b,i,u,v)
	gmu_inv = torch.inverse(gmu)
	chris = torch.einsum('bal,bmln->bamn',gmu_inv,dgmu)+torch.einsum('bal,bnlm->bamn',gmu_inv,dgmu)-torch.einsum('bal,blmn->bamn',gmu_inv,dgmu)
	return chris/2	

def d_Christoffel_array(gmu,dgmu,ddgmu,gmu_inv,dgmu_inv):
# gmu (b,u,v)
# dgmu (b,i,u,v)
# ddgmu(b,i,j,u,v)
	gmu_inv = torch.inverse(gmu)
	chris1 = torch.einsum('bmln,bilj->bijmn',dgmu,dgmu_inv)+torch.einsum('bnlm,bilj->bijmn',dgmu,dgmu_inv)-torch.einsum('blmn,bilj->bijmn',dgmu,dgmu_inv)
	chris2 = torch.einsum('bjl,bimln->bijmn',gmu_inv,ddgmu)+torch.einsum('bjl,binlm->bijmn',gmu_inv,ddgmu)-torch.einsum('bjl,bilmn->bijmn',gmu_inv,ddgmu)
	return (chris1+chris2)/2

def RiemannTensor(ADM,coord):
	gmu,dgmu,ddgmu,gmu_inv,dgmu_inv = get_metric(ADM,coord)
	chris = Christoffel_array(gmu,dgmu,gmu_inv)
	d_chris = d_Christoffel_array(gmu,dgmu,ddgmu,gmu_inv,dgmu_inv)
	deriv_term = d_chris-d_chris.permute(0,4,2,3,1)
	prod_term = torch.einsum('balm,blcn->bacmn',chris,chris)-torch.einsum('baln,blcm->bacmn',chris,chris)
	return deriv_term-prod_term

def EinsteinTensor(gmu,gmu_inv,RiemannTensor):
	Ricci_tensor = torch.einsum('baman->bmn',RiemannTensor)
	Ricci_scalar = torch.einsum('bmn,bmn->b',gmu_inv,Ricci_tensor)
	output = Ricci_tensor - gmu*Ricci_scalar[:,None,None]/2	
	return output

coord = torch.rand(1000,4)
coord.requires_grad = True
L1 = nn.Linear(4,10)
Relu1 = Swish()
L2 = nn.Linear(10,10)
Relu2 = Swish()
ADM = Relu2(L2(Relu1(L1(coord))))
metric,dmetric,ddmetric,metric_inv,dmetrc_inv = get_metric(ADM,coord)
RT = RiemannTensor(ADM,coord)
ET = EinsteinTensor(metric,metric_inv,RT)

