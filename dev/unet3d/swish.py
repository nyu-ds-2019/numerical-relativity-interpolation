import torch
import torch.nn

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
#         return torch.nn.LeakyReLU(x)
