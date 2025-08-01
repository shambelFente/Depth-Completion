# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
# inverse distance weighting
import numpy as np
import matplotlib.pyplot as plot
import torch
import copy
import torch.nn as nn
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
import numpy as np

def inverse_distance_weighted(k1, power):
    kernel = np.empty((k1,k1))
    center = np.array((0,0))
    kernel = []
    weights = []
    
    for value in power:
        kernel = []
        for x_axis in np.arange(-(k1-1)/2, (k1+1)/2, 1):
            dist = []
            for y_axis in np.arange((k1-1)/2,-(k1+1)/2, -1):
                d = np.linalg.norm(center - np.array((x_axis, y_axis)))
                if d != 0:
                    weight = 1/(d**value)
                else:
                    weight = 0
                dist.append(weight)
            kernel.append(dist)
            
        # changing to tensor and reshaping it
        kernel_tensor = torch.tensor(np.array(kernel).astype(dtype = np.float32))
        kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(1)
    
        # adding the tensor to a list
        weights.append(kernel_tensor)
        
    weights = torch.stack(weights).squeeze(1) # removing the extra channel
    return weights




#inverse_distance_weight parameters
power = np.linspace(2.0, 4.0, num=16)#2,4
weights = inverse_distance_weighted(31, power) 


class inverseDistWeight(nn.Module):
    def __init__(self):
        super().__init__()
        # convolution layer for Inverse distance weighted interpolation
        self.inverse_dist_weight = nn.Conv2d(
            1,
            16,  
            kernel_size=31,
            padding = 'same',
            bias=False) 

        self.inverse_dist_weight.weight = nn.Parameter(
                                 data = weights, 
                                 requires_grad = False)
        
    def forward(self, x, mask):
        idw_f = self.inverse_dist_weight(x)
        normalizer = self.inverse_dist_weight(mask)
        normalizer[normalizer==0] = -1
        normalizer = 1/normalizer
        normalizer[normalizer==-1] = 0        
        idw_f = idw_f * normalizer
        idw_final = x + (1-mask)*idw_f
        return  idw_final