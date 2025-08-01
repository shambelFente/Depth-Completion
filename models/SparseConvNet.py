import torch.nn as nn
import torch
import torch.nn.functional as fn
import torchvision.transforms as tf
from torchvision.utils import make_grid
from mpl_toolkits.axes_grid1 import AxesGrid
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import imageio 
import numpy as np 
from args import get_arguments
#from models.newSparseConvNet import SparseConvNet as sparseNew
cmap = plt.cm.get_cmap("Blues")
# Get the arguments
args = get_arguments()
device = torch.device(args.device)
from inverse_distance_weight import inverseDistWeight
idw = inverseDistWeight().to(device)
    
# ≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠≠
class SparseConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super().__init__()

        padding = kernel_size//2
        
        # convolution layer for the conventional convolution
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels), 
            requires_grad=True)
        
        # convolution layer for sparse convolution
        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        # kernel for sparse convolution
        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel, 
            requires_grad=False)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim = 1)
        self.batch = nn.BatchNorm2d(out_channels)
        

        self.max_pool = nn.MaxPool2d(
            kernel_size, 
            stride=1, 
            padding=padding)

        
    
    def forward(self, x, mask=None, idw_x = None, idw_mask = None, last_layers=False):
        epsilon = 1.0e-20
        # computing channelwise multiplication with IDW
        if idw_x != None:
            x = x*mask
            x = self.conv(x)
            normalizer = self.sparsity(mask)
            normalizer[normalizer==0] = -1
            normalizer = 1/normalizer
            normalizer[normalizer==-1] = 0
            x = x*normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            x = self.batch(x)

            y = self.softmax(x)
            idw_f = idw(idw_x, idw_mask)
            idw_x= y * idw_f
            #idw_x = torch.sum(idw_x, 1, keepdim=True)
            return idw_x, y


        if idw_x == None and last_layers == False:
            x = self.conv(x)
            x = x + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            x = self.batch(x)
            x = self.relu(x)
            return x
        
        if idw_x == None and last_layers ==True:
            x = x*mask
            x = self.conv(x)
            normalizer = self.sparsity(mask)
            normalizer[normalizer==0] = -1
            normalizer = 1/normalizer
            normalizer[normalizer==-1] = 0
            x = x*normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            x = self.batch(x)
            x = self.relu(x)
            mask = self.max_pool(mask)
            return x, mask

        

#=========================================================================================================================================
class SparseConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.SparseLayer1 = SparseConv(1, 16, 11)
        self.SparseLayer2 = SparseConv(16, 16, 7) #7
        
        self.SparseLayer3 = SparseConv(16,  16, 7)
        self.SparseLayer4 = SparseConv(16, 16, 7)
        self.SparseLayer5 = SparseConv(16, 16, 5)
        self.SparseLayer6 = SparseConv(16, 16, 3)
        self.SparseLayer7 = SparseConv(16, 16, 3) 
        self.SparseLayer8 = SparseConv(16, 1, 1)


    def forward(self, x, mask):
        y, maskY = self.SparseLayer1(x, mask, None, None,True)
        y, softmax = self.SparseLayer2(y, maskY,  x, mask)
        arg_max = torch.argmax(softmax.detach(), dim = 1, keepdim=True).cpu().numpy()
        y= self.SparseLayer3(y)
        y= self.SparseLayer4(y)
        y = self.SparseLayer5(y)
        y= self.SparseLayer6(y)
        y = self.SparseLayer7(y)
        y= self.SparseLayer8(y)
        return y, arg_max


