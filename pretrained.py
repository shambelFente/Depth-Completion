import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as tf
from torchvision.utils import make_grid
import torchvision
import imageio
import numpy as np
from args import get_arguments
#from models.newSparseConvNet import SparseConvNet as sparseNew
# Get the arguments
args = get_arguments()
device = torch.device(args.device)

from inverse_distance_weight import inverse_distance_weighted
power = [3, 3.2, 3.5, 3.8, 4, 4.2, 4.5] 
#power = [2, 2.2 ,2.5, 2.8, 3, 3.2, 3.5] 
weights = inverse_distance_weighted(31, power)
output_depth_dir = '/Users/shambelfentemengistu/depth_completion/SparConv/softmax'

#=======================================================================================================================================

# Non-negativity enforcement class        
class EnforcePos(object):
    def __init__(self, pos_fn, name):
        self.name = name
        self.pos_fn = pos_fn


    @staticmethod
    def apply(module, name, pos_fn):
        fn = EnforcePos(pos_fn, name)
        
        module.register_forward_pre_hook(fn)                    

        return fn

    def __call__(self, module, inputs):
       if module.training:
            for layer_name, layer in module.named_children():
                if isinstance(layer_name, nn.Conv2d):
                    weight = getattr(layer.conv, self.name)
                    weight.data = self._pos(weight).data
       else:
            pass

    def _pos(self, p):
        pos_fn = self.pos_fn.lower()
        if pos_fn == 'softmax':
            p_sz = p.size()
            p = p.view(p_sz[0],p_sz[1], -1)
            p = F.softmax(p, -1)
            return p.view(p_sz)
        elif pos_fn == 'exp':
            return torch.exp(p)
        elif pos_fn == 'softplus':
            return F.softplus(p, beta=10)
        elif pos_fn == 'sigmoid':
            return F.sigmoid(p)
        else:
            print('Undefined positive function!')
            return 
#=======================================================================================================================================
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
            #padding = 'same',
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels), 
            requires_grad=True)
        
        # convolution layer for sparse convolution
        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            #padding = 'same',
            padding=padding,
            bias=False)

        # convolution layer for Inverse distance weighted interpolation
        self.inverse_dist_weight = nn.Conv2d(
            1,
            7,
            kernel_size=31,
            padding = 'same',
            bias=False) 

        self.inverse_dist_weight.weight = nn.Parameter(
                                 data = weights, 
                                 requires_grad = False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel, 
            requires_grad=False)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim = 1)
        #self.softmax = nn.Softmax2d() # the same as softmax(dim =1)
        self.batch_norm = nn.BatchNorm2d(out_channels)


        self.max_pool = nn.MaxPool2d(
            kernel_size, 
            stride=1, 
            padding=padding)

        

    def forward(self, data, conf):
        eps = 1.0e-20
        # Normalized Convolution
        denom = self.conv(conf)        
        nomin = self.conv(data*conf)        
        nconv = nomin / (denom+eps)
        
        
        # Add bias
        b = self.bias
        sz = b.size(0)
        b = b.view(1,sz,1,1)
        b = b.expand_as(nconv)
        nconv += b
        
        # Propagate confidence
        cout = denom
        sz = cout.size()
        cout = cout.view(sz[0], sz[1], -1)
        
        k = self.weight
        k_sz = k.size()
        k = k.view(k_sz[0], -1)
        s = torch.sum(k, dim=-1, keepdim=True)        

        cout = cout / s
        cout = cout.view(sz)
        
        return nconv, cout
    
        '''
        x = x*mask
        x = self.conv(x)
 
        #normalizer = np.divide(1, self.sparsity(mask).detach().cpu().numpy(), 
                     #out=np.zeros_like(self.sparsity(mask).detach().cpu().numpy()), where=self.sparsity(mask).detach().cpu().numpy()!=0)

        x = x + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
     
        x = self.relu(x)
        mask = self.max_pool(mask)
        return x, mask
        '''


class SparseConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.SparseLayer1 = SparseConv(1, 16, 11)
        self.SparseLayer2 = SparseConv(16, 16, 7) 
        self.SparseLayer3 = SparseConv(16, 16, 5)
        self.SparseLayer4 = SparseConv(16, 16, 3)
        self.SparseLayer5 = SparseConv(16, 16, 3) 
        self.SparseLayer6 = SparseConv(16, 1, 1)


    def forward(self, x, mask, epochs, step):
        features = []
        x, mask = self.SparseLayer1(x, mask)
        x, mask = self.SparseLayer2(x, mask)
        x, mask = self.SparseLayer3(x, mask)
        x, mask = self.SparseLayer4(x, mask)
        x, mask = self.SparseLayer5(x, mask)
        if epochs != 100:
            if step <16:
                for i in range(0,16):
                    features.append(x[0][i].cpu().detach().unsqueeze(0))
                file_path = output_depth_dir + '/' + str(step) + str(epochs) + '.png'
                Grid = make_grid(features, nrow=8, padding=25, pad_value=0.9)   

                # display result
                img = torchvision.transforms.ToPILImage()(Grid)
                img.save(file_path)
        x, mask = self.SparseLayer6(x, mask)
        return x