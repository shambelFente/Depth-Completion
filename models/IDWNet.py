# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
output_depth_dirT = '/home/shambel/SparConv/featuremaps/image'
output_depth_dirV = '/home/shambel/SparConv/featuremaps/mask_density'

# %%

def get_distances( kernelsize ):
    XX,YY = np.meshgrid( np.arange(kernelsize), np.arange(kernelsize ))
    ksh = kernelsize//2
    d = np.sqrt( (XX-ksh)**2 + (YY-ksh)**2 )
    d[ ksh, ksh ] = 1
    return torch.unsqueeze( torch.unsqueeze( torch.tensor(d, requires_grad=False, dtype=torch.float32 ), 0 ), 0 )

# %%


class IDWBlock( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.Nexp = 0

        # K1
        exp = torch.tensor( [1.0, 3.0, 7.0 ], requires_grad=False, dtype=torch.float32 )
        self.Nexp += exp.shape[0]
        exp = torch.unsqueeze( exp, -1 )
        exp = torch.unsqueeze( exp, -1 )
        exp = torch.unsqueeze( exp, -1 )
        self.exponents_K1 = nn.Parameter( data=exp, requires_grad=True )
        distances_K1 = get_distances(5)
        self.register_buffer( "distances_K1", distances_K1 )
        ones_K1 = distances_K1*0+1.0
        self.register_buffer( "ones_K1", ones_K1 )
        del exp

        # K2
        exp = torch.tensor( [0.5, 1.0, 2.0, 3.0, 4.0 ], requires_grad=False, dtype=torch.float32 )
        self.Nexp += exp.shape[0]
        exp = torch.unsqueeze( exp, -1 )
        exp = torch.unsqueeze( exp, -1 )
        exp = torch.unsqueeze( exp, -1 )
        self.exponents_K2 = nn.Parameter( data=exp, requires_grad=True )
        distances_K2 = get_distances(17)
        self.register_buffer( "distances_K2", distances_K2 )
        ones_K2 = distances_K2*0+1.0
        self.register_buffer( "ones_K2", ones_K2 )
        del exp

        # K3
        exp = torch.tensor( [0.7, 2.0, 4.0 ], requires_grad=False, dtype=torch.float32 )
        self.Nexp += exp.shape[0]
        exp = torch.unsqueeze( exp, -1 )
        exp = torch.unsqueeze( exp, -1 )
        exp = torch.unsqueeze( exp, -1 )
        self.exponents_K3 = nn.Parameter( data=exp, requires_grad=True )
        distances_K3 = get_distances(37)
        self.register_buffer( "distances_K3", distances_K3 )
        ones_K3 = distances_K3*0+1.0
        self.register_buffer( "ones_K3", ones_K3 )
        del exp

        self.maskconv1 = nn.Conv2d( 3, self.Nexp*2, 5, padding="same", bias=True )
        self.maskconv2 = nn.Conv2d( self.Nexp*2, self.Nexp , 3, padding="same", bias=True )
        self.maskconv3 = nn.Conv2d( self.Nexp, self.Nexp , 3, padding="same", bias=True )
        self.relu = nn.ReLU( )

    def print_summary(self):
        print(" - K1 %s, exp: %s"%(str(self.distances_K1.squeeze().shape),str(self.exponents_K1.detach().cpu().numpy().flatten())))
        print(" - K2 %s, exp: %s"%(str(self.distances_K2.squeeze().shape),str(self.exponents_K2.detach().cpu().numpy().flatten())))
        print(" - K3 %s, exp: %s"%(str(self.distances_K3.squeeze().shape),str(self.exponents_K3.detach().cpu().numpy().flatten())))


    def forward( self, x, mask, debug=False ):

        idw_kernels = 1.0 / ( torch.pow( self.distances_K1, self.exponents_K1 ) )
        x_num = torch.nn.functional.conv2d( x, idw_kernels, bias=None, padding="same" )
        x_den_K1 = torch.nn.functional.conv2d( mask, idw_kernels, bias=None, padding="same" )
        x_idw_K1 = mask*x + (1.0-mask)*(x_num/(x_den_K1+1E-10))
        valid_idw_K1 = torch.sign( x_den_K1 )

        idw_kernels = 1.0 / ( torch.pow( self.distances_K2, self.exponents_K2 ) )
        x_num = torch.nn.functional.conv2d( x, idw_kernels, bias=None, padding="same" )
        x_den_K2 = torch.nn.functional.conv2d( mask, idw_kernels, bias=None, padding="same" )
        x_idw_K2 = mask*x + (1.0-mask)*(x_num/(x_den_K2+1E-10))
        valid_idw_K2 = torch.sign( x_den_K2 )

        idw_kernels = 1.0 / ( torch.pow( self.distances_K3, self.exponents_K3 ) )
        x_num = torch.nn.functional.conv2d( x, idw_kernels, bias=None, padding="same" )
        x_den_K3 = torch.nn.functional.conv2d( mask, idw_kernels, bias=None, padding="same" )
        x_idw_K3 = mask*x + (1.0-mask)*(x_num/(x_den_K3+1E-10))
        valid_idw_K3 = torch.sign( x_den_K3 )


        if debug:
            # %%

            plt.figure(figsize=(7,3))
            plt.subplot( 1,3,1 )
            plt.imshow( x[0,0,...].detach().cpu().numpy(), interpolation="none")
            plt.subplot( 1,3,2 )
            plt.imshow( x_idw_K1[0,0,...].detach().cpu().numpy(), interpolation="none")
            plt.title("idw K1")
            plt.subplot( 1,3,3 )
            plt.imshow( valid_idw_K1[0,0,...].detach().cpu().numpy(), interpolation="none")
            plt.title("valid idw K1")
            plt.tight_layout()
            plt.savefig("dbg/idw_k1.png")
            plt.close()

            plt.figure(figsize=(7,3))
            plt.subplot( 1,3,1 )
            plt.imshow( x[0,0,...].detach().cpu().numpy(), interpolation="none")
            plt.subplot( 1,3,2 )
            plt.imshow( x_idw_K2[0,0,...].detach().cpu().numpy(), interpolation="none")
            plt.title("idw K2")
            plt.subplot( 1,3,3 )
            plt.imshow( valid_idw_K2[0,0,...].detach().cpu().numpy(), interpolation="none")
            plt.title("valid idw K2")
            plt.tight_layout()
            plt.savefig("dbg/idw_k2.png")
            plt.close()

            plt.figure(figsize=(7,3))
            plt.subplot( 1,3,1 )
            plt.imshow( x[0,0,...].detach().cpu().numpy(), interpolation="none")
            plt.subplot( 1,3,2 )
            plt.imshow( x_idw_K3[0,0,...].detach().cpu().numpy(), interpolation="none")
            plt.title("idw K3")
            plt.subplot( 1,3,3 )
            plt.imshow( valid_idw_K3[0,0,...].detach().cpu().numpy(), interpolation="none")
            plt.title("valid idw K3")
            plt.tight_layout()
            plt.savefig("dbg/idw_k3.png")
            plt.close()
            # %%

        x_idw = torch.concatenate( [x_idw_K1, x_idw_K2, x_idw_K3], dim=1 )
        x_idw_valid = torch.concatenate( [valid_idw_K1, valid_idw_K2, valid_idw_K3], dim=1 )

        local_density_K1 = torch.nn.functional.conv2d( mask, self.ones_K1, bias=None, padding="same" )
        local_density_K2 = torch.nn.functional.conv2d( mask, self.ones_K2, bias=None, padding="same" )
        local_density_K3 = torch.nn.functional.conv2d( mask, self.ones_K3, bias=None, padding="same" )

        local_densities = torch.concatenate( [local_density_K1, local_density_K2, local_density_K3], dim=1 )
        #local_densities = torch.concatenate( [x_den_K1, x_den_K2, x_den_K3], dim=1 )

        mx = self.maskconv1(local_densities )
        mx = self.relu( mx )
        mx = self.maskconv2( mx )
        mx = self.relu( mx )
        mx = self.maskconv3( mx )
        mx = self.relu(mx)*x_idw_valid + 1E-11
        mx = mx / torch.sum( mx, dim=1, keepdim=True )

        mixed = torch.sum( x_idw * mx, dim=1, keepdim=True )
        maskout = torch.sign( local_density_K2 )

        return mixed, maskout 

# %%


class IDWNet( nn.Module ):
    def __init__( self, num_idwblocks=3 ):
        super().__init__()

        self.idw_blocks = nn.ModuleList()
        for ii in range( num_idwblocks ):
            self.idw_blocks.append( IDWBlock() )
        
        self.c1 = nn.Conv2d( 1, 32, 7, padding="same", bias=True )
        self.c2 = nn.Conv2d( 32, 16, 5, padding="same", bias=True )
        self.c3 = nn.Conv2d( 16, 16, 5, padding="same", bias=True )
        self.c4 = nn.Conv2d( 16, 16, 3, padding="same", bias=True )
        self.c5 = nn.Conv2d( 16, 1, 3, padding="same", bias=True )
        self.relu = nn.ReLU( )


    def print_summary( self ):
        print("")
        print("IDWNet summary")
        print("========================================================")
        for ii,block in enumerate(self.idw_blocks):
            print(" IDW block %d: "%ii)
            block.print_summary()
        print("========================================================")
        print("")


    def forward( self, x, mask ):
        x_, mask_ = x, mask
        for idwblock in self.idw_blocks:
            x_, mask_ = idwblock( x_*mask_, mask_ )

        x_ = x*mask + (1.0-mask)*x_

        x_ = self.c1( x_ )
        x_ = self.relu( x_ )
        x_ = self.c2( x_ )
        x_ = self.relu( x_ )
        x_ = self.c3( x_ )
        x_ = self.relu( x_ )
        x_ = self.c4( x_ )
        x_ = self.relu( x_ )
        x_ = self.c5( x_ )
        x_ = self.relu( x_ )

        #x_ = x*mask + (1.0-mask)*x_
        return x_, mask_




"""
def test():

    # %%

    exponents = torch.tensor( [1.0, 2.0, 3.0, 4.0 ], requires_grad=True, dtype=torch.float32 )
    exponents = torch.unsqueeze( exponents, -1 )
    exponents = torch.unsqueeze( exponents, -1 )
    exponents = torch.unsqueeze( exponents, -1 )
    idw_kernels = 1.0 / ( torch.pow( get_distances(7),exponents) )

    # %%

    I = torch.Tensor( np.load( "/home/fibe/projects/SparConv_nyu/dbg/in.npy") )
    mask = torch.Tensor( np.load( "/home/fibe/projects/SparConv_nyu/dbg/mask.npy") )
    out = np.load( "/home/fibe/projects/SparConv_nyu/dbg/out.npy")

    # %%
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(I.squeeze(), interpolation="none")
    plt.figure()
    plt.imshow(mask.squeeze(), interpolation="none")
    plt.colorbar()


    # %%


    I_num = torch.nn.functional.conv2d( I, idw_kernels, bias=None, padding="same" )
    I_den = torch.nn.functional.conv2d( mask, idw_kernels, bias=None, padding="same" ) + 1E-10
    I_idw = mask*I + (1.0-mask)*(I_num/I_den)

    for ii in range(I_idw.shape[1]):
        plt.figure()
        plt.imshow( I_idw[0,ii,...].detach().numpy(), interpolation="none")


    # %%

"""