import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm

class Train:
    def __init__(self, model, data_loader, optim, criterion, device, lr_updater):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.device = device
        self.lr_updater = lr_updater
        

    def run_epoch(self, epochs, iteration_loss=False):
        
        self.model.train()
        epoch_loss = 0.0
        loss_smoothed = None 
        
        # updating the learning rate
        # lr_rate =0.01 *((1-(epochs/35))**0.9) # best so far 1.8
        print("Learning rates: ", end="" )
        for param_group in self.optim.param_groups:
            print(" %1.6f "%param_group['lr'], end="" )
        print("")


        tq = tqdm.tqdm(self.data_loader) 
        for step, batch_data in enumerate( tq ): # get_item() of dataset class called

            input = (batch_data[0]).to(self.device, dtype=torch.float32).unsqueeze(0)
            label = (batch_data[1]).to(self.device, dtype=torch.float32).unsqueeze(0)
            inputs = input.transpose(1,0)
            labels = label.transpose(1,0)
            label_mask  = torch.ne(labels , 0).to(self.device, dtype=torch.float32)
            mask  = torch.ne(inputs , 0).to(self.device, dtype=torch.float32)

            """
            # %%
            np.save("dbg/in", input.cpu().numpy() )
            np.save("dbg/out", labels.cpu().numpy() )
            np.save("dbg/mask", mask.cpu().numpy() )
            plt.figure()
            plt.imshow( inputs.cpu().numpy().squeeze() )
            plt.colorbar()
            plt.savefig( "dbg/inputs.png")

            plt.figure()
            plt.imshow( labels.cpu().numpy().squeeze() )
            plt.colorbar()
            plt.savefig( "dbg/labels.png")

            plt.figure()
            plt.imshow( label_mask.cpu().numpy().squeeze() )
            plt.colorbar()
            plt.savefig( "dbg/label_mask.png")
            # %%
            """

            #============================
        
            self.model.zero_grad()

            # Forward propagation
            outputs, _ = self.model(inputs, mask)

            # %%
            if step % 20 == 0:
                oo = outputs[0,0,:,:].detach().cpu().numpy()
                gg = labels[0,0,:,:].detach().cpu().numpy()
                ii = inputs[0,0,:,:].detach().cpu().numpy()
                vmin = np.amin(ii)
                vmax = np.amax(ii)
                plt.figure( figsize=(12,4))
                plt.subplot(1,3,1)
                plt.imshow( ii, interpolation="none", vmin=vmin, vmax=vmax )
                plt.title("input")
                plt.subplot(1,3,2)
                plt.imshow( oo, interpolation="none", vmin=vmin, vmax=vmax )
                plt.title("ouput")
                plt.subplot(1,3,3)
                plt.imshow( gg, interpolation="none", vmin=vmin, vmax=vmax )
                plt.title("gt")
                plt.tight_layout()
                plt.savefig( "dbg/oo.png")
                plt.close()
            # %%

            # Compute loss
            #loss = torch.sum((self.criterion(outputs*label_mask, labels*label_mask)))/torch.sum(label_mask)
            loss = torch.sum(torch.abs(outputs*label_mask - labels*label_mask))/torch.sum(label_mask)
            #loss = torch.mean(torch.abs(outputs - labels))

            lossval = loss.item()

            # Backpropagation
            loss.backward( retain_graph=False )
            self.optim.step()

            # Compute stats
            epoch_loss += lossval
            loss_smoothed = lossval if loss_smoothed is None else 0.99*loss_smoothed + 0.01*lossval

            tq.set_description("Batch loss: %.4f, smoothed: %.4f" % (loss.item(), loss_smoothed))


        #self.lr_updater.step()
        return epoch_loss / len(self.data_loader)
