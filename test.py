import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as tf
from sklearn import preprocessing as pre
import cv2
import csv
import numpy as np
from os import listdir
from os.path import isfile, join
import torch.nn.functional as F
from args import get_arguments
from metric import evaluate 
import tqdm

args = get_arguments()
test_mode = args.mode
batch_s = args.batch_size


class Test:

    def __init__(self, model, data_loader, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device
    
    def run_epoch(self,  epochs, iteration_loss=False):
        if test_mode == 'test':
            mypath = '/data/Datasets/test'
            #onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            #onlyfiles = sorted(onlyfiles)
            # getting metric class object
            rmse = 0
            mae = 0
            absrel = 0
            delta2 = 0
            delta3 = 0
            delta1 = 0
            count = 0
            mae_list = []
            rmse_list = []
            channels_activated = np.zeros([16])

        self.model.eval()
        epoch_loss = 0.0
        st = 0

        tq = tqdm.tqdm(self.data_loader) 
        for step, batch_data in enumerate(tq):

            input = (batch_data[0]).to(self.device, dtype=torch.float32).unsqueeze(0)
            label = (batch_data[1]).to(self.device, dtype=torch.float32).unsqueeze(0)
            inputs = input.transpose(1,0)
            labels = label.transpose(1,0)
            label_mask  = torch.ne(labels , 0).to(self.device, dtype=torch.float32)
            mask  = torch.ne(inputs , 0).to(self.device, dtype=torch.float32)
            
            with torch.no_grad():
                # Forward propagation
                #
                #outputs = self.model(inputs, mask, st)
                st = st + 1
                
                outputs, _ = self.model(inputs, mask)

                if test_mode == 'test':
                                        
                    # computing the metrics
                    rmse1, mae1, absrel1, delta11, delta22, delta33 = evaluate(outputs,labels)
                    mae_list.append(mae1)
                    rmse_list.append(rmse1)
                    rmse += rmse1
                    mae += mae1
                    absrel += absrel1
                    delta1 += delta11
                    delta2 += delta22
                    delta3 += delta33
                    count += 1
                    
                    # %% Save images 
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
                    plt.savefig("%s/figs/%d.png"%(args.save_dir,step))
                    plt.close()
                    # %%
                            
                    
                # Loss computation
                #loss = torch.sum((self.criterion(outputs*label_mask, labels*label_mask)))/torch.sum(label_mask)
                loss = torch.sum(torch.abs(outputs*label_mask - labels*label_mask))/torch.sum(label_mask)
            
            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            tq.set_description("Iteration loss: %.4f" % loss.item())


        if test_mode=='test':
            pixel = channels_activated.reshape(-1,1)
            return (epoch_loss / len(self.data_loader) , rmse/count, mae/count, absrel/count, delta1 /count, delta2 /count, delta3 /count)
        else:
            return epoch_loss / len(self.data_loader)
