import os
import json
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
#import torchvision.transforms as transforms
import transform_nyu as transforms
from torchvision import transforms as transform_t
import matplotlib.pyplot as plt
from PIL import Image
import copy
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
import transforms as ext_transforms
from models.IDWNet import IDWNet

from train import Train
from test import Test
from args import get_arguments
import utils
from data import CamVid as dataset

# Get the arguments
args = get_arguments()

device = torch.device(args.device)
test_mode = args.mode

def load_dataset(dataset):
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)
    

    train_set = dataset(
        args.dataset_dir,
        transform=lambda x: x)

    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last = True)

    val_set = dataset(
        args.dataset_dir,
        transform=lambda x:x,
        mode='val')
    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True)

    test_set = dataset(
        args.dataset_dir,
        transform=lambda x:x,
        mode='test')
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
    
        
    return train_loader, val_loader, test_loader


def train(train_loader, val_loader):
    train_loss_dict = []
    val_loss_dict = []
    train_loss_json = {}
    val_loss_json = {}
    
    model = IDWNet().to(device)

    criterion = nn.MSELoss(reduction='none')
    #criterion = nn.SmoothL1Loss(reduction='none')

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate
        ) # weight_decay=args.weight_decay
    
    # Learning rate decay scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)
    
    
    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_loss = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} "
              "| Best mean loss = {1:.4f}".format(start_epoch, best_loss))
    else:
        start_epoch = 0
        best_loss = 999999999.9

    model.print_summary() 
        
    # Start Training
    print()
    train = Train(model, train_loader, optimizer, criterion, device, lr_updater)
    val = Test(model, val_loader, criterion, device)

    for epoch in range(start_epoch, args.epochs):
        print(">>>> [Epoch: {%d/%d}] Training"%(epoch+1,args.epochs))

        epoch_loss = train.run_epoch( epoch,  args.print_step)
        train_loss_dict.append(epoch_loss)
        train_loss_json[epoch] = epoch_loss

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f}".format(epoch+1, epoch_loss))

        if (epoch + 1) % 1 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss = val.run_epoch(epoch, args.print_step)
            val_loss_dict.append(loss)
            val_loss_json[epoch] = loss

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f}".
                  format(epoch, loss))

            model.print_summary()

            # Save the model if it's the best thus far
            if loss < best_loss:
                print("\n !!! Best model thus far.\n")
                best_loss = loss
            
            print("Saving checkpoint...")
            utils.save_checkpoint(model, optimizer, epoch + 1, best_loss, args.save_dir)

    
    # ploting training /validation loss
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    epoc = np.arange(0, args.epochs, 1)
    ax.plot(epoc, list(np.log(train_loss_dict).round(2)), color='tab:blue', marker='o', label ='train' )
    ax.plot(epoc, list(np.log(val_loss_dict).round(2)),  color='tab:orange', marker='o', label = 'val')
    ax.set_title('traing vs validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('log(loss)')
    plt.legend(loc='upper right')
    fig.savefig('%s/loss.png'%args.save_dir )   # loss_idw & loss
    plt.close(fig)    

    save_loss(train_loss_json, val_loss_json)    
    return model


def test(model,test_loader):
    print("\nTesting...\n")

    criterion = nn.MSELoss(reduction='none')
    #criterion = nn.HuberLoss(reduction='none')

    # Test the trained model on the test set
    test = Test(model,  test_loader, criterion, device)

    print(">>>> Running test dataset")
    loss,  rmse, mae, absrel, delta1, delta2, delta3 = test.run_epoch( args.epochs, args.print_step)
    print(">>>> Avg. loss: {0:.4f} ".format(loss))
    
    print()
    print(">>>> RMSE: {} ".format(rmse))
    print(">>>> MAE: {} ".format(mae))
    print(">>>> REL: {} ".format(absrel))
    print(">>>> \u03B4 1: {} ".format(delta1))
    print(">>>> \u03B4 2: {} ".format(delta2))
    print(">>>> \u03B4 3: {} ".format(delta3))


def save_loss(train_dict, val_dict):
    with open('%s/loss.json'%args.save_dir, 'w') as json_file:
        json.dump([train_dict,val_dict], json_file)
        json_file.close()


def main(): 
    # setting seed to have the same weight initialization, everytime we train the model, have no any performance effect
    torch.manual_seed(52)

    # Fail fast if the dataset directory doesn't exist
    assert os.path.isdir(
        args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
            args.dataset_dir)

    # Fail fast if the saving directory doesn't exist
    assert os.path.isdir(
        args.save_dir), "The directory \"{0}\" doesn't exist.".format(
            args.save_dir)
 
    # %%
    """
    train_set = dataset(
        args.dataset_dir,
        transform=lambda x: x)

    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last = True)

    for d in train_loader:
        print(d[0].shape)
        print(d[1].shape)
    
    # %%
    return
    """

    train_loader, val_loader, test_loader = load_dataset(dataset)

    if args.mode.lower() in {'train', 'full'}:
        model = train(train_loader, val_loader)

    if args.mode.lower() in {'test', 'full'}:
        torch.manual_seed(52)
        model = IDWNet().to(device)

        optimizer = optim.Adam(model.parameters())
        model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name)[0]
        model.print_summary()
        test(model,test_loader)



'''>>>>>>>>>>>>>>>>>>MAIN Starts Here>>>>>>>>>>>>>>>>>>'''
# Run only if this module is being run directly
if __name__ == '__main__':
    main()