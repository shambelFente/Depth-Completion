import os
import cv2
import torch
from PIL import Image
import numpy as np
import torchvision.transforms.functional as f
from sklearn import preprocessing as pre
#import h5py
import matplotlib.pyplot as plt
from args import get_arguments

# Get the arguments
args = get_arguments()

device = torch.device(args.device)
mode = args.mode


def get_files(folder, name_filter=None, extension_filter=None):
    """Helper function that returns the list of files in a specified folder
    with a specified extension.

    Keyword arguments:
    - folder (``string``): The path to a folder.
    - name_filter (```string``, optional): The returned files must contain
    this substring in their filename. Default: None; files are not filtered.
    - extension_filter (``string``, optional): The desired file extension.
    Default: None; files are not filtered

    """
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))

    # Filename filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files that do not
    # contain "name_filter"
    if name_filter is None:
        # This looks hackish...there is probably a better way
        name_cond = lambda filename: True
    else:
        name_cond = lambda filename: name_filter in filename

    # Extension filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files whose extension
    # is not "extension_filter"
    if extension_filter is None:
        # This looks hackish...there is probably a better way
        ext_cond = lambda filename: True
    else:
        ext_cond = lambda filename: filename.endswith(extension_filter)

    filtered_files = []

    # Explore the directory tree to get files that contain "name_filter" and
    # with extension "extension_filter"
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)

    return filtered_files


def h5_loader(data_path, label_path):

    data = cv2.imread(data_path,cv2.IMREAD_ANYDEPTH).astype(np.float32)/256.0
    label = cv2.imread(label_path,cv2.IMREAD_ANYDEPTH).astype(np.float32)/256.0
    
    
    # the following piece of code is used for sparsifying the data to test the model robustness for different sparsity level
    '''
    mask = (data>0).astype(float)
    for i in np.arange(0,mask.shape[0], 32, dtype = int):
        for j in np.arange(0, mask.shape[1], 32, dtype=int):
            # slicing a patch of size 9x18 from the mask
            patch = mask[i:(i+32), j:(j+32)]
            #count the number of ones in the patch and abandon 40% of depth points, i.e, selecting only 60% of them(60% rate)
            num_replaced = int(np.count_nonzero(patch) * (20/100))
            #num_replaced = int(np.count_nonzero(patch) - np.count_nonzero(patch)* (80/100))  # 9x18 is the patch size
            # storing iindices all ones in the patch
            indices = np.where(patch ==1)

            # if all the elements in the patch are zero, do nothing
            if num_replaced>0: 
                indices_x = []
                indices_y = []
    
                indices_random = np.arange(0, len(indices[0]))
                p_range = np.arange(1, len(indices[0])+1)
                prob = np.random.dirichlet(p_range, 1)[0]
                
                indices_choosen = np.random.choice(indices_random, num_replaced, replace=False, p=prob)
                for k in(indices_choosen):
                    indices_x.append(indices[0][k])
                    indices_y.append(indices[1][k])
                    
                patch[indices_x, indices_y] = 0
                mask[i:(i+32), j:(j+32)] = patch    
    data = data * mask
    '''
    
    
    # cropping the height to 272 to make it easier for the center crop(avoids croping the bottom part which has importatnt information in the center crop)
    # cropping only for training and validation
    if mode == 'val' or mode == 'train':
        data = data[(data.shape[0]-272):, :]  # 352, 256
        label= label[(label.shape[0]-272):, :]

    data=Image.fromarray(data)
    label=Image.fromarray(label)
    
    # resizing the image to make evenely divisible dimension for upsampling and downsampling
    if mode == 'val' or mode == 'train':
        data = f.center_crop(data, output_size=(272,1240))   # 352, 1224
        label = f.center_crop(label, output_size=(272, 1240))
    
    return data, label


