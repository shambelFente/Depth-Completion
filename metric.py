import torch
import torch.nn as nn
import math

def evaluate(output, target):
    valid_mask = target > 0

    output_mm = output[valid_mask]
    target_mm = target[valid_mask]

    abs_diff = (output_mm - target_mm).abs()

    mse = float((torch.pow(abs_diff, 2)).mean())
    rmse = math.sqrt(mse)
    mae = float(abs_diff.mean())
    absrel = float((abs_diff / target_mm).mean())
    
    maxRatio = torch.max(output_mm / target_mm, target_mm / output_mm)
    delta1 = float((maxRatio < 1.25).float().mean())
    delta2 = float((maxRatio < 1.25 ** 2).float().mean())
    delta3 = float((maxRatio < 1.25 ** 3).float().mean())
    '''    
    # convert from meters to km
    inv_output_km = (1e-3 * output[valid_mask])
    inv_target_km = (1e-3 * target[valid_mask])

    inv_output_km[inv_output_km==0] = -1
    inv_target_km[inv_target_km==0] = -1

    inv_output_km = 1/inv_output_km
    inv_target_km = 1/inv_target_km

    inv_output_km[inv_output_km==-1] = 0
    inv_target_km[inv_target_km==-1] = 0


    
    abs_inv_diff = (inv_output_km - inv_target_km).abs()
        
    irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
    imae = float(abs_inv_diff.mean())
    '''    
    return(rmse, mae, absrel, delta1, delta2, delta3)
    
        
