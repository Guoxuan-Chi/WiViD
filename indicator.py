import torch
from gaussian_diffusion import *
import math

def compute_errors(gt, pred, valid_mask):
    """ Computation of error metrics between predicted and ground truth depths
    """
    gt /= 1000
    pred /= 1000

    valid_mask = valid_mask > 0
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    
    thresh = torch.max((gt / pred), (pred / gt))
    d1 = float((thresh < 1.25).float().mean())
    d2 = float((thresh < 1.25 ** 2).float().mean())
    d3 = float((thresh < 1.25 ** 3).float().mean())
        
    rmse = (gt - pred) ** 2
    rmse = math.sqrt(rmse.mean())
    
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = math.sqrt(rmse_log.mean())
    
    abs_rel = ((gt - pred).abs() / gt).mean()
    sq_rel = (((gt - pred) ** 2) / gt).mean()

    return abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

