import torch

def depth_nll(pred_depth, target_depth, pred_std, std_eps=1e-8):
    assert std_eps > 0
    nll = torch.log(pred_std + std_eps) + ((pred_depth - target_depth) / (pred_std + std_eps))**2
    return nll.mean()

def std_scaled_depth_mse(pred_depth, target_depth, pred_std, std_eps=1e-8):
    assert std_eps > 0
    mse = ((pred_depth - target_depth) / (pred_std + std_eps))**2
    mse = mse.mean()
    return mse

def depth_mse(pred_depth, target_depth):
    mse = (pred_depth - target_depth)**2
    mse = mse.mean()
    return mse 

