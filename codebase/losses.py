import torch
import torch.nn as nn


class BaseWeightedLoss(nn.Module):
    def __init__(self, type_of_loss, lambda_loss, norm, device, size_average, reduction, reduce):
        super(BaseWeightedLoss, self).__init__()
        self.lambda_loss  = lambda_loss
        self.norm         = norm
        self.device       = device
        self.reduction    = reduction
        self.reduce       = reduce
        self.size_average = size_average
        self.type_of_loss = type_of_loss

    def forward(self, pred, target, weights=None):

        """
        new_idx = torch.isinf(new_target)
        (Pdb) new_idx = torch.any(new_idx, dim=1)
        (Pdb) new_target_ = new_target[new_idx!=True, :]
        (Pdb) new_target_.size()
        """
        if self.norm:
            pred   = pred / torch.norm(pred, 1)
            target = target / torch.norm(target, 1)

        """
        if torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred)) \
                or torch.any(torch.isnan(target)) or torch.any(torch.isinf(target)):
            # print("[LOSSES-MODULE]: Nans, ", torch.any(torch.isnan(pred)), torch.any(torch.isinf(pred)), torch.any(torch.isnan(target)), torch.any(torch.isinf(target)))
            if pred.dim() == 1:
                dim = 1; one_dim_vector = True
            else:
                dim = pred.size(-1); one_dim_vector = False

            pred     = pred.reshape(-1, dim)
            target   = target.reshape(-1, dim)
            new_idx1 = torch.isinf(target)
            new_idx1 = torch.any(new_idx1, dim=1)
            new_idx2 = torch.isnan(target)
            new_idx2 = torch.any(new_idx2, dim=1)
            new_idx3 = torch.isinf(pred)
            new_idx3 = torch.any(new_idx3, dim=1)
            new_idx4 = torch.isnan(pred)
            new_idx4 = torch.any(new_idx4, dim=1)

            new_idx = torch.logical_or(new_idx1, new_idx2)
            new_idx = torch.logical_or(new_idx,  new_idx3)
            new_idx = torch.logical_or(new_idx,  new_idx4)
            pred    = pred[new_idx    != True, :]
            target  = target[new_idx  != True, :]
            if weights is not None:
                weights = weights[new_idx != True, :]
        """
        
        if self.type_of_loss == 'l1':
            out = torch.abs(pred - target)

        elif self.type_of_loss in ['l2', 'mse']:
            out = (pred - target) ** 2

        elif self.type_of_loss in 'smooth_l1':
            diff = pred - target
            abs_ = torch.abs(diff)
            out  = torch.where(abs_ < 1., 0.5 * diff ** 2, abs_ - 0.5)
        else:
            raise NotImplementedError

        if weights is not None:
            out = out * weights

        if self.size_average or self.reduction in ['mean', 'avg'] or self.reduce:
            ret_val = out.mean()
        else:
            if self.reduction == 'none':
                ret_val = out
            else:
                ret_val = out.sum()

        return ret_val * self.lambda_loss


class MSELoss(BaseWeightedLoss):
    def __init__(self, lambda_loss, norm, device, size_average, reduce, reduction):
        super(MSELoss, self).__init__(lambda_loss=lambda_loss, norm=norm, device=device, size_average=size_average,
                                      reduce=reduce, reduction=reduction, type_of_loss='l2')


class L1Loss(BaseWeightedLoss):
    def __init__(self, lambda_loss, norm, device, size_average, reduce, reduction):
        super(L1Loss, self).__init__(lambda_loss=lambda_loss, norm=norm, device=device, size_average=size_average,
                                     reduce=reduce, reduction=reduction, type_of_loss='l1')


class SmoothL1Loss(BaseWeightedLoss):
    def __init__(self, lambda_loss, norm, device, size_average, reduce, reduction):
        super(SmoothL1Loss, self).__init__(lambda_loss=lambda_loss, norm=norm, device=device, size_average=size_average,
                                           reduce=reduce, reduction=reduction, type_of_loss='smooth_l1')


class NSE(nn.Module):
    def __init__(self, lambda_loss):
        super(NSE, self).__init__()
        self.criterion   = nn.MSELoss(reduce=False)
        self.lambda_loss = lambda_loss

    def forward(self, p1, p2, weights=None):
        np1 = torch.norm(p1, dim=1, keepdim=True)
        np2 = torch.norm(p2, dim=1, keepdim=True)
        x   = self.criterion(p1 / np1, p2 / np2)
        if weights is None:
            x = torch.mean(x)
        else:
            x = torch.mean(x * weights[:, :, None])
        return x * self.lambda_loss

class Equalize_bone_pairs(nn.Module):
    def __init__(self, number_of_joints, bone_pairs, median_bch=False):
        super(Equalize_bone_pairs, self).__init__()
        self.median_bch       = median_bch
        self.bone_pairs       = bone_pairs
        self.number_of_joints = number_of_joints

    @staticmethod
    def compute_bone(p_start, p_end, eps=1e-5):
        """
        Input points are of the shape B x 3
        """
        bone_vector   = p_end - p_start
        b_length      = torch.norm(bone_vector, dim=1, keepdim=True)  # must be of size B x 1
        n_bone_vector = bone_vector / (b_length + eps)  # unit vector
        return n_bone_vector, b_length

    def equalize_bones_pair(self, joints3d, bones_pairs):
        """
        Function performs equalization of a pair of bones under several assumptions:
        - "start" joint must be fixed. Only "end" point is tuned. 
        - the uncertainty (or confidence) in both symmetric bones length prediction is same.
        - direction of a bone is fixed, as if it correctly predicted
        - previous point implies that uncertainty in direction decreases with the distance from the root joints 

        params:
            :joints3d    - (B x num_joints x 3), for each B this is a 3-dimensional joints array,
            :bones_pairs - list (or tuple) of two lists (tuples) with indices of bones,
                where they start and end. Assume that first entry is for the "left", second - for the "right".
            :median_bch - if set to True it takes the median of the bone length of the mini-batch
                as the target bone length
        """
        l_start, l_end = bones_pairs[0]
        r_start, r_end = bones_pairs[1]
        joints = {}
        for idx, elem in enumerate((l_start, l_end, r_start, r_end)):
            if isinstance(elem, int):
                joints[idx] = joints3d[:, elem, :]
            else:
                joints[idx] = (joints3d[:, elem[0], :] + joints3d[:, elem[1], :]) / 2.

        l_p_start, l_p_end, r_p_start, r_p_end = joints[0], joints[1], joints[2], joints[3]
        l_n_bone, l_b_len = self.compute_bone(l_p_start, l_p_end)  # unit vectors of bones and their lengths
        r_n_bone, r_b_len = self.compute_bone(r_p_start, r_p_end)
        if self.median_bch:
            bch_b_len = torch.cat((l_b_len, r_b_len), dim=0)
            median    = torch.median(bch_b_len)
            delta_l   = l_b_len - median
            l_p_end   = l_p_end - delta_l * l_n_bone
            delta_r   = r_b_len - median
            r_p_end   = r_p_end - delta_r * r_n_bone
        else:
            delta   = (l_b_len - r_b_len) / 2  # here we used that uncertainty is 50-50% between two symmetric bones lengths prediction
            l_p_end = l_p_end - delta * l_n_bone
            r_p_end = r_p_end + delta * r_n_bone

        joints3d[:, l_end, :] = l_p_end
        joints3d[:, r_end, :] = r_p_end

    def forward(self, joints_3d):
        joints3d_out = joints_3d.view(-1, self.number_of_joints, 3)
        for bone_pair in self.bone_pairs:
            self.equalize_bones_pair(joints3d_out, bone_pair)
        return joints3d_out


def get_individual_loss_module(dict_vals_loss, loss_name, lambda_loss, string):
    if loss_name == 'l1':
        print("[LOSSES-MODULE]: Will be Using L1 loss for {}.".format(string))
        loss_module = L1Loss(**dict_vals_loss)
    elif loss_name in ['l2', 'mse']:
        print("[LOSSES-MODULE]: Will be Using L2/MSE loss for {}.".format(string))
        loss_module = MSELoss(**dict_vals_loss)
    elif loss_name == 'nse':
        print("[LOSSES-MODULE]: Will be using NSE loss for {}.".format(string))
        loss_module = NSE(lambda_loss=lambda_loss)
    elif loss_name == 'smooth_l1':
        print("[LOSSES-MODULE]: Will be using Smooth L1 loss {}.".format(string))
        loss_module = SmoothL1Loss(**dict_vals_loss)
    elif loss_name == 'cross_entropy' or loss_name == 'xent':
        print("[LOSSES-MODULE]: Will be using Cross Entropy loss {}.".format(string))
        loss_module = torch.nn.CrossEntropyLoss()
    elif loss_name == 'bce' or loss_name == 'binary_cross_entropy':
        print("[LOSSES-MODULE]: Will be using Binary Cross Entropy loss {}.".format(string))
        loss_module = torch.nn.BCELoss()
    else:
        loss_module = None
    return loss_module


def func_to_obtain_sup_3d_loss(config, basic_lost_dict_vals):
    if config.calculate_loss_supervised_3d:
        assert config.experimental_setup in ['semi', 'fully']
        print("[LOSSES-MODULE]: Will be calculating the supervised 3D loss.")
        dict_vals_anno_3d   = {'lambda_loss': config.lambda_loss_supervised_3d, **basic_lost_dict_vals}
        loss_anno_3d_module = get_individual_loss_module(dict_vals_loss=dict_vals_anno_3d,   lambda_loss=config.lambda_loss_supervised_3d,
                                                         loss_name=config.loss_supervised_3d, string='supervised 3D loss')
    else:
        loss_anno_3d_module = None
    return loss_anno_3d_module


def func_to_obtain_sup_2d_loss(config, basic_lost_dict_vals):
    if config.calculate_loss_supervised_2d:
        print("[LOSSES-MODULE]: Will be calculating the supervised 2D loss.")
        dict_vals_anno_2d   = {'lambda_loss': config.lambda_loss_supervised_2d,  **basic_lost_dict_vals}
        loss_anno_2d_module = get_individual_loss_module(dict_vals_loss=dict_vals_anno_2d, loss_name=config.loss_supervised_2d,
                                                         lambda_loss=config.lambda_loss_supervised_2d,
                                                         string='supervised 2D loss')
    else:
        loss_anno_2d_module = None
    return loss_anno_2d_module


def func_to_obtain_uncertainty_loss(config, basic_lost_dict_vals):
    if config.calculate_uncertainty_loss:
        print("[LOSSES-MODULE]: Will be calculating the supervised 2D loss.")
        uncertainty_loss_module = UncertaintyLoss(
            device=basic_lost_dict_vals['device'],
            num_joints=config.n_joints,
            mixed_loss=config.mixed_loss,
            mixed_loss_lambda=config.mixed_loss_lambda)
    else:
        uncertainty_loss_module = None
    return uncertainty_loss_module

def get_losses_module_pose_lifting_net_together(config, device, total_loss_key):
    """
    Function to return the various loss modules to be used in the experiment.
    :param config: The Experimental Configuration file.
    :param device: The device id (cpu or gpu).
    :param total_loss_key : A key for the collecting the total loss module.
    :return:
    """

    losses = {}; losses_keys = {}; print_loss_keys = {}
    basic_lost_dict_vals = {'norm': config.norm, 'device': device, 'size_average': config.size_average, 'reduce': config.reduce, 'reduction': config.reduction}
    loss_anno_3d_module  = func_to_obtain_sup_3d_loss(config=config, basic_lost_dict_vals=basic_lost_dict_vals)
    if loss_anno_3d_module is not None:
        losses['loss_sup_3d']            = loss_anno_3d_module
        losses_keys['loss_sup_3d']       = ('pred_3d', 'target_3d', 'weights_3d')
        print_loss_keys['loss_sup_3d']   = 'Supervised Loss 3D = {:.5f} (Avg={:.5f})'

    loss_anno_2d_module = func_to_obtain_sup_2d_loss(config=config, basic_lost_dict_vals=basic_lost_dict_vals)
    if loss_anno_2d_module is not None:
        losses['loss_dlt_2d']            = loss_anno_2d_module
        losses_keys['loss_dlt_2d']       = ('pose_2d', 'target_2d', 'weights_2d')
        print_loss_keys['loss_dlt_2d']   = 'Supervised Loss 2D = {:.5f} (Avg={:.5f})'

    uncertainty_loss_module = func_to_obtain_uncertainty_loss(config=config, basic_lost_dict_vals=basic_lost_dict_vals)
    if uncertainty_loss_module is not None:
        losses['uncertainty_loss']          = uncertainty_loss_module
        losses_keys['uncertainty_loss']     = ('pred_rel_depth_and_uncertainty', 'target_rel_depth', 'weights')
        print_loss_keys['uncertainty_loss'] = 'Uncertainty Loss 2D = {:.5f} (Avg={:.5f})'

    if len(list(losses.keys())) == 0 and config.perform_test is False:
        raise ValueError("No Losses have been defined")

    print_loss_keys[total_loss_key]  = 'Total Loss ={:.5f} (Avg={:.5f}).'
    return losses, losses_keys, print_loss_keys

class UncertaintyLoss(nn.Module):
    def __init__(self, num_joints, device, mixed_loss: bool =False, mixed_loss_lambda: float =1.0) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.device     = device
        self.mixed_loss = mixed_loss
        self.mixed_loss_lambda = mixed_loss_lambda
    
    def forward(self, pred, target, weights=None, min_std=1e-5):
        """
        :param pred: Input Tensor of size Batch_size x Num_Joints x 2 which represents the predicted depth mean and standard deviation.
        :param target: Input Tensor of size Batch_size x Num_Joints which represents the target depth mean values.
        :param weights: Input Tensor of size Num_Joints which represents the weights for each joint.
        :param min_std: Minimum standard deviation value.
        :return: The loss value Tensor.
        """
        assert pred.dim() == 3
        assert target.dim() == 2 and target.shape == pred.shape[:-1]
        assert (weights is None) \
            or (weights.dim() == 1 and len(weights) == pred.shape[1]  and (weights >= 0).all().item())
        
        # mean depths (B x J x 1), and depth stds (B x J x 1)
        pred_depth_vals = pred[..., 0] 
        pred_std_vals   = pred[..., 1]
        
        # loss per joint (B x J)
        pred_std_vals = pred_std_vals.clamp(min=min_std)
        loss = torch.log(pred_std_vals) + ((target - pred_depth_vals) / (pred_std_vals))**2
        if self.mixed_loss:
            loss += self.mixed_loss_lambda * 0.5 * (target - pred_depth_vals)**2
            
        # compute the per-batch-element weighted average of the losses using the per-joint weights if provided
        if weights is not None:
            loss = (loss * weights).sum(dim=1)
        # if the weights are provided, the next instruction computes the average loss over all batch element losses
        # otherwise it directly averages the loss over all joints of all the elements in the batch
        loss = loss.mean()
        
        if torch.isinf(loss).any() or torch.isnan(loss).any():
            print("[LOSSES-MODULE]: found inf or nan in the loss value.")
            
            print(f"[LOSSES-MODULE]: min(pred_depth_vals) = {pred_depth_vals.min().item()}")
            print(f"[LOSSES-MODULE]: max(pred_depth_vals) = {pred_depth_vals.max().item()}")
            
            print(f"[LOSSES-MODULE]: min(pred_std_vals) = {pred_std_vals.min().item()}")
            print(f"[LOSSES-MODULE]: max(pred_std_vals) = {pred_std_vals.max().item()}")
            
            abs_diff_target_pred_depth = (target - pred_depth_vals).abs()
            print(f"[LOSSES-MODULE]: max(abs(target - pred_depth_vals)) = {abs_diff_target_pred_depth.max().item()}")
            print(f"[LOSSES-MODULE]: min(abs(target - pred_depth_vals)) = {abs_diff_target_pred_depth.abs().min().item()}")
            
            assert False, "Found inf or nan in the loss value."
        
        return loss
        
