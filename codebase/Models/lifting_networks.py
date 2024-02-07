from Models.lifting_net_utils  import function_to_obtain_3D_poses_from_depth
from Models.modulated_gcn_lifting import ModulatedGCNResNetDepthAndUncertaintyEstimator
from Models.residual_lifting   import ResNetDepthAndUncertaintyEstimator

import torch.nn as nn
import torch

class UncertaintyLifter(nn.Module):
    def __init__(
        self,
        network,
        inp_det_keypoints,
        use_view_info,
        with_uncertainty: bool =True):
        super(UncertaintyLifter, self).__init__()
        
        self.network             = network
        self.inp_det_keypoints   = inp_det_keypoints
        
        self.use_view_info       = use_view_info
        
        self.with_uncertainty    = with_uncertainty
        
        if self.inp_det_keypoints:
            print("[LIFTING-NET]: The input to the Lifting Net are 2D Keypoints.")
        else:
            print("[LIFTING-NET]: The input to the Lifting Net are 2D Keypoints normalized to [-1, 1].")

    def forward(self, inp):
        x_y_coords  = inp['inp_lifting']
        
        joints_mask = inp['joints_mask']
        if joints_mask is not None:
            assert joints_mask.size()[:2] == x_y_coords.size()[:2]
            # joints mask: value 0 = visible, value 1 = masked
            x_y_coords = x_y_coords * (~ joints_mask).unsqueeze(-1).float()
        
        batch_size = x_y_coords.size(0)
        x_y_coords  = x_y_coords.reshape(batch_size, -1)
        
        if self.use_view_info:
            view_info         = inp['view_info'] / 10.0
            view_info         = view_info.reshape(-1, 1)
            x_y_coords = torch.cat((x_y_coords, view_info), dim=1)
      
        # these are the detected 2D poses in the image coordinates of shape
        # num_samples x (number of joints * 2) if config.remove_head_view_info is True
        # OR num_samples x (number of joints * 2 + 2) when config.remove_head_view_info is False.
        projection_dict_val = {'R': inp['R'], 't': inp['t'], 'K': inp['K'], 'dist': inp['dist']}
        pelvis_cam_z = inp['pelvis_cam_z']
        det_2d_poses = inp['det_poses_2d']
        img = inp['inp_images_lifting_net']
        
        # forward pass through the lifting network
        inp_dict_vals = {"x_y_coords" : x_y_coords , "img" : img, "with_uncertainty": self.with_uncertainty}
        out_lifter    = self.network(**inp_dict_vals)
        out_depth     = out_lifter['depth']
        out_uncertain = (
            out_lifter['uncertainty']
            if out_lifter['uncertainty'] is not None
            else torch.ones_like(out_depth, requires_grad=False))
        
        # out_depth should be of shape num_samples x number of joints
        pred_z = (out_depth + pelvis_cam_z).view(batch_size, -1, 1)
        
        # pred_z is of shape num_samples x number of joints x 1
        # det_2d_poses is of shape num_samples x number of joints x 2
        out_3d_world, out_3d_cam = function_to_obtain_3D_poses_from_depth(
            inp_2d_poses_dist=det_2d_poses,
            pred_z=pred_z,
            **projection_dict_val)
        
        # out_3d_world should be of shape num_samples x number of joints x 3
        # out_3d_cam should be of shape num_samples x number of joints x 3
        out = {
            'pose_3d_world' : out_3d_world,
            'pose_3d_cam' : out_3d_cam,
            'rel_depth' : out_depth,
            'uncertainty': out_uncertain}
        return out
    
def obtain_Lifting_Network_with_Uncertainty(config):
    n_joints = config.n_joints
    
    # TODO: need to adapt input and output when using view info
    # first dimension is for depth and 2nd dimension is for uncertainty
    input_dim  = 2 * n_joints 
    output_dim = 2 * n_joints
    
    network = None
    match config.type_lifting_network:
        case "resnet":
            base_n_res_blocks = 2; head_n_res_blocks = 1
            base_hidden_dim = 272; head_hidden_dim = 272
            n_layers_per_res_block = 2
            network = ResNetDepthAndUncertaintyEstimator(
                input_dim=input_dim,
                base_hidden_dim=base_hidden_dim,
                head_hidden_dim=head_hidden_dim,
                n_joints=n_joints,
                base_n_res_blocks=base_n_res_blocks,
                head_n_res_blocks=head_n_res_blocks,
                n_layers_per_res_block=n_layers_per_res_block,
                base_normalization=config.lifting_backbone_normalization)
        case "modulated_gcn":
            base_n_res_blocks = 2; head_n_res_blocks = 1
            base_hidden_dim = 272; head_hidden_dim = 272
            n_layers_per_res_block = 2
            network = ModulatedGCNResNetDepthAndUncertaintyEstimator(
                base_hidden_dim=base_hidden_dim,
                head_hidden_dim=head_hidden_dim,
                n_joints=n_joints,
                base_n_res_blocks=base_n_res_blocks,
                head_n_res_blocks=head_n_res_blocks,
                n_layers_per_res_block=n_layers_per_res_block)
        case _:
            raise NotImplementedError(f"Unknown type of lifting network: {config.type_lifting_network}")
    
    print("[LIFTING-NET]: The Input of the 2D-3D lifting network are {} dimensional vectors.".format(input_dim))
    print("[LIFTING-NET]: The Output of the 2D-3D lifting network are {} dimensional vectors.".format(output_dim))
    print("[LIFTING-NET]: The Number of Residual Blocks in this ResNet (or ResNet base) is base={}, heads={},{}".format(
        base_n_res_blocks, head_n_res_blocks, head_n_res_blocks))
    print("[LIFTING-NET]: The Number of Trainable Parameters in the 2D-3D lifting network are {}".format(
        sum(p.numel() for p in network.parameters() if p.requires_grad)))
    
    uncertainty_lifter = UncertaintyLifter(
        network=network, 
        use_view_info=config.use_view_info_lifting_net,
        inp_det_keypoints=config.inp_det_keypoints,
        with_uncertainty=not(config.calculate_depth_only))
    
    return uncertainty_lifter