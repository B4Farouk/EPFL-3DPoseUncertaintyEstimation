# MODULATED GCN
# source: https://github.com/ZhimingZo/Modulated-GCN/tree/main/Modulated_GCN/Modulated_GCN_gt/models

from typing import Optional

import numpy as np
import torch
from torch import nn

from Models.residual_lifting import ResNetDepthHead, ResNetDepthUncertaintyHead


class ScaledSigmoid(nn.Module):
    def __init__(self):
        super(ScaledSigmoid, self).__init__()
        self.scalar = nn.Parameter(torch.Tensor([1.0]))
        self.sigmoid = nn.Sigmoid()
    
    def __constrained_scalar(self):
        if self.scalar.item() > 0.0:
            return self.scalar
        else:
            self.scalar.data = torch.clamp_min(self.scalar.data, min=1e-8)
            return self.scalar
    
    def __call__(self, input):
        scalar = self.__constrained_scalar() 
        return scalar * self.sigmoid(input) 


def get_activ_fn(name: Optional[str] =None) -> nn.Module:
    assert name is None or isinstance(name, str)
    
    activ_fn = None
    match name:
        case "gelu":
            activ_fn = nn.GELU()
        case "silu":
            activ_fn = nn.SiLU()
        case "leaky_relu":
            activ_fn = nn.LeakyReLU()
        case "relu": 
            activ_fn = nn.ReLU()
        case "scaled_sigmoid":
            activ_fn = ScaledSigmoid()
        case "softplus":
            activ_fn = nn.Softplus()
        case "identity":
            activ_fn = nn.Identity()
        case None:
            activ_fn = nn.Identity()
        case _:
            raise NotImplementedError("Unknown activation function: {}".format(name))
        
    return activ_fn


def get_norma_layer(name: Optional[str] =None, **kwargs) -> nn.Module:
    assert name is None or isinstance(name, str)
    
    match name:
        case "layer":
            norma_layer = nn.LayerNorm(kwargs["dim"])
        case "batch":
            norma_layer = nn.BatchNorm1d(kwargs["dim"])
        case None:
            norma_layer = nn.Identity()
        case _:
            raise NotImplementedError("Unknown normalization layer: {}".format(name))
        
    return norma_layer
  

class ModulatedGraphConvLayer(nn.Module):
    
    def __init__(self,
                 input_dim: int, output_dim: int,
                 affinity_mat: torch.Tensor,
                 use_bias: bool =True):
        super(ModulatedGraphConvLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.self_weights = nn.Parameter(torch.zeros((input_dim, output_dim), dtype=torch.float))
        self.cross_weights = nn.Parameter(torch.zeros((input_dim, output_dim), dtype=torch.float))
        nn.init.xavier_uniform_(self.self_weights.data, gain=1.414)
        nn.init.xavier_uniform_(self.cross_weights.data, gain=1.414)
        
        self.modulation_weights = nn.Parameter(torch.ones((affinity_mat.size(0), output_dim), dtype=torch.float))
        
        self.base_affinity_mat = affinity_mat
        self.adapt_affinity_mat = nn.Parameter(torch.ones_like(affinity_mat, dtype=torch.float))    
        nn.init.constant_(self.adapt_affinity_mat, 1e-6)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros((1, 1, output_dim), dtype=torch.float))
            stdv = 1. / np.sqrt(output_dim)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter("bias", None)
            
    
    @staticmethod
    def __make_symmetric(matrix):
        return (matrix.T + matrix) / 2

    def forward(self, input: torch.Tensor):
        batch_size = input.size(0)
        
        self.base_affinity_mat = self.base_affinity_mat.to(self.adapt_affinity_mat.device)
        affinity_mat = ModulatedGraphConvLayer.__make_symmetric(
            self.base_affinity_mat + self.adapt_affinity_mat)
        
        # compute affinity matrices for each case
        identity_mat = torch.eye(affinity_mat.size(0), dtype=torch.float).to(input.device)
        self_affinity_mat = identity_mat * affinity_mat
        cross_affinity_mat = (1 - identity_mat) * affinity_mat
        
        # compute output
        input = input.view(batch_size, -1, self.input_dim)
        h_self = torch.matmul(input, self.self_weights)
        h_cross = torch.matmul(input, self.cross_weights)
        output = (
            torch.matmul(self_affinity_mat, self.modulation_weights * h_self)
            + torch.matmul(cross_affinity_mat, self.modulation_weights * h_cross))
        
        # add bias
        output = output + self.bias if self.bias is not None else output
        
        return output
    

class ModulatedGCNResBlock(nn.Module):
    def __init__(
        self,
        affinity_mat: torch.Tensor,
        n_layers: int,
        hidden_dim: int,
        activation: str ="relu",
        normalization: Optional[str] =None,
        activate_last: bool =False):
        
        assert isinstance(affinity_mat, torch.Tensor)
        assert isinstance(n_layers, int) and n_layers > 1
        assert isinstance(hidden_dim, int) and hidden_dim > 0
        assert isinstance(activation, str) 
        assert normalization is None or isinstance(normalization, str)
        
        super(ModulatedGCNResBlock, self).__init__()
        
        layers = nn.ModuleList([])      
        # all layers except last
        for _ in range(n_layers - 1): # -1 because the last layer is added below
            layers.append(get_norma_layer(normalization, dim=hidden_dim))
            layers.append(ModulatedGraphConvLayer(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                affinity_mat=affinity_mat))
            layers.append(get_activ_fn(activation))
        # last layer
        layers.append(get_norma_layer(normalization, dim=hidden_dim))
        layers.append(ModulatedGraphConvLayer(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                affinity_mat=affinity_mat))
        if activate_last:
            layers.append(get_activ_fn(activation))
            
        self.gcn = nn.Sequential(*layers)
        
        # save attributes
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.activaté_last = activate_last
        self.normalization = normalization
               
    def forward(self, input):
        return input + self.gcn(input)
    

class ModulatedGCNResNet(nn.Module):
    def __init__(
        self,
        affinity_mat: torch.Tensor,
        n_res_blocks: int,
        n_layers_per_res_block: int,
        hidden_dim: int,
        activation: str ="relu",
        normalization: Optional[str] =None,
        resblock_activate_last: bool =False):

        assert isinstance(affinity_mat, torch.Tensor)
        assert isinstance(n_res_blocks, int) and n_res_blocks > 0
        assert isinstance(n_layers_per_res_block, int) and n_layers_per_res_block > 0
        assert isinstance(hidden_dim, int) and hidden_dim > 0
        assert isinstance(activation, str)
        assert normalization is None or isinstance(normalization, str)
                
        super(ModulatedGCNResNet, self).__init__()
        
        layers = nn.ModuleList([])
        layers.extend([
            ModulatedGCNResBlock(
                affinity_mat=affinity_mat,
                n_layers=n_layers_per_res_block,
                hidden_dim=hidden_dim,
                activation=activation,
                normalization=normalization,
                activate_last=resblock_activate_last)
            for _ in range(n_res_blocks)])
        self.resnet = nn.Sequential(*layers)
        
        # save attributes
        self.n_res_layers = n_res_blocks
        self.n_layers_per_res_block = n_layers_per_res_block
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.normalization = normalization
        self.resblock_activate_last = resblock_activate_last
        
    def forward(self, input):
        return self.resnet(input)


class ModulatedGCNResNetBackbone(ModulatedGCNResNet):
    def __init__(self,
                 affinity_mat: torch.Tensor,
                 n_res_blocks: int,
                 n_layers_per_res_block: int,
                 hidden_dim: int):
        
        super().__init__(
            affinity_mat=affinity_mat,
            n_res_blocks=n_res_blocks,
            n_layers_per_res_block=n_layers_per_res_block,
            hidden_dim=hidden_dim,
            activation="silu",
            normalization="layer") # cannot use batch norm here
    
    def forward(self, input, img=None):
        return super().forward(input)


class JointEmbeddingLayer(nn.Module):
    def __init__(self, n_joints: int, embedding_dim: int):
        super(JointEmbeddingLayer, self).__init__()
        
        self.n_joints = n_joints
        self.embed = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.LayerNorm(embedding_dim))
        
    def forward(self, joints: torch.Tensor):
        batch_size = joints.size(0)
        joints = joints.view(batch_size, self.n_joints, 2)
        return self.embed(joints)


class ModulatedGCNResNetDepthAndUncertaintyEstimator(nn.Module):
    @staticmethod
    def __get_affinity_matrix(dim: int, min_affinity: float =0.0):
        assert dim >= 2
        assert min_affinity >= 0.0 and min_affinity <= 1.0
        
        bones = [
            # upper body    
            (9, 10), # head to site
            (8, 9),  # neck to head
            (7, 8),  # spine1 to neck
                
            (8, 11), # neck to left arm
            (8, 14), # neck to right arm
                
            (11, 12), # left arm to left fore arm
            (12, 13), # left fore arm to left hand
                
            (14, 15), # right arm to right fore arm
            (15, 16), # right fore arm to right hand
                
            # lower body
                
            (0, 7), # hips to spine1
                
            (0, 1), # hips to right up leg
            (1, 2), # right up leg to right leg
            (2, 3), # right leg to right foot
                
            (0, 4), # hips to left up leg
            (4, 5), # left up leg to left leg
            (5, 6) # left leg to left foot
        ]
        
        affinity_mat = min_affinity * torch.ones((dim, dim),
                                   dtype=torch.float, requires_grad=False)
        for bone in bones:
            affinity_mat[bone[0], bone[1]] = 1
            affinity_mat[bone[1], bone[0]] = 1
         
        return affinity_mat
    
    def __init__(
        self,
        base_hidden_dim: int,
        head_hidden_dim: int,
        n_joints: int,
        base_n_res_blocks: int =1,
        head_n_res_blocks: int =1,
        n_layers_per_res_block: int =2):
                
        super().__init__()
        
        # joint embedding
        self.joint_emb_layer = JointEmbeddingLayer(n_joints, base_hidden_dim)
        
        # backbone
        self.backbone = ModulatedGCNResNetBackbone(
            affinity_mat=ModulatedGCNResNetDepthAndUncertaintyEstimator.__get_affinity_matrix(
                dim=n_joints, min_affinity=0.0),
            n_res_blocks=base_n_res_blocks,
            n_layers_per_res_block=n_layers_per_res_block,
            hidden_dim=base_hidden_dim)
        self.backbone_out_proj = nn.Linear(n_joints, 1)
        
        # depth head
        self.depth_head = ResNetDepthHead(
            n_res_blocks=head_n_res_blocks,
            n_layers_per_res_block=n_layers_per_res_block,
            input_dim=base_hidden_dim,
            hidden_dim=head_hidden_dim,
            output_dim=n_joints)
        
        # uncertainty head
        self.uncertainty_head = ResNetDepthUncertaintyHead(
                n_res_blocks=head_n_res_blocks,
                n_layers_per_res_block=n_layers_per_res_block,
                input_dim=base_hidden_dim,
                hidden_dim=head_hidden_dim,
                output_dim=n_joints)
        # save attributes
        self.base_n_res_blocks = base_n_res_blocks
        self.head_n_res_blocks = head_n_res_blocks
        self.n_layers_per_res_block = n_layers_per_res_block
        self.base_hidden_dim = base_hidden_dim
        self.head_hidden_dim = head_hidden_dim
        self.n_joints = n_joints
        
    def forward(self, x_y_coords, img=None, with_uncertainty: bool =True):
        """
        Args:
            x_y_coords: tensor of shape (batch_size, n_joints * 2)
            img: The batch of images. Defaults to None.
        """
        batch_size = x_y_coords.size(0)
        
        # get the joint embeddings
        joint_embeddings = self.joint_emb_layer(x_y_coords)
        
        # pass the coordinates through the backbone
        backbone_output = self.backbone(joint_embeddings, img)
        backbone_output = self.backbone_out_proj(
            backbone_output.transpose(-2, -1)).squeeze()
              
        # pass the backbone output through the depth head and depth uncertainty head
        backbone_output = backbone_output.view(batch_size, -1)
        depth = self.depth_head(backbone_output)
        uncertainty = self.uncertainty_head(backbone_output) if with_uncertainty else None
        
        outputs = {'depth': depth, 'uncertainty': uncertainty}
        return outputs