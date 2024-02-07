import torch
import torch.nn as nn

from typing import Optional


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


class ResBlock(nn.Module):
    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        activation: str ="relu",
        normalization: Optional[str] =None,
        activate_last: bool =False):

        assert isinstance(n_layers, int) and n_layers > 1
        assert isinstance(hidden_dim, int) and hidden_dim > 0
        assert isinstance(activation, str) 
        assert normalization is None or isinstance(normalization, str)
        
        super(ResBlock, self).__init__()
        
        layers = nn.ModuleList([])        
        # all layers except last
        for _ in range(n_layers - 1): # -1 because the last layer is added below
            layers.append(get_norma_layer(normalization, dim=hidden_dim))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(get_activ_fn(activation))
        # last layer
        layers.append(get_norma_layer(normalization, dim=hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if activate_last:
            layers.append(get_activ_fn(activation))
        self.ff = nn.Sequential(*layers)
        
        # save attributes
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.activaté_last = activate_last
        self.normalization = normalization
        
    def forward(self, input):
        return input + self.ff(input)


class ResNet(nn.Module):
    def __init__(
        self,
        n_res_blocks: int,
        n_layers_per_res_block: int,
        hidden_dim: int,
        activation: str ="relu",
        normalization: Optional[str] =None,
        resblock_activate_last: bool =False):
        
        super(ResNet, self).__init__()
        
        assert isinstance(n_res_blocks, int) and n_res_blocks > 0
        assert isinstance(n_layers_per_res_block, int) and n_layers_per_res_block > 0
        assert isinstance(hidden_dim, int) and hidden_dim > 0
        assert isinstance(activation, str)
        assert normalization is None or isinstance(normalization, str)
        
        layers = nn.ModuleList([])
        layers.extend([
            ResBlock(
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


class ResNetBackbone(ResNet):
    def __init__(
        self,
        n_res_blocks: int,
        n_layers_per_res_block: int,
        input_dim: int,
        hidden_dim: int,
        normalization: Optional[str] =None):
        
        super(ResNetBackbone, self).__init__(
            n_res_blocks=n_res_blocks,
            n_layers_per_res_block=n_layers_per_res_block,
            hidden_dim=hidden_dim,
            activation="silu",
            normalization=normalization)
        
        self.input_layer = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim
            else nn.Identity())

        # save attributes
        self.input_dim = input_dim
        
    def forward(self, input, img=None):
        input = self.input_layer(input)
        return super().forward(input)


class ResNetDepthHead(ResNet):
    def __init__(
        self,
        n_res_blocks: int,
        n_layers_per_res_block: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        skipped: bool =True):
        
        super(ResNetDepthHead, self).__init__(
            n_res_blocks=n_res_blocks,
            n_layers_per_res_block=n_layers_per_res_block,
            hidden_dim=hidden_dim,
            activation="silu",
            normalization=None)
        
        self.lin_adapt = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim
            else nn.Identity())
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.skipped = skipped
        
    def forward(self, input, img=None):
        input = self.lin_adapt(input)
        output = super().forward(input)
        output = input + output if self.skipped else output
        output = self.output_layer(output)
        return output
 

class ResNetDepthUncertaintyHead(ResNet):
    def __init__(
        self,
        n_res_blocks: int,
        n_layers_per_res_block: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        out_activ_fn: str ="softplus",
        skipped: bool =True):
        
        super(ResNetDepthUncertaintyHead, self).__init__(
            n_res_blocks=n_res_blocks,
            n_layers_per_res_block=n_layers_per_res_block,
            hidden_dim=hidden_dim,
            activation="silu",
            normalization=None)
        
        self.lin_adapt = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim
            else nn.Identity())
            
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            get_activ_fn(out_activ_fn))
        
        self.output_activation = out_activ_fn
        self.skipped = skipped
        
    def forward(self, input, img=None):
        input = self.lin_adapt(input)
        output = super().forward(input)
        output = input + output if self.skipped else output
        output = self.output_layer(output)
        return output


class ResNetDepthAndUncertaintyEstimator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        base_hidden_dim: int,
        head_hidden_dim: int,
        n_joints: int,
        base_n_res_blocks: int,
        head_n_res_blocks: int,
        n_layers_per_res_block: int =2,
        base_normalization: Optional[str] =None):
        
        super(ResNetDepthAndUncertaintyEstimator, self).__init__()
        
        self.backbone = ResNetBackbone(
            n_res_blocks=base_n_res_blocks,
            n_layers_per_res_block=n_layers_per_res_block,
            input_dim=input_dim,
            hidden_dim=base_hidden_dim,
            normalization=base_normalization)
        
        self.depth_head = ResNetDepthHead(
            n_res_blocks=head_n_res_blocks,
            n_layers_per_res_block=n_layers_per_res_block,
            input_dim=base_hidden_dim,
            hidden_dim=head_hidden_dim,
            output_dim=n_joints,
            skipped=True)
        
        self.uncertainty_head = ResNetDepthUncertaintyHead(
                n_res_blocks=head_n_res_blocks,
                n_layers_per_res_block=n_layers_per_res_block,
                input_dim=base_hidden_dim,
                hidden_dim=head_hidden_dim,
                output_dim=n_joints,
                skipped=True)
    
        # save attributes
        self.input_dim = input_dim
        self.base_hidden_dim = base_hidden_dim
        self.head_hidden_dim = head_hidden_dim
        self.output_dim = n_joints
        self.base_n_res_blocks = base_n_res_blocks
        self.head_n_res_blocks = head_n_res_blocks
        self.n_layers_per_res_block = n_layers_per_res_block

    def forward(self, x_y_coords, img=None, with_uncertainty: bool=True):
        """
        Args:
            x_y_coords: tensor of shape (batch_size, n_joints * 2)
            img: The batch of images. Defaults to None.
        """        
        batch_size = x_y_coords.size(0)
        x_y_coords = x_y_coords.view(batch_size, -1)
        
        backbone_output = self.backbone(x_y_coords, img)
        depth = self.depth_head(backbone_output, img)
        uncertainty = self.uncertainty_head(backbone_output, img) if with_uncertainty else None
        
        outputs = {'depth': depth, 'uncertainty': uncertainty}
        return outputs
 