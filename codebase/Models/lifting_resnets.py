import torch

import torchvision.models as models
import torch.nn           as nn

from Models.residual_lifting import ResNetDepthUncertaintyEstimator
from Models.MLP_Lifting      import MLP


class Network_ResNet(nn.Module):
    def __init__(self, arch_name, drop_rate, embedding_layer_size, batch_norm_eval_imagenet_backbone,
                 number_of_joints, num_inp_lifting, num_out_lifting, load_pretrained_weights, use_residual_lifting_in_resnet,
                 use_batch_norm_mlp):
        super(Network_ResNet, self).__init__()
        self.arch_name            = arch_name #opt.type_lifting_network
        self.drop_rate            = drop_rate #opt.encoder_dropout
        self.embedding_layer_size = embedding_layer_size #opt.embedding_layer_size
        self.batch_norm_eval      = batch_norm_eval_imagenet_backbone #opt.batch_norm_eval_imagenet_backbone
        self.pretrained           = load_pretrained_weights #opt.load_pretrained_weights
        self.number_of_joints     = number_of_joints
        self.num_inp_lifting      = num_inp_lifting
        self.num_out_lifting      = num_out_lifting
        self.use_batch_norm_mlp   = use_batch_norm_mlp
        self.use_residual_lifting_in_resnet = use_residual_lifting_in_resnet #opt.use_residual_lifting_in_resnet

        print('{}'.format(self.arch_name.upper()))
        if self.pretrained:
            print("-- PRETRAINED WITH IMAGENET WEIGHTS.")
        if self.arch_name.lower() == 'resnet18':
            self.model = models.resnet18(pretrained=self.pretrained)
        elif self.arch_name.lower() == 'resnet50':
            self.model = models.resnet50(pretrained=self.pretrained)

        in_feat = self.model.fc.in_features  # Size of the output of the last fc layer (should be 4096)
        if self.batch_norm_eval:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None
        del self.model.fc
        self.dropout = nn.Dropout(self.drop_rate)
        embed_sizes  = [in_feat] + self.embedding_layer_size
        modules      = []
        N_           = len(embed_sizes) - 1
        for i in range(0, N_):
            modules.append(nn.Linear(embed_sizes[i], embed_sizes[i+1]))
        self.embedding_layer = nn.Sequential(*modules)

        if not self.use_residual_lifting_in_resnet:
            print("Will be using the standard MLP designed by Leo.")
            self.final_layer = MLP(d_in=embed_sizes[-1]+self.num_inp_lifting, d_hidden=2048,
                                   d_out=self.num_out_lifting, n_hidden=2, dropout=0.2,
                                   use_batch_norm_mlp=use_batch_norm_mlp)
        else:
            print("Will be using the residual MLP designed by Farouk.")
            self.final_layer = ResNetDepthUncertaintyEstimator(num_joints=self.num_out_lifting, inp_layer=embed_sizes[-1] + self.num_inp_lifting)

    def forward(self, img, x_y_coords):
        # img -- Tensor of size B x 3 x H x W
        # x_y_coords - Tensor of size B x 29
        N   = img.size(0)
        x   = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(img))))
        x   = self.model.layer1(x)
        x   = self.model.layer2(x)
        x   = self.model.layer3(x)
        x   = self.model.layer4(x)
        x   = self.model.avgpool(x)
        x   = x.view(N, -1)
        x   = self.embedding_layer(x)
        x   = torch.cat((x, x_y_coords), dim=1)
        out = self.final_layer(x)
        return out


def get_resnets(config, num_inp_lifting, num_out_lifting, number_of_joints,
                use_batch_norm_mlp):
    lifting_net = Network_ResNet(num_inp_lifting=num_inp_lifting, num_out_lifting=num_out_lifting,
                                 number_of_joints=number_of_joints, arch_name=config.type_lifting_network,
                                 drop_rate=config.encoder_dropout, embedding_layer_size=config.embedding_layer_size,
                                 batch_norm_eval_imagenet_backbone=config.batch_norm_eval_imagenet_backbone,
                                 use_residual_lifting_in_resnet=config.use_residual_lifting_in_resnet,
                                 load_pretrained_weights=config.load_pretrained_weights,
                                 use_batch_norm_mlp=use_batch_norm_mlp)
    return lifting_net