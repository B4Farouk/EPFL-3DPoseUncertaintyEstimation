import torch
import torch.nn as nn

from Models.kcs_utils import KCS_util

functions       = {"leakyrelu": nn.LeakyReLU(), "relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "Tanh": nn.Tanh()}


class KCSi_discriminator(nn.Module):
    def __init__(self, activation, in_features, channel, mid_channel, predict_with_sigmoid=True):
        super(KCSi_discriminator, self).__init__()
        self.residual_layer_1     = nn.Linear(in_features, channel, bias=True)
        self.residual_layer_2     = nn.Linear(channel, channel, bias=True)
        self.residual_layer_3     = nn.Linear(channel, channel, bias=True)
        self.mlp_layer_1          = nn.Linear(channel, mid_channel, bias=True)
        self.mlp_layer_pred       = nn.Linear(mid_channel, 1, bias=False)
        self.activation           = activation
        self.predict_with_sigmoid = predict_with_sigmoid

    def forward(self, x):
        res1  = self.activation(self.residual_layer_1(x))
        res2  = self.activation(self.residual_layer_2(res1))
        res3  = self.activation(self.residual_layer_3(res2) + res1)
        mlp_1 = self.activation(self.mlp_layer_1(res3))
        if self.predict_with_sigmoid:
            mlp_pred = nn.Sigmoid()(self.mlp_layer_pred(mlp_1))
        else:
            mlp_pred = self.activation(self.mlp_layer_pred(mlp_1))

        return mlp_pred


class Pos3dDiscriminator(nn.Module):
    def __init__(self, num_joints: int, joints_ordering: dict, bones: list, partitions: dict, channel: int, mid_channel: int, activation: str):
        super(Pos3dDiscriminator, self).__init__()
        activation       = functions[activation]
        self.num_joints  = num_joints
        self.channel     = channel
        self.mid_channel = mid_channel
        self.activation  = activation
        
        self.kcs_util    = KCS_util(num_joints=num_joints, joints_ordering=joints_ordering, bones=bones, partitions=partitions)
        features         = self.kcs_util.compute_features()

        # activation, in_features, channel, mid_channel
        dict_vals = {'activation' : self.activation, 'in_features' : features,
                     'channel' : self.channel, 'mid_channel' : self.mid_channel}

        self.kcsi_ll    = KCSi_discriminator(**dict_vals) # activation=self.activation, features, self.channel, self.mid_channel)
        self.kcsi_rl    = KCSi_discriminator(**dict_vals) # activation=self.activation, features, self.channel, self.mid_channel)
        self.kcsi_torso = KCSi_discriminator(**dict_vals) # activation=self.activation, features, self.channel, self.mid_channel)
        self.kcsi_lh    = KCSi_discriminator(**dict_vals) # activation=self.activation, features, self.channel, self.mid_channel)
        self.kcsi_rh    = KCSi_discriminator(**dict_vals) # activation=self.activation, features, self.channel, self.mid_channel)
        self.optimizer  = None

    def forward(self, inputs_3d):
        Num_samples   = inputs_3d.size(0)
        ext_inputs_3d = self.kcs_util.extend(inputs_3d)
        ext_inputs_3d = self.kcs_util.center(ext_inputs_3d)
        bv            = self.kcs_util.bone_vectors(ext_inputs_3d)
        kcs_ll        = self.kcs_util.kcs_layer(bv, "ll")
        kcs_rl        = self.kcs_util.kcs_layer(bv, "rl")
        kcs_torso     = self.kcs_util.kcs_layer(bv, "torso")
        kcs_lh        = self.kcs_util.kcs_layer(bv, "lh")
        kcs_rh        = self.kcs_util.kcs_layer(bv, "rh")
        ll_pred       = self.kcsi_ll(kcs_ll.view((Num_samples, -1))) #inputs_3d.size(0), -1)))
        rl_pred       = self.kcsi_rl(kcs_rl.view((Num_samples, -1)))
        torso_pred    = self.kcsi_torso(kcs_torso.view((Num_samples, -1)))
        lh_pred       = self.kcsi_lh(kcs_lh.view((Num_samples, -1)))
        rh_pred       = self.kcsi_rh(kcs_rh.view((Num_samples, -1)))
        out           = torch.cat([ll_pred, rl_pred, torso_pred, lh_pred, rh_pred], dim=1)
        # out           = torch.stack([ll_pred, rl_pred, torso_pred, lh_pred, rh_pred])
        return out





def obtain_discriminator_3d_network(num_joints, joints_ordering, bones, partitions, config):
    print("Obtaining the 3D pose discriminators based on Part-Based KCS discriminator.")
    assert config.type_3d_discriminator == 'mehdi_kcs'
    channel     = config.mehdi_channel
    mid_channel = config.mehdi_mid_channel
    activation  = config.activation_mehdi
    disc_net    = Pos3dDiscriminator(num_joints=num_joints, joints_ordering=joints_ordering, bones=bones, partitions=partitions,
                                     channel=channel, mid_channel=mid_channel, activation=activation)
    return disc_net