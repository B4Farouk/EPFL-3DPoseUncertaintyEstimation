import torch


class kcs_util:
    def __init__(self, joints, bones, partitions, pelvis_name=None, num_joints=None):
        self.num_joints  = num_joints
        self.joints      = joints
        self.bones       = bones
        self.partitions  = partitions
        self.C           = None
        self.pelvis_name = pelvis_name or 'Hips'

    def extend(self, x):
        return x

    def compute_features(self):
        return (self.num_joints - 1) ** 2

    def c(self, joint_i, joint_j):
        ls                       = [0 for i in range(self.num_joints)]
        ls[self.joints[joint_i]] = 1
        ls[self.joints[joint_j]] = -1
        return ls

    def bone_vectors(self, ext_input):
        device = ext_input.device
        if self.C is None:
            self.C = torch.tensor([self.c(bone[0], bone[1]) for bone in self.bones]).transpose(1, 0).type(torch.FloatTensor).to(device=device) 
            # type_as(ext_input)#type(torch.FloatTensor)

        C         = self.C.repeat([ext_input.size(0), 1, 1])
        ext_input = ext_input.permute(0, 2, 1).type(torch.FloatTensor).to(device=device) 
        # type(torch.FloatTensor)
        B         = torch.matmul(ext_input, C)
        B         = B.permute(0, 2, 1)
        return B

    def center(self, inputs_3d):
        index = self.joints[self.pelvis_name]
        return inputs_3d - inputs_3d[:, index: index + 1, :]

    def kcs_layer(self, bv, region):
        index             = self.partitions[region]
        mask              = torch.zeros_like(bv)
        mask[:, index, :] = 1
        bv                = bv * mask
        kcs               = torch.matmul(bv, bv.permute(0, 2, 1))
        return kcs


class kcs_util17(kcs_util):
    def __init__(self, joints, bones, partitions):
        super().__init__(joints, bones, partitions, num_joints=17)


class kcs_util13(kcs_util):
    def __init__(self, joints, bones, partitions):
        super().__init__(joints, bones, partitions, num_joints=16)

    def extend(self, x):
        X                             = torch.empty((x.size(0), self.num_joints, 3))
        X[:, 0:x.size()[1], :]        = x
        X[:, self.joints['Hips'], :]  = (x[:, self.joints['Lhip'], :] + x[:, self.joints['Rhip'], :]) / 2
        X[:, self.joints['Neck'], :]  = (x[:, self.joints['Lshoulder'], :] + x[:, self.joints['Rshoulder'], :]) / 2
        X[:, self.joints['Spine'], :] = (X[:, self.joints['Neck'], :] + X[:, self.joints['Hips'], :]) / 2
        return X


def KCS_util(num_joints: int, joints_ordering: dict, bones: list, partitions: dict):
    if num_joints == 13:
        func = kcs_util13
    else:
        func = kcs_util17
    out = func(joints_ordering, bones, partitions)
    return out


