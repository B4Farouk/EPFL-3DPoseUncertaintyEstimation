import os
import json
import yaml
import pickle
import torch
import numpy as np
import errno
import sys
import cv2
import dgl


import torch.nn.functional as F
from PIL       import Image
from itertools import zip_longest

# seeding
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)

IMAGENET_MEAN, IMAGENET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
Image.MAX_IMAGE_PIXELS      = None
image_shape_resnet152       = [384, 384]
key_val_2d_info             = '2D_information'


def json_read(filename):
    try:
        with open(os.path.abspath(filename)) as f:
            data = json.load(f)
        return data
    except:
        raise ValueError("Unable to read JSON {}".format(filename))


def json_write(filename, data):
    try:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.abspath(filename), 'w') as f:
            json.dump(data, f, indent=2)
    except:
        raise ValueError("Unable to write JSON {}".format(filename))


def yaml_read(filename):
    with open(filename, 'r') as f:
        data = yaml.load(f)
    return data


def yaml_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.abspath(filename), 'w') as f:
        yaml.dump(data, f, default_flow_style=False, width=1000)


def pickle_read(filename, **kwargs):
    with open(filename, "rb") as f:
        data = pickle.load(f, **kwargs)
    return data


def pickle_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_state_dict_from_multiple_gpu_to_single(old_state_dict):
    """
    Function to convert the parameters stored in multiple GPU to single GPU. Basically remove the torch.nn.module
    :param old_state_dict: The dictionary containing the parameters which are stored in the multiple gpu setup.
    :return:
    """
    new_state_dict = {}
    for key in old_state_dict:
        if 'module.' in key:
            new_key = key.replace('module.', '')
        else:
            new_key = key
        new_state_dict[new_key] = old_state_dict[key]
    return new_state_dict


if torch.cuda.is_available():
    device   = 'cuda:0'
    use_cuda = True
else:
    device   = 'cpu'
    use_cuda = False


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file    = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()



def get_optimizer_scheduler(
    params : list,
    optimizer_type : str,
    scheduler_type : str,
    step_val : list,
    gamma_val : float,
    string_val: str = 'all the parameters'):
    """
    :param optimizer_type: A string depicting the optimizer that will be used to learn the learnable parameters in params.
    :param params: A list of parameters that are needed to be learnt.
    :param scheduler_type: A string denoting the scheduler that will be used to anneal the learning rate of the parameters.
    :param step_val: The step size to be used for Step Learning Scheduler.
    :param gamma_val: The Annealing Factor for the Scheduler.
    :param string_val :
    :return: The optimizer and the Scheduler.
    """
    if len(params) == 0:
        optimizer = None
    else:
        if optimizer_type.lower() == 'adam':
            print("Using the Adam Optimizer for learning {}.".format(string_val))
            optimizer = torch.optim.Adam(params, amsgrad=True)

        elif optimizer_type.lower() == 'adamw':
            print("Using the AdamW Optimizer for learning {}.".format(string_val))
            optimizer = torch.optim.AdamW(params, amsgrad=True)

        elif optimizer_type.lower() == 'rmsprop':
            print("Using the RMSPROP Optimizer for learning {}.".format(string_val))
            optimizer = torch.optim.RMSprop(params)

        elif optimizer_type.lower() == 'rmsprop_added':
            print("Using the RMSPROP Optimizer with added functionalities for learning {}.".format(string_val))
            optimizer = torch.optim.RMSprop(params, alpha=0.9, momentum=0.9)

        else:
            assert optimizer_type.lower() == 'sgd'
            print("Using the SGD Optimizer for learning.")
            optimizer = torch.optim.SGD(params, momentum=0.9)

    if scheduler_type == 'step':
        print("Using the Step Scheduler for the optimization process for learning {}.".format(string_val))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma_val, step_size=step_val[0])
    elif scheduler_type == 'exp':
        print("Using the Exp Scheduler for the optimization process for learning {}.".format(string_val))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_val)
    elif scheduler_type == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=5e-5,
            max_lr=5e-4,
            step_size_up=2,
            mode="exp_range",
            cycle_momentum=False,
            #base_momentum=0.8,
            #max_momentum=0.9,
            gamma=0.97)
    else:
        print("No scheduler for the optimization process for learning {}.".format(string_val))
        scheduler = None

    print("\n")
    return optimizer, scheduler



optimizer_2d_pose_estimator_name = 'optimizer_2d_pose_estimator'
scheduler_2d_pose_estimator_name = 'scheduler_2d_pose_estimator'
optimizer_refine_net_name        = 'optimizer_refine_net'
scheduler_refine_net_name        = 'scheduler_refine_net'
optimizer_lifting_net_name       = 'optimizer_lifting_net'
scheduler_lifting_net_name       = 'scheduler_lifting_net'
optimizer_embedding_network_name = 'optimizer_embedding_network'
scheduler_embedding_network_name = 'scheduler_embedding_network'

optimizer_discriminator_3d_network_name = 'optimizer_discriminator_3d_network'
scheduler_discriminator_3d_network_name = 'scheduler_discriminator_3d_network'


def get_state_dict_from_multiple_gpu_to_single(old_state_dict):
    new_state_dict = {}
    for key in old_state_dict:
        if 'module.' in key:
            new_key = key.replace('module.', '')
        else:
            new_key = key
        new_state_dict[new_key] = old_state_dict[key]
    return new_state_dict


def transfer_partial_weights(model, pretrained_state_dict):
    model_state_dict = model.state_dict()
    for name in model_state_dict:
        if name in pretrained_state_dict:
            if pretrained_state_dict[name].size() == model_state_dict[name].size():
                model_state_dict[name] = pretrained_state_dict[name]
            else:
                print("Even though the parameter {} is present, the sizes of the tensor are not matching and thus can't be loaded.".format(name))
                print("The size of parameter {} in model is ".format(name), model_state_dict[name].size(),
                      " and that of pretrained dict is ", pretrained_state_dict[name].size())
                print("\n")
        else:
            print("The {} parameter is missing.".format(name))
    model.load_state_dict(model_state_dict)
    return model


def perform_softmax_over_heatmaps_and_get_normalized_keypoints(hm_i, temp):
    N, num_joints, dim_y, dim_x = hm_i.size(0), hm_i.size(1), hm_i.size(2), hm_i.size(3)
    hm_i_soft_max = hm_i.view(N, num_joints, -1)
    hm_i_soft_max = F.softmax(hm_i_soft_max / temp, dim=2)
    hm_i_soft_max = hm_i_soft_max.view(N, num_joints, dim_y, dim_x)
    device        = hm_i_soft_max.device

    grid    = torch.arange(dim_x).float().to(device) / dim_x * 2 - 1
    grid    = grid[None, None]
    point_x = torch.sum(hm_i_soft_max.sum(dim=2) * grid, dim=-1)

    grid    = torch.arange(dim_y).float().to(device) / dim_y * 2 - 1
    grid    = grid[None, None]
    point_y = torch.sum(hm_i_soft_max.sum(dim=3) * grid, dim=-1)

    keypoints_2d_normalized = torch.cat([point_x.unsqueeze(dim=2), point_y.unsqueeze(dim=2)], dim=2)
    return keypoints_2d_normalized


def obtain_heatmaps_from_2d_pose_estimator_model(pose_model, inp_pose, batch_size, num_joints, num_cameras):
    heatmaps, confidences = pose_model(inp_pose)# TODO. Confidences.
    if confidences is not None:
        # confidences are of shape batch_size*num_cameras x num_joints.
        confidences = confidences.view(batch_size, num_cameras, num_joints)
        # confidences are of shape batch_size x num_cameras x num_joints.
        confidences = confidences.transpose(1, 2)
        # confidences are of shape batch_size x num_joints x num_cameras.

    return heatmaps, confidences


def area(boxA, boxB):
    min_xA, min_yA, max_xA, max_yA = boxA[0], boxA[1], boxA[2], boxA[3]
    min_xB, min_yB, max_xB, max_yB = boxB[0], boxB[1], boxB[2], boxB[3]
    dx = min(max_xA, max_xB) - max(min_xA, min_xB)
    dy = min(max_yA, max_yB) - max(min_yA, min_yB)
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return 0.0


def make_circle(center=(0, 0, 0), radius=0.16, N=20):
    angles = np.linspace(0, 2 * np.pi, N)
    points = []
    for a in angles:
        points.append((np.cos(a) * radius + center[0], np.sin(a) * radius + center[1], center[2]))
    return np.array(points)


def bbox_from_points(points_2d, pb=0.2):
    xmin, ymin = points_2d.min(axis=0)
    xmax, ymax = points_2d.max(axis=0)
    s    = np.mean([xmax-xmin, ymax-ymin])*pb
    xmin = xmin-s
    xmax = xmax+s
    ymin = ymin-s
    ymax = ymax+s
    bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
    return bbox


def print_num_samples(mode, labels):
    N = len(labels); num_anno = sum(labels)
    print("----------"*10)
    print("The number of Samples for {} is {}".format(mode, N))
    print("The number of Annotated Samples for {} is {}.".format(mode, num_anno))
    print("The number of UnAnnotated Samples for {} is {}.".format(mode, N-num_anno))
    print("----------"*10, end='\n')


def compute_intersection(bbox, bboxes):
    _bboxes = np.array(bboxes)
    area    = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    xmins   = np.maximum(bbox[0], _bboxes[:, 0])
    ymins   = np.maximum(bbox[1], _bboxes[:, 1])
    xmaxs   = np.minimum(bbox[2], _bboxes[:, 2])
    ymaxs   = np.minimum(bbox[3], _bboxes[:, 3])
    w             = np.maximum(0.0, xmaxs - xmins + 10)
    h             = np.maximum(0.0, ymaxs - ymins + 10)
    intersections = w * h
    return (intersections / area).tolist()


def im_to_torch(img):
    img = np.array(img)
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def torch_to_im(img):
    if img.dim() == 3:
        img = to_numpy(img)
        img = np.transpose(img, (1, 2, 0))  # C*H*W
    else:
        img = to_numpy(img)
        img = np.transpose(img, (0, 2, 3, 1))
    return img


def cropBox(img, ul, br, resH, resW):
    ul   = ul.int()
    br   = (br - 1).int()
    lenH = max((br[1] - ul[1]).item(), (br[0] - ul[0]).item() * resH / resW)
    lenW = lenH * resW / resH
    if img.dim() == 2:
        img = img[np.newaxis, :]

    box_shape = [(br[1] - ul[1]).item(), (br[0] - ul[0]).item()]
    pad_size  = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]
    # Padding Zeros
    if ul[1] > 0:
        img[:, :ul[1], :] = 0
    if ul[0] > 0:
        img[:, :, :ul[0]] = 0
    if br[1] < img.shape[1] - 1:
        img[:, br[1] + 1:, :] = 0
    if br[0] < img.shape[2] - 1:
        img[:, :, br[0] + 1:] = 0

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :]  = np.array([ul[0] - pad_size[1], ul[1] - pad_size[0]], np.float32)
    src[1, :]  = np.array([br[0] + pad_size[1], br[1] + pad_size[0]], np.float32)
    dst[0, :]  = 0
    dst[1, :]  = np.array([resW - 1, resH - 1], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    trans      = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    dst_img    = cv2.warpAffine(torch_to_im(img), trans, (resW, resH), flags=cv2.INTER_LINEAR)
    return im_to_torch(torch.Tensor(dst_img))


def crop_from_dets(img, boxes, inps, pt1, pt2, inputResH, inputResW, scaleRate, bg_img=None):
    imght    = img.size(1)
    imgwidth = img.size(2)
    tmp_img  = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft      = torch.Tensor((float(box[0]), float(box[1])))
        bottomRight = torch.Tensor((float(box[2]), float(box[3])))
        ht          = bottomRight[1] - upLeft[1]
        width       = bottomRight[0] - upLeft[0]

        upLeft[0]      = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1]      = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)
        try:
            inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, inputResH, inputResW)
            if bg_img is not None:
                tmp_bg_img = bg_img
                tmp_bg_img[0].add_(-0.406)
                tmp_bg_img[1].add_(-0.457)
                tmp_bg_img[2].add_(-0.480)
                bg_inp = cropBox(tmp_bg_img.clone(), upLeft, bottomRight, inputResH, inputResW)
            else:
                bg_inp = None

        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight
    return inps, pt1, pt2, bg_inp


def square_the_bbox(bbox):
    left, top, right, bottom = bbox
    width  = right - left
    height = bottom - top
    if height < width:
        center = (top + bottom) * 0.5
        top = int(round(center - width * 0.5))
        bottom = top + width
    else:
        center = (left + right) * 0.5
        left   = int(round(center - height * 0.5))
        right  = left + height
    return left, top, right, bottom


def scale_bbox_resnet152(bbox, scale):
    left, upper, right, lower = bbox
    width, height             = right - left, lower - upper

    x_center, y_center        = (right + left) // 2, (lower + upper) // 2
    new_width, new_height     = int(scale * width), int(scale * height)
    new_left                  = x_center - new_width // 2
    new_right                 = new_left + new_width

    new_upper = y_center - new_height // 2
    new_lower = new_upper + new_height
    bbox_new  = [new_left, new_upper, new_right, new_lower]
    return bbox_new


def crop_image_resnet152(image, bbox):
    """Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros
    Args:
        image numpy array of shape (height, width, 3): input image
        bbox tuple of size 4: input bbox (left, upper, right, lower)
    Returns:
        cropped_image numpy array of shape (height, width, 3): resulting cropped image
    """
    image_pil = Image.fromarray(image)
    image_pil = image_pil.crop(bbox)
    return np.asarray(image_pil)


def resize_image_resnet152(image, shape):
    image_resized = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return image_resized


def normalize_image_resnet152(image):
    """Normalizes image using ImageNet mean and std
    Args:
        image numpy array of shape (h, w, 3): image
    Returns normalized_image numpy array of shape (h, w, 3): normalized image
    """
    return (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD



def obtain_parameters_to_optimize(model_2d_pose_estimator, lifting_network, embedding_network, config):
    """
    Function to Return the Learnable parameters of various models that we need to optimize.
    :param model_2d_pose_estimator: The 2D pose Estimator Model.
    :param lifting_network:         The 2D to 3D Lifting Network.
    :param embedding_network:       The 3D Pose to Embedding and then to the 2D Pose Network.
    :param config:                  The Configuration File consisting of the arguments of training.
    :return:                        The Parameters that are to be optimized using an optimizer.
    """
    params = []; params_dict = {}
    if config.train_2d_pose_estimator:
        assert model_2d_pose_estimator is not None
        print("We will be Learning the Parameters of the 2D Pose Estimator Model with the "
              "Learning Rate = {} and Weight Decay = {}.".format(config.pose_lr, config.pose_wd))
        params_model_2d_pose_estimator = {'params': model_2d_pose_estimator.parameters(), 'lr' : config.pose_lr, 'weight_decay': config.pose_wd}
        params.append(params_model_2d_pose_estimator)
        params_dict['model_2d_pose_estimator'] = [params_model_2d_pose_estimator]
    else:
        print("We are not Learning the Parameters of the 2D Pose Estimator Model.")

    if config.train_lifting_net and lifting_network is not None:
        # assert lifting_network is not None
        if config.type_lifting_network in ['mlp', 'resnet', 'temporal_Pavllo', 'modulated_gcn']:
            print("We will be Learning the Parameters of the 2D to 3D Lifting Network with the "
                  "Learning Rate = {} and Weight Decay = {}.".format(config.lr_lifting_net, config.wd_lifting_net))
            params_lifting_net = {'params': lifting_network.parameters(), 'lr': config.lr_lifting_net, 'weight_decay': config.wd_lifting_net}
            params.append(params_lifting_net)
            params_dict['lifting_network'] = [params_lifting_net]
        else:
            print("We will be Learning the Parameters of the ResNet Based 2D to 3D Lifting Network with the "
                  "Learning Rate = {} and Weight Decay = {} "
                  "for the BackBone Layers.".format(config.lifting_lr_resnets_backbone, config.lifting_wd_resnets_backbone))
            params_lifting_net_backbone = {'params': lifting_network.network.model.parameters(), 'lr': config.lifting_lr_resnets_backbone,
                                           'weight_decay': config.lifting_wd_resnets_backbone}
            params.append(params_lifting_net_backbone)
            params_lifter_dict         = [params_lifting_net_backbone]

            print("We will be Learning the Parameters of Lifter present in the ResNet Based 2D to 3D Lifting Network with the "
                  "Learning Rate = {} and Weight Decay = {}.".format(config.lifting_lr_resnets_lifter, config.lifting_wd_resnets_lifter))
            params_lifter                   = list(lifting_network.network.embedding_layer.parameters()) \
                                              + list(lifting_network.network.final_layer.parameters())
            params_lifting_net_not_backbone = {'params': params_lifter, 'lr': config.lifting_lr_resnets_lifter,
                                               'weight_decay': config.lifting_wd_resnets_lifter}
            params.append(params_lifting_net_not_backbone)
            params_lifter_dict.append(params_lifting_net_not_backbone)
            params_dict['lifting_network'] = params_lifter_dict

    else:
        print("We are not Learning the Parameters of the 2D to 3D Lifting Network.")

    if config.train_embedding_network and not(embedding_network is None):
        print("We will be Learning the Parameters of the Embedding Network with the "
              "Learning Rate = {} and Weight Decay = {}.".format(config.lr_embedding_network, config.wd_embedding_network))
        params_embedding_network = {'params': embedding_network.parameters(), 'lr': config.lr_embedding_network, 'weight_decay': config.wd_embedding_network}
        params.append(params_embedding_network)
        params_dict['embedding_network'] = [params_embedding_network]
    else:
        print("We are not Learning the Parameters of the Embedding Network.")

    if not config.use_different_optimizers:
        return params
    else:
        return params_dict


parts_considered_for_triangulation_crowd_pose_sport_center = torch.from_numpy(np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))

def pass_through_2D_Pose_Estimator(model_2d_pose_estimator: torch.nn.Module, inp_pose_model: torch.Tensor,
                                   number_of_joints : int, number_of_cameras : int, batch_size : int,
                                   temp_softmax: float, use_2D_GT_poses_directly: bool,
                                   use_2D_DET_poses_directly : bool, target_pose_2d_norm: torch.Tensor,
                                   device: str, dataset_name: str, pose_model_name: str,
                                   det_pose_2d_norm: torch.Tensor = None):
    if model_2d_pose_estimator is not None:
        heatmaps, conf_pre = obtain_heatmaps_from_2d_pose_estimator_model(pose_model=model_2d_pose_estimator,
                                                                          inp_pose=inp_pose_model,
                                                                          num_joints=number_of_joints,
                                                                          num_cameras=number_of_cameras,
                                                                          batch_size=batch_size)
        # if conf_pre is not None it's shape should be batch_size x number_of_joints x number_of_cameras.
        keypoints_det_norm = perform_softmax_over_heatmaps_and_get_normalized_keypoints(hm_i=heatmaps, temp=temp_softmax)
        # keypoints_det_norm will be of shape batch_size*number_of_cameras x number of joints x 2.
        if dataset_name.lower() == 'sport_center':
            # and pose_model_name.lower() == 'crowd_pose':
            indexes            = parts_considered_for_triangulation_crowd_pose_sport_center.long().to(device)
            keypoints_det_norm = keypoints_det_norm[:, indexes, :]
        else:
            keypoints_det_norm = keypoints_det_norm[:, 0:number_of_joints, ...]

    else:
        if use_2D_GT_poses_directly:
            keypoints_det_norm = target_pose_2d_norm
        else:
            if use_2D_DET_poses_directly:
                keypoints_det_norm = det_pose_2d_norm if det_pose_2d_norm is not None else target_pose_2d_norm
            else:
                keypoints_det_norm = None

        keypoints_det_norm = keypoints_det_norm.view(-1, number_of_joints, 2)
        conf_pre           = torch.ones(batch_size, number_of_joints, number_of_cameras)
    return keypoints_det_norm, conf_pre


def get_predefined_edge_relation(edge_weights, relations, batch_graphs, device, drop_edges, edges_drop_rate, type_data):
    if edge_weights == 'ones' or edge_weights == 'random':
        pre_def_edge_relations = {}
        for relation in relations:
            num_edges_relation = batch_graphs.num_edges(relation)
            if edge_weights == 'ones':
                edge_weights_relation = torch.ones(num_edges_relation, 1)
            else:
                edge_weights_relation = torch.randn(num_edges_relation, 1)
            if drop_edges:
                print("Dropping a certain {} % of edges for every relation.".format(edges_drop_rate))
                num_edges_to_drop                 = int(num_edges_relation * edges_drop_rate)
                indexes                           = torch.randint(low=0, high=num_edges_relation, size=(num_edges_to_drop, 1)).long()
                edge_weights_relation[indexes, :] = 0
            pre_def_edge_relations[relation]      = edge_weights_relation.type_as(type_data).to(device)
    else:
        raise NotImplementedError
    return pre_def_edge_relations


def get_pelvis(pose, pelvis_idx, rhip_idx, lhip_idx, return_z):
    # pose should be of shape Number of Different Poses x number of joints x 3
    if pose.dim() == 2:
        pose = pose.unsqueeze(dim=0)

    if pelvis_idx != -1:
        pelvis = pose[:, pelvis_idx, :]
    else:
        assert rhip_idx != -1 and lhip_idx != -1
        lhip   = pose[:, lhip_idx, :]
        rhip   = pose[:, rhip_idx, :]
        pelvis = (rhip + lhip) / 2

    if return_z:
        return pelvis[:, -1]
    else:
        return pelvis


def obtain_inp_target_2d_loss(target_pose_2d_norm, keypoints_det_norm, pose_2d_proj_norm, target_2d,
                              pose_2d_proj, keypoints_det_dist, labels, number_of_joints,
                              not_use_norm_pose_2d_in_loss: bool, use_dets_for_labeled_in_loss: bool,
                              swap_inp_tar_unsup: bool):
    """
    Function to obtain the predictions and targets for 2D loss.
    :param target_pose_2d_norm: The target 2D poses in the normalized coordinates.
    :param keypoints_det_norm: The detected 2D poses in the normalized coordinates.
    :param pose_2d_proj_norm: The 2D poses projected from the triangulated 3D in the normalized coordinates.
    :param target_2d: The 2D poses in the image coordinates.
    :param pose_2d_proj: The 2D poses projected from the triangulated 3D in the image coordinates.
    :param keypoints_det_dist: The detected 2D poses in the image coordinates.
    :param labels: The supervisory labels.
    :param number_of_joints: The number of joints present in each of the poses.
    :param not_use_norm_pose_2d_in_loss: If True, the losses will be calculated using the poses in image coordinates,
                                         else it will be calculated using the poses in the normalized coordinates.
    :param use_dets_for_labeled_in_loss: If True, we will use the detected 2D poses as an input for the supervised/labeled
                                         samples in the loss calculation, else it will be the projection of the 3D pose.
    :param swap_inp_tar_unsup : TODO
    :return:
    """
    target_pose_2d_norm = target_pose_2d_norm.reshape(-1, number_of_joints, 2) if target_pose_2d_norm is not None else None
    keypoints_det_norm  = keypoints_det_norm.reshape(-1, number_of_joints, 2)  if keypoints_det_norm  is not None else None
    pose_2d_proj_norm   = pose_2d_proj_norm.reshape(-1, number_of_joints, 2)   if pose_2d_proj_norm   is not None else None
    target_2d           = target_2d.reshape(-1,  number_of_joints, 2)          if target_2d           is not None else None
    pose_2d_proj        = pose_2d_proj.reshape(-1, number_of_joints, 2)        if pose_2d_proj        is not None else None
    keypoints_det_dist  = keypoints_det_dist.reshape(-1, number_of_joints, 2)  if keypoints_det_dist  is not None else None
    labels              = labels.reshape(-1, 1, 1)
    if not_use_norm_pose_2d_in_loss is False: # print("Using Poses in Normalized Image Coordinates in The Loss.")
        targets = [target_pose_2d_norm]
        if not use_dets_for_labeled_in_loss: #print("A")
            inputs = [pose_2d_proj_norm]
            # For the labeled Samples.
        else: #print("B")
            inputs = [keypoints_det_norm]
            # For the labeled Samples.

        # For the Unlabeled Samples.
        if not swap_inp_tar_unsup: #print("C")
            # Input is the normalized 2D keypoints obtained by projecting the triangulated 3D pose.
            # Target is the normalized 2D detected keypoints.
            inputs.append(pose_2d_proj_norm)
            targets.append(keypoints_det_norm)
        else: #print("D")
            # Input is the 2D detected keypoints.
            # Target is the 2D keypoints obtained by projecting the triangulated 3D pose.
            inputs.append(keypoints_det_norm)
            targets.append(pose_2d_proj_norm)

    else: # print("Using Poses in the Original Image Coordinates in The Loss.")
        targets = [target_2d]
        if not use_dets_for_labeled_in_loss:
            inputs = [pose_2d_proj]  # For the labeled Samples.
        else:
            inputs = [keypoints_det_dist]  # For the labeled Samples.

        # For the Unlabeled Samples.
        if not swap_inp_tar_unsup:
            # Input is the 2D keypoints obtained by projecting the triangulated 3D pose.
            # Target is the 2D detected keypoints.
            inputs.append(pose_2d_proj)
            targets.append(keypoints_det_dist)
        else:
            # Input is the 2D detected keypoints.
            # Target is the 2D keypoints obtained by projecting the triangulated 3D pose.
            inputs.append(keypoints_det_dist)
            targets.append(pose_2d_proj)

    inp_2d_loss    = (labels * inputs[0]) if inputs[0]   is not None else 0.0 + ((1 - labels) * inputs[1])  if inputs[1]  is not None else 0.0
    target_2d_loss = (labels * targets[0]) if targets[0] is not None else 0.0 + ((1 - labels) * targets[1]) if targets[1] is not None else 0.0
    return inp_2d_loss, target_2d_loss



def get_windows(N, half_window_size, delta_t_0, extend_last, sampling_rate):
    """
    :param N: The total Number of Samples.
    :param half_window_size: The number of samples in each half of a window. The total number of samples in a window is 2*half_window_size + 1.
    :param delta_t_0: The sampling rate of center of every window.
    :param extend_last: If true, we will append the missing samples within a window.
    :param sampling_rate: The sampling rate withing each window.
    :return:
    """
    win     = half_window_size * sampling_rate * 2 + 1
    dl      = delta_t_0
    sampler = lambda ls: [  # I select images from beginning of window to it's end with step sampling_rate
        ls[i:i + win:sampling_rate]  # Every iteration I make delta_t_0 step until end
        for i in range(0, N, dl)
    ]
    images         = np.arange(N) # get this sampling for images and gts
    images_windows = sampler(images)
    if not images_windows:
        return []

    num_samples_in_each_window = 2 * half_window_size + 1
    final_windows              = []
    for window in images_windows:
        k_window = num_samples_in_each_window - len(window)
        if 0 < k_window < num_samples_in_each_window//2:
            if extend_last:
                window_ = np.append(window, [images[-1]] * k_window)
            else:
                window_ = []
        else:
            window_ = window
        if len(window_) == num_samples_in_each_window:
            final_windows.append(window_)
    return final_windows


def function_to_obtain_root_relative_depth(pose_3d, pelvis_cam_z):
    pose_3d_z = pose_3d[:, :, -1] # z coordinates of the poses.
    depth     = pose_3d_z - pelvis_cam_z
    return depth


def MPJPE_(preds_, targets_):
    assert preds_.shape == targets_.shape
    mpjpe = torch.mean(torch.norm(preds_ - targets_, dim=len(targets_.shape) - 1))
    return mpjpe


def NMPJPE_(preds_, targets_):
    assert preds_.shape == targets_.shape
    norm_predicted = torch.mean(torch.sum(preds_ ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target    = torch.mean(torch.sum(targets_ * preds_, dim=3, keepdim=True), dim=2, keepdim=True)
    scale          = norm_target / norm_predicted
    return MPJPE_(scale * preds_, targets_)


def PMPJPE_(preds_, targets_):
    predicted = preds_
    target    = targets_
    assert predicted.shape == target.shape
    dim_2, dim_1 = predicted.shape[-2], predicted.shape[-1]
    predicted    = predicted.cpu().numpy().reshape(-1, dim_2, dim_1)
    target       = target.cpu().numpy().reshape(-1, dim_2, dim_1)

    muX = np.mean(target,    axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H        = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V        = Vt.transpose(0, 2, 1)
    R        = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR    = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1]    *= sign_detR.flatten()
    R            = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a  = tr * normX / normY  # Scale
    t  = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    pmpjpe = np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)
    return pmpjpe


def calculate_mpjpe_(predictions, targets, n_mpjpe, p_mpjpe, print_string):
    mpjpe       = MPJPE_(preds_=predictions, targets_=targets)
    print("The MPJPE {} is {:.2f} cms".format(print_string, mpjpe * 100.0))
    if n_mpjpe:
        nmpjpe = NMPJPE_(preds_=predictions, targets_=targets)
        print("The NMPJPE {} is {:.2f} cms".format(print_string, nmpjpe * 100.0))
    if p_mpjpe:
        pmpjpe = PMPJPE_(preds_=predictions, targets_=targets)
        pmpjpe = np.mean(pmpjpe)
        print("The PMPJPE {} is {:.2f} cms".format(print_string, pmpjpe * 100.0))
    return mpjpe

#### ---------- PCK ---------- ###
pck_thresholds = [0.1, 0.15, 0.2, 0.3]

def PCK(pred, target, bbox=None, scale=None):
    """
    Percentage of Correct Keypoints (PCK) metric
    A joint is considered correct if it is within a certain
    threshold from the corresponding ground-truth one.
    The threshold can vary.

    If bbox is provided, we normalize the poses such that
    the distance between the bottom and the top of the box is 1.

    Another normalization option if the camera parameters are known,
    would be to scale the poses so that the threshold is in mm for example.
    """
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        h                      = ymax - ymin
        pred                   = pred / h
        target                 = target / h

    if scale is not None:
        pred   = pred * scale
        target = target * scale

    dists = np.linalg.norm(pred - target, axis=1) # pcks  = [np.mean(dists < th) for th in ths]
    return dists.tolist()


def calculate_pck(preds, targets):
    raise NotImplementedError("This has not been implemented yet.")
    """
    print("Calculating the PCK for MPI 3DHP dataset for SINGLE VIEW SETUP.")
    dist_singleview_3d = []
    for pred_, target_ in zip(preds, targets):  # , bboxes_enc_trained, indexes_enc_trained, views_enc_trained):
        pred_   = np.array(pred_)
        target_ = np.array(target_)
        dist_3d = PCK(pred=pred_, target=target_)
        dist_singleview_3d.extend(dist_3d)
    dist_singleview_3d = np.array(dist_singleview_3d)
    singleview_pck     = [np.mean(dist_singleview_3d < th) * 100.0 for th in pck_thresholds]
    print("The value of PCK for different thresholds are as follows-")
    for pck_single_view, threshold in zip(singleview_pck, pck_thresholds):
        print("Threshold = {:.2f} mms  (SINGLE VIEW) ---> 3DPCK = {:.2f}".format(threshold * 1000.0, pck_single_view))
    """

def MPJPE(preds_, targets_):
    """
    Function to calculate the MPJPE.
    :param preds_: A list of numpy arrays, each element contains the prediction of the network.
    :param targets_: A list of numpy arrays, each element contains the corresponding target.
    :return: A list of mpjpe scores for every pair of prediction and its corresponding target.
    """
    scores = []
    for p, t in zip(preds_, targets_):
        mpjpe = np.linalg.norm(p-t, axis=1).mean()
        scores.append(mpjpe)
    return scores


def NMPJPE(preds_, targets_):
    """
    Function to calculate the NMPJPE.
    :param preds_: A list of numpy arrays, each element contains the prediction of the network.
    :param targets_: A list of numpy arrays, each element contains the corresponding target.
    :return: A list of nmpjpe scores for every pair of prediction and its corresponding target.
    """

    scores = []
    for p, t in zip(preds_, targets_):
        p_norm       = np.linalg.norm(p, ord='fro')
        t_norm       = np.linalg.norm(t, ord='fro')
        p_normalized = p * t_norm / p_norm
        nmpjpe       = np.linalg.norm(p_normalized-t, axis=1).mean()
        scores.append(nmpjpe)
    return scores


def PMPJPE(preds_, targets_):
    """
    Function to calculate the PMPJPE.
    :param preds_: A list of numpy arrays, each element contains the prediction of the network.
    :param targets_: A list of numpy arrays, each element contains the corresponding target.
    :return: A list of pmpjpe scores for every pair of prediction and its corresponding target.
    # _, pred_alligned, _= procrustes(target, pred)
    # compute_mpjpe(pred_alligned, target)
    """
    scores = []
    for p, t in zip(preds_, targets_):
        _, p_aligned, _ = procrustes(t, p)
        pmpjpe          = np.linalg.norm(p_aligned-t, axis=1).mean()
        scores.append(pmpjpe)

    return scores


def calculate_mpjpe(predictions, targets, n_mpjpe, p_mpjpe):
    predictions_wo_nans = []
    targets_wo_nans = []
    for x, y in zip(predictions, targets):
        x    = np.array(x)
        y    = np.array(y)
        idx1 = np.any(np.isnan(x), axis=1)
        idx2 = np.any(np.isnan(y), axis=1)
        idx3 = np.any(np.isinf(x), axis=1)
        idx4 = np.any(np.isinf(y), axis=1)

        idx  = np.logical_or(idx1, idx2)
        idx  = np.logical_or(idx,  idx3)
        idx  = np.logical_or(idx,  idx4)
        x    = x[idx != True]
        y    = y[idx != True]
        predictions_wo_nans.append(x)
        targets_wo_nans.append(y)
        
    nmpjpe = None
    pmpjpe = None
    mpjpe  = MPJPE(preds_=predictions_wo_nans, targets_=targets_wo_nans)
    mpjpe  = np.mean(mpjpe)
    if n_mpjpe:
        nmpjpe = NMPJPE(preds_=predictions_wo_nans, targets_=targets_wo_nans)
        nmpjpe = np.mean(nmpjpe)
    if p_mpjpe:
        pmpjpe = PMPJPE(preds_=predictions_wo_nans, targets_=targets_wo_nans)
        pmpjpe = np.mean(pmpjpe)
    return mpjpe, nmpjpe, pmpjpe

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Procrustes' analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """

    n, m   = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A        = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V        = Vt.T
    T        = np.dot(V, U.T)

    if reflection != 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1]    *= -1
            T         = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standardized distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c     = muX - b*np.dot(muY, T)
    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}
    return d, Z, tform


def function_to_project_3D_world_to_camera(pose_3d_world, R, t): ## This is working fine.
    """
    Function to project a 3D pose in the world to the Camera Coordinates.
    :param pose_3d_world: The 3D pose in the world coordinates is of size num_samples x num_joints x 3.
    :param R: The Rotation Matrix is of shape num_samples x 3 x 3.
    :param t: The translation Matrix is of shape num_samples x 3.
    :return: The 3D pose in the Camera Coordinates.
    """
    num_samples, num_joints, \
        joints_dim = pose_3d_world.size(0), pose_3d_world.size(1), pose_3d_world.size(2)
    pose_3d_world  = pose_3d_world.reshape(-1, num_joints, joints_dim)

    R, t           = R.reshape(-1, 3, 3), t.reshape(-1, 1, 3)
    pose_3d_cam    = torch.bmm(R, pose_3d_world.transpose(1, 2)).transpose(1, 2) + t
    return pose_3d_cam


def project_point_radial_batch(point_3d, R, T, f, c, k, p):
    """
    Function to project the 3D points in x to the Distorted Image Coordinates.
    :param point_3d: The 3D points in the World Coordinates of the size N x number_of_joints x 3.
    :param R: The Rotation Matrices of the size N x 3 x 3.
    :param T: The Translation Matrices of the size N x 3 x 1.
    :param f: The Focal length Matrices of the size N x 2 x 1.
    :param c: The Camera Center Matrices of the size N x 2 x 1.
    :param k: The Camera Radial Distortion Matrices of the size N x 3 x 1.
    :param p: The Camera Tangential Distortion Matrices of the size N x 3 x 1.
    :return:  The 2D Image Coordinates of the Batch.
    """
    N, J   = point_3d.size(0), point_3d.size(1)
    xcam   = torch.bmm(R, torch.transpose(point_3d, 1, 2) - T)
    y      = xcam[:, :2] / xcam[:, 2].unsqueeze(dim=1)
    kexp   = k.repeat(1, 1, J)
    r2     = torch.sum(y**2, 1, keepdim=True)
    r2exp  = torch.cat([r2, r2**2, r2**3], 1)
    radial = 1 + torch.einsum('bij,bij->bj', kexp, r2exp)
    tan    = 2 * p[:, 0] * y[:, 1] + 2 * p[:, 1] * y[:, 0]
    corr   = (radial + tan).unsqueeze(dim=1).repeat((1, 2, 1))

    p1     = torch.cat([p[:, 1], p[:, 0]], dim=1).view(N, -1)
    r2_    = r2.view(N, -1)
    a      = torch.bmm(p1.unsqueeze(dim=2), r2_.unsqueeze(1))
    y      = y * corr + a
    ypixel = (f * y) + c
    ypixel = ypixel.transpose(1, 2)
    return ypixel


def invert_Rt_batch(R, t):
    """
    Function to Invert the Translation and the Rotation Vectors.
    :param R: The Rotation Vector of size batch_size x 3 x 3
    :param t: The Translation Vector of size batch_size x 3
    :return: The Inverse of the Rotation and the Translation Vectors.
    """
    Ri = R.transpose(1, 2)
    ti = torch.matmul(-Ri, t)
    return Ri, ti


def perform_projection_from_3D_to_2D(points_3d: torch.Tensor, R: torch.Tensor, t: torch.Tensor, K: torch.Tensor, dist: torch.Tensor):
    """
    Function to Project the 3D keypoints in the World Coordinates to the Image Coordinates in 2D.
    :param points_3d: The 3D keypoints in the World Coordinates of size batch_size x number_of_joints x 3
    :param R: The Rotation Tensor of size (batch_size * number_of_cameras) x 3 x 3
    :param t: The Translation Tensor of size (batch_size * number_of_cameras) x 3.
    :param K: The Camera Intrinsics Tensor of size (batch_size * number_of_cameras) x 3 x 3.
    :param dist: The Distortion Tensor of size (batch_size * number_of_cameras) x 5.
    :return: The 2D keypoints in the Image Coordinates of batch_size x number_of_cameras x number_of_joints x 2.
    """
    batch_size  = points_3d.size(0)
    num_cameras = R.size(0) // batch_size
    num_joints  = points_3d.size(1)
    N           = batch_size*num_cameras
    _, t_inv    = invert_Rt_batch(R=R, t=t)
    points_3d   = points_3d.unsqueeze(dim=1).repeat(1, num_cameras, 1, 1).view(N, num_joints, 3)

    f = K[:, [0, 1], [0, 1]].reshape(N, 2, 1)
    c = K[:, 0:2, 2].reshape(N, 2, 1)
    k = dist[:, [0, 1, 4]].reshape(N, 3, 1)
    p = dist[:, [2, 3]].reshape(N, 2, 1)
    points_2d = project_point_radial_batch(point_3d=points_3d, R=R, T=t_inv, f=f, c=c, k=k, p=p)
    points_2d = points_2d.view(batch_size, num_cameras, num_joints, 2)
    return points_2d




def check_list_of_array_format(points):
    check_shapes_compatibility(points, -1)


def check_list_of_list_of_array_format(points):
    # each element of `points` is a list of arrays of compatible shapes
    components = zip_longest(*points, fillvalue=torch.Tensor())
    for i, component in enumerate(components):
        check_shapes_compatibility(component, i)


def check_shapes_compatibility(list_of_arrays, i):
    arr0 = list_of_arrays[0]
    if not isinstance(arr0, torch.Tensor):
        raise ValueError("Expected points of format list of `torch.Tensor`s.", f"Got {type(arr0)} for component {i} of point 0.")
    shape = arr0.shape
    for j, arr in enumerate(list_of_arrays[1:]):
        if not isinstance(arr, torch.Tensor):
            raise ValueError(f"Expected points of format list of `torch.Tensor`s. Got {type(arr)}", f"for component {i} of point {j + 1}.")
        if arr.shape != shape:
            raise ValueError(f"Expected shape {shape} for component {i} of point {j + 1}.", f"Got shape {arr.shape} instead.")


def function_to_extend_dim(data, which_dim, which_keys):
    if data is not None:
        data_keys  = list(data.keys())
        which_keys = data_keys if (which_keys == [] or which_keys is None) else which_keys
        for key in which_keys:
            if key in data_keys:
                data_key  = data[key]
                data_key  = data_key.unsqueeze(dim=which_dim)
                data[key] = data_key
    return data
