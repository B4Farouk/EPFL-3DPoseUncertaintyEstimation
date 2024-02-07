import torch
import numpy as np
import hashlib
from torch.autograd import Variable
import dgl
    

def deterministic_random(min_value, max_value, data):
    digest    = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def test_calculation(predicted, target, action, error_sum, data_type, subject, MAE=False):
    error_sum     = mpjpe_by_action_p1(predicted, target, action, error_sum)
    if not MAE:
        error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    return error_sum


def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    batch_num = predicted.size(0)
    frame_num = predicted.size(1)
    dist      = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item()*batch_num*frame_num, batch_num*frame_num)
    else:
        for i in range(batch_num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(torch.mean(dist[i]).item()*frame_num, frame_num)
            
    return action_error_sum


def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num     = predicted.size(0)
    pred    = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt      = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist    = p_mpjpe(pred, gt)
    
    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)
            
    return action_error_sum


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX   = np.mean(target, axis=1, keepdims=True)
    muY   = np.mean(predicted, axis=1, keepdims=True)

    X0    = target - muX
    Y0    = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0   /= normX
    Y0   /= normY

    H         = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt  = np.linalg.svd(H)
    V         = Vt.transpose(0, 2, 1)
    R         = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR    = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1]    *= sign_detR.flatten()
    
    R  = np.matmul(V, U.transpose(0, 2, 1))
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a  = tr * normX / normY
    t  = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)


def define_actions( action ):

  actions = ["Directions","Discussion","Eating","Greeting", "Phoning","Photo","Posing","Purchases", "Sitting","SittingDown","Smoking","Waiting",
             "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all" or action == '*':
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]


def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: {'p1':AccumLoss(), 'p2':AccumLoss()} for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val
        self.count += n
        self.avg    = self.sum / self.count
        

def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = print_error_action(action_error_sum, is_train)
    return mean_error_p1, mean_error_p2


def print_error_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all  = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")
            
        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
            mean_error_all['p2'].avg))
    
    return mean_error_all['p1'].avg, mean_error_all['p2'].avg


def save_model(previous_name, save_dir,epoch, data_threshold, model, model_name):
    # if os.path.exists(previous_name):
    #     os.remove(previous_name)

    torch.save(model.state_dict(), '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))
    previous_name = '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100)
    return previous_name
    
def save_model_new(save_dir,epoch, data_threshold, lr, optimizer, model, model_name):
    # if os.path.exists(previous_name):
    #     os.remove(previous_name)
    # torch.save(model.state_dict(),
    #            '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))
    torch.save({'epoch': epoch, 'lr': lr, 'optimizer': optimizer.state_dict(), 'model_pos': model.state_dict(),}, 
                '%s/%s_%d_%d.pth' % (save_dir, model_name, epoch, data_threshold * 100))


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]


def image_coordinates(X, w, h):
	assert X.shape[-1] == 2

	# Reverse camera frame normalization
	return (X + [1, h / w]) * w / 2

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) 
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) 



def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

def wrap(func, *args, unsqueeze=False):
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
    
    result = func(*args)
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
	    
        
        return tuple(result)
	
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
	
    else:
	    return result

def qrot(q, v):
	assert q.shape[-1]  == 4
	assert v.shape[-1]  == 3
	assert q.shape[:-1] == v.shape[:-1]

	qvec = q[..., 1:]
	uv   = torch.cross(qvec, v, dim=len(q.shape) - 1)
	uuv  = torch.cross(qvec, uv, dim=len(q.shape) - 1)
	return (v + 2 * (q[..., :1] * uv + uuv))


def qinverse(q, inplace=False):
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w   = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape) - 1)


def get_uvd2xyz(uvd, gt_3D, cam):
    N, T, V,_   = uvd.size()

    dec_out_all = uvd.view(-1, T, V, 3).clone()  
    root        = gt_3D[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1).clone()
    enc_in_all  = uvd[:, :, :, :2].view(-1, T, V, 2).clone() 

    cam_f_all   = cam[..., :2].view(-1,1,1,2).repeat(1,T,V,1)
    cam_c_all   = cam[..., 2:4].view(-1,1,1,2).repeat(1,T,V,1)

    z_global           = dec_out_all[:, :, :, 2]
    z_global[:, :, 0]  = root[:, :, 0, 2]
    z_global[:, :, 1:] = dec_out_all[:, :, 1:, 2] + root[:, :, 1:, 2]
    z_global           = z_global.unsqueeze(-1) 
    
    uv         = enc_in_all - cam_c_all 
    xy         = uv * z_global.repeat(1, 1, 1, 2) / cam_f_all  
    xyz_global = torch.cat((xy, z_global), -1)
    xyz_offset = (xyz_global - xyz_global[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1))

    return xyz_offset


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


def create_graph(num_samples_inside_one_window: int, strides: list, number_of_joints: int, edges_idx: list,
                 number_of_bones: int, nodes_name : str):
    verts          = np.tensordot(torch.Tensor(edges_idx), np.ones(num_samples_inside_one_window), axes=0) \
                     + np.arange(num_samples_inside_one_window)*number_of_joints
    edges          = verts.transpose([1, 0, 2])
    edges_reshaped = edges.reshape(2, number_of_bones*num_samples_inside_one_window).astype(np.int32)
    edges_reshaped = list(edges_reshaped)
    joints         = {(nodes_name, 'spatial', nodes_name) : (edges_reshaped[0], edges_reshaped[1])}
    for stride in strides:
        temp_joints = ([], [])
        for i in range(num_samples_inside_one_window-stride):
            u = list(np.arange(number_of_joints) + number_of_joints*i)
            v = list(np.arange(number_of_joints) + number_of_joints*(i+stride))
            temp_joints[0].extend(u)
            temp_joints[1].extend(v)
        joints[(nodes_name, 'temporal-{}'.format(stride), nodes_name)] = (temp_joints[0], temp_joints[1])
    graph = dgl.heterograph(joints)
    graph = dgl.to_bidirected(graph)
    return graph