import torch
import numpy as np

from dlt_functions.triangulation       import triangulate_from_multiple_views_svd as DLT_func
from itertools                         import combinations
from Geometric_median.geometric_median import compute_geometric_median_ours
from scipy.stats                       import multivariate_normal
from dlt_functions.eccv_sub            import get_weights_for_dlt_eccv


def calc_weights_by_geometric_median_pairwise(fixed_cov_matrix: bool, fix_cov_val: float, minimum_pairs_needed: int,
                                              points_2d_undist: torch.Tensor, consider : torch.Tensor,
                                              projection_matrices : torch.Tensor):
    """

    :param fixed_cov_matrix: A boolean variable. If True, we will use a fixed covariance  matrix; otherwise we will calculate the Covariance Matrix from the data.
    :param fix_cov_val: The value to be added multiplied with the identity covariance matrix.
    :param minimum_pairs_needed: The minimum number of pairs needed to calculate the weights for a given person.
    :param points_2d_undist: The Undistorted 2D Keypoints in the Image Coordinates of size batch_size x Num_Cameras x Number_of_joints x 2.
    :param consider: A tensor of 1's and 0's of size batch_size x Num_Cameras denoting which cameras are to be considered for 3D pose calculations.
    :param projection_matrices: The Projection Matrices of size batch_size x Number_of_joints x Num_Cameras x 3 x 4.
    :return:
    """
    batch_size, num_cameras, num_joints = points_2d_undist.size(0), points_2d_undist.size(1), points_2d_undist.size(2)
    assert projection_matrices.size(0) == batch_size
    assert projection_matrices.size(1) == num_joints
    assert projection_matrices.size(2) == num_cameras


    cameras       = ['ace_{}'.format(x) for x in range(num_cameras)]
    final_weights = []
    dim           = 3
    cov_mat       = np.eye(dim) * fix_cov_val
    for n_i in range(batch_size):
        possible_pairs_i     = list(filter(lambda e: consider[n_i][cameras.index(e[0])] * consider[n_i][cameras.index(e[1])] == 1, combinations(cameras, 2)))
        camera_pairs_indices = list(map(lambda e: (cameras.index(e[0]), cameras.index(e[1])), possible_pairs_i))
        if len(possible_pairs_i) < minimum_pairs_needed:
            weights = np.zeros(shape=(num_joints, num_cameras))
        else:
            indexes_i = [[index1 for index1, value1 in enumerate(possible_pairs_i) for index2, value2 in enumerate(value1)
                          if value2 == check] for check in cameras]
            assert len(indexes_i) == num_cameras
            pairwise_3d_th = function_to_calculate_pairwise_3D_pose(points_2d_undist_i=points_2d_undist[n_i],
                                                                    projection_matrices_i=projection_matrices[n_i],
                                                                    camera_pairs_indices=camera_pairs_indices)
            geo_med        = compute_geometric_median_ours(points=pairwise_3d_th)
            confidences    = []
            assert len(geo_med) == num_joints
            for n_j in range(num_joints):
                joint_th = pairwise_3d_th[n_j]
                if not fixed_cov_matrix:
                    cov_mat   = np.cov(joint_th)
                mean_val_th   = geo_med[n_j]
                mean_val_th   = mean_val_th.view(-1, 3)
                inp_gaussian  = torch.cat((joint_th, mean_val_th), dim=0)
                weights_joint = multivariate_normal.pdf(x=inp_gaussian.cpu().numpy(), mean=mean_val_th.view(-1).cpu().numpy(), cov=cov_mat)
                weights_joint = weights_joint / weights_joint[-1]
                weights_joint = weights_joint[0:-1]
                assert weights_joint.size == len(possible_pairs_i)
                confidences.append(weights_joint)

            confidences = np.stack(confidences)
            weights     = [func_to_generate_weights(confidences, idx) for idx in indexes_i]
            weights     = np.stack(weights, axis=-1)
        final_weights.append(weights)
    final_weights = np.stack(final_weights, axis=0)
    final_weights = torch.from_numpy(final_weights)
    final_weights = final_weights.view(batch_size, num_joints, num_cameras, 1)
    return final_weights


def function_to_obtain_weights(calc_weights_need : dict, P_th_b, points_2d_undist, consider, eccv_calculations: dict):
    """
    Function to obtain the weights for Triangulation.
    :param calc_weights_need: A dictionary containing the various parameters needed for calculating the weights.
    :param P_th_b: The Projection Matrices of size batch_size x Number_of_joints x Num_Cameras x 3 x 4.
    :param points_2d_undist: The Undistorted 2D Keypoints in the Image Coordinates of size batch_size x Num_Cameras x Number_of_joints x 2.
    :param consider: A tensor of 1's and 0's of size batch_size x Num_Cameras denoting which cameras are to be considered for 3D pose calculations.
    :param eccv_calculations: TODO
    :return: A Matrix of size Num_Keypoints x Num_Cameras representing the weights for each per keypoint.
    """
    if calc_weights_need['weighted_dlt'] == 'geo-med':
        if calc_weights_need['method_of_geometric_median'] == 'now':
            confidences = calc_weights_by_geometric_median_pairwise(fixed_cov_matrix=calc_weights_need['fixed_cov_matrix'],
                                                                    fix_cov_val=calc_weights_need['fix_cov_val'],
                                                                    minimum_pairs_needed=calc_weights_need['minimum_pairs_needed'],
                                                                    projection_matrices=P_th_b, points_2d_undist=points_2d_undist,
                                                                    consider=consider)
        elif calc_weights_need['method_of_geometric_median'] == 'prev':
            cov_matrix_   = np.eye(3, 3) * calc_weights_need['fix_cov_val']
            rotation      = eccv_calculations['rotation']
            translation   = eccv_calculations['translation']
            intrinsics    = eccv_calculations['intrinsics']
            distortions   = eccv_calculations['distortions']
            pose_2d_dists = eccv_calculations['pose_2d_dists']
            confidences   = get_weights_for_dlt_eccv(cov_matrix_=cov_matrix_, consider=consider, rotation=rotation,
                                                     translation=translation, intrinsics=intrinsics,
                                                     distortions=distortions, pose_2d_dists=pose_2d_dists)

        else:
            # confidences = None
            raise NotImplementedError

        # confidences are of shape batch_size x number_of_joints x number of cameras x 1
        if calc_weights_need['add_epsilon_to_weights']:
            confidences += calc_weights_need['epsilon_value']

        if calc_weights_need['normalize_confidences_by_sum']:
            # Will be normalizing the summing the weights of every camera per joint in every sample.
            confidences  = confidences / confidences.sum(dim=2, keepdim=True)
            confidences += calc_weights_need['epsilon_value']
    else:
        # confidences = None
        raise NotImplementedError("Not Implemented Yet.")

    device = points_2d_undist.device
    if points_2d_undist.is_cuda:
        use_cuda = True
    else:
        use_cuda = False

    if use_cuda:
        confidences = confidences.cuda(device=device)

    return confidences



func_to_generate_weights = lambda x, y: np.quantile(x[:, y], axis=-1, q=0.5) if len(y) > 0 else np.zeros(x.shape[0])

"""
def undistort(ypixel, f, c, k, p, N=5):
    # Args
    #     ypixel: N_kpts * 2 points distorted image in pixels
    #     f: 2x1 Camera focal length
    #     c: 2x1 Camera center
    #     k: 3x1 Camera radial distortion coefficients
    #     p: 2x1 Camera tangential distortion coefficients
    # Returns
    #     xpixel: Nx2 points undistorted image in pixels
    y      = (ypixel.t() - c)/f
    n_kpts = ypixel.shape[0]
    kexp   = k.repeat((1, n_kpts))
    y0     = y.clone()
    tan    = 2 * p[0] * y[1] + 2 * p[1] * y[0]

    for _ in range(N):
        r2     = torch.sum(y**2, 0, keepdim=True)
        r2exp  = torch.cat([r2, r2**2, r2**3], 0)
        radial = 1 + torch.einsum('ij,ij->j', kexp, r2exp)
        corr   = (radial + tan).repeat((2, 1))
        y      = (y0 - torch.ger(torch.cat([p[1], p[0]]).view(-1), r2.view(-1)))/corr
    xpixel = (f * y) + c
    return xpixel.t()
"""


def undistort_batch(points_2d_dist, f, c, k, p, N_iters=5, dont_project=False):
    """
    Function
    :param points_2d_dist: The Distorted Keypoints in 2D Image Coordinates of size batch_size x number_of_joints x 2
    :param f: The Focal length Matrices of the size batch_size x 2 x 1.
    :param c: The Camera Center Matrices of the size batch_size x 2 x 1.
    :param k: The Camera Radial Distortion Matrices of the size batch_size x 3 x 1.
    :param p: The Camera Tangential Distortion Matrices of the size N x 3 x 1.
    :param N_iters: The Number of Iterations for convergence.
    :param dont_project : TODO
    :return: The 2D points in the Undistorted Image Coordinates.
    """
    N, J = points_2d_dist.size(0), points_2d_dist.size(1)
    y    = (points_2d_dist.transpose(1, 2) - c) / f
    kexp = k.repeat(1, 1, J)
    y0   = y.clone()
    tan  = 2 * p[:, 0] * y[:, 1] + 2 * p[:, 1] * y[:, 0]

    for i in range(N_iters):
        r2     = torch.sum(y ** 2, 1, keepdim=True)
        r2exp  = torch.cat([r2, r2 ** 2, r2 ** 3], 1)
        radial = 1 + torch.einsum('bij,bij->bj', kexp, r2exp)
        corr   = (radial + tan).unsqueeze(dim=1).repeat((1, 2, 1))
        p1     = torch.cat([p[:, 1], p[:, 0]], dim=1).view(N, -1)
        r2_    = r2.view(N, -1)
        a      = torch.bmm(p1.unsqueeze(dim=2), r2_.unsqueeze(1))
        y      = (y0 - a) / corr
    points_2d_undist = y if dont_project else (f * y) + c
    points_2d_undist = points_2d_undist.transpose(1, 2)
    return points_2d_undist


def perform_dlt(R: torch.Tensor, t: torch.Tensor, K: torch.Tensor, dist: torch.Tensor, consider: torch.Tensor,
                points_2d_dist: torch.Tensor, calc_weights: bool, calc_weights_need: dict,
                pre_computed_conf: torch.Tensor = None):
    """
    Function to perform DLT (with or without weights) and obtain the 3D pose in the world coordinates.
    :param R                 : The Rotation Tensor of size batch_size x number_of_cameras x 3 x 3
    :param t                 : The Translation Tensor of size batch_size x number_of_cameras x 3.
    :param K                 : The Camera Intrinsics Tensor of size batch_size x number_of_cameras x 3 x 3
    :param dist              : The Distortion Tensor of size batch_size x number_of_cameras x 5.
    :param points_2d_dist    : The 2D keypoints in the Distorted Image Coordinates of size batch_size x number_of_cameras x number_of_joints x 2.
    :param calc_weights      : If True, we will calculate the weights.
    :param calc_weights_need : A dictionary containing the important parameters to calculate the weights of the keypoints per camera.
    :param consider          : A Tensor of size batch_size x number_of_cameras; with its elements 0
                                (don't consider that camera for triangulation) and 1 (consider that camera for triangulation).
    :param pre_computed_conf : if not None, it represents the pre-computed weights by the mode
                                of shape batch_size x number of joints x number_of_cameras.
    :return: The 3D pose of size  batch_size x number_of_joints x 2 in the world coordinate system.

    """
    #print("Inside the Perform DLT Function.")
    batch_size, num_cameras, num_joints = points_2d_dist.size(0), points_2d_dist.size(1), points_2d_dist.size(2)
    N                 = batch_size * num_cameras
    R                 = R.view(N, 3, 3) #R.size(-2), R.size(-1))
    t                 = t.view(N, 3)  #t.size(-1))
    K                 = K.view(N, 3, 3)  #K.size(-2), K.size(-1))
    dist              = dist.view(N, 5)  #dist.size(-1))
    eccv_calculations = {}
    if calc_weights:
        if calc_weights_need['method_of_geometric_median'] == 'prev':
            points_2d_dist_copy = points_2d_dist.cpu()
            R_copy              = R.view(batch_size, num_cameras, 3, 3).cpu()
            t_copy              = t.view(batch_size, num_cameras, 3).cpu()
            K_copy              = K.view(batch_size, num_cameras, 3, 3).cpu()
            dist_copy           = dist.view(batch_size, num_cameras, 5).cpu()
            eccv_calculations   = {'rotation': R_copy, 'translation' : t_copy, 'intrinsics' : K_copy,
                                   'distortions' : dist_copy, 'pose_2d_dists' : points_2d_dist_copy}

    f = K[:, [0, 1], [0, 1]].reshape(N, 2, 1)
    c = K[:, 0:2, 2].reshape(N, 2, 1)
    k = dist[:, [0, 1, 4]].reshape(N, 3, 1)
    p = dist[:, [2, 3]].reshape(N, 2, 1)

    points_2d_dist        = points_2d_dist.view(N, num_joints, 2)
    points_2d_undist      = undistort_batch(points_2d_dist=points_2d_dist, f=f, c=c, k=k, p=p)
    # points_2d_undist will be of the size N x num_joints x 2
    points_2d_undist      = points_2d_undist.view(batch_size, num_cameras, num_joints, 2)
    points_2d_undist_copy = points_2d_undist.clone() # size is batch_size x number of cameras x number of joints x 2.
    points_2d_undist      = points_2d_undist.transpose(1, 2).reshape(-1, num_cameras, 2)
    # points_2d_undist will be of the size (batch_size*num_joints) x num_cameras x 2.
    a_th_b      = torch.cat([R, t.reshape(N, 3, 1)], dim=-1)
    P_th_b      = torch.matmul(K, a_th_b)
    P_th_b      = P_th_b.view(batch_size, num_cameras, 3, 4)
    P_th_b      = P_th_b.unsqueeze(dim=1).repeat(1, num_joints, 1, 1, 1)
    P_th_b_copy = P_th_b.clone() # size is batch_size x num_joints x num_cameras x 3 x 4.
    P_th_b      = P_th_b.view(batch_size*num_joints, num_cameras, 3, 4)
    # P_th_b will be of size (batch_size*num_joints) x number_of_cameras x 3 x 4.

    if pre_computed_conf is None:
        if calc_weights is False:
            weights_dlt = None
            weights_2d  = torch.ones(batch_size, num_joints, num_cameras, 1).type_as(consider)
            """
            weights_dlt  = consider.unsqueeze(dim=1)
            weights_dlt  = weights_dlt.repeat(1, num_joints, 1)
            # weights_dlt is of shape batch_size x num_joints x num_cameras
            weights_dlt  = weights_dlt.view(batch_size, num_joints, num_cameras, 1)
            # weights_dlt is of shape batch_size x num_joints x num_cameras x 1.
            if not use_weights_in_2d_loss:
                weights_2d  = weights_dlt
                weights_dlt = None
            else:
                weights_2d = torch.ones_like(weights_dlt)
                # weights_2d is of shape batch_size x number_of_joints x number of cameras x 1.
            """
        else:
            weights_dlt = function_to_obtain_weights(P_th_b=P_th_b_copy, points_2d_undist=points_2d_undist_copy,
                                                     calc_weights_need=calc_weights_need, consider=consider,
                                                     eccv_calculations=eccv_calculations)
            # weights is of shape batch_size x number_of_joints x number of cameras x 1
            weights_2d = weights_dlt.clone()
            # weights_2d is of shape batch_size x number_of_joints x number of cameras x 1.

        if weights_dlt is not None:
            weights_dlt = weights_dlt.view(-1, num_cameras, 1, 1) # This goes to the DLT function.
            # weights_dlt is of shape  (N * number_of_joints) x number of cameras x 1 x 1.
    else:
        # pre_computed_conf is of shape batch_size x number of joints x number_of_cameras
        pre_computed_conf = pre_computed_conf.unsqueeze(dim=-1)
        # pre_computed_conf is of shape batch_size x number of joints x number_of_cameras x 1
        weights_2d        = pre_computed_conf
        # weights_2d is of shape batch_size x number_of_joints x number of cameras x 1.
        weights_dlt       = pre_computed_conf.reshape(batch_size*num_joints, num_cameras, 1, 1)
        # weights_dlt is of shape (N * number_of_joints) x number of cameras x 1 x 1.

    # weights_2d is of shape batch_size x number_of_joints x number of cameras x 1.
    # which is what we need for the loss calculation.

    points_3d = DLT_func(proj_matrices=P_th_b, points=points_2d_undist, weights=weights_dlt)
    points_3d = points_3d.view(batch_size, num_joints, 3)
    ret_val   = {'points_3d' : points_3d, 'weights_2d': weights_2d.squeeze(dim=-1)}
    # 'points_3d' should be of shape N x number_of_joints x 3.
    # 'weights_2d' should be of shape N x number of cameras x number_of_joints.
    return ret_val


def function_to_calculate_pairwise_3D_pose(points_2d_undist_i: torch.Tensor, projection_matrices_i: torch.Tensor, camera_pairs_indices: list):
    pairwise_3d = []
    for camera_pair in camera_pairs_indices:
        points_2d_undist_i_pair    = points_2d_undist_i[camera_pair, :, :] # Its size is 2(number of cameras) x number of joint x 2
        projection_matrices_i_pair = projection_matrices_i[:, camera_pair, :, :] # Its size is number of joints x 2(number of cameras) x 3 x 4.
        points_2d_undist_i_pair    = points_2d_undist_i_pair.transpose(0, 1)
        points_3d_pair             = DLT_func(proj_matrices=projection_matrices_i_pair, points=points_2d_undist_i_pair, weights=None)
        pairwise_3d.append(points_3d_pair)
    pairwise_3d  = torch.stack(pairwise_3d, dim=1)
    return pairwise_3d