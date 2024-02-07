import torch
import numpy as np
import cv2

from scipy.optimize         import minimize
from scipy.spatial.distance import cdist
from scipy.stats            import multivariate_normal



def get_pairs_of_views(consider_n_i): # Function to get the pairs of views or cameras to be used for weighting mechanism
    pairs_views = []; views_considered = []
    num_cameras = len(consider_n_i)
    for i in range(num_cameras):
        for j in range(i + 1,  num_cameras):
            vid_1 = 'ace_{}'.format(i) ## First  view.
            vid_2 = 'ace_{}'.format(j) ## Second view.
            if consider_n_i[i] * consider_n_i[j] == 1:
                flag_pairs = True
            else:
                flag_pairs = False

            if flag_pairs:
                pairs_views.append((vid_1, vid_2))
                if vid_1 not in views_considered:
                    views_considered.append(vid_1)
                if vid_2 not in views_considered:
                    views_considered.append(vid_2)
    return pairs_views, views_considered


def minimize_method(points, options={}):
    """
    Geometric median as a convex optimization problem.
    """
    # objective function
    def aggregate_distance(x):
        return cdist([x], points).sum()
    # initial guess: centroid
    centroid        = points.mean(axis=0)
    optimize_result = minimize(aggregate_distance, centroid, method='COBYLA')
    return optimize_result.x


def get_weights_for_dlt_eccv(consider, pose_2d_dists, rotation, translation, intrinsics, distortions, cov_matrix_):
    weights_mine = []
    Ns           = consider.size(0)
    for n_i in range(Ns):
        consider_n_i           = consider[n_i]
        pairs_views_i, \
            views_considered_i = get_pairs_of_views(consider_n_i)
        dist_points_i          = pose_2d_dists[n_i]
        num_views              = dist_points_i.size(0)
        R_i    = rotation[n_i]
        t_i    = translation[n_i]
        K_i    = intrinsics[n_i]
        dist_i = distortions[n_i]
        dist_points_dict, Rs_dict, ts_dict, Ks_dict, dists_dict = {}, {}, {}, {}, {}
        for x in range(num_views):
            key_val             = 'ace_{}'.format(x)
            Rs_dict[key_val]    = R_i[x]
            ts_dict[key_val]    = t_i[x]
            dists_dict[key_val] = dist_i[x]
            Ks_dict[key_val]    = K_i[x]
            dist_points_dict[key_val] = dist_points_i[x]
        weights_mine_calc = get_weights_for_dlt(dist_points=dist_points_dict, pairs_views=pairs_views_i,
                                                views_considered=views_considered_i, dists_dict=dists_dict,
                                                fixed_cov_matrix=True, weighted_dlt='geo-med',
                                                weights_for_weighted_dlt='med', Rs_dict=Rs_dict,
                                                ts_dict=ts_dict, Ks_dict=Ks_dict, cov_matrix_=cov_matrix_)
        weights_mine.append(weights_mine_calc)
    weights_mine = torch.stack(weights_mine)
    return weights_mine



def get_weights_for_dlt(dist_points : dict, pairs_views : list, views_considered : list,  dists_dict : dict,
                             cov_matrix_, fixed_cov_matrix, Rs_dict : dict, ts_dict : dict, Ks_dict : dict,
                             weighted_dlt='geo-med', weights_for_weighted_dlt='med'):
    key                         = list(dist_points.keys())[0]
    num_joints                  = dist_points[key].size(0) #self.num_labeled_joints
    num_views_considered        = len(views_considered)
    weights_                    = {}
    for cam in views_considered:
        weights_[cam] = [[] for _ in range(num_joints)]
    counts                   = torch.zeros(num_joints, num_views_considered)
    confidences              = torch.zeros(num_joints, num_views_considered)
    poses_3d_pairwise_joints = [torch.Tensor()]*num_joints

    for pair_view in pairs_views:
        view1 = pair_view[0]; view2 = pair_view[1]
        R1 = Rs_dict[view1]; t1 = ts_dict[view1]; K1 = Ks_dict[view1]; dist1 = dists_dict[view1]
        R2 = Rs_dict[view2]; t2 = ts_dict[view2]; K2 = Ks_dict[view2]; dist2 = dists_dict[view2]

        pts1   = dist_points[view1].numpy()
        pts2   = dist_points[view2].numpy()
        R1, R2 = R1.numpy(), R2.numpy()
        t1, t2 = t1.numpy(), t2.numpy()
        K1, K2 = K1.numpy(), K2.numpy()
        dist1  = dist1.numpy()
        dist2  = dist2.numpy()

        pts1_       = np.reshape(pts1, (-1, 2)).copy()
        pts2_       = np.reshape(pts2, (-1, 2)).copy()
        pts1_undist = cv2.undistortPoints(pts1_, K1, dist1, P=K1)[:, 0]
        pts2_undist = cv2.undistortPoints(pts2_, K2, dist2, P=K2)[:, 0]

        P1       = np.dot(K1, np.hstack([R1, t1.reshape(3, 1)]))
        P2       = np.dot(K2, np.hstack([R2, t2.reshape(3, 1)]))
        point_3d = cv2.triangulatePoints(P1, P2, pts1_undist.T, pts2_undist.T)
        point_3d = point_3d[:3] / point_3d[3]
        point_3d = point_3d.T
        point_3d = torch.from_numpy(point_3d)
        for j_idx in range(0, num_joints):
            joint                           = point_3d[j_idx, :].view(-1, 3)  ## Get the joint in 3D for the pairwise view.
            aa                              = poses_3d_pairwise_joints[j_idx] ## Get the tensor for storing the pairwise information.
            aa                              = torch.cat((aa, joint.type_as(aa)), dim=0) ## append the tensor
            poses_3d_pairwise_joints[j_idx] = aa ## Put the tensor back to the list.

    for joint_idx in range(0, num_joints):
        joint_pairwise_3d = poses_3d_pairwise_joints[joint_idx]
        assert joint_pairwise_3d.size(0) == len(pairs_views)
        if weighted_dlt.lower() == 'geo-med':
            med_joint = geometric_median_previous(joint_pairwise_3d.numpy())
            med_joint = torch.from_numpy(med_joint)

        elif weighted_dlt.lower() == 'med':
            p_dist = torch.cdist(joint_pairwise_3d, joint_pairwise_3d, p=2).sum(dim=1).view(-1)
            _, idx = torch.sort(p_dist, dim=-1, descending=False)
            med_joint = joint_pairwise_3d[idx[0], :]

        else:
            raise NotImplementedError("Weighted DLT with geo-med, dbscan and med is only implemented.")

        if med_joint is not None:
            med_joint = med_joint.view(-1, 3)
            if not fixed_cov_matrix:
                cov_joint = np.cov(joint_pairwise_3d.t().numpy())
            else:
                cov_joint = cov_matrix_

            gauss_dist_joint = multivariate_normal(mean=med_joint.view(-1).numpy(), cov=cov_joint)
            inp_gaussian     = torch.cat((joint_pairwise_3d, med_joint), dim=0)
            weights_joint    = gauss_dist_joint.pdf(x=inp_gaussian)
            weights_joint    = torch.from_numpy(weights_joint)
            weights_joint    = weights_joint / weights_joint[-1]
            weights_joint    = weights_joint[0:-1]
            assert weights_joint.numel() == len(pairs_views)
        else:
            weights_joint = torch.zeros(len(pairs_views))

        for i1, pair_view in enumerate(pairs_views):
            vid_1 = pair_view[0]
            vid_2 = pair_view[1]
            weights_[vid_1][joint_idx].append(weights_joint[i1].item())
            weights_[vid_2][joint_idx].append(weights_joint[i1].item())
            idx_1 = views_considered.index(vid_1)
            idx_2 = views_considered.index(vid_2)
            counts[joint_idx, idx_1] += 1
            counts[joint_idx, idx_2] += 1

    for view_conf in views_considered:
        view_idx = views_considered.index(view_conf)
        for joint_idx_conf in range(num_joints):
            weights_view_joints_ = weights_[view_conf][joint_idx_conf]
            weights_view_joints_ = torch.Tensor([weights_view_joints_]).view(-1)
            if weights_for_weighted_dlt == 'geo-mean':
                conf_view = torch.pow(weights_view_joints_.prod(), 1.0 / weights_view_joints_.numel())
            elif weights_for_weighted_dlt == 'med':
                conf_view = torch.median(weights_view_joints_)
                # get_median_val(inp=weights_view_joints_) ## To get the mean of the two medians for even number of pairs for a given joint.
            else:
                raise NotImplementedError
            confidences[joint_idx_conf, view_idx] = conf_view
            assert len(weights_[view_conf][joint_idx_conf]) == counts[joint_idx_conf, view_idx]
    return confidences


def weiszfeld_method(points, options={}):
    """
    Weiszfeld's algorithm as described on Wikipedia.
    """
    default_options = {'maxiter': 1000, 'tol': 1e-7}
    default_options.update(options)
    options         = default_options

    def distance_func(x):
        return cdist([x], points)

    # initial guess: centroid
    guess = points.mean(axis=0)
    iters = 0
    while iters < options['maxiter']:
        distances = distance_func(guess).T
        # catch divide by zero
        # TODO: Wikipedia cites how to deal with distance 0
        distances      = np.where(distances == 0, 1, distances)
        guess_next     = (points/distances).sum(axis=0) / (1./distances).sum(axis=0)
        guess_movement = np.sqrt(((guess - guess_next)**2).sum())
        guess          = guess_next
        if guess_movement <= options['tol']:
            break
        iters += 1
    return guess


_methods = {'minimize': minimize_method, 'weiszfeld': weiszfeld_method}


def geometric_median_previous(points, options={}):
    """
    Calculates the geometric median of an array of points.
    method specifies which algorithm to use:
        * 'auto'      -- uses a heuristic to pick an algorithm
        * 'minimize'  -- scipy.optimize the sum of distances
        * 'weiszfeld' -- Weiszfeld's algorithm
    """
    points = np.asarray(points)
    if len(points.shape) == 1:
        # geometric_median((0, 0)) has too much potential for error.
        # Did the user intend a single 2D point or two scalars?
        # Use np.median if you meant the latter.
        raise ValueError("Expected 2D array")
    # if method == 'auto':
    if points.shape[1] > 2:
        method = 'weiszfeld'
    else:
        method = 'minimize'
    return _methods[method](points, options)