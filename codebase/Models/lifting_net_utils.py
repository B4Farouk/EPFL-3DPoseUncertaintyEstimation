import torch
from dlt_functions.perform_dlt      import undistort_batch


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


def function_to_obtain_3D_poses_from_depth(inp_2d_poses_dist, pred_z, R, t, K, dist):
    """
    Function to convert 2D poses in 2.5 representation to a 3D pose in the camera and the world coordinates.
    :param inp_2d_poses_dist: The input 2D poses of shape num_samples x number of joints x 2.
    :param pred_z : The predicted Z from the lifting network (That is predicted depth + the depth of the root) of shape num_samples x number of joints
    :param R: The Rotation Matrix is of shape num_samples x 3 x 3.
    :param t: The translation Matrix is of shape num_samples x 3.
    :param K: The Camera Intrinsic Matrix is of shape num_samples x 3 x 3.
    :param dist: The Camera Distortion Matrix  is of num_samples x 5.
    :return: The 3D pose in the World and Camera Coordinates respectively.

    """
    num_samples      = inp_2d_poses_dist.size(0)
    number_of_joints = inp_2d_poses_dist.size(1)

    f             = K[:, [0, 1], [0, 1]].reshape(num_samples, 2, 1)
    c             = K[:, 0:2, 2].reshape(num_samples, 2, 1)
    k             = dist[:, [0, 1, 4]].reshape(num_samples, 3, 1)
    p             = dist[:, [2, 3]].reshape(num_samples, 2, 1)
    pred_cam      = undistort_batch(points_2d_dist=inp_2d_poses_dist, f=f, c=c, k=k, p=p, dont_project=True)
    pred_cam      = torch.cat((pred_cam, torch.ones(num_samples, number_of_joints, 1).type_as(pred_cam)), dim=-1)
    pred_cam      = pred_cam * pred_z
    pred_world_th = torch.matmul(R.transpose(1, 2), pred_cam.transpose(1, 2) - t.reshape(num_samples, 3, 1)).transpose(1, 2)
    return pred_world_th, pred_cam
