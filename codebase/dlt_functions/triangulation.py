import torch


def homogeneous_to_euclidean(points):
    """Converts torch homogeneous points to euclidean
    Args:
        points torch tensor of shape (N, M + 1): N homogeneous points of dimension M
    Returns:
        torch tensor of shape (N, M): euclidean points
    """
    if torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("This methods expects a PyTorch tensor.")


def triangulate_from_multiple_views_svd(proj_matrices, points, weights=None):
    """
    This module lifts B 2d detections obtained from N viewpoints to 3D using the Direct Linear Transform method.
    It computes the eigenvector associated to the smallest eigenvalue using Singular Value Decomposition.
    :param proj_matrices: torch tensor of shape (B, N, 3, 4): sequence of projection matrices (3x4).
    :param points: torch tensor of shape (B, N, 2): sequence of points' coordinates.
    :param weights: If not None, torch tensor of shape should be of size B x N x 1 x 1.
    :return:
    """
    batch_size = proj_matrices.shape[0]
    n_views    = proj_matrices.shape[1]
    A          = proj_matrices[:, :, 2:3].expand(batch_size, n_views, 2, 4) * points.view(-1, n_views, 2, 1)
    A         -= proj_matrices[:, :, :2]
    if weights is not None:
        A = A*weights
    A             = A.view(batch_size, -1, 4)
    _, _, vh      = torch.svd(A)
    point_3d_homo = -vh[:, :, 3]
    point_3d      = homogeneous_to_euclidean(point_3d_homo)
    return point_3d
