import torch


def obtain_normalized_keypoints_in_2D(points_2d_dist: torch.Tensor, bboxes : torch.Tensor):
    """
    Function to obtain Keypoints in the normalized coordinates [-1, 1] from the Normalized keypoints in the Image Coordinates.
    :param points_2d_dist: The Distorted Keypoints in the Image Coordinates of shape (N*number_of_cameras) x number of joints x 2.
    :param bboxes: The Bounding Boxes for every Image of Shape (N*number_of_cameras) x 4.
    :return: The Keypoints in the normalized coordinates [-1, 1] of shape (N * number_of_cameras) x number of joints x 2.
    """
    # N, C, J        = points_2d_dist.size(0), points_2d_dist.size(1), points_2d_dist.size(2)
    # points_2d_dist = points_2d_dist.contiguous().view(-1, J, 2)
    bboxes      = bboxes.int()
    xmin        = bboxes[:, 0]
    ymin        = bboxes[:, 1]
    xmax        = bboxes[:, 2]
    ymax        = bboxes[:, 3]
    crop_width  = xmax - xmin
    crop_height = ymax - ymin
    crop_size   = torch.stack((crop_width.unsqueeze(dim=1), crop_height.unsqueeze(dim=1)), dim=2)
    crop_shift  = torch.stack((xmin.unsqueeze(dim=1), ymin.unsqueeze(dim=1)), dim=2)
    points_norm = ((points_2d_dist - crop_shift) / crop_size) * 2 - 1.0
    # points_norm = points_norm.view(N, C, J, 2)
    return points_norm


def obtain_distorted_keypoints_in_2D(bboxes : torch.Tensor, points_norm : torch.Tensor):
    """
    Function to obtain Distorted keypoints in the Image Coordinates from the Keypoints in the Normalized Coordinates.
    :param bboxes: The Bounding Boxes for every Image of Shape N*number_of_cameras x 4.
    :param points_norm: The Keypoints in the Normalized coordinates [-1, 1] of shape N x number_of_cameras x number of joints x 2.
    :return: The Distorted Keypoints in the Image Coordinates of shape N x number_of_cameras x number of joints x 2.
    """
    # N, C, J     = points_norm.size(0), points_norm.size(1), points_norm.size(2)
    # points_norm = points_norm.contiguous().view(-1, J, 2)
    bboxes      = bboxes.int()
    xmin        = bboxes[:, 0]
    ymin        = bboxes[:, 1]
    xmax        = bboxes[:, 2]
    ymax        = bboxes[:, 3]
    crop_width  = xmax - xmin
    crop_height = ymax - ymin
    crop_size   = torch.stack((crop_width.unsqueeze(dim=1), crop_height.unsqueeze(dim=1)), dim=2)
    crop_shift  = torch.stack((xmin.unsqueeze(dim=1), ymin.unsqueeze(dim=1)), dim=2)
    points_dist = torch.mul((points_norm + 1) / 2, crop_size) + crop_shift
    # points_dist = points_dist.view(N, C, J, 2)
    return points_dist
