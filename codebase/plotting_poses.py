import numpy as np
import os
import cv2
from utils import mkdir_if_missing

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def get_pose_with_pelvis(pose, lhip_idx, rhip_idx):
    pelvis = (pose[lhip_idx] + pose[rhip_idx]) / 2
    pelvis = pelvis.reshape(1, pose.shape[-1])
    pose_with_pelvis = np.vstack((pose, pelvis))
    return pose_with_pelvis


def get_hip(points_key, lhip_idx, rhip_idx):
    l_hip = points_key[lhip_idx]
    r_hip = points_key[rhip_idx]
    pelvis = [(x + y) / 2 for x, y in zip(l_hip, r_hip)]  # l_hip + r_hip / 2
    return pelvis


def get_pelvis(points_key, joint_present, lhip_idx, rhip_idx, neck_idx):
    if lhip_idx in joint_present and rhip_idx in joint_present:
        l_hip = points_key[lhip_idx]
        r_hip = points_key[rhip_idx]
        pelvis = [(x + y) / 2 for x, y in zip(l_hip, r_hip)]
        # l_hip + r_hip / 2
        # show_pelvis = 1

    elif lhip_idx in joint_present and rhip_idx not in joint_present and neck_idx in joint_present:
        l_hip = points_key[lhip_idx]
        head = points_key[0]
        pelvis = [head[0], l_hip[1]]
        # show_pelvis = 1

    elif rhip_idx in joint_present and lhip_idx not in joint_present and neck_idx in joint_present:
        r_hip = points_key[rhip_idx]
        head = points_key[0]
        pelvis = [head[0], r_hip[1]]
        # show_pelvis = 1
    else:
        pelvis = [0.0, 0.0]
        # show_pelvis = 0
    return pelvis  # , show_pelvis


def plot_poses(save_folder: str, pose_2d_information: dict, key_val_plot: str, vals_to_plot: list, phase: str,
               suffix: str, bones: list, rhip_idx: int, lhip_idx: int, pelvis_idx: int, neck_idx: int):
    """
    Function to plot the detected, target and projected 2D keypoints.
    :param save_folder:   The Folder to save the images.
    :param pose_2d_information: A dictionary containing the necessary values to plot in 2D.
    :param key_val_plot: The Key of the image/subject/action that we are considering at present.
    :param vals_to_plot: A tuple consisting of the various values that we need to plot in 2D.
    :param phase:        The current phase of plotting, i.e. either Test or Train.
    :param suffix:       A suffix string that is to  be added to the file names of the stored images.
    :param bones:        The bones needed to plot the pose.
    :param rhip_idx:     An integer denoting the index of the Right Hip if present, otherwise it will be -1.
    :param lhip_idx:     An integer denoting the index of the Lef Hip if present, otherwise it will be -1.
    :param pelvis_idx:   An integer denoting the index of the Pelvis if present, otherwise it will be -1.
    :param neck_idx:     An integer denoting the index of the Neck if present, otherwise it will be -1.
    :return:
    """
    color_text  = (1, 0, 0)
    folder_save = os.path.join(save_folder, phase)
    mkdir_if_missing(folder_save)
    cameras     = list(pose_2d_information.keys())
    # color_codes = {'Det_2D': (255, 0, 0), 'Tar_2D': (0, 255, 0), 'Proj_2D': (255, 255, 255)}
    color_codes = {'Det_2D': (255, 0, 0), 'Tar_2D': (255, 255, 255), 'Proj_2D': (0, 255, 0)}
    line_widths = {'Det_2D': 3, 'Tar_2D': 3, 'Proj_2D': 3}
    legends     = {'Det_2D': ('DET', 'r'), 'Tar_2D': ('TAR', 'g'), 'Proj_2D': ('PROJ', 'b')}
    fig         = plt.figure()
    counter     = 1
    flag        = False
    num_cameras = len(cameras)
    num_cols    = int(np.ceil(float(num_cameras / 2)))
    legend_elements = []
    for plot_val in vals_to_plot:
        legend_val = legends[plot_val]
        legend_elements.append(Line2D([0], [0], color=legend_val[-1], lw=4, label=legend_val[0]))

    for camera in cameras:
        pose_2d_cam  = pose_2d_information[camera]
        if pose_2d_cam is not None:
            img_path = pose_2d_cam['Image_Path_2D']
            # print(img_path)
            # if not os.path.exists(img_path):
            #     breakpoint()
            img_orig = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img_orig = np.array(img_orig.astype(float))
            for plot_val in vals_to_plot:
                pose_2d_legend      = pose_2d_cam[plot_val]
                bones_to_plot       = bones.copy()
                pose_2d_legend      = list(pose_2d_legend)
                num_original_joints = len(pose_2d_legend)
                if pelvis_idx == -1:
                    # pelvis is not present. we need to calculate it.
                    assert lhip_idx != -1
                    assert rhip_idx != -1
                    assert neck_idx != -1
                    pelvis_ = get_hip(points_key=pose_2d_legend, lhip_idx=lhip_idx, rhip_idx=rhip_idx)
                    pose_2d_legend.append(pelvis_)
                    extra_bones_with_pelvis = [[neck_idx, num_original_joints]],  # Between Neck and Pelvis.
                    bones_to_plot.extend(extra_bones_with_pelvis)

                for bone in bones_to_plot:
                    idx_1 = bone[0]; idx_2 = bone[1]
                    x1 = int(pose_2d_legend[idx_1][0]); y1 = int(pose_2d_legend[idx_1][1])
                    x2 = int(pose_2d_legend[idx_2][0]); y2 = int(pose_2d_legend[idx_2][1])
                    cv2.line(img_orig, (x1, y1), (x2, y2), color_codes[plot_val], thickness=line_widths[plot_val], lineType=8)
            bbox     = pose_2d_cam['bbox']
            ax       = fig.add_subplot(2, num_cols, counter)
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            img_orig = img_orig[ymin:ymax, xmin:xmax]
            h, w     = img_orig.shape[0:2]
            if not (h == 0 or w == 0):
                flag = True
                ax.imshow(img_orig / 255.0)  #
                ax.set_title('CAM-{}'.format(camera), color=color_text)
            counter += 1

    if flag:
        fig.legend(handlesx=legend_elements, loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(folder_save, "{}-{}-Joints.jpg".format(key_val_plot, suffix)))
        plt.close()


def plot_poses_2D_3D(save_folder, pose_information, key_val_plot, vals_to_plot_2D, phase, suffix,
                     vals_to_plot_3D, rhip_idx, lhip_idx, neck_idx, pelvis_idx, bones):
    """
    Function to plot the detected, target and projected 2D keypoints and 3D Keypoints in Single View.
    :param save_folder: The Folder to save the images.
    :param pose_information: A dictionary containing the necessary values to plot in 2D and 3D.
    :param key_val_plot: The Key of the image/subject/action that we are considering at present.
    :param vals_to_plot_2D: A tuple consisting of the various values that we need to plot in 2D.
    :param phase: The current phase of plotting, i.e. either Test or Train.
    :param suffix: A suffix string that is to  be added to the file names of the stored images.
    :param vals_to_plot_3D: A tuple consisting of the various values that we need to plot in 3D.
    :param rhip_idx: An integer denoting the index of the Right Hip if present, otherwise it will be -1.
    :param lhip_idx: An integer denoting the index of the Left Hip if present, otherwise it will be -1.
    :param neck_idx: An integer denoting the index of the Neck if present, otherwise it will be -1.
    :param pelvis_idx: An integer denoting the index of the Pelvis if present, otherwise it will be -1.
    :param bones: The bones needed to plot the pose.
    :return:
    """
    folder_save = os.path.join(save_folder, 'Images', phase, key_val_plot)
    mkdir_if_missing(folder_save)
    color_text      = (1, 0, 0)
    camera          = pose_information['camera_id']  # This is a string.
    subject         = pose_information['subject_id']  # This is a string.
    action          = pose_information['action_id']  # This is a string.
    image_file_path = pose_information['Image_Path_2D']  # This is a string.
    info_3D         = pose_information['info_3D']  # This is a dictionary
    keys_3D_present = list(info_3D.keys())
    info_2D         = pose_information['info_2D']  # This is a dictionary
    keys_2D_present = list(info_2D.keys())
    bbox_2D         = pose_information['bbox_2D']  # This is a list of integers.
    image_id        = pose_information['image_id']  # This is an integer.
    color_codes_2D  = {'Det_2D': (255, 0, 0), 'GT_2D': (0, 255, 0), 'Proj_2D-Lift': (0, 0, 255)}
    color_codes_3D  = {'Anno-3D': (1, 0, 0), 'Tri-3D': (0, 1, 0), 'Pred-3D': (0, 0, 1)}
    line_widths_2D  = {'Det_2D': 2, 'Tar_2D': 2, 'Proj_2D': 2}
    legends         = {'Det_2D'  : ('DET-2D', 'r'),  'GT_2D'  : ('GT-2D', 'g'),  'Proj_2D-Lift' : ('PROJ-LIFT', 'b'),
                       'Anno-3D' : ('ANNO-3D', 'r'), 'Tri-3D' : ('TAR-3D', 'g'), 'Pred-3D'      : ('PRED-LIFT', 'b')}
    flag       = False
    fig        = plt.figure()
    ax1        = fig.add_subplot(121, projection='3d')
    legends_3D = ([], [])
    for plot_val_3d in vals_to_plot_3D:
        if plot_val_3d in keys_3D_present:
            pose_3d_val = info_3D[plot_val_3d]
            if pose_3d_val is not None:
                bones_to_plot = bones.copy()
                pose_3d_val   = list(pose_3d_val)
                num_joints    = len(pose_3d_val)
                if pelvis_idx == -1:  # pelvis is not present. we need to calculate it.
                    assert lhip_idx != -1
                    assert rhip_idx != -1
                    assert neck_idx != -1
                    pelvis_3d = get_hip(points_key=pose_3d_val, lhip_idx=lhip_idx, rhip_idx=rhip_idx)
                    pose_3d_val.append(pelvis_3d)
                    extra_bones_with_pelvis = [[neck_idx, num_joints]]
                    bones_to_plot.extend(extra_bones_with_pelvis)
                for bone in bones_to_plot:
                    start_bone_idx = bone[0]
                    end_bone_idx   = bone[1]
                    start_bone     = pose_3d_val[start_bone_idx]
                    end_bone       = pose_3d_val[end_bone_idx]
                    P,             = ax1.plot3D([start_bone[0], end_bone[0]], [start_bone[1], end_bone[1]],
                                                [start_bone[2], end_bone[2]], c=color_codes_3D[plot_val_3d])
                legends_3D[0].append(P)
                legends_3D[1].append(plot_val_3d)
    ax1.legend(legends_3D[0], legends_3D[1])

    img_orig   = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
    img_orig   = np.array(img_orig.astype(float))
    ax2        = fig.add_subplot(122)
    legends_2D = ([], [])
    for plot_val_2d in vals_to_plot_2D:
        if plot_val_2d in keys_2D_present:
            pose_2d_val = info_2D[plot_val_2d]
            if pose_2d_val is not None:
                legends_2D[0].append(plot_val_2d)
                legend_val_2d = legends[plot_val_2d]
                legends_2D[1].append(Line2D([0], [0], color=legend_val_2d[-1], lw=4, label=legend_val_2d[0]))
                bones_to_plot = bones.copy()
                pose_2d_val   = list(pose_2d_val)
                num_original_joints = len(pose_2d_val)
                if pelvis_idx == -1:  # pelvis is not present. we need to calculate it.
                    assert lhip_idx != -1
                    assert rhip_idx != -1
                    assert neck_idx != -1
                    pelvis_ = get_hip(points_key=pose_2d_val, lhip_idx=lhip_idx, rhip_idx=rhip_idx)
                    pose_2d_val.append(pelvis_)
                    extra_bones_with_pelvis = [[neck_idx, num_original_joints]]
                    bones_to_plot.extend(extra_bones_with_pelvis)

                for bone in bones_to_plot:
                    idx_1 = bone[0]; idx_2 = bone[1]
                    x1 = int(pose_2d_val[idx_1][0]); y1 = int(pose_2d_val[idx_1][1])
                    x2 = int(pose_2d_val[idx_2][0]); y2 = int(pose_2d_val[idx_2][1])
                    cv2.line(img_orig, (x1, y1), (x2, y2), color_codes_2D[plot_val_2d],
                             thickness=line_widths_2D[plot_val_2d], lineType=8)

    xmin, ymin, xmax, ymax = int(bbox_2D[0]), int(bbox_2D[1]), int(bbox_2D[2]), int(bbox_2D[3])
    img_orig               = img_orig[ymin:ymax, xmin:xmax]
    h, w                   = img_orig.shape[0:2]
    if not (h == 0 or w == 0):
        fig.legend(handles=legends_2D[1], loc='upper right')
        flag = True
        ax2.imshow(img_orig / 255.0)  #
    if flag:
        plt.tight_layout()
        fig.suptitle('Subject {} in Camera {} performing {} \n in Frame {:06d}.'.format(subject, camera, action, image_id),
                     color=color_text)
        plt.savefig(os.path.join(folder_save, "{}-{}-Joints.jpg".format(key_val_plot, suffix)))
        plt.close()
