import cv2
import torch
import random
import dgl

import numpy                  as np
import torchvision.transforms as transforms

from Models.get_alphapose_models import test_transform
from torch.utils.data.dataloader import default_collate
from torch.utils.data            import Dataset, sampler
from utils                       import (im_to_torch, crop_from_dets, square_the_bbox, scale_bbox_resnet152, area,
                                         crop_image_resnet152, image_shape_resnet152, resize_image_resnet152,
                                         normalize_image_resnet152, get_windows, bbox_from_points, function_to_extend_dim,
                                         compute_intersection)

graph_key                 = 'batch_graphs'
keys_to_return_for_graphs = ['inp_pose_model', 'bboxes', 'target_pose_2d', 'R', 't', 'K', 'dist', 'target_pose_3d',
                             'image_paths', 'camera_ids', 'frame_ids', 'action_ids', 'subject_ids', 'consider',
                             'pose_2d_tar_norm', 'pelvis_cam_z', 'labels', 'consider_for_triangulation', 'batch_graphs',
                             'target_root_rel_depth', 'target_pose_3d_camera_coordinate']


class Batch_sampler(sampler.Sampler):
    def __init__(self, labels, batch_size, randomize, num_anno_samples_per_batch):
        super(Batch_sampler, self).__init__(data_source=labels)
        """
        Function to Create a Sampler having a fixed amount of annotated samples <num_anno_samples_per_batch> in a batch of size "batch_size".
        :param labels: The Labels of the each individual instance of Data.
        :param batch_size: The Batch Size.
        :param randomize: If True, we will randomize the instances in the each mini batch. TODO
        :param num_anno_samples_per_batch: The number of annotated samples to be present in each mini batch.
        """
        self.labels                          = np.asarray(labels)
        self.batch_size                      = batch_size
        self.randomize                       = randomize
        self.num_anno_samples_per_batch      = num_anno_samples_per_batch
        self.labeled_candidates_idx          = np.where(self.labels == 1)[0]
        self.unlabeled_candidates_idx        = np.where(self.labels == 0)[0]
        self.num_labeled_candidates          = self.labeled_candidates_idx.size
        self.num_unlabeled_candidates        = self.unlabeled_candidates_idx.size
        self.labeled_candidates_batch_size   = num_anno_samples_per_batch  # int(np.floor(self.batch_size / 2))
        self.unlabeled_candidates_batch_size = self.batch_size - self.labeled_candidates_batch_size

        x1                = np.ceil(self.num_labeled_candidates / float(self.labeled_candidates_batch_size))
        x2                = np.ceil(self.num_unlabeled_candidates) / float(self.unlabeled_candidates_batch_size)
        limit             = max(x1, x2)
        self.limit        = int(limit)
        total_num_batches = self.limit
        self.total_count  = batch_size * total_num_batches

    def __len__(self):
        return self.total_count

    def __iter__(self):
        final_indices                   = []
        labeled_candidates_idx          = np.copy(self.labeled_candidates_idx)
        unlabeled_candidates_idx        = np.copy(self.unlabeled_candidates_idx)
        count_num_candidates_considered = 0
        while count_num_candidates_considered < self.limit:
            labeled_indices_batch            = labeled_candidates_idx[0:self.labeled_candidates_batch_size]
            unlabeled_indices_batch          = unlabeled_candidates_idx[0:self.unlabeled_candidates_batch_size]
            count_num_candidates_considered += 1
            final_indices.extend(labeled_indices_batch)
            final_indices.extend(unlabeled_indices_batch)
            labeled_candidates_idx   = np.roll(labeled_candidates_idx, -self.labeled_candidates_batch_size)
            unlabeled_candidates_idx = np.roll(unlabeled_candidates_idx, -self.unlabeled_candidates_batch_size)
        return iter(final_indices)



def chunks(lst, n, extend_last):
    """Yield successive n-sized chunks from lst."""
    len_ = len(lst)
    for i in range(0, len_, n):
        x = lst[i:i + n]
        if len(x) != n and extend_last:
            rem_len = n-len(x)
            x.extend(lst[0:rem_len])
        yield x



class Sampler_Combine(sampler.Sampler):
    def __init__(self, lens: list, batch_size : int, randomize: bool, extend_last: bool,
                 labels: list, only_annotations: bool, batch_construction_constraint: bool, num_anno_samples_per_batch: int):
        """
        :param lens:
        :param batch_size:
        :param randomize:
        :param extend_last:
        :param labels:
        :param only_annotations:
        :param num_anno_samples_per_batch:
        """
        super(Sampler_Combine, self).__init__(data_source=None)
        self.lens                          = lens
        self.batch_size                    = batch_size
        self.randomize                     = randomize
        self.extend_last                   = extend_last
        self.labels                        = labels
        self.only_annotations              = only_annotations
        self.batch_construction_constraint = batch_construction_constraint
        self.num_anno_samples_per_batch    = num_anno_samples_per_batch
        counter = 0
        batches = []
        for len_, label_ in zip(self.lens, self.labels):
            assert len_ == len(label_)
            start = counter
            end   = counter + len_
            lst   = list(range(start, end))
            if self.only_annotations:
                assert self.batch_construction_constraint is False
                lst_ = []
                N    = len(label_)
                for i in range(N):
                    if label_[i] == 1:
                        lst_.append(lst[i])
                lst         = lst_
                flag_chunks = True
                extend_last = True

            elif self.batch_construction_constraint:
                assert self.only_annotations is False
                labeled_candidates_idx          = np.where(np.array(label_) == 1)[0] + start
                unlabeled_candidates_idx        = np.where(np.array(label_) == 0)[0] + start
                num_labeled_candidates          = labeled_candidates_idx.size
                num_unlabeled_candidates        = unlabeled_candidates_idx.size
                labeled_candidates_batch_size   = self.num_anno_samples_per_batch
                unlabeled_candidates_batch_size = self.batch_size - labeled_candidates_batch_size

                x2 = np.ceil(num_unlabeled_candidates) / float(unlabeled_candidates_batch_size)
                if x2 == 0:
                    labeled_candidates_batch_size = batch_size
                x1    = np.ceil(num_labeled_candidates / float(labeled_candidates_batch_size))
                limit = int(max(x1, x2))
                count_num_candidates_considered = 0
                while count_num_candidates_considered < limit:
                    labeled_indices_batch            = labeled_candidates_idx[0:labeled_candidates_batch_size]
                    unlabeled_indices_batch          = unlabeled_candidates_idx[0:unlabeled_candidates_batch_size]
                    count_num_candidates_considered += 1
                    list_indexes                     = list(labeled_indices_batch) + list(unlabeled_indices_batch)
                    batches.append(list_indexes)
                    labeled_candidates_idx   = np.roll(labeled_candidates_idx,   -labeled_candidates_batch_size)
                    unlabeled_candidates_idx = np.roll(unlabeled_candidates_idx, -unlabeled_candidates_batch_size)
                flag_chunks = False
                counter    += len_
                extend_last = False
            else:
                flag_chunks = True
                extend_last = self.extend_last

            if flag_chunks:
                chunk    = chunks(lst, n=self.batch_size, extend_last=extend_last)
                counter += len_
                batches.extend(list(chunk))

        batches      = [batch for batch in batches if len(batch) == batch_size]
        self.batches = batches
        self.length  = len(self.batches) * batch_size

    def __len__(self):
        return self.length

    def __iter__(self):
        batches = self.batches
        if self.randomize:
            random.shuffle(batches)
        final_batches = []
        for x in batches:
            final_batches.extend(x)
        return iter(final_batches)


class Dataset_base(Dataset):
    def get_pose_3d_common(self, subject_idx, seq_idx, frame_idx):
        """
        :param subject_idx: It is the id of the Subject.
        :param seq_idx: For MPI, it is the sequence id, for H36M it is the action id.
        :param frame_idx: This is the frame index.
        :return:
        """
        return np.array([])

    def get_filename_image_common(self, subject_idx, seq_idx, cam_idx, frame_idx):
        """
        :param subject_idx: It is the id of the Subject.
        :param seq_idx: For MPI, it is the sequence id, for H36M it is the action id.
        :param cam_idx: This is the id of the camera.
        :param frame_idx : This is the id of the frame.
        :return:
        """
        return ''


    def get_calibration_common(self, subject_idx, seq_idx, cam_idx, frame_idx, rot_vector=True):
        """
        :param subject_idx: It is the id of the Subject.
        :param seq_idx: For MPI, it is the sequence id, for H36M it is the action id.
        :param frame_idx: This is the id of the frame.
        :param cam_idx: This is the id of the camera
        :param rot_vector: If True, we will perform cv2.Rodrigues on the Rotation Matrix.
        :return:
        """
        return np.array([]), np.array([]), np.array([]), np.array([]), []


    def get_pose_2d_common(self, subject_idx, seq_idx, cam_idx, frame_idx, from_3d):
        """
        :param subject_idx: It is the id of the Subject.
        :param seq_idx: For MPI, it is the sequence id, for H36M it is the action id.
        :param frame_idx: This is the id of the frame.
        :param cam_idx: This is the id of the camera
        :param from_3d: If True, we will calculate the 2D pose by projecting the 3D pose, or else we will use the manually detected 2D pose.
        :return:
        """
        return np.array([])

    def bbox_from_points_(self, pose_2d_vid, subject_id, image_idx, camera_name):
        bbox = bbox_from_points(pose_2d_vid, pb=self.p)
        return bbox


    def obtain_mv_samples(self, subject_idx: str, seq_idx: str, frame_idx, cameras : list, label):
        """
        Functions <get_filename_image_common>, <get_calibration_common>, <get_pose_2d_common>, <get_pose_3d_common>
        need to be created for every dataset.

        :param subject_idx: It is the id of the Subject.
        :param seq_idx: For MPI, it is the sequence id, for H36M it is the action id.
        :param cameras: A list of cameras
        :param frame_idx : This is the id of the frame.
        :param label: This is the label of the frame.
        :return:
        """
        target_pose_3d     = self.get_pose_3d_common(subject_idx=subject_idx, seq_idx=seq_idx, frame_idx=frame_idx)
        if target_pose_3d is not None:
            target_pose_3d = np.float32(target_pose_3d)
            mask_valid     = ~np.isnan(target_pose_3d[:, 0])
        else:
            mask_valid = None

        bboxes = []; inp_pose_model = []; consider = []; R = []; t = []; K = []; dist = []; target_pose_2d = []
        pose_2d_tar_norm = []; image_paths = []; camera_ids = []; inp_images_lifting_net = []; pelvis_cam_z = []
        target_root_rel_depth = []; target_pose_3d_camera_coordinate = []

        num_cameras      = len(cameras)
        for i in range(num_cameras):
            cam_id                    = cameras[i]
            img_file_path             = self.get_filename_image_common(subject_idx=subject_idx, seq_idx=seq_idx, cam_idx=cam_id,
                                                                       frame_idx=frame_idx)
            R_cam, t_cam, K_cam, \
                dist_cam, shape_view  = self.get_calibration_common(subject_idx=subject_idx, seq_idx=seq_idx,
                                                                    cam_idx=cam_id, frame_idx=frame_idx, rot_vector=False)
            pose_2d_vid               = self.get_pose_2d_common(subject_idx=subject_idx, seq_idx=seq_idx, cam_idx=cam_id,
                                                                frame_idx=frame_idx, from_3d=self.from_3d)
            bbox_old                  = self.bbox_from_points_(pose_2d_vid=pose_2d_vid, subject_id=subject_idx,
                                                               image_idx=frame_idx, camera_name=cam_id)
            """
            # xmin, ymin, xmax, ymax    = bbox
            # xmin = max(xmin - self.margin, 0)
            # ymin = max(ymin - self.margin, 0)
            # xmax = min(xmax + self.margin, shape_view[1])
            # ymax = min(ymax + self.margin, shape_view[0])
            # bbox_old               = [int(xmin), int(ymin), int(xmax), int(ymax)]
            """
            pose_input, bbox_new   = self.function_to_obtain_pose_input(image_file_path=img_file_path, bbox=bbox_old)  # It is just a single string.
            xmin, ymin, xmax, ymax = int(bbox_new[0]), int(bbox_new[1]), int(bbox_new[2]), int(bbox_new[3])
            if self.inp_lifting_net_is_images:
                lifting_net_input_image = self.function_to_obtain_lifting_net_input(image_file_path=img_file_path, bbox=bbox_new)
                inp_images_lifting_net.append(lifting_net_input_image)

            """
            area_A = (xmax - xmin) * (ymax - ymin)
            recB   = [0, 0, shape_view[1], shape_view[0]]
            if area_A == 0.0:
                consider.append(0)
            else:
                area_overlapped = area(boxA=bbox_new, boxB=recB) / float(area_A)
            """
            bbox_for_visibility = [xmin, ymin, xmax, ymax]
            bbox_img            = [0, 0, shape_view[1], shape_view[0]]
            area_overlapped     = float(compute_intersection(bbox_for_visibility, [bbox_img])[0])
            if area_overlapped < 0.7:
                consider.append(0)
            else:
                consider.append(1)

            pose_2d_vid_norm    = self.get_norm_2d_pose(pose_2d=pose_2d_vid, bbox=bbox_new)
            camera_ids.append('{}'.format(cam_id))
            bboxes.append(bbox_new)
            inp_pose_model.append(pose_input)
            target_pose_2d.append(pose_2d_vid)
            pose_2d_tar_norm.append(pose_2d_vid_norm)
            image_paths.append(img_file_path)
            R.append(R_cam)
            t.append(t_cam)
            K.append(K_cam)
            dist.append(dist_cam)

            R_cam, t_cam    = np.array(R_cam), np.array(t_cam)
            tar_pose_3d_cam = np.dot(R_cam, target_pose_3d.T).T + t_cam.reshape(1, 3)
            if self.pelvis_idx != -1:
                pelvis_cam = tar_pose_3d_cam[self.pelvis_idx]
            else:
                pelvis_cam = np.mean([tar_pose_3d_cam[self.lhip_idx], tar_pose_3d_cam[self.rhip_idx]], axis=0)
            pelvis_cam_z_cam = pelvis_cam[2]
            pelvis_cam_z.append(pelvis_cam_z_cam)

            tar_root_relative_depth = tar_pose_3d_cam[:, 2] - pelvis_cam_z_cam
            target_root_rel_depth.append(tar_root_relative_depth)
            target_pose_3d_camera_coordinate.append(tar_pose_3d_cam)

        if ((sum(consider) >= self.minimum_views_needed_for_triangulation) and (self.minimum_views_needed_for_triangulation > 0)) \
                or self.minimum_views_needed_for_triangulation == 0 :
            consider_for_triangulation       = 1
            pose_2d_tar_norm                 = torch.from_numpy(np.stack(pose_2d_tar_norm))
            inp_pose_model                   = torch.from_numpy(np.stack(inp_pose_model))
            bboxes                           = torch.from_numpy(np.stack(bboxes))
            target_pose_2d                   = torch.from_numpy(np.stack(target_pose_2d))
            R                                = torch.from_numpy(np.stack(R))
            t                                = torch.from_numpy(np.stack(t))
            K                                = torch.from_numpy(np.stack(K))
            dist                             = torch.from_numpy(np.stack(dist))
            target_pose_3d                   = torch.from_numpy(target_pose_3d)
            target_pose_3d_camera_coordinate = torch.from_numpy(np.stack(target_pose_3d_camera_coordinate))
            consider                         = torch.Tensor(consider)
            label                            = torch.from_numpy(np.array(label))
            pelvis_cam_z                     = torch.from_numpy(np.array(pelvis_cam_z))
            label                            = label.repeat(num_cameras)
            target_root_rel_depth            = torch.from_numpy(np.array(target_root_rel_depth))
            target_pose_3d                   = target_pose_3d.unsqueeze(dim=0)
            target_pose_3d                   = target_pose_3d.repeat(num_cameras, 1, 1)
            """
            target_pose_3d_camera_coordinate = target_pose_3d_camera_coordinate.unsqueeze(dim=0)
            print(target_pose_3d_camera_coordinate.size(), target_pose_3d.size())
            target_pose_3d_camera_coordinate = target_pose_3d_camera_coordinate.repeat(num_cameras, 1, 1)
            """
            retval                           = {'inp_pose_model' : inp_pose_model, 'bboxes' : bboxes, 'target_pose_2d' : target_pose_2d,
                                                'R' : R, 't' : t, 'K' : K, 'dist' : dist, 'target_pose_3d' : target_pose_3d,
                                                'image_paths' : image_paths, 'camera_ids': camera_ids,  'frame_ids': int(frame_idx),
                                                'action_ids' : seq_idx, 'subject_ids' : subject_idx, 'consider' : consider,
                                                'pose_2d_tar_norm' : pose_2d_tar_norm, 'mask_valid' : mask_valid,
                                                'consider_for_triangulation': consider_for_triangulation, 'labels' : label,
                                                'pelvis_cam_z' : pelvis_cam_z, 'target_root_rel_depth': target_root_rel_depth,
                                                'target_pose_3d_camera_coordinate' : target_pose_3d_camera_coordinate
                                                }
            if self.inp_lifting_net_is_images:
                inp_images_lifting_net           = torch.from_numpy(np.stack(inp_images_lifting_net))
                retval['inp_images_lifting_net'] = inp_images_lifting_net

            if self.training_pose_lifting_net_without_graphs:
                retval = function_to_extend_dim(data=retval, which_dim=0, which_keys=self.keys_to_extend_dim)

        else:
            retval = None
        return retval

    def obtain_mv_samples_sequences(self, subject_idx: str, seq_idx: str, cameras : list,
                                    frames_window: list, labels_window: list, target_frame_window,
                                    time_pred):
        """
        :param subject_idx: It is the id of the Subject.
        :param seq_idx: For MPI, it is the sequence id, for H36M it is the action id.
        :param cameras: A list of cameras
        :param frames_window: A list of frames present inside the window.
        :param labels_window: A list of labels of each frame present inside the window.
        :param target_frame_window: The target frame index inside the window.
        :param time_pred: The index of the target inside each window.
        :return:
        """
        bboxes = []; inp_pose_model = []; consider = []; R = []; t = []; K = []; dist = []; target_pose_2d = []
        pose_2d_tar_norm = []; inp_images_lifting_net = []; labels = []; predicted_frame_index = []
        image_paths_predicted_frame = []; camera_ids_predicted_frame = []; action_predicted_frame = []
        target_pose_3d = []; subject_predicted_frame = []; count_frames = -1; pelvis_cam_z = []; target_root_rel_depth = []
        num_frames     = len(frames_window); num_cameras = len(cameras); target_pose_3d_camera_coordinate = []

        for frame_idx, label_idx in zip(frames_window, labels_window):
            count_frames += 1  # This counting starts from 0 here.
            bboxes_frames = []; inp_pose_model_frames = []; consider_frames = []; R_frames = []; t_frames = []
            K_frames = []; dist_frames = []; target_pose_2d_frames = []; pose_2d_tar_norm_frames = []
            inp_images_lifting_net_frames = []; labels_frames = []; pelvis_cam_z_frames = []; target_root_rel_depth_frames = []
            target_pose_3d_camera_coordinate_frames = []

            target_pose_3d_frame = self.get_pose_3d_common(subject_idx=subject_idx, seq_idx=seq_idx, frame_idx=frame_idx)
            for i in range(num_cameras):
                cam_id                   = cameras[i]
                img_file_path            = self.get_filename_image_common(subject_idx=subject_idx, cam_idx=cam_id,
                                                                          frame_idx=frame_idx, seq_idx=seq_idx)
                R_frame_cam, t_frame_cam, K_frame_cam, dist_frame_cam, \
                    shape_view = self.get_calibration_common(subject_idx=subject_idx, seq_idx=seq_idx, cam_idx=cam_id,
                                                             frame_idx=frame_idx, rot_vector=False)
                pose_2d_vid             = self.get_pose_2d_common(subject_idx=subject_idx, seq_idx=seq_idx, cam_idx=cam_id,
                                                                  frame_idx=frame_idx, from_3d=self.from_3d)
                bbox_old                = self.bbox_from_points_(pose_2d_vid=pose_2d_vid, subject_id=subject_idx,
                                                                 image_idx=frame_idx, camera_name=cam_id)
                """
                # xmin, ymin, xmax, ymax  = bbox
                # xmin = max(xmin - self.margin, 0)
                # ymin = max(ymin - self.margin, 0)
                # xmax = min(xmax + self.margin, shape_view[1])
                # ymax = min(ymax + self.margin, shape_view[0])
                # bbox_old                   = [int(xmin), int(ymin), int(xmax), int(ymax)]
                """
                pose_input, bbox_frame_new = self.function_to_obtain_pose_input(image_file_path=img_file_path, bbox=bbox_old)  # It is just a single string.
                pose_input                 = np.array(pose_input)
                xmin, ymin, xmax, ymax     = int(bbox_frame_new[0]), int(bbox_frame_new[1]), int(bbox_frame_new[2]), int(bbox_frame_new[3])
                if self.inp_lifting_net_is_images:
                    lifting_net_input_image = self.function_to_obtain_lifting_net_input(image_file_path=img_file_path, bbox=bbox_frame_new)
                    lifting_net_input_image = np.array(lifting_net_input_image)
                    inp_images_lifting_net_frames.append(lifting_net_input_image)
                """
                area_A = (xmax - xmin) * (ymax - ymin)
                recB   = [0, 0, shape_view[1], shape_view[0]]
                if area_A == 0.0:
                    consider_frames.append(0)
                else:
                    area_overlapped = area(boxA=bbox_frame_new, boxB=recB) / float(area_A)
                """
                bbox_for_visibility = [xmin, ymin, xmax, ymax]
                bbox_img            = [0, 0, shape_view[1], shape_view[0]]
                area_overlapped     = float(compute_intersection(bbox_for_visibility, [bbox_img])[0])
                if area_overlapped < 0.7:
                    consider_frames.append(0)
                else:
                    consider_frames.append(1)

                pose_2d_frame_vid_norm = self.get_norm_2d_pose(pose_2d=pose_2d_vid, bbox=bbox_frame_new)
                inp_pose_model_frames.append(pose_input)
                target_pose_2d_frames.append(pose_2d_vid)
                pose_2d_tar_norm_frames.append(pose_2d_frame_vid_norm)
                R_frames.append(R_frame_cam)
                t_frames.append(t_frame_cam)
                K_frames.append(K_frame_cam)
                dist_frames.append(dist_frame_cam)
                bboxes_frames.append(bbox_frame_new)
                labels_frames.append(label_idx)

                R_cam, t_cam              = np.array(R_frame_cam), np.array(t_frame_cam)
                target_pose_3d_frame_cam  = np.dot(R_cam, target_pose_3d_frame.T).T + t_cam.reshape(1, 3)
                if self.pelvis_idx != -1:
                    pelvis_cam_frame_cam = target_pose_3d_frame_cam[self.pelvis_idx]
                else:
                    pelvis_cam_frame_cam = np.mean([target_pose_3d_frame_cam[self.lhip_idx], target_pose_3d_frame_cam[self.rhip_idx]], axis=0)

                pelvis_cam_z_frame = pelvis_cam_frame_cam[2]
                pelvis_cam_z_frames.append(pelvis_cam_z_frame)

                tar_root_relative_depth_frame_cam = target_pose_3d_frame_cam[:, 2] - pelvis_cam_z_frame
                target_root_rel_depth_frames.append(tar_root_relative_depth_frame_cam)
                target_pose_3d_camera_coordinate_frames.append(target_pose_3d_frame_cam)

                if count_frames == time_pred: # count_frames == time_pred: TODO-CHECK
                    assert (frame_idx == target_frame_window)
                    image_paths_predicted_frame.append(img_file_path)
                    camera_ids_predicted_frame.append(cam_id)
                    action_predicted_frame.append(seq_idx)
                    subject_predicted_frame.append(subject_idx)
                    predicted_frame_index.append(frame_idx)

            target_pose_3d.append(target_pose_3d_frame)
            bboxes.append(bboxes_frames)
            inp_pose_model.append(inp_pose_model_frames)
            consider.append(consider_frames)
            R.append(R_frames)
            t.append(t_frames)
            K.append(K_frames)
            dist.append(dist_frames)
            target_pose_2d.append(target_pose_2d_frames)
            pose_2d_tar_norm.append(pose_2d_tar_norm_frames)
            labels.append(labels_frames)
            pelvis_cam_z.append(pelvis_cam_z_frames)
            target_root_rel_depth.append(target_root_rel_depth_frames)
            target_pose_3d_camera_coordinate.append(target_pose_3d_camera_coordinate_frames)

            if self.inp_lifting_net_is_images:
                inp_images_lifting_net.append(inp_images_lifting_net_frames)

        pose_2d_tar_norm                 = torch.from_numpy(np.stack(pose_2d_tar_norm))
        bboxes                           = torch.from_numpy(np.stack(bboxes))
        target_pose_2d                   = torch.from_numpy(np.stack(target_pose_2d))
        pose_2d_tar_norm                 = torch.from_numpy(np.stack(pose_2d_tar_norm))
        R                                = torch.from_numpy(np.stack(R))
        t                                = torch.from_numpy(np.stack(t))
        K                                = torch.from_numpy(np.stack(K))
        dist                             = torch.from_numpy(np.stack(dist))
        target_pose_3d                   = torch.from_numpy(np.stack(target_pose_3d))
        target_pose_3d_camera_coordinate = torch.from_numpy(np.stack(target_pose_3d_camera_coordinate))
        consider                         = torch.Tensor(consider)
        labels                           = torch.Tensor(labels)
        inp_pose_model                   = torch.from_numpy(np.stack(inp_pose_model))
        pelvis_cam_z                     = torch.from_numpy(np.stack(pelvis_cam_z))

        target_pose_3d                   = target_pose_3d.unsqueeze(dim=1)
        target_pose_3d                   = target_pose_3d.repeat(1, num_cameras, 1, 1)
        """
        target_pose_3d_camera_coordinate = target_pose_3d_camera_coordinate.unsqueeze(dim=1)
        target_pose_3d_camera_coordinate = target_pose_3d_camera_coordinate.repeat(1, num_cameras, 1, 1)
        """
        target_root_rel_depth            = torch.from_numpy(np.array(target_root_rel_depth))
        ret_val                          = {'inp_pose_model': inp_pose_model, 'bboxes': bboxes,
                                            'target_pose_2d': target_pose_2d, 'R': R, 't': t, 'K': K, 'dist': dist,
                                            'target_pose_3d': target_pose_3d, 'image_paths': image_paths_predicted_frame,
                                            'camera_ids': camera_ids_predicted_frame, 'frame_ids': predicted_frame_index,
                                            'action_ids': action_predicted_frame, 'subject_ids': subject_predicted_frame,
                                            'consider': consider, 'pose_2d_tar_norm': pose_2d_tar_norm,
                                            'pelvis_cam_z': pelvis_cam_z, 'labels': labels,
                                            'consider_for_triangulation': torch.ones(num_frames, 1),
                                            'target_root_rel_depth' : target_root_rel_depth,
                                            'target_pose_3d_camera_coordinate' : target_pose_3d_camera_coordinate
                                            }
        if self.inp_lifting_net_is_images:
            inp_images_lifting_net = torch.from_numpy(np.stack(inp_images_lifting_net))
            ret_val['inp_images_lifting_net'] = inp_images_lifting_net
        return ret_val

    def __getitem__(self, item):
        return None

    def __init__(self,
                 dataset_folder,
                 phase,
                 num_joints,
                 overfit,
                 randomize,
                 only_annotations,
                 ten_percent_3d_from_all,
                 random_seed_for_ten_percent_3d_from_all,
                 augmentations,
                 resnet152_backbone,
                 get_2d_gt_from_3d,
                 minimum_views_needed_for_triangulation,
                 joints_order,
                 pose_model_type,
                 input_size_alphapose,
                 pelvis_idx: int,
                 neck_idx: int,
                 lhip_idx: int,
                 rhip_idx: int,
                 crop_size: tuple,
                 training_pose_lifting_net_without_graphs: bool,
                 inp_lifting_net_is_images: bool,
                 consider_unused_annotated_as_unannotated=False,
                 every_nth_frame=None,
                 every_nth_frame_train_annotated=None,
                 every_nth_frame_train_unannotated=None):
        """
        A Base Class for Every Dataset in order to train the 2D Pose Estimator.
        :param dataset_folder: The Dataset Folder.
        :param phase: The current phase of learning, i.e. train or test/validation.
        :param num_joints: The number of joints present in each pose.
        :param overfit: If True, we will be training only on one sample of dataset to overfit on it.
        :param only_annotations: If True, we will only be using the annotated samples.
        :param ten_percent_3d_from_all: If True, we will be sampling 10% for the entire dataset.
        :param random_seed_for_ten_percent_3d_from_all: The random seed for sampling the 10% of the dataset.
        :param augmentations: If True, we will be using augmentations on the images.
        :param resnet152_backbone: If True, we are running the experiments using the preprocessing of the learnable triangulation paper.
        :param get_2d_gt_from_3d: If True, we are obtaining the GT 2D by projecting the GT 3D on each image; otherwise the 2D manual annotations will be used.
        :param minimum_views_needed_for_triangulation: The minimum number of cameras needed for triangulation.
        :param consider_unused_annotated_as_unannotated: If True, the samples unused in the annotated set will be considered as unannotated. This is pure bullshit.
        :param every_nth_frame: The Sampling Rate for the test/validation images.
        :param every_nth_frame_train_annotated: The Sampling Rate for Annotated Training Samples.
        :param every_nth_frame_train_unannotated: The Sampling Rate for UnAnnotated Training Samples.
        """
        super(Dataset_base, self).__init__()
        self.dataset_folder        = dataset_folder
        self.phase                 = phase.lower()
        self.num_joints            = num_joints
        self.p                     = 0.2
        self.margin                = 20
        self.resnet152_backbone    = resnet152_backbone
        self.from_3d               = get_2d_gt_from_3d
        self.overfit               = overfit
        self.only_annotations      = only_annotations
        self.augmentations         = augmentations
        self.randomize             = randomize
        self.labels                = []
        self.short                 = 512
        self.joints_order          = joints_order
        self.pose_model_type       = pose_model_type
        self.pelvis_idx            = pelvis_idx
        self.neck_idx              = neck_idx
        self.lhip_idx              = lhip_idx
        self.rhip_idx              = rhip_idx
        self.crop_size             = crop_size
        self.input_size_alphapose  = input_size_alphapose
        self.inp_lifting_net_is_images                 = inp_lifting_net_is_images
        self.training_pose_lifting_net_without_graphs  = training_pose_lifting_net_without_graphs
        self.every_nth_frame                           = every_nth_frame
        self.every_nth_frame_train_annotated           = every_nth_frame_train_annotated
        self.every_nth_frame_train_unannotated         = every_nth_frame_train_unannotated
        self.ten_percent_3d_from_all                   = ten_percent_3d_from_all
        self.random_seed_for_ten_percent_3d_from_all   = random_seed_for_ten_percent_3d_from_all
        self.consider_unused_annotated_as_unannotated  = consider_unused_annotated_as_unannotated
        self.minimum_views_needed_for_triangulation    = minimum_views_needed_for_triangulation
        self.inputResH, self.inputResW, self.scaleRate = 320, 256, 0.1
        self.augmentations_transforms                  = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.07)
        if phase.lower() in ['training', 'train']:
            # self.augmentations_transforms = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.07)
            print("Obtaining the Transformations for the Train Phase for Lifting network only.")
            self.transforms_lifting_net   = transforms.Compose([transforms.Resize(size=self.crop_size),
                                                                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.07),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                     std=[0.229, 0.224, 0.225]),
                                                                ])
        else:
            # self.augmentations_transforms = None
            self.transforms_lifting_net   = transforms.Compose([transforms.Resize(size=self.crop_size),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                     std=[0.229, 0.224, 0.225])
                                                                ])
        keys_to_extend_dim = ['inp_pose_model', 'bboxes', 'pose_2d_tar_norm', 'target_pose_2d',
                              'target_pose_3d', 'labels', 'consider', 'R', 't', 'K', 'dist', 'pelvis_cam_z',
                              'target_root_rel_depth', 'target_pose_3d_camera_coordinate']

        if self.inp_lifting_net_is_images:
            keys_to_extend_dim.append('inp_images_lifting_net')
        self.keys_to_extend_dim = keys_to_extend_dim

    def function_to_obtain_lifting_net_input(self, image_file_path, bbox):
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        pil_image              = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
        pil_image              = pil_image[ymin:ymax, xmin:xmax]
        pil_image              = pil_image[ymin:ymax, xmin:xmax]
        if pil_image.size != 0:
            pil_image             = im_to_torch(pil_image)
            image                 = self.transforms_lifting_net(pil_image)
        else:
            image = torch.zeros(3, self.crop_size[0], self.crop_size[1])
        return image

    def return_labels(self):
        """
        Function to return the labels of every sample.
        """
        return self.labels

    @staticmethod
    def get_norm_2d_pose(pose_2d, bbox: list):
        """
        Function to normalize the 2D pose in image coordinates to [-1, 1] using BBoxes.
        :param pose_2d: The 2D pose in Image Coordinates.
        :param bbox: The Bounding Boxes.
        :return: The 2D pose in Normalized Coordinates.
        """
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        crop_width   = (xmax - xmin)
        crop_height  = (ymax - ymin)
        crop_size    = np.array([crop_width, crop_height])
        crop_shift   = np.array([xmin, ymin])
        pose_2d_norm = ((pose_2d - crop_shift) / crop_size) * 2 - 1.0
        return pose_2d_norm

    def function_to_load_alphapose(self, image_file_path: str, bbox : list):
        """
        Function to provide the input images to a pose estimator alphapose model.
        :param image_file_path: A single string which denotes the path of a single image.
        :param bbox: A single list of 4 elements of type [xmin, ymin, xmax, ymax]  of a bounding box.
        :return: A single preprocessed image as an input to the pose estimator model, and the new upscaled bounding box.

        class_IDs                = [[0]]
        bboxs                    = [bbox]
        scores                   = [[1]]
        bboxs                    = mx.nd.array(bboxs)
        class_IDs                = mx.nd.array(class_IDs)
        scores                   = mx.nd.array(scores)
        _, img                   = load_test_yolo(image_file_path, short=self.short)
        pose_input, upscale_bbox = detector_to_alpha_pose(img, class_IDs, scores, bboxs)
        pose_input = pose_input.numpy()
        pose_input = pose_input.squeeze()
        return pose_input, upscale_bbox # Only for one image.
        """
        orig_img_k = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
        pose_model_inp, \
            bbox_new = test_transform(orig_img_k, bbox, input_size=self.input_size_alphapose)
        return pose_model_inp, bbox

    def function_to_load_crowdpose(self, bbox : list, image_file_path: str):
        """
        Function to provide the input images to a CrowdPose Pose Estimator Model.
        :param bbox: A single list of 4 elements of type [xmin, ymin, xmax, ymax] of a bounding box.
        :param image_file_path: A single string which denotes the path of a single image.
        :return: A single preprocessed image as an input to the pose estimator model, and the bounding box.
        """
        orig_image_cv2 = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
        bbox_new       = bbox
        bbox           = torch.Tensor([bbox])
        img            = im_to_torch(orig_image_cv2)
        if self.augmentations and self.augmentations_transforms is not None:
            img = self.augmentations_transforms(img)

        inps       = torch.zeros(bbox.size(0), 3, self.inputResH, self.inputResW)
        pt1        = torch.zeros(bbox.size(0), 2)
        pt2        = torch.zeros(bbox.size(0), 2)
        X, _, _, _ = crop_from_dets(img, bbox, inps, pt1, pt2, self.inputResH, self.inputResW, scaleRate=self.scaleRate)
        X          = X.squeeze(dim=0)
        return X, bbox_new

    @staticmethod
    def function_to_load_resnet152(bbox : list, image_file_path: str):
        """
        Function to provide the input images to Pose Estimator Model of Learnable Triangulation Paper.
        :param bbox: A single list of 4 elements of type [xmin, ymin, xmax, ymax] of a bounding box.
        :param image_file_path: A single string which denotes the path of a single image.
        :return: A single preprocessed image as an input to the pose estimator model, and the new squared bounding box.
        """
        bbox_squared   = square_the_bbox(bbox=bbox)
        bbox_new       = scale_bbox_resnet152(bbox=bbox_squared, scale=1.0)
        image          = cv2.imread(image_file_path) # orig_image_cv2 = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        image          = crop_image_resnet152(image, bbox_new)
        image          = resize_image_resnet152(image, image_shape_resnet152)
        image          = normalize_image_resnet152(image)
        image          = np.expand_dims(image, axis=0)
        image          = np.transpose(image, (0, 3, 1, 2))
        image          = torch.from_numpy(image).float()
        image          = image.squeeze(dim=0)
        return image, bbox_new

    def function_to_obtain_pose_input(self, bbox : list, image_file_path: str):
        """
        Function to obtain the input images to Pose Estimator Model along with its bounding box.
        :param bbox: A single list of 4 elements of type [xmin, ymin, xmax, ymax] of a bounding box.
        :param image_file_path: A single string which denotes the path of a single image.
        :return: A single preprocessed image as an input to the pose estimator model, and the new squared bounding box.
        """
        if self.pose_model_type == 1: # For Alphapose
            pose_input, bbox_new = self.function_to_load_alphapose(bbox=bbox, image_file_path=image_file_path)
        elif self.pose_model_type == 2: # For CrowdPose
            pose_input, bbox_new = self.function_to_load_crowdpose(bbox=bbox, image_file_path=image_file_path)
        elif self.pose_model_type == 3: # For ResNet152
            pose_input, bbox_new = self.function_to_load_resnet152(bbox=bbox, image_file_path=image_file_path)
        else:
            raise NotImplementedError
        return pose_input, bbox_new


def get_sampler(labels, batch_size, randomize, num_anno_samples_per_batch):
    """
    Function to Obtain a Sampler containing a fixed number of annotated samples in a mini-batch of size <batch_size>.
    :param labels:     The Labels of every sample.
    :param batch_size: The batch size of every mini-batch.
    :param randomize:  If True, we will randomize the samples within each mini-batch. #TODO
    :param num_anno_samples_per_batch: The number of annotated samples per mini batch.
    :return: A Sampler that return a mini batch containing <num_anno_samples_per_batch> of annotated samples in mini-batch size of <batch_size>.
    """
    assert (0 < num_anno_samples_per_batch <= batch_size)
    return Batch_sampler(batch_size=batch_size, randomize=randomize, labels=labels, num_anno_samples_per_batch=num_anno_samples_per_batch)


class Batch_sampler_last(sampler.Sampler):
    def __init__(self, num_samples, batch_size, randomize):
        super(Batch_sampler_last, self).__init__(data_source=num_samples)
        self.batch_size         = batch_size
        self.randomize          = randomize
        self.num_samples        = num_samples # num_samples  = len(self.labels) # self.labels = labels
        sample_indexes_original = list(np.arange(0, self.num_samples))
        print("The Total Number of Original Samples are {}".format(self.num_samples))
        print("The Batch Size is set to {}.".format(self.batch_size))
        print("Therefore, originally we will have {} complete batches".format(num_samples // self.batch_size), end='')
        num_samples_last_batch = self.num_samples % self.batch_size
        sample_indexes_final   = sample_indexes_original.copy()
        if num_samples_last_batch == 0:
            print(".")
            final_num_batches = self.num_samples // self.batch_size
        else:
            print(" , and the last incomplete batch has {} samples.".format(num_samples_last_batch))
            final_num_batches = self.num_samples / float(self.batch_size)
            final_num_batches = int(np.ceil(final_num_batches))
            total_count       = final_num_batches * self.batch_size
            num_extra_samples_to_finish_the_batch = total_count - self.num_samples
            print("Therefore, we will need {} extra samples to finish the last batch.".format(num_extra_samples_to_finish_the_batch))
            for i in range(num_extra_samples_to_finish_the_batch):
                new_index_i = random.randint(0, self.num_samples)
                sample_indexes_final.append(new_index_i)
            num_new_samples = len(sample_indexes_final)
            print("Therefore, now we will have all complete batches with a total of {} samples.".format(num_new_samples))
            assert (num_new_samples % self.batch_size) == 0
            assert (num_new_samples // self.batch_size) == final_num_batches
            # breakpoint()
            # print(sample_indexes_original, "\n", sample_indexes_final)

        self.total_count          = final_num_batches * self.batch_size
        self.sample_indexes_final = sample_indexes_final


    def __len__(self):
        return self.total_count

    def __iter__(self):
        final_indices = self.sample_indexes_final.copy()
        if self.randomize:
            random.shuffle(final_indices)
        return iter(final_indices)


def get_sampler_train(labels, use_annotations, only_annotations, num_anno_samples_per_batch, batch_size,
                      randomize, shuffle, num_samples: int, extend_last_batch_graphs: bool):
    """
    Function to obtain the Sampler of the Training Dataset.
    :param labels                    : The Labels of every sample.
    :param use_annotations           : If True, the dataset will consist of labeled and unlabeled samples.
    :param only_annotations          : If True, we will only be considering the labeled samples.
    :param num_anno_samples_per_batch: The number of annotated samples per mini batch.
    :param batch_size                : The batch size of every mini-batch.
    :param randomize                 : If True, we will randomize the samples within each mini-batch of the Sampler. #TODO.
    :param shuffle                   : If True, we will shuffle the mini-batches inside every iteration.
    :param extend_last_batch_graphs  : If True, we will extend the last batch for the graphs, very important for using the Mask Ensemble using.
    :param num_samples               : The number of samples.
    :return: Sampler and Shuffle of the Train DataLoader.

    """
    if extend_last_batch_graphs:
        print("E" * 100)
        sampler_train = Batch_sampler_last(num_samples=num_samples, randomize=randomize, batch_size=batch_size)
        shuffle_train = False
    else:
        if use_annotations:
            if only_annotations:
                print("B" * 100)
                print(" ONly Using Annotations for Warm Startup.")
                sampler_train = None
                shuffle_train = shuffle
            else:
                if num_anno_samples_per_batch > 0:
                    print("A" * 100)
                    print("Every {} samples in a mini batch of size {} will be annotated.".format(num_anno_samples_per_batch, batch_size))
                    sampler_train = get_sampler(batch_size=batch_size, randomize=randomize, labels=labels,
                                                num_anno_samples_per_batch=num_anno_samples_per_batch)
                    shuffle_train = False
                else:
                    print("D" * 100)
                    sampler_train = None
                    shuffle_train = shuffle
        else:
            print("C" * 100)
            sampler_train = None
            shuffle_train = shuffle
    return sampler_train, shuffle_train


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


class collate_func_pose_net_triangulation(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            batch = [{'None': []}]
        return default_collate(batch)


class collate_with_graphs(object):
    def __init__(self, keys_to_consider, graph_key):
        self.keys_to_consider    = keys_to_consider
        self.graph_key           = graph_key
        self.new_batch_dicts     = {key: [] for key in self.keys_to_consider}
        self.key_no_modification = ['image_paths', 'camera_ids', 'frame_ids', 'action_ids', 'subject_ids', graph_key]

    def __call__(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            batch = [{'None': []}]

        samples, graphs = map(list, zip(*batch))
        new_graphs      = []
        for graph in graphs:
            for item in graph:
                new_graphs.append(item)
        final_samples                 = default_collate(samples)
        final_samples['batch_graphs'] = dgl.batch(new_graphs)
        return final_samples


def obtain_windows(subject_ids, actions_to_consider, samples_data_arrangement, labels_data_arrangement,
                   half_window_size, delta_t_0, extend_last, sampling_rate, time_pred, annotated_subjects: list):
    subj_dict                = {}
    all_windows              = []
    all_label_windows        = []
    label_windows_pred_frame = []# This is the label of the predicted frame in a given window.
    for i, subject_id in enumerate(subject_ids):
        subj_dict[subject_id] = {}
        for act in actions_to_consider:
            label_subject_act          = labels_data_arrangement[subject_id][act]
            frames_subject_act         = samples_data_arrangement[subject_id][act]
            min_n                      = len(frames_subject_act)
            window_i_act               = get_windows(min_n, half_window_size, delta_t_0, extend_last, sampling_rate)
            subj_dict[subject_id][act] = (min_n, len(window_i_act), window_i_act)
            for win in window_i_act:
                if len(win) > 0:
                    frames_win         = [frames_subject_act[w] for w in win]
                    labels_win         = [label_subject_act[w] for w in win]
                    all_windows       += [(frames_win, subject_id, act)]
                    all_label_windows += [labels_win]  # all_label_windows.append(label_subject_act[win[time_pred]])
                    # label_windows_pred_frame.append(1 if subject_id in annotated_subjects else 0)
                    label_windows_pred_frame.append(labels_win[time_pred])
    assert len(all_windows) == len(all_label_windows) == len(label_windows_pred_frame)
    return all_windows, all_label_windows, label_windows_pred_frame


def get_samples_labels_arrangements(actions_to_consider, subjects_to_consider, samples, labels):
    samples_data_arrangement = {subject: {action: [] for action in actions_to_consider} for subject in subjects_to_consider}
    labels_data_arrangement  = {subject: {action: [] for action in actions_to_consider} for subject in subjects_to_consider}
    for sample, label_sample in zip(samples, labels):
        subject_sample = sample[0]
        action_sample  = sample[1]
        frame_idx      = sample[2]
        if action_sample not in actions_to_consider:
            continue
        samples_data_arrangement[subject_sample][action_sample].append(frame_idx)
        labels_data_arrangement[subject_sample][action_sample].append(label_sample)
    return samples_data_arrangement, labels_data_arrangement
