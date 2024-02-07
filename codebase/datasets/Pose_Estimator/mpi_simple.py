import os
import numpy as np
import cv2
import imageio
import itertools
import multiprocessing
import warnings
import h5py

from scipy                 import io
from operator              import itemgetter
from abc                   import abstractmethod
from utils                 import (json_read, print_num_samples, bbox_from_points, compute_intersection)
from datasets.dataset_base import Dataset_base, get_sampler_train, collate_func_pose_net_triangulation
from torch.utils.data      import DataLoader
from datasets.mpi_basics   import (intrinsics_test_sequences, parse_camera_calibration, training_subjects_all,
                                   joint_idxs_28to17ours_learnable_h36m_train, joint_idxs_28to17_original_train,
                                   joint_idxs_17to17ours_learnable_h36m_test, joint_idxs_17to17_according_to_h36m_test,
                                   joint_idxs_17to17_original_test, joint_names_all_17_according_to_h36m,
                                   joint_names_all_17_original, joint_names_17_learnable, joint_idxs_28to17_according_to_h36m_train,
                                   function_to_calculate_new_intrinsics)


def get_files_in(folder):
    """

    :param folder:
    :return:
    """
    return [name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]


def _prepare_poses_train_parallel(base, subject, seq, calibrations):
    """
    :param base:
    :param subject:
    :param seq:
    :param calibrations:
    :return:
    """
    mat               = io.loadmat(os.path.join(base, subject, seq, 'annot.mat'))
    arr               = np.stack(mat['annot2'].flatten())
    poses_2d          = arr.reshape((arr.shape[0], arr.shape[1], 28, 2))
    data2             = {'idx_frames': list(range(poses_2d.shape[1]))}
    data2['poses_2d'] = {}
    for cam_id, poses in enumerate(poses_2d):
        data2['poses_2d'][cam_id] = poses
    arr      = np.stack(mat['annot3'].flatten())
    poses_3d = arr.reshape((arr.shape[0], arr.shape[1], 28, 3))
    poses_3d = poses_3d/1000
    # since the 3D poses are in camera coordinate, we simply take the poses
    # of the first camera and convert them to world.
    cam_id   = 0
    with open(os.path.join(base, subject, seq, "camera.calibration"), "r") as f:
        calibration = parse_camera_calibration(f, y_z_swapped=False)[cam_id]
    poses_3d_cam0 = poses_3d[cam_id]
    R             = np.array(calibration['R'])
    t             = np.array(calibration['t'])
    poses_world   = np.tensordot(R.T, poses_3d_cam0.T-t.ravel()[:,None,None], axes=([1],[0])).T
    # The height in the original data is the y-axis. We change it so that the z-axis is the height.
    # To do so we apply a 90° rotation around the x-axis.
    # We also change the rotation and translation matrices in the function parse_camera_calibration
    data2['poses_3d'] = poses_world[:, : , [0, 2, 1]]*np.array([1, -1, 1])[None,None]
    return data2



class MPI_INF_3DHP_base(Dataset_base):
    def __init__(self,
                 phase,
                 dataset_folder,
                 num_joints,
                 overfit,
                 randomize,
                 only_annotations,
                 ten_percent_3d_from_all,
                 random_seed_for_ten_percent_3d_from_all,
                 augmentations,
                 resnet152_backbone,
                 get_2d_gt_from_3d,
                 joints_order,
                 minimum_views_needed_for_triangulation,
                 consider_unused_annotated_as_unannotated,
                 every_nth_frame,
                 pose_model_type,
                 every_nth_frame_train_annotated,
                 every_nth_frame_train_unannotated,
                 input_size_alphapose,
                 training_pose_lifting_net_without_graphs,
                 pelvis_idx: int,
                 neck_idx: int,
                 lhip_idx: int,
                 rhip_idx: int,
                 crop_size: tuple,
                 inp_lifting_net_is_images: bool,
                 ):
        """

        :param phase:
        :param dataset_folder:
        :param num_joints:
        :param overfit:
        :param randomize:
        :param only_annotations:
        :param ten_percent_3d_from_all:
        :param random_seed_for_ten_percent_3d_from_all:
        :param augmentations:
        :param resnet152_backbone:
        :param get_2d_gt_from_3d:
        :param joints_order:
        :param minimum_views_needed_for_triangulation:
        :param consider_unused_annotated_as_unannotated:
        :param every_nth_frame:
        :param pose_model_type:
        :param every_nth_frame_train_annotated:
        :param every_nth_frame_train_unannotated:
        """
        self.samples                 = []
        self.subjects                = []
        self.sequences               = []
        self.camera_ids              = []
        self.multiview_sample        = True
        self.views                   = []
        self.calibrations            = {}
        self.data                    = {}
        self.joint_indexes_to_return = []

        super(MPI_INF_3DHP_base, self).__init__(dataset_folder=dataset_folder, phase=phase, num_joints=num_joints,
                                                overfit=overfit, only_annotations=only_annotations,
                                                ten_percent_3d_from_all=ten_percent_3d_from_all,
                                                random_seed_for_ten_percent_3d_from_all=random_seed_for_ten_percent_3d_from_all,
                                                augmentations=augmentations, resnet152_backbone=resnet152_backbone,
                                                get_2d_gt_from_3d=get_2d_gt_from_3d, joints_order=joints_order,
                                                consider_unused_annotated_as_unannotated=consider_unused_annotated_as_unannotated,
                                                every_nth_frame=every_nth_frame, randomize=randomize,
                                                every_nth_frame_train_annotated=every_nth_frame_train_annotated,
                                                every_nth_frame_train_unannotated=every_nth_frame_train_unannotated,
                                                minimum_views_needed_for_triangulation=minimum_views_needed_for_triangulation,
                                                pose_model_type=pose_model_type, crop_size=crop_size,
                                                input_size_alphapose=input_size_alphapose,
                                                training_pose_lifting_net_without_graphs=training_pose_lifting_net_without_graphs,
                                                pelvis_idx=pelvis_idx, rhip_idx=rhip_idx, lhip_idx=lhip_idx, neck_idx=neck_idx,
                                                inp_lifting_net_is_images=inp_lifting_net_is_images)


    def __len__(self):
        """
        :return:
        """
        return len(self.samples)

    @abstractmethod
    def get_filename_image(self, subject, seq, cam_id, idx):
        """
        :param subject:
        :param seq:
        :param cam_id:
        :param idx:
        :return:
        """
        # idx: It is the frame idx.
        pass

    def get_image(self, subject, seq, cam_id, idx):
        """
        :param subject:
        :param seq:
        :param cam_id:
        :param idx:
        :return:
        """
        return imageio.imread(self.get_filename_image(subject, seq, cam_id, idx))

    def get_calibration(self, subject, seq, cam_id, rot_vector=True):
        """
        :param subject:
        :param seq:
        :param cam_id:
        :param rot_vector:
        :return:
        """
        calibration = self.calibrations[subject][seq][cam_id]
        r           = np.array(calibration['R'])
        if rot_vector:
            r = cv2.Rodrigues(r)[0]
        t    = np.array(calibration['t']).ravel()
        K    = np.array(calibration['K'])
        dist = np.array(calibration['dist'])
        return r, t, K, dist, (int(calibration['height']), int(calibration['width']))

    @staticmethod
    def get_ground():
        return [[-2, -2, 0], [-2, 2, 0], [2, 2, 0], [2, -2, 0]]

    def get_pose_3d(self, subject, seq, frame_idx):
        """
        :param subject:
        :param seq:
        :param frame_idx:
        :return:
        """
        indexes = self.data[subject][seq]['idx_frames']
        if frame_idx not in indexes:
            return None
        pose_3d = self.data[subject][seq]['poses_3d'][indexes.index(frame_idx)]
        pose_3d = pose_3d[self.joint_indexes_to_return]
        return pose_3d

    def get_all_poses_3d(self):
        """
        :return:
        """
        poses = list(itertools.chain(*[x['poses_3d'][:, self.joint_indexes_to_return].tolist()
                                       for subject in self.subjects for x in self.data[subject].values()]))
        return np.float32(poses)

    def get_pose_2d(self, subject, seq, cam_id, frame_idx, from_3d=False):
        """
        :param subject:
        :param seq:
        :param cam_id:
        :param frame_idx:
        :param from_3d:
        :return:
        """
        if from_3d:
            pose_3d = self.get_pose_3d(subject, seq, frame_idx)
            if pose_3d is None:
                return None
            rvec, t, K, dist, _ = self.get_calibration(subject=subject, seq=seq, cam_id=cam_id, rot_vector=True)
            pose_2d             = cv2.projectPoints(pose_3d, rvec, t, K, dist)[0].reshape(-1, 2)
        else:
            indexes = self.data[subject][seq]['idx_frames']
            if frame_idx not in indexes:
                return None
            pose_2d = self.data[subject][seq]['poses_2d'][cam_id][indexes.index(frame_idx)]
            pose_2d = pose_2d[self.joint_indexes_to_return]
        return pose_2d

    def get_position(self, subject, seq, frame_idx):
        """
        :param subject:
        :param seq:
        :param frame_idx:
        :return:
        """
        pose_3d = self.get_pose_3d(subject, seq, frame_idx)
        if pose_3d is None:
            return None
        return np.mean(pose_3d, axis=0)[:2]

    def get_bbox(self, subject, seq, cam_id, idx, p=0.3):
        """
        :param subject:
        :param seq:
        :param cam_id:
        :param idx:
        :param p:
        :return:
        """
        pose_2d = self.get_pose_2d(subject, seq, cam_id, idx)
        if pose_2d is None:
            return None
        bbox = bbox_from_points(pose_2d, pb=p)
        return bbox

    def get_calibration_by_index(self, index):
        """
        :param index:
        :return:
        """
        if self.multiview_sample:
            subject_idx, seq_idx, frame_idx = self.samples[index] # subject_idx, seq_idx, _, frame_idx = self.samples[index]
            views                           = self.views
        else:
            subject_idx, seq_idx, cam_idx, frame_idx = self.samples[index]
            views = [cam_idx]

        calibration = {}
        for view in views:
            R, t, K, dist, _  = self.get_calibration(subject=subject_idx, seq=seq_idx, cam_id=view, rot_vector=False)
            calibration[view] = {'R' : R, 't' : t, 'K' : K, 'dist': dist}
        return calibration

    def get_calibration_per_view(self, view, rot_vector, index):
        """
        :param view:
        :param rot_vector:
        :param index:
        :return:
        """
        if self.multiview_sample:
            subject_idx, seq_idx, frame_idx = self.samples[index]
            cam_idx                         = view
        else:
            subject_idx, seq_idx, cam_idx, frame_idx = self.samples[index]
            #
        R, t, K, dist, _  = self.get_calibration(subject=subject_idx, seq=seq_idx, cam_id=cam_idx, rot_vector=rot_vector)
        return R, t, K, dist

    @abstractmethod
    def _prepare_data(self):
        pass

    def get_shapes(self):
        """
        :return:
        """
        shape_val = {}
        for cam_id in self.camera_ids:
            shape_val[cam_id] = (2048, 2048)
        return shape_val

    def visibility(self, subject, seq, view, idx):
        """
        :param subject:
        :param seq:
        :param view:
        :param idx:
        :return:
        """
        pose                = self.get_pose_2d(subject, seq, view, idx, from_3d=True)
        bbox_for_visibility = bbox_from_points(pose, pb=0.05)
        # # subject, seq, cam_id
        _, _, _, _, shape   = self.get_calibration(subject=subject, seq=seq, cam_id=view, rot_vector=False)
        bbox_img            = [0, 0, shape[1], shape[0]]
        return float(compute_intersection(bbox_for_visibility, [bbox_img])[0])

    def sample(self, index):
        """
        :param index:
        :return:
        """
        output          = {}
        label           = self.labels[index]
        output['label'] = label
        if self.multiview_sample:
            subject_idx, seq_idx, frame_idx = self.samples[index]
            views                           = self.views
        else:
            subject_idx, seq_idx, cam_idx, frame_idx = self.samples[index]
            views                                    = [cam_idx]
        """
        target_pose_3d = self.get_pose_3d(subject_idx, seq_idx, frame_idx)
        if target_pose_3d is not None:
            target_pose_3d = np.float32(target_pose_3d)
            mask_valid     = ~np.isnan(target_pose_3d[:, 0])
        else:
            mask_valid     = None
        bboxes = []; inp_pose_model = []; consider = []; R = []; t = []; K = []; dist = []; target_pose_2d = []
        pose_2d_tar_norm = []; image_paths = []; camera_ids = []; action = ''; inp_images_lifting_net = []
        num_cameras      = len(views)
        for i in range(num_cameras):
            cam_id                    = views[i]
            img_file_path             = self.get_filename_image(subject_idx, seq_idx, cam_id, frame_idx)
            R_cam, t_cam, K_cam, \
                 dist_cam, shape_view = self.get_calibration(subject_idx, seq_idx, cam_id, rot_vector=False)
            pose_2d_vid = self.get_pose_2d(subject_idx, seq_idx, cam_id, frame_idx, from_3d=self.from_3d)
            bbox        = bbox_from_points(pose_2d_vid, pb=self.p)

            xmin, ymin, xmax, ymax = bbox
            xmin = max(xmin - self.margin, 0)
            ymin = max(ymin - self.margin, 0)
            xmax = min(xmax + self.margin, shape_view[1])
            ymax = min(ymax + self.margin, shape_view[0])

            bbox_old               = [int(xmin), int(ymin), int(xmax), int(ymax)]
            pose_input, bbox_new   = self.function_to_obtain_pose_input(image_file_path=img_file_path, bbox=bbox_old)  # It is just a single string.
            xmin, ymin, xmax, ymax = int(bbox_new[0]), int(bbox_new[1]), int(bbox_new[2]), int(bbox_new[3])
            if self.inp_lifting_net_is_images:
                lifting_net_input_image = self.function_to_obtain_lifting_net_input(image_file_path=img_file_path, bbox=bbox_new)
                inp_images_lifting_net.append(lifting_net_input_image)

            area_A          = (xmax - xmin) * (ymax - ymin)
            recB            = [0, 0, shape_view[1], shape_view[0]]
            if area_A == 0.0:
                consider.append(0)
            else:
                area_overlapped = area(boxA=bbox_new, boxB=recB) / float(area_A)
                if area_overlapped < 0.7:
                    consider.append(0)
                else:
                    consider.append(1)

            pose_2d_vid_norm = self.get_norm_2d_pose(pose_2d=pose_2d_vid, bbox=bbox_new)
            camera_ids.append('ace_{}'.format(cam_id))
            bboxes.append(bbox_new)
            inp_pose_model.append(pose_input)
            target_pose_2d.append(pose_2d_vid)
            pose_2d_tar_norm.append(pose_2d_vid_norm)
            image_paths.append(img_file_path)
            R.append(R_cam)
            t.append(t_cam)
            K.append(K_cam)
            dist.append(dist_cam)

        if sum(consider) >= self.minimum_views_needed_for_triangulation:
            consider_for_triangulation = 1
            pose_2d_tar_norm           = torch.from_numpy(np.stack(pose_2d_tar_norm))
            inp_pose_model             = torch.from_numpy(np.stack(inp_pose_model))
            bboxes                     = torch.from_numpy(np.stack(bboxes))
            target_pose_2d             = torch.from_numpy(np.stack(target_pose_2d))
            R                          = torch.from_numpy(np.stack(R))
            t                          = torch.from_numpy(np.stack(t))
            K                          = torch.from_numpy(np.stack(K))
            dist                       = torch.from_numpy(np.stack(dist))
            target_pose_3d             = torch.from_numpy(target_pose_3d)
            consider                   = torch.Tensor(consider)
            label                      = torch.Tensor(label)
            pelvis_cam_z               = get_pelvis(pose=target_pose_3d, lhip_idx=self.lhip_idx, rhip_idx=self.rhip_idx,
                                                    pelvis_idx=self.pelvis_idx, return_z=True)
            pelvis_cam_z               = pelvis_cam_z.view(-1).repeat(num_cameras)
            label                      = label.repeat(num_cameras)
            target_pose_3d             = target_pose_3d.unsqueeze(dim=0)
            target_pose_3d             = target_pose_3d.repeat(num_cameras, 1, 1)

            retval = {'inp_pose_model' : inp_pose_model, 'bboxes' : bboxes, 'target_pose_2d' : target_pose_2d,
                      'R' : R, 't' : t, 'K' : K, 'dist' : dist, 'target_pose_3d' : target_pose_3d,
                      'image_paths' : image_paths, 'camera_ids': camera_ids,  'frame_ids': int(frame_idx),
                      'action_ids' : seq_idx, 'subject_ids' : subject_idx, 'consider' : consider,
                      'pose_2d_tar_norm' : pose_2d_tar_norm, 'mask_valid' : mask_valid,
                      'consider_for_triangulation': consider_for_triangulation, 'labels' : label,
                      'pelvis_cam_z' : pelvis_cam_z}
            if self.inp_lifting_net_is_images:
                inp_images_lifting_net           = torch.from_numpy(np.stack(inp_images_lifting_net))
                retval['inp_images_lifting_net'] = inp_images_lifting_net

            if self.training_pose_lifting_net_without_graphs:
                retval = function_to_extend_dim(data=retval, which_dim=0, which_keys=self.keys_to_extend_dim)
        else:
            retval = None
        """
        retval = self.obtain_mv_samples(subject_idx=subject_idx, seq_idx=seq_idx,
                                        frame_idx=frame_idx, cameras=views, label=label)
        return retval

    def get_samples(self):
        return self.samples

    def get_labels(self):
        return self.labels

    def obtain_subjects(self):
        return self.subjects

    def obtain_actions(self):
        return self.sequences  # TODO- THis is not Action.

    def obtain_cameras(self):
        return self.camera_ids

    def get_pose_3d_common(self, subject_idx, seq_idx, frame_idx):
        """
        :param subject_idx: It is the id of the Subject.
        :param seq_idx: It is the id of the Sequence.
        :param frame_idx: This is the id of the frame.
        :return:
        """
        pose_3d = self.get_pose_3d(subject=subject_idx, seq=seq_idx, frame_idx=frame_idx)
        return pose_3d

    def get_filename_image_common(self, subject_idx, seq_idx, cam_idx, frame_idx):
        """
        :param subject_idx: It is the id of the Subject.
        :param seq_idx: It is the id of the Sequence.
        :param frame_idx: This is the id of the frame.
        :param cam_idx: This is the id of the camera.
        :return:
        """
        file_path = self.get_filename_image(subject=subject_idx, seq=seq_idx, cam_id=cam_idx, idx=frame_idx)
        return file_path


    def get_pose_2d_common(self, subject_idx, seq_idx, cam_idx, frame_idx, from_3d):
        """
        :param subject_idx: It is the id of the Subject.
        :param seq_idx: It is the id of the Sequence.
        :param frame_idx: This is the id of the frame.
        :param cam_idx: This is the id of the camera.
        :param from_3d: If True, we will calculate the 2D pose by projecting the 3D pose, or else we will use the manually detected 2D pose.
        :return:
        """
        pose_2d_cam = self.get_pose_2d(subject=subject_idx, seq=seq_idx, cam_id=cam_idx, frame_idx=frame_idx, from_3d=from_3d)
        return pose_2d_cam

    def get_calibration_common(self, subject_idx, seq_idx, cam_idx, frame_idx, rot_vector=True):
        """
        :param subject_idx: It is the id of the Subject.
        :param seq_idx: It is the id of the Sequence.
        :param frame_idx: This is the id of the frame.
        :param cam_idx: This is the id of the camera.
        :param rot_vector: If True, we will perform cv2.Rodrigues on the Rotation Matrix.
        :return:
        """
        R, t, K, dist, shape = self.get_calibration(subject=subject_idx, seq=seq_idx, cam_id=cam_idx, rot_vector=rot_vector)
        return R, t, K, dist, shape


class MPI_INF_3DHP_train(MPI_INF_3DHP_base):
    def __init__(self,
                 base,
                 base_images,
                 subjects,
                 only_chest_height_cameras,
                 multiview_sample,
                 min_visibility,
                 joint_indexes_to_return,
                 root_index,
                 load_from_cache,
                 path_cache,
                 every_nth_frame_annotated,
                 every_nth_frame_unannotated,
                 use_annotations_only,
                 resnet152_backbone,
                 ten_percent_3d_from_all,
                 num_joints,
                 overfit,
                 use_annotations,
                 randomize,
                 augmentations,
                 get_2d_gt_from_3d,
                 random_seed_for_ten_percent_3d_from_all,
                 minimum_views_needed_for_triangulation,
                 consider_unused_annotated_as_unannotated,
                 pose_model_type,
                 ignore_sub8_seq2,
                 just_sub8_seq2,
                 input_size_alphapose,
                 training_pose_lifting_net_without_graphs,
                 pelvis_idx: int,
                 neck_idx: int,
                 lhip_idx: int,
                 rhip_idx: int,
                 crop_size: tuple,
                 inp_lifting_net_is_images : bool,
                 calculate_K : bool,
                 every_nth_frame=None,
                 annotated_subjects=None):
        """

        :param base:
        :param base_images:
        :param subjects:
        :param only_chest_height_cameras:
        :param multiview_sample:
        :param min_visibility:
        :param joint_indexes_to_return:
        :param root_index:
        :param load_from_cache:
        :param path_cache:
        :param every_nth_frame_annotated:
        :param every_nth_frame_unannotated:
        :param use_annotations_only:
        :param resnet152_backbone:
        :param ten_percent_3d_from_all:
        :param num_joints:
        :param overfit:
        :param use_annotations:
        :param randomize:
        :param augmentations:
        :param get_2d_gt_from_3d:
        :param random_seed_for_ten_percent_3d_from_all:
        :param minimum_views_needed_for_triangulation:
        :param consider_unused_annotated_as_unannotated:
        :param pose_model_type:
        :param ignore_sub8_seq2:
        :param just_sub8_seq2:
        :param every_nth_frame:
        :param annotated_subjects:
        """

        super(MPI_INF_3DHP_train, self).__init__(phase='train', dataset_folder=base, num_joints=num_joints,
                                                 overfit=overfit, randomize=randomize, only_annotations=use_annotations_only,
                                                 ten_percent_3d_from_all=ten_percent_3d_from_all,
                                                 random_seed_for_ten_percent_3d_from_all=random_seed_for_ten_percent_3d_from_all,
                                                 augmentations=augmentations, resnet152_backbone=resnet152_backbone,
                                                 get_2d_gt_from_3d=get_2d_gt_from_3d, joints_order=joint_indexes_to_return,
                                                 minimum_views_needed_for_triangulation=minimum_views_needed_for_triangulation,
                                                 consider_unused_annotated_as_unannotated=consider_unused_annotated_as_unannotated,
                                                 every_nth_frame=every_nth_frame, every_nth_frame_train_annotated=every_nth_frame_annotated,
                                                 every_nth_frame_train_unannotated=every_nth_frame_unannotated,
                                                 pose_model_type=pose_model_type, crop_size=crop_size,
                                                 input_size_alphapose=input_size_alphapose,
                                                 training_pose_lifting_net_without_graphs=training_pose_lifting_net_without_graphs,
                                                 pelvis_idx=pelvis_idx, rhip_idx=rhip_idx, lhip_idx=lhip_idx, neck_idx=neck_idx,
                                                 inp_lifting_net_is_images=inp_lifting_net_is_images)
        self.base               = base  # "/cvlabsrc1/cvlab/MPI_INF_3D/mpi_inf_3dhp/"
        self.base_images        = base_images  # "/cvlabsrc1/cvlab/dataset_mpi_inf_3dhp/"
        self.min_visibility     = min_visibility
        self.ignore_sub8_seq2   = ignore_sub8_seq2
        self.just_sub8_seq2     = just_sub8_seq2
        self.use_annotations    = use_annotations
        self.annotated_subjects = [] if annotated_subjects is None else annotated_subjects
        if len(subjects) == 0 or subjects is None:
            self.subjects = ['S{}'.format(i) for i in range(1, 9)]
        else:
            self.subjects = subjects

        print("The following subjects are used for training {}.".format(', '.join(self.subjects)))
        if annotated_subjects is not None:
            print("The following subjects are annotated {}".format(', '.join(self.annotated_subjects)))

        self.sequences = ['Seq1', 'Seq2']
        print("The following sequences are used {}".format(', '.join(self.sequences)))

        if only_chest_height_cameras:
            self.camera_ids = [0, 2, 4, 7, 8]
        else:
            self.camera_ids = [0, 1, 2, 4, 5, 6, 7, 8]
        print("The following camera ids are used for training - {}".format(self.camera_ids))
        self.views                     = self.camera_ids
        self.multiview_sample          = multiview_sample
        self.joint_indexes_to_return   = np.int32(joint_indexes_to_return)
        self.only_chest_height_cameras = only_chest_height_cameras
        if not isinstance(root_index, (list, tuple)):
            self.root_index = [root_index]
        self.root_index   = np.int32(root_index)
        self.calibrations = {}
        self.calculate_K  = calculate_K

        for subject in self.subjects:
            self.calibrations[subject] = {}
            for seq in self.sequences:
                if self.just_sub8_seq2:
                    if subject == 'S8' and seq == 'Seq2':
                        get_cal = True
                    else:
                        get_cal = False

                elif self.ignore_sub8_seq2:
                    if subject == 'S8' and seq == 'Seq2':
                        get_cal = False
                    else:
                        get_cal = True

                else:
                    get_cal = True

                if get_cal:
                    with open(os.path.join(self.base, subject, seq, "camera.calibration"), "r") as f:
                        calibration = parse_camera_calibration(f)
                        if self.calculate_K:
                            print("Calculating the New K for {} in {} for the 2D pose Estimator in the Train Set.".format(seq, subject))
                            calibration = function_to_calculate_new_intrinsics(seq=seq, subject=subject, base=self.base, old_calibration=calibration)

                    self.calibrations[subject][seq] = calibration

        self.data               = self._prepare_data()
        filename_cached_samples = os.path.join(path_cache, "MPI_INF_3DHP_train_samples_{}.json".format(training_subjects_all))
        self.load_from_cache    = load_from_cache

        if load_from_cache and os.path.isfile(filename_cached_samples):
            print("Loading samples from the cache from {}.".format(filename_cached_samples))
            samples_loaded = json_read(filename_cached_samples)
            print("Done Loading {} samples.".format(len(samples_loaded)))
            samples = []; key_vals_loaded = []; N_samples_loaded = len(samples_loaded); counter = 0
            for sample in samples_loaded:
                subject_idx, seq_idx, cam_idx, frame_idx = sample
                if self.just_sub8_seq2:
                    if subject_idx == 'S8' and seq_idx == 'Seq2':
                        todo_1 = True
                    else:
                        todo_1 = False

                elif self.ignore_sub8_seq2:
                    if subject_idx == 'S8' and seq_idx == 'Seq2':
                        todo_1 = False
                    else:
                        todo_1 = True

                else:
                    todo_1 = True

                if todo_1:
                    key_val  = 'Sub{}-Seq{}-Frame{:05d}'.format(subject_idx, seq_idx, frame_idx)
                    counter += 1
                    if key_val not in key_vals_loaded:
                        key_vals_loaded.append(key_val)
                        if multiview_sample:
                            sample = (subject_idx, seq_idx, frame_idx)
                            samples.append(sample)
                        else:
                            for cam_idx in self.camera_ids:
                                if self.visibility(subject_idx, seq_idx, cam_idx, frame_idx) > min_visibility:
                                    samples.append((subject_idx, seq_idx, cam_idx, frame_idx))

                    if counter % 10000 == 0:
                        print("{}-{}".format(counter, N_samples_loaded))
        else:
            # """
            subjects = self.annotated_subjects + self.subjects
            subjects = list(set(subjects))
            subjects.sort()
            samples  = []
            for subject in subjects:
                for seq in self.sequences:
                    if self.just_sub8_seq2:
                        if subject == 'S8' and seq == 'Seq2':
                            todo_2 = True
                        else:
                            todo_2 = False

                    elif self.ignore_sub8_seq2:
                        if subject == 'S8' and seq == 'Seq2':
                            todo_2 = False
                        else:
                            todo_2 = True
                    else:
                        todo_2 = True

                    if todo_2:
                        print("subject - {}-{}".format(subject, seq))
                        idx_frames_ = None
                        for cam_id in self.camera_ids:
                            path_images = os.path.join(self.base_images, subject, seq, 'imageSequence', 'video_{}'.format(cam_id), 'frames')
                            idx_frames = [int(f.split('.')[0].split('_')[-1]) for f in get_files_in(path_images)]
                            if idx_frames_ is None:
                                idx_frames_ = idx_frames
                            else:
                                diff = set(idx_frames_).difference(idx_frames)
                                if len(diff) > 0:
                                    print(subject, seq, cam_id, diff)
                                    raise RuntimeError()
                        for idx in idx_frames:
                            if self.multiview_sample:
                                samples.append((subject, seq, idx))
                            else:
                                for cam_id in self.camera_ids:
                                    if self.visibility(subject, seq, cam_id, idx) > min_visibility:
                                        samples.append((subject, seq, cam_id, idx))

        print("Before Sampling Total Samples loaded are {}".format(len(samples)))
        if not self.ten_percent_3d_from_all:
            samples_annotated = []; samples_unannotated = []
            for sample in samples:
                subject_id = sample[0]
                if subject_id in self.annotated_subjects:  # labels.append(1)
                    samples_annotated.append(sample)
                elif subject_id in self.subjects:  # labels.append(0)
                    samples_unannotated.append(sample)
        else:
            N_labels = len(samples)
            print("Performing 10% of all {} Samples to be Annotated.. Rest all are UnAnnotated.".format(N_labels))
            labels = [0] * N_labels
            label  = 1
            ii     = 0
            for index in range(0, N_labels, 10):
                labels[index] = label
                ii           += 1
            assert self.every_nth_frame_train_annotated == 1
            print("After performing 10% sampling of all samples.. we have {} samples annotated and rest all are unannotated.".format(ii))
            print("----------" * 20)
            idx_labeled = np.where(np.array(labels) == 1)[0]
            print("Before Sampling -- The number of Annotated Samples are {}".format(len(idx_labeled)))
            if len(idx_labeled) > 0:
                print("Sampling Every {} frame for Labeled samples for {} mode.".format(self.every_nth_frame_train_annotated, 'TRAIN'))
                samples_annotated = list(itemgetter(*idx_labeled)(samples))
                print("After Sampling .. Total Number of Annotated Samples are {}".format(len(samples_annotated)))
            else:
                samples_annotated = []

            idx_unlabeled = np.where(np.array(labels) == 0)[0]
            print("Before Sampling -- The number of UnAnnotated Samples are {}".format(len(idx_unlabeled)))
            if len(idx_unlabeled) > 0:
                print("Sampling Every {} frame for UnLabeled samples for {} mode.".format(self.every_nth_frame_train_unannotated, 'TRAIN'))
                samples_unannotated = list(itemgetter(*idx_unlabeled)(samples))
                print("After Sampling .. Total Number of UnAnnotated Samples are {}".format(len(samples_unannotated)))
            else:
                samples_unannotated = []

        assert self.every_nth_frame_train_annotated is not None
        assert self.every_nth_frame_train_unannotated is not None
        annotated_samples   = samples_annotated[::self.every_nth_frame_train_annotated]
        unannotated_samples = samples_unannotated[::self.every_nth_frame_train_unannotated]
        labels_annotated    = [1] * len(annotated_samples)
        labels_unannotated  = [0] * len(unannotated_samples)
        if self.only_annotations:
            print("will be using annotated samples only")
            self.samples = annotated_samples
            self.labels  = labels_annotated
            print("Number of Annotated samples only is {}".format(len(annotated_samples)))
        else:
            print("will be using annotated samples and unannotated samples")
            self.samples = annotated_samples + unannotated_samples
            self.labels  = labels_annotated + labels_unannotated
            print("Number of Annotated samples is {} + UnAnnotated samples is {}.".format(len(annotated_samples), len(unannotated_samples)))

    def get_key_by_index(self, index):
        """

        :param index:
        :return:
        """
        subject_idx, seq_idx, frame_idx = self.samples[index]
        key = '{}-{}-{}'.format(frame_idx, subject_idx, seq_idx)
        return key

    def get_filename_image(self, subject, seq, cam_id, idx):
        """

        :param subject:
        :param seq:
        :param cam_id:
        :param idx:
        :return:
        """
        return os.path.join(self.base_images, subject, seq, "imageSequence", 'video_{}'.format(cam_id),
                            "frames", "frame_{:05d}.jpg".format(idx))

    def _prepare_data(self, threads=16):
        """

        :param threads:
        :return:
        """
        inputs = []
        for subject in self.subjects:
            for seq in self.sequences:
                if self.just_sub8_seq2:
                    if subject == 'S8' and seq == 'Seq2':
                        get_input = True
                        inputs.append((self.base, subject, seq, self.calibrations))
                    else:
                        get_input = False
                elif self.ignore_sub8_seq2:
                    if subject == 'S8' and seq == 'Seq2':
                        get_input = False
                    else:
                        get_input = True
                else:
                    get_input = True

                if get_input:
                    inputs.append((self.base, subject, seq, self.calibrations))

        with multiprocessing.Pool(threads) as pool:
            res = pool.starmap(_prepare_poses_train_parallel, inputs)
        assert len(inputs) == len(res)
        data = {}
        for (_, subject, seq, _), data2 in zip(inputs, res):
            if subject not in data:
                data[subject] = {}
            data[subject][seq] = data2
            assert len(data2['idx_frames']) == len(data2['poses_3d'])
            print("Loaded {} 3D poses for subject:{} seq:{}.".format(len(data2['poses_3d']), subject, seq))
        return data

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        output = self.sample(index)
        return output




class MPI_INF_3DHP_test(MPI_INF_3DHP_base):

    def get_key_by_index(self, index):
        """

        :param index:
        :return:
        """
        subject_idx, seq_idx, _, frame_idx = self.samples[index]
        key = '{}-{}-{}'.format(frame_idx, subject_idx, seq_idx)
        return key

    def get_calibration_by_index(self, index):
        """

        :param index:
        :return:
        """
        sample      = self.samples[index]
        subject     = sample[0]
        calib       = self.calibrations[subject][self.sequences][self.camera_ids[0]]
        return {'ace_0': calib}

    def __init__(self,
                 phase,
                 base,
                 pose_model_type,
                 base_images,
                 min_visibility,
                 root_index,
                 num_joints,
                 overfit,
                 augmentations,
                 resnet152_backbone,
                 get_2d_gt_from_3d,
                 joint_indexes_to_return,
                 minimum_views_needed_for_triangulation,
                 every_nth_frame,
                 input_size_alphapose,
                 pelvis_idx: int,
                 neck_idx: int,
                 lhip_idx: int,
                 rhip_idx: int,
                 crop_size: tuple,
                 inp_lifting_net_is_images: bool,
                 calculate_K: bool,
                 training_pose_lifting_net_without_graphs: bool):
        """

        :param phase:
        :param base:
        :param pose_model_type:
        :param base_images:
        :param min_visibility:
        :param root_index:
        :param num_joints:
        :param overfit:
        :param augmentations:
        :param resnet152_backbone:
        :param get_2d_gt_from_3d:
        :param joint_indexes_to_return:
        :param minimum_views_needed_for_triangulation:
        :param every_nth_frame:
        """
        super(MPI_INF_3DHP_test, self).__init__(phase=phase, dataset_folder=base, num_joints=num_joints,
                                                overfit=overfit, randomize=False, only_annotations=True,
                                                ten_percent_3d_from_all=False, random_seed_for_ten_percent_3d_from_all=0,
                                                augmentations=augmentations, resnet152_backbone=resnet152_backbone,
                                                get_2d_gt_from_3d=get_2d_gt_from_3d, joints_order=joint_indexes_to_return,
                                                minimum_views_needed_for_triangulation=minimum_views_needed_for_triangulation,
                                                consider_unused_annotated_as_unannotated=False,
                                                every_nth_frame=every_nth_frame, every_nth_frame_train_annotated=0,
                                                every_nth_frame_train_unannotated=0, pose_model_type=pose_model_type,
                                                input_size_alphapose=input_size_alphapose, crop_size=crop_size,
                                                training_pose_lifting_net_without_graphs=training_pose_lifting_net_without_graphs,
                                                pelvis_idx=pelvis_idx, rhip_idx=rhip_idx, lhip_idx=lhip_idx, neck_idx=neck_idx,
                                                inp_lifting_net_is_images=inp_lifting_net_is_images)
        self.base                      = base  # "/cvlabsrc1/cvlab/MPI_INF_3D/mpi_inf_3dhp/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set"
        self.base_images               = base_images
        self.min_visibility            = min_visibility
        self.only_chest_height_cameras = True
        self.subjects                  = ['TS{}'.format(i) for i in range(1, 7)]
        self.sequences                  = 'seq'
        self.camera_ids                = [0]
        self.views            = self.camera_ids
        self.multiview_sample = False
        self.every_nth_frame  = every_nth_frame
        self.joint_indexes_to_return = np.int32(joint_indexes_to_return)

        if not isinstance(root_index, (list, tuple)):
            self.root_index = [root_index]

        self.root_index   = np.int32(root_index)
        self.calibrations = {}
        for subject in self.subjects:
            K, dist, R, t, h, w        = intrinsics_test_sequences(base=self.base, calculate_K=calculate_K, seq=subject)
            self.calibrations[subject] = {}
            self.calibrations[subject][self.sequences] = {}
            self.calibrations[subject][self.sequences][self.camera_ids[0]] = {'R': R, 't': t, 'K': K, 'dist': dist, 'height': h, 'width': w}

        self.data = self._prepare_data()
        self.samples = []
        for subject in self.subjects:
            idx_frames = self.data[subject][self.sequences]['idx_frames']  # idx_frames are the ones that are valid
            for idx in idx_frames:
                if self.visibility(subject, self.sequences, self.camera_ids[0], idx) > min_visibility:
                    self.samples.append((subject, self.sequences, self.camera_ids[0], idx))

        self.samples = self.samples[::self.every_nth_frame]
        num_samples  = len(self.samples)
        labels       = [1] * num_samples
        self.labels  = labels
        print("The number of samples {} for {}.".format(num_samples, self.phase.upper()))
        print_num_samples(mode=self.phase.lower(), labels=self.labels)

    def get_filename_image(self, subject, seq, cam_id, idx):
        """

        :param subject:
        :param seq:
        :param cam_id:
        :param idx:
        :return:
        """
        # idx: It is the frame idx.
        return os.path.join(self.base_images, subject, "imageSequence", "img_{:06d}.jpg".format(idx + 1))

    def get_position(self, subject, seq, frame_idx):
        """

        :param subject:
        :param seq:
        :param frame_idx:
        :return:
        """
        # idx: It is the frame idx.
        warnings.warn("Positions/3D poses in the test set are in camera coordinate not world!")
        pose_3d = self.get_pose_3d(subject, seq, frame_idx)
        if pose_3d is None:
            return None
        return np.mean(pose_3d, axis=0)

    def _prepare_data(self):
        """
        :return:
        """
        def f(subject):
            mat          = h5py.File(os.path.join(self.base, subject, 'annot_data.mat'), 'r')
            poses_2d     = np.array(mat['annot2'])[:, 0]
            poses_3d     = np.array(mat['annot3'])[:, 0] / 1000
            valid_frames = np.array(mat['valid_frame']).ravel() > 0
            return {'idx_frames': np.where(valid_frames)[0].tolist(),
                    'poses_2d': {self.camera_ids[0]: poses_2d[valid_frames]},
                    'poses_3d': poses_3d[valid_frames]}
        data = {}
        for subject in self.subjects:
            data[subject] = {self.sequences: f(subject)}
            print("Loaded {} frames for subject:{}.".format(len(data[subject][self.sequences]['idx_frames']), subject))
        return data

    def get_pose_3d_test(self, subject, seq, frame_idx):
        """
        :param subject:
        :param seq:
        :param frame_idx:
        :return:
        """
        indexes = self.data[subject][seq]['idx_frames']
        if frame_idx not in indexes:
            return None
        pose_3d = self.data[subject][seq]['poses_3d'][indexes.index(frame_idx)]
        return pose_3d[self.joint_indexes_to_return]

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        output = self.sample(index)
        return output


def get_mpi_simple_data_loaders(config, pose_model_type, training_pose_lifting_net_without_graphs):
    """
    :param config: The Configuration File.
    :param pose_model_type: The type of Pose Model used in this experiment.
    :param training_pose_lifting_net_without_graphs: If True, we will make number of frames = 1
    :return: The dataloaders for train, validation, test and train (without shuffle) sets for training with MPI dataset.
    """
    print("\n\n")
    print('#####' * 20)
    subjects   = config.annotated_subjects + config.unannotated_subjects
    subjects   = list(set(subjects))
    subjects.sort()
    crop_size  = (256, 256)
    pelvis_idx = config.pelvis_idx
    neck_idx   = config.neck_idx
    lhip_idx   = config.lhip_idx
    rhip_idx   = config.rhip_idx

    only_chest_height_cameras   = config.only_chest_height_cameras
    multiview_sample            = config.multiview_sample
    min_visibility              = config.min_visibility
    resnet152_backbone          = True if 'resnet152' in config.type_of_2d_pose_model else False
    annotated_subjects          = config.annotated_subjects
    randomize                   = config.randomize
    root_index                  = [config.lhip_idx, config.rhip_idx]
    load_from_cache             = config.load_from_cache
    path_cache                  = '/cvlabdata2/home/citraro/code/hpose/hpose/datasets' # config.path_cache
    base                        = '/cvlabsrc1/cvlab/MPI_INF_3D/mpi_inf_3dhp' # config.dataset_folder
    base_images                 = "/cvlabsrc1/cvlab/dataset_mpi_inf_3dhp/" # config.image_folder
    every_nth_frame_annotated   = config.every_nth_frame_train_annotated
    every_nth_frame_unannotated = config.every_nth_frame_train_unannotated
    use_annotations             = True if config.experimental_setup in ['semi', 'fully', 'weakly'] else False
    get_2d_gt_from_3d           = True if (config.experimental_setup in ['semi', 'fully'] or config.pretraining_with_annotated_2D is True) else False
    only_annotations            = True if (config.train_with_annotations_only is True and config.perform_test is False) else False
    # only_annotations            = True if (config.train_with_annotations_only is True or config.pretraining_with_annotated_2D is True) else False
    if resnet152_backbone:
        joint_indexes_train = joint_idxs_28to17ours_learnable_h36m_train
    else:
        if config.learning_use_h36m_model:
            joint_indexes_train = joint_idxs_28to17_according_to_h36m_train
        else:
            joint_indexes_train = joint_idxs_28to17_original_train

    num_joints    = config.number_of_joints
    overfit       = config.overfit_dataset
    augmentations = config.use_augmentations
    calculate_K   = config.calculate_K
    print("Obtaining the Train Datasets for MPI DATASET.")
    minimum_views_needed_for_triangulation = config.minimum_views_needed_for_triangulation
    inp_lifting_net_is_images              = config.inp_lifting_net_is_images
    train_dataset                          = MPI_INF_3DHP_train(base=base, base_images=base_images, subjects=subjects,
                                                                only_chest_height_cameras=only_chest_height_cameras,
                                                                multiview_sample=multiview_sample, min_visibility=min_visibility,
                                                                joint_indexes_to_return=joint_indexes_train, root_index=root_index,
                                                                load_from_cache=load_from_cache, path_cache=path_cache,
                                                                every_nth_frame_annotated=every_nth_frame_annotated,
                                                                every_nth_frame_unannotated=every_nth_frame_unannotated,
                                                                use_annotations_only=only_annotations, resnet152_backbone=resnet152_backbone,
                                                                num_joints=num_joints, overfit=overfit, use_annotations=use_annotations,
                                                                randomize=randomize, augmentations=augmentations, crop_size=crop_size,
                                                                annotated_subjects=annotated_subjects, get_2d_gt_from_3d=get_2d_gt_from_3d,
                                                                ten_percent_3d_from_all=config.ten_percent_3d_from_all,
                                                                ignore_sub8_seq2=config.ignore_sub8_seq2, just_sub8_seq2=False,
                                                                minimum_views_needed_for_triangulation=minimum_views_needed_for_triangulation,
                                                                consider_unused_annotated_as_unannotated=config.consider_unused_annotated_as_unannotated,
                                                                random_seed_for_ten_percent_3d_from_all=config.random_seed_for_ten_percent_3d_from_all,
                                                                pose_model_type=pose_model_type, input_size_alphapose=config.input_size_alphapose,
                                                                training_pose_lifting_net_without_graphs=training_pose_lifting_net_without_graphs,
                                                                pelvis_idx=pelvis_idx, rhip_idx=rhip_idx, lhip_idx=lhip_idx, neck_idx=neck_idx,
                                                                inp_lifting_net_is_images=inp_lifting_net_is_images,
                                                                calculate_K=calculate_K)
    train_labels                 = train_dataset.return_labels()
    shuffle                      = config.shuffle
    num_anno_samples_per_batch   = config.num_anno_samples_per_batch
    num_samples_train            = len(train_dataset)
    sampler_train, shuffle_train = get_sampler_train(use_annotations=use_annotations, labels=train_labels,
                                                     only_annotations=only_annotations, shuffle=shuffle,
                                                     num_anno_samples_per_batch=num_anno_samples_per_batch,
                                                     batch_size=config.batch_size, randomize=randomize,
                                                     extend_last_batch_graphs=False, num_samples=num_samples_train)
    my_collate                   = collate_func_pose_net_triangulation()
    train_loader                 = DataLoader(dataset=train_dataset, batch_size=config.batch_size, sampler=sampler_train,
                                              shuffle=shuffle_train, num_workers=config.num_workers, collate_fn=my_collate)
    train_loader_wo_shuffle      = DataLoader(dataset=train_dataset, batch_size=config.batch_size, sampler=None, #sampler_train,
                                              shuffle=False, num_workers=config.num_workers, collate_fn=my_collate)
    if config.evaluate_on_S8_seq2_mpi:
        assert config.ignore_sub8_seq2 is True
        # if config.evaluate_learnt_model is False:
        print("Getting the samples of S8 sampled at {}.".format(config.every_nth_frame_validation))
        validation_dataset = MPI_INF_3DHP_train(base=base, base_images=base_images, subjects=['S8'],
                                                only_chest_height_cameras=only_chest_height_cameras,
                                                multiview_sample=multiview_sample, min_visibility=min_visibility,
                                                joint_indexes_to_return=joint_indexes_train,
                                                root_index=root_index, load_from_cache=load_from_cache, path_cache=path_cache,
                                                every_nth_frame_annotated=config.every_nth_frame_validation,
                                                every_nth_frame_unannotated=every_nth_frame_unannotated,
                                                use_annotations_only=True, resnet152_backbone=resnet152_backbone,
                                                num_joints=num_joints, overfit=overfit, use_annotations=True,
                                                randomize=False, augmentations=augmentations, annotated_subjects=['S8'],
                                                get_2d_gt_from_3d=get_2d_gt_from_3d,
                                                ten_percent_3d_from_all=False, ignore_sub8_seq2=False, just_sub8_seq2=True,
                                                minimum_views_needed_for_triangulation=minimum_views_needed_for_triangulation,
                                                consider_unused_annotated_as_unannotated=False,
                                                random_seed_for_ten_percent_3d_from_all=0,
                                                pose_model_type=pose_model_type, crop_size=crop_size,
                                                input_size_alphapose=config.input_size_alphapose,
                                                training_pose_lifting_net_without_graphs=training_pose_lifting_net_without_graphs,
                                                pelvis_idx=pelvis_idx, rhip_idx=rhip_idx, lhip_idx=lhip_idx, neck_idx=neck_idx,
                                                inp_lifting_net_is_images=inp_lifting_net_is_images,
                                                calculate_K=calculate_K)
        test_dataset = validation_dataset
        mpjpe_poses_in_camera_coordinates = False
    else:
        print("HI HI " * 10)
        if resnet152_backbone:
            joint_indexes_test = joint_idxs_17to17ours_learnable_h36m_test
        else:
            if config.learning_use_h36m_model:
                joint_indexes_test = joint_idxs_17to17_according_to_h36m_test
            else:
                joint_indexes_test = joint_idxs_17to17_original_test
        base_test_validation        = "/cvlabsrc1/cvlab/MPI_INF_3D/mpi_inf_3dhp/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set"
        base_images_test_validation = "/cvlabsrc1/cvlab/MPI_INF_3D/mpi_inf_3dhp/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set"
        every_nth_frame_validation  = config.every_nth_frame_validation
        validation_dataset          = MPI_INF_3DHP_test(base=base_test_validation, base_images=base_images_test_validation,
                                                        min_visibility=min_visibility, root_index=root_index,
                                                        joint_indexes_to_return=joint_indexes_test,
                                                        every_nth_frame=every_nth_frame_validation,
                                                        resnet152_backbone=resnet152_backbone, num_joints=num_joints,
                                                        overfit=overfit, augmentations=augmentations, get_2d_gt_from_3d=True,
                                                        minimum_views_needed_for_triangulation=minimum_views_needed_for_triangulation,
                                                        phase='validation', pose_model_type=pose_model_type,
                                                        input_size_alphapose=config.input_size_alphapose, crop_size=crop_size,
                                                        training_pose_lifting_net_without_graphs=training_pose_lifting_net_without_graphs,
                                                        pelvis_idx=pelvis_idx, rhip_idx=rhip_idx, lhip_idx=lhip_idx, neck_idx=neck_idx,
                                                        inp_lifting_net_is_images=inp_lifting_net_is_images,
                                                        calculate_K=calculate_K)

        test_dataset                = MPI_INF_3DHP_test(base=base_test_validation, base_images=base_images_test_validation,
                                                        min_visibility=min_visibility, root_index=root_index,
                                                        joint_indexes_to_return=joint_indexes_test,
                                                        every_nth_frame=every_nth_frame_validation,
                                                        resnet152_backbone=resnet152_backbone, num_joints=num_joints,
                                                        overfit=overfit, augmentations=augmentations, get_2d_gt_from_3d=True,
                                                        minimum_views_needed_for_triangulation=minimum_views_needed_for_triangulation,
                                                        phase='validation', pose_model_type=pose_model_type,
                                                        input_size_alphapose=config.input_size_alphapose, crop_size=crop_size,
                                                        training_pose_lifting_net_without_graphs=training_pose_lifting_net_without_graphs,
                                                        pelvis_idx=pelvis_idx, rhip_idx=rhip_idx, lhip_idx=lhip_idx, neck_idx=neck_idx,
                                                        inp_lifting_net_is_images=inp_lifting_net_is_images,
                                                        calculate_K=calculate_K)
        mpjpe_poses_in_camera_coordinates = True

    validation_loader = DataLoader(dataset=validation_dataset, batch_size=config.batch_size_test, sampler=None,
                                   shuffle=False, num_workers=config.num_workers, collate_fn=my_collate)
    test_loader       = DataLoader(dataset=test_dataset, batch_size=config.batch_size_test, sampler=None,
                                   shuffle=False, num_workers=config.num_workers, collate_fn=my_collate)
    print("The batch size for Train and Test is set to {} and {} respectively".format(config.batch_size, config.batch_size_test))
    print("The number of Batches for Train and Test set is {} and {} respectively.".format(len(train_loader), len(test_loader)), end='\n')
    return train_loader, validation_loader, test_loader, train_loader_wo_shuffle, mpjpe_poses_in_camera_coordinates



def get_mpi_dataset_characteristics(config):
    """
    Function to obtain certain characteristics of the MPI-INF-3DHP which we will be using.
    param config: The Configuration File consisting of the arguments of training.
    :return: number_of_joints --> The number of joints used in our framework.
             cameras          --> The various camera ids.
             bone_pairs       --> The pairs of bones between two joints.
             rhip_idx         --> The index of the Right Hip.
             lhip_idx         --> The index of the Left Hip.
             neck_idx         --> The index of the Neck.
             pelvis_idx       --> The index of the Pelvis.
    """
    resnet152_backbone = True if 'resnet152' in config.type_of_2d_pose_model else False
    if resnet152_backbone:
        print("We are using the ResNet152 Pretrained on MS-COCO+H36M+MPI as our backbone 2D Pose Estimator Model.")
        """
        ['RightFoot'    --> 'right_ankle', 0
        'RightLeg'      --> 'right_knee, 1
        'RightUpLeg'    --> right_hip, 2
        'LeftUpLeg'     --> 'left_hip', 3
        'LeftLeg'       --> 'left_knee', 4
        'LeftFoot'      --> 'left_ankle', 5 
        'Hips'          --> 'pelvis', 6
        'Spine1'        --> 'spine', 7
        'Neck'          --> 'neck', 8
        'Site-head'     --> 'head_top', 9 
        'RightHand'     --> 'right_wrist', 10
        'RightForeArm'  --> 'right_elbow', 11 
        'RightArm'      --> 'right_shoulder', 12
        'LeftArm'       --> 'left_shoulder', 13
        'LeftForeArm'   --> 'left_elbow', 14
        'LeftHand'      --> 'left_wrist', 15
        'Head'          --> 'head'] 16
        """
        joints_names = joint_names_17_learnable
        """
        ['right_ankle', 'right_knee', 'right_hip', 'left_hip', 'left_knee', 'left_ankle',
        'pelvis', 'spine', 'neck', 'head_top', 'right_wrist', 'right_elbow',
        'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist', 'head']
        """
    else:
        if config.learning_use_h36m_model:
            print("Will be considering 17 joints for the AlphaPose model.")
            """
            ['pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee',
            'left_ankle', 'spine', 'neck', 'head', 'head_top', 'left_shoulder',
            'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist']
            """
            joints_names = joint_names_all_17_according_to_h36m
        else:
            """
            ['head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
            'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee', 'right_ankle',
            'left_hip', 'left_knee', 'left_ankle', 'pelvis', 'spine', 'head']
            """
            joints_names = joint_names_all_17_original
    bones_pairs    = []
    bones          = [('pelvis', 'right_hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'), # Right Leg is done
                      ('pelvis', 'left_hip'),  ('left_hip', 'left_knee'),   ('left_knee', 'left_ankle'), # Left Leg is done
                      ('pelvis', 'spine'), ('spine', 'neck'), ('neck', 'head'), ('head', 'head_top'), # Spine is done
                      ('neck', 'left_shoulder'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'), # Left Arm is done
                      ('neck', 'right_shoulder'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist') # Right Arm is done
                      ]
    rl_bone_idx    = [0, 1, 2, 6, 3]
    ll_bone_idx    = [3, 4, 5, 6, 0]
    torso_bone_idx = [0, 3, 6, 7, 8, 9, 10, 13]
    lh_bone_idx    = [10, 11, 12, 7]
    rh_bone_idx    = [13, 14, 15, 7]

    for bone in bones:
        bone_start = bone[0]; bone_start_idx = joints_names.index(bone_start)
        bone_end   = bone[1]; bone_end_idx   = joints_names.index(bone_end)
        bones_pairs.append([bone_start_idx, bone_end_idx])

    bone_pairs_symmetric = [[('pelvis', 'right_hip'),           ('pelvis', 'left_hip')],
                            [('right_hip', 'right_knee'),       ('left_hip', 'left_knee')],
                            [('right_knee', 'right_ankle'),     ('left_knee', 'left_ankle')],
                            [('neck', 'right_shoulder'),        ('neck', 'left_shoulder')],
                            [('right_shoulder', 'right_elbow'), ('left_shoulder', 'left_elbow')],
                            [('right_elbow', 'right_wrist'),    ('left_elbow', 'left_wrist')]
                            ]
    bone_pairs_symmetric_indexes = []
    for bone_pair_sym in bone_pairs_symmetric:
        right_bone, left_bone = bone_pair_sym[0], bone_pair_sym[1]
        right_bone_start = right_bone[0]
        right_bone_end   = right_bone[1]
        left_bone_start  = left_bone[0]
        left_bone_end    = left_bone[1]
        index            = ([joints_names.index(right_bone_start), joints_names.index(right_bone_end)],
                            [joints_names.index(left_bone_start), joints_names.index(left_bone_end)])
        bone_pairs_symmetric_indexes.append(index)

    number_of_joints = len(joints_names)
    lhip_idx         = joints_names.index('left_hip')
    rhip_idx         = joints_names.index('right_hip')
    neck_idx         = joints_names.index('neck')
    pelvis_idx       = joints_names.index('pelvis')
    head_idx         = joints_names.index('head')
    return number_of_joints, bones_pairs, rhip_idx, lhip_idx, neck_idx, pelvis_idx, bone_pairs_symmetric_indexes, \
           ll_bone_idx, rl_bone_idx, lh_bone_idx, rh_bone_idx, torso_bone_idx, head_idx


# def get_mpi_simple_data_loaders_lifting_net_only(config):
#     """
#     TODO
#     :param config:
#     :return:
#     """
#     print("\n\n")
#     print('#####' * 20)
#     subjects                    = config.annotated_subjects + config.unannotated_subjects
#     only_chest_height_cameras   = config.only_chest_height_cameras
#     multiview_sample            = config.multiview_sample
#     min_visibility              = config.min_visibility
#     resnet152_backbone          = True if 'resnet152' in config.type_of_2d_pose_model else False
#     annotated_subjects          = config.annotated_subjects
#     randomize                   = config.randomize
#     root_index                  = [config.lhip_idx, config.rhip_idx]
#     load_from_cache             = config.load_from_cache
#     path_cache                  = '/cvlabdata2/home/citraro/code/hpose/hpose/datasets'  # config.path_cache
#     base                        = '/cvlabsrc1/cvlab/MPI_INF_3D/mpi_inf_3dhp'  # config.dataset_folder
#     base_images                 = "/cvlabsrc1/cvlab/dataset_mpi_inf_3dhp/"  # config.image_folder
#     every_nth_frame_annotated   = config.every_nth_frame_train_annotated
#     every_nth_frame_unannotated = config.every_nth_frame_train_unannotated
#     use_annotations             = True if config.experimental_setup in ['semi', 'fully', 'weakly'] else False
#     get_2d_gt_from_3d           = True if (config.experimental_setup in ['semi','fully'] or config.pretraining_with_annotated_2D is True) else False
#     only_annotations            = True if (config.train_with_annotations_only is True or config.pretraining_with_annotated_2D is True) else False
#     if resnet152_backbone:
#         joint_indexes_train = joint_idxs_28to17ours_learnable_h36m_train
#     else:
#         if config.learning_use_h36m_model:
#             joint_indexes_train = joint_idxs_28to17_according_to_h36m_train
#         else:
#             joint_indexes_train = joint_idxs_28to17_original_train
#
#     num_joints    = config.number_of_joints
#     overfit       = config.overfit_dataset
#     augmentations = config.use_augmentations
#     print("Obtaining the Train Datasets for MPI DATASET.")
#
#
#     train_dataset      = None # TODO
#     test_dataset       = None # TODO
#     validation_dataset = None # TODO
#     train_labels                 = train_dataset.return_labels()
#     shuffle                      = config.shuffle
#     num_anno_samples_per_batch   = config.num_anno_samples_per_batch
#     sampler_train, shuffle_train = get_sampler_train(use_annotations=use_annotations, labels=train_labels,
#                                                      only_annotations=only_annotations, shuffle=shuffle,
#                                                      num_anno_samples_per_batch=num_anno_samples_per_batch,
#                                                      batch_size=config.batch_size, randomize=randomize)
#     train_loader                 = DataLoader(dataset=train_dataset, batch_size=config.batch_size, sampler=sampler_train,
#                                               shuffle=shuffle_train, num_workers=config.num_workers, collate_fn=my_collate)
#     train_loader_wo_shuffle      = DataLoader(dataset=train_dataset, batch_size=config.batch_size, sampler=sampler_train,
#                                               shuffle=False, collate_fn=my_collate, num_workers=config.num_workers)
#     validation_loader            = DataLoader(dataset=validation_dataset, batch_size=config.batch_size_test, sampler=None,
#                                               shuffle=False, collate_fn=my_collate, num_workers=config.num_workers)
#     test_loader                  = DataLoader(dataset=test_dataset, batch_size=config.batch_size_test, sampler=None,
#                                               shuffle=False, collate_fn=my_collate, num_workers=config.num_workers)
#     print("The batch size for Train and Test is set to {} and {} respectively".format(config.batch_size, config.batch_size_test))
#     print("The number of Batches for Train and Test set is {} and {} respectively.".format(len(train_loader), len(test_loader)), end='\n')
#     mpjpe_poses_in_camera_coordinates = True
#     return train_loader, validation_loader, test_loader, train_loader_wo_shuffle, mpjpe_poses_in_camera_coordinates
