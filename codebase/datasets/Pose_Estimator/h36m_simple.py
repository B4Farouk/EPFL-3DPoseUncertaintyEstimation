import os
import cv2
import numpy as np


from utils                 import print_num_samples, bbox_from_points
from torch.utils.data      import DataLoader
from datasets.dataset_base import Dataset_base, get_sampler_train, collate_func_pose_net_triangulation
from datasets.h36m_basics  import (image_shapes, camera_names, joint_idxs_32to17_learnable, joint_names_17, joint_idxs_32to17,
                                   joint_names_17_learnable, H36M_dataset, get_labeled_samples)



class H36M_dataloader(Dataset_base):
    def __init__(self,
                 dataset,
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

        :param dataset:
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
        super(H36M_dataloader, self).__init__(dataset_folder=dataset_folder, phase=phase, num_joints=num_joints,
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
                                              pose_model_type=pose_model_type, 
                                              input_size_alphapose=input_size_alphapose,
                                              training_pose_lifting_net_without_graphs=training_pose_lifting_net_without_graphs,
                                              pelvis_idx=pelvis_idx, rhip_idx=rhip_idx, lhip_idx=lhip_idx, neck_idx=neck_idx,
                                              crop_size=crop_size, inp_lifting_net_is_images=inp_lifting_net_is_images,
                                              )

        self.dataset   = dataset
        self.num_views = 4
        self.shapes    = image_shapes
        print("Obtaining the Data for Phase {}".format(self.phase))
        calibration_phase, data_phase, subject_ids_phase, samples_phase, actions_phase, labels_phase = self.dataset.get_values(phase=phase.lower())

        self.cameras     = camera_names
        self.calibration = calibration_phase
        self.data        = data_phase
        self.subject_ids = subject_ids_phase
        self.actions     = actions_phase
        samples, labels  = get_labeled_samples(samples_phase=samples_phase, labels_phase=labels_phase, phase=phase,
                                               every_nth_frame=self.every_nth_frame, overfit=self.overfit,
                                               ten_percent_3d_from_all=self.ten_percent_3d_from_all,
                                               every_nth_frame_train_annotated=self.every_nth_frame_train_annotated,
                                               every_nth_frame_train_unannotated=self.every_nth_frame_train_unannotated,
                                               only_annotations=self.only_annotations, randomize=self.randomize)
        self.samples     = samples
        self.labels      = labels
        print_num_samples(mode=self.phase.lower(), labels=self.labels)

    def get_samples(self):
        return self.samples

    def get_labels(self):
        return self.labels

    def obtain_subjects(self):
        return self.subject_ids

    def obtain_actions(self):
        return self.actions

    def obtain_cameras(self):
        return self.cameras

    def get_filename_image(self, subject, idx, camera_name, action):
        """
        :param subject:
        :param action:
        :param idx:
        :param camera_name:
        :return:
        """
        return os.path.join(self.dataset_folder, subject, action, "imageSequence", camera_name, "img_{:06d}.jpg".format(idx))

    def ret(self):
        """
        :return:
        """
        return self.calibration, self.data, self.subject_ids, self.samples, self.actions, self.labels

    def __len__(self):
        """
        :return:
        """
        num_files = len(self.labels)
        return num_files

    def get_pose_3d(self, subject, action, idx):
        """

        :param subject:
        :param action:
        :param idx:
        :return:
        """
        indexes = self.data[subject][action]['idx_frames']
        if idx not in indexes:
            return None
        pose_3d = self.data[subject][action]['poses_3d'][indexes.index(idx)]
        pose_3d = pose_3d[self.joints_order]
        return pose_3d

    def get_bbox(self, subject, action, idx, camera_name):
        """

        :param subject:
        :param action:
        :param idx:
        :param camera_name:
        :return:
        """
        pose_2d = self.get_pose_2d(subject, action, idx, camera_name)
        if pose_2d is None:
            return None
        bbox = bbox_from_points(pose_2d, pb=self.p)
        return bbox

    def get_pose_2d_from_3d(self, subject, camera_name, pose_3d):
        """

        :param subject:
        :param camera_name:
        :param pose_3d:
        :return:
        """
        rvec, t, K, dist, _ = self.get_calibration_subject_camera(subject, camera_name, rot_vector=True)
        pose_2d             = cv2.projectPoints(pose_3d, rvec, t, K, dist)[0].reshape(-1, 2)
        return pose_2d

    def get_pose_2d(self, subject, action, idx, camera_name, from_3d=False):
        """

        :param subject:
        :param action:
        :param idx:
        :param camera_name:
        :param from_3d:
        :return:
        """
        if from_3d:
            pose_3d = self.get_pose_3d(subject, action, idx)
            if pose_3d is None:
                return None
            rvec, t, K, dist, _ = self.get_calibration_subject_camera(subject, camera_name, rot_vector=True)
            pose_2d             = cv2.projectPoints(pose_3d, rvec, t, K, dist)[0].reshape(-1, 2)
        else:
            indexes = self.data[subject][action]['idx_frames']
            if idx not in indexes:
                return None
            pose_2d = self.data[subject][action]['poses_2d'][camera_name][indexes.index(idx)]
            pose_2d = pose_2d[self.joints_order]
        return pose_2d


    def get_pose_3d_common(self, subject_idx, seq_idx, frame_idx):
        """
        :param subject_idx: It is the id of the Subject.
        :param seq_idx: It is the action id.
        :param frame_idx: This is the id of the frame.
        :return:
        """
        pose_3d = self.get_pose_3d(subject=subject_idx, action=seq_idx, idx=frame_idx)
        return pose_3d


    def get_filename_image_common(self, subject_idx, seq_idx, cam_idx, frame_idx):
        """
        :param subject_idx: It is the id of the Subject.
        :param seq_idx: It is the action id.
        :param cam_idx: This is the id of the camera.
        :param frame_idx: This is the id of the frame.
        :return:
        """
        file_name = self.get_filename_image(subject=subject_idx, idx=frame_idx, camera_name=cam_idx, action=seq_idx)
        return file_name

    def get_pose_2d_common(self, subject_idx, seq_idx, cam_idx, frame_idx, from_3d):
        """
        :param subject_idx: It is the id of the Subject.
        :param seq_idx: It is the action id.
        :param cam_idx: This is the id of the camera.
        :param frame_idx: This is the id of the frame.
        :param from_3d: If True, we will calculate the 2D pose by projecting the 3D pose, or else we will use the manually detected 2D pose.
        :return:
        """
        pose_2d = self.get_pose_2d(subject=subject_idx, camera_name=cam_idx, action=seq_idx, idx=frame_idx, from_3d=from_3d)
        return pose_2d

    def get_calibration_common(self, subject_idx, seq_idx, cam_idx, frame_idx, rot_vector=True):
        """
        :param subject_idx: It is the id of the Subject.
        :param seq_idx: It is the action id.
        :param cam_idx: This is the id of the camera.
        :param frame_idx: This is the id of the frame.
        :param rot_vector: If True, we will perform cv2.Rodrigues on the Rotation Matrix.
        :return:
        """
        R, t, K, dist, shape = self.get_calibration_subject_camera(subject=subject_idx, camera_name=cam_idx, rot_vector=rot_vector)
        return R, t, K, dist, shape





    def get_position(self, subject, action, idx):
        """

        :param subject:
        :param action:
        :param idx:
        :return:
        """
        pose_3d = self.get_pose_3d(subject, action, idx)
        if pose_3d is None:
            return None
        return np.mean(pose_3d, axis=0)[:2]

    def get_calibration_per_view(self, view, index, rot_vector=True):
        """

        :param view:
        :param index:
        :param rot_vector:
        :return:
        """
        subject          = self.samples[index][0]
        r, t, K, dist, _ = self.get_calibration_subject_camera(subject=subject, camera_name=view, rot_vector=rot_vector)
        return r, t, K, dist

    def get_calibration_subject_camera(self, subject, camera_name, rot_vector=True):
        """

        :param subject:
        :param camera_name:
        :param rot_vector:
        :return:
        """
        calib_values = self.calibration[subject][camera_name]
        r            = calib_values['R']
        if rot_vector:
            r = cv2.Rodrigues(r)[0]
        t    = calib_values['t']
        K    = calib_values['K']
        dist = calib_values['dist']
        return r, t, K, dist, self.shapes[camera_name]

    def get_subject_action_frame_label_by_index(self, index):
        """

        :param index:
        :return:
        """
        sample_index      = self.samples[index]
        subject           = sample_index[0]
        action            = sample_index[1]
        frame_idx         = sample_index[2]
        labeled_candidate = self.labels[index]
        return subject, action, frame_idx, labeled_candidate

    def get_extra_by_index(self, index):
        """

        :param index:
        :return:
        """
        return self.get_action_by_index(index=index)

    def get_subject_by_index(self, index):
        """

        :param index:
        :return:
        """
        sample_index = self.samples[index]
        subject      = sample_index[0]
        return subject

    def get_action_by_index(self, index):
        """

        :param index:
        :return:
        """
        sample_index = self.samples[index]
        action       = sample_index[1]
        return action

    def get_calibration_by_index(self, index):
        """

        :param index:
        :return:
        """
        subject = self.get_subject_by_index(index=index)
        return self.calibration[subject]

    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        subject_idx, action, frame_idx,\
            labeled_candidate = self.get_subject_action_frame_label_by_index(index)
        retval                = self.obtain_mv_samples(subject_idx=subject_idx, seq_idx=action,
                                                       frame_idx=frame_idx, cameras=self.cameras, label=labeled_candidate)
        """
        target_pose_3d = self.get_pose_3d(subject=subject, action=action, idx=frame_idx)
        target_pose_3d = target_pose_3d[self.joints_order]
        if target_pose_3d is not None:
            target_pose_3d = np.float32(target_pose_3d)
            mask_valid     = ~np.isnan(target_pose_3d[:, 0])
        else:
            mask_valid     = None
        bboxes = []; inp_pose_model = []; consider = []; R = []; t = []; K = []; dist = []; target_pose_2d = []
        pose_2d_tar_norm = []; image_paths = []; camera_ids = []; inp_images_lifting_net = []
        # for i in range(self.num_views):
        #     cam_id        = self.cameras[i]
        for cam_id in self.cameras:
            img_file_path = self.get_filename_image(subject=subject, action=action, idx=frame_idx, camera_name=cam_id)
            shape_view    = self.shapes[cam_id]
            pose_2d_vid   = self.get_pose_2d(subject=subject, action=action, idx=frame_idx, camera_name=cam_id, from_3d=self.from_3d)
            pose_2d_vid   = pose_2d_vid[self.joints_order]
            bbox          = bbox_from_points(pose_2d_vid, pb=self.p)

            xmin, ymin, xmax, ymax = bbox
            xmin = max(xmin - self.margin, 0)
            ymin = max(ymin - self.margin, 0)
            xmax = min(xmax + self.margin, shape_view[1])
            ymax = min(ymax + self.margin, shape_view[0])

            bbox_old               = [int(xmin), int(ymin), int(xmax), int(ymax)]
            pose_input, bbox_new   = self.function_to_obtain_pose_input(image_file_path=img_file_path, bbox=bbox_old) # It is just a single string.
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

            R_cam, t_cam, K_cam, dist_cam, _ = self.get_calibration_subject_camera(subject=subject, camera_name=cam_id, rot_vector=False)
            pose_2d_vid_norm                 = self.get_norm_2d_pose(pose_2d=pose_2d_vid, bbox=bbox_new)
            camera_ids.append(cam_id)
            inp_pose_model.append(pose_input)
            target_pose_2d.append(pose_2d_vid)
            pose_2d_tar_norm.append(pose_2d_vid_norm)
            image_paths.append(img_file_path)
            R.append(R_cam)
            t.append(t_cam)
            K.append(K_cam)
            dist.append(dist_cam)
            bboxes.append(bbox_new)

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
            labeled_candidate          = torch.Tensor(labeled_candidate)
            labeled_candidate          = labeled_candidate.repeat(self.num_views)
            pelvis_cam_z               = get_pelvis(pose=target_pose_3d, lhip_idx=self.lhip_idx, rhip_idx=self.rhip_idx,
                                                    pelvis_idx=self.pelvis_idx, return_z=True)
            pelvis_cam_z               = pelvis_cam_z.view(-1).repeat(self.num_views)
            # if self.training_pose_lifting_net_without_graphs:
            target_pose_3d             = target_pose_3d.unsqueeze(dim=0)
            target_pose_3d             = target_pose_3d.repeat(self.num_views, 1, 1)


            retval = {'inp_pose_model' : inp_pose_model, 'bboxes' : bboxes, 'target_pose_2d' : target_pose_2d,
                      'R' : R, 't' : t, 'K' : K, 'dist' : dist, 'target_pose_3d' : target_pose_3d,
                      'image_paths' : image_paths, 'camera_ids': camera_ids, 'frame_ids': int(frame_idx),
                      'action_ids' : action, 'subject_ids' : subject, 'consider' : consider,
                      'pose_2d_tar_norm' : pose_2d_tar_norm, 'mask_valid' : mask_valid,
                      'pelvis_cam_z' : pelvis_cam_z, 'labels' : labeled_candidate,
                      'consider_for_triangulation': consider_for_triangulation}
            if self.inp_lifting_net_is_images:
                inp_images_lifting_net           = torch.from_numpy(np.stack(inp_images_lifting_net))
                retval['inp_images_lifting_net'] = inp_images_lifting_net

            if self.training_pose_lifting_net_without_graphs:
                retval = function_to_extend_dim(data=retval, which_dim=0, which_keys=self.keys_to_extend_dim)
        else:
            retval = None
        """
        return retval


def get_h36m_simple_data_loaders(config, pose_model_type, training_pose_lifting_net_without_graphs):
    """
    :param config: The Configuration File.
    :param pose_model_type: The type of Pose Model used in this experiment.
    :param training_pose_lifting_net_without_graphs: If True, we will make number of frames = 1
    :return: The dataloaders for train, validation, test and train (without shuffle) sets for training with H36M dataset.
    """
    print("\n\n")
    print('#####' * 20)
    minimum_views_needed_for_triangulation = config.minimum_views_needed_for_triangulation
    dataset_folder      = '/cvlabsrc1/cvlab/dataset_H36M/h36m_orig_sampled/'
    calibration_folder  = '/cvlabsrc1/cvlab/dataset_H36M/h36m_orig_sampled'
    use_annotations     = True if config.experimental_setup in ['semi', 'fully', 'weakly'] else False
    get_2d_gt_from_3d   = True if (config.experimental_setup in ['semi', 'fully'] or config.pretraining_with_annotated_2D is True) else False
    only_annotations    = True if (config.train_with_annotations_only is True and config.perform_test is False) else False
    # only_annotations    = True if (config.train_with_annotations_only is True or config.pretraining_with_annotated_2D is True) else False
    path_cache          = '/cvlabdata2/home/citraro/code/hpose/hpose/datasets'
    annotated_subjects  = config.annotated_subjects
    load_from_cache     = config.load_from_cache
    resnet152_backbone  = True if 'resnet152' in config.type_of_2d_pose_model else False
    shuffle             = config.shuffle
    randomize           = config.randomize
    crop_size           = (256, 256)
    pelvis_idx          = config.pelvis_idx
    neck_idx            = config.neck_idx
    lhip_idx            = config.lhip_idx
    rhip_idx            = config.rhip_idx

    if resnet152_backbone:
        joints_order = joint_idxs_32to17_learnable
    else:
        joints_order = joint_idxs_32to17
    training_subjects         = annotated_subjects + config.unannotated_subjects
    training_subjects         = list(set(training_subjects))
    inp_lifting_net_is_images = config.inp_lifting_net_is_images
    training_subjects.sort()
    dataset           = H36M_dataset(dataset_folder=dataset_folder, path_cache=path_cache, use_annotations=use_annotations,
                                     calibration_folder=calibration_folder, use_annotations_only=only_annotations,
                                     annotated_subjects=annotated_subjects, load_from_cache=load_from_cache,
                                     training_subjects=training_subjects)
    train_dataset     = H36M_dataloader(dataset_folder=dataset_folder, dataset=dataset, phase='Train',
                                        num_joints=config.number_of_joints, overfit=config.overfit_dataset,
                                        randomize=randomize, only_annotations=only_annotations,
                                        every_nth_frame=None, ten_percent_3d_from_all=config.ten_percent_3d_from_all,
                                        random_seed_for_ten_percent_3d_from_all=config.random_seed_for_ten_percent_3d_from_all,
                                        every_nth_frame_train_unannotated=config.every_nth_frame_train_unannotated,
                                        every_nth_frame_train_annotated=config.every_nth_frame_train_annotated,
                                        augmentations=config.use_augmentations, resnet152_backbone=resnet152_backbone,
                                        consider_unused_annotated_as_unannotated=config.consider_unused_annotated_as_unannotated,
                                        get_2d_gt_from_3d=get_2d_gt_from_3d, joints_order=joints_order,
                                        minimum_views_needed_for_triangulation=minimum_views_needed_for_triangulation,
                                        pose_model_type=pose_model_type, crop_size=crop_size,
                                        input_size_alphapose=config.input_size_alphapose,
                                        training_pose_lifting_net_without_graphs=training_pose_lifting_net_without_graphs,
                                        pelvis_idx=pelvis_idx, rhip_idx=rhip_idx, lhip_idx=lhip_idx, neck_idx=neck_idx,
                                        inp_lifting_net_is_images=inp_lifting_net_is_images
                                        )

    train_labels                 = train_dataset.return_labels()
    num_samples_train            = len(train_dataset)
    sampler_train, shuffle_train = get_sampler_train(use_annotations=use_annotations, only_annotations=only_annotations,
                                                     num_anno_samples_per_batch=config.num_anno_samples_per_batch,
                                                     batch_size=config.batch_size, randomize=randomize,
                                                     shuffle=shuffle, labels=train_labels, extend_last_batch_graphs=False,
                                                     num_samples=num_samples_train)

    my_collate              = collate_func_pose_net_triangulation()
    train_loader            = DataLoader(dataset=train_dataset, batch_size=config.batch_size, sampler=sampler_train,
                                         shuffle=shuffle_train, num_workers=config.num_workers, collate_fn=my_collate)
    train_loader_wo_shuffle = DataLoader(dataset=train_dataset, batch_size=config.batch_size, sampler=None, #sampler_train,
                                         shuffle=False, num_workers=config.num_workers, collate_fn=my_collate)
    validation_dataset      = H36M_dataloader(dataset_folder=dataset_folder, dataset=dataset, phase='Validation',
                                              num_joints=config.number_of_joints, overfit=False,
                                              randomize=randomize, only_annotations=False,
                                              every_nth_frame=config.every_nth_frame_validation,
                                              ten_percent_3d_from_all=False,
                                              random_seed_for_ten_percent_3d_from_all=config.random_seed_for_ten_percent_3d_from_all,
                                              every_nth_frame_train_annotated=None, every_nth_frame_train_unannotated=None,
                                              augmentations=config.use_augmentations, resnet152_backbone=resnet152_backbone,
                                              consider_unused_annotated_as_unannotated=False,
                                              get_2d_gt_from_3d=True, joints_order=joints_order,
                                              minimum_views_needed_for_triangulation=minimum_views_needed_for_triangulation,
                                              pose_model_type=pose_model_type, crop_size=crop_size,
                                              input_size_alphapose=config.input_size_alphapose,
                                              training_pose_lifting_net_without_graphs=training_pose_lifting_net_without_graphs,
                                              pelvis_idx=pelvis_idx, rhip_idx=rhip_idx, lhip_idx=lhip_idx, neck_idx=neck_idx,
                                              inp_lifting_net_is_images=inp_lifting_net_is_images)

    validation_loader = DataLoader(dataset=validation_dataset, batch_size=config.batch_size_test, sampler=None,
                                   shuffle=False, num_workers=config.num_workers, collate_fn=my_collate)
    test_dataset      = H36M_dataloader(dataset_folder=dataset_folder, dataset=dataset, phase='Test',
                                        num_joints=config.number_of_joints, overfit=False, randomize=randomize,
                                        only_annotations=False, every_nth_frame=config.every_nth_frame_test,
                                        ten_percent_3d_from_all=False, joints_order=joints_order,
                                        random_seed_for_ten_percent_3d_from_all=config.random_seed_for_ten_percent_3d_from_all,
                                        every_nth_frame_train_annotated=None, every_nth_frame_train_unannotated=None,
                                        augmentations=config.use_augmentations, resnet152_backbone=resnet152_backbone,
                                        consider_unused_annotated_as_unannotated=False, get_2d_gt_from_3d=True,
                                        minimum_views_needed_for_triangulation=minimum_views_needed_for_triangulation,
                                        pose_model_type=pose_model_type, crop_size=crop_size,
                                        input_size_alphapose=config.input_size_alphapose,
                                        training_pose_lifting_net_without_graphs=training_pose_lifting_net_without_graphs,
                                        pelvis_idx=pelvis_idx, rhip_idx=rhip_idx, lhip_idx=lhip_idx, neck_idx=neck_idx,
                                        inp_lifting_net_is_images=inp_lifting_net_is_images)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size_test, sampler=None, shuffle=False,
                             num_workers=config.num_workers, collate_fn=my_collate)
    print("The batch size for Train and Test is set to {} and {} respectively".format(config.batch_size, config.batch_size_test))
    print("The number of Batches for Train and Test set is {} and {} respectively.".format(len(train_loader), len(test_loader)), end='\n')
    print('#####' * 20)
    print("\n\n")
    mpjpe_poses_in_camera_coordinates = False
    return train_loader, validation_loader, test_loader, train_loader_wo_shuffle, mpjpe_poses_in_camera_coordinates


def get_h36m_dataset_characteristics(config):
    """
    Function to obtain certain characteristics of the H3.6M which we will be using.
    param config: The Configuration File consisting of the arguments of training.
    :return: number_of_joints --> The number of joints used in our framework.
             cameras          --> The various camera ids.
             bone_pairs       --> The pairs of bones between two joints.
             rhip_idx         --> The index of the Right Hip.
             lhip_idx         --> The index of the Left Hip.
             neck_idx         --> The index of the Neck.
             pelvis_idx       --> The index of the Pelvis.
    """
    resnet152_backbone = True if 'resnet152' in config.type_of_2d_pose_model else False #False if 'crowd_pose' in config.type_of_2d_pose_model else True
    if resnet152_backbone:
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
        print("We are using the ResNet152 Pretrained on MSCOCO+H36M+MPI as our backbone 2D Pose Estimator Model.")
        joints_names = joint_names_17_learnable
        """
        ['RightFoot', 'RightLeg', 'RightUpLeg', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Hips', 'Spine1',
        'Neck', 'Site-head', 'RightHand', 'RightForeArm', 'RightArm', 'LeftArm', 'LeftForeArm', 'LeftHand', 'Head']
        """
    else:
        print("Will be considering 17 joints for the AlphaPose model.")
        """
        ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Spine1', 
        'Neck', 'Head', 'Site-head', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand']
        """
        joints_names = joint_names_17
    bones_pairs          = []
    bones                = [('Hips', 'RightUpLeg'), ('RightUpLeg', 'RightLeg'), ('RightLeg', 'RightFoot'),    # Right Leg is done
                            ('Hips', 'LeftUpLeg'), ('LeftUpLeg', 'LeftLeg'), ('LeftLeg', 'LeftFoot'),         # Left Leg is done
                            ('Hips', 'Spine1'), ('Spine1', 'Neck'), ('Neck', 'Head'), ('Head', 'Site-head'),  # Spine is done
                            ('Neck', 'LeftArm'), ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'),     # Left Arm is done
                            ('Neck', 'RightArm'), ('RightArm', 'RightForeArm'), ('RightForeArm', 'RightHand') # Right Arm is done
                            ]
    rl_bone_idx          = [0, 1, 2, 6, 3]
    ll_bone_idx          = [3, 4, 5, 6, 0]
    torso_bone_idx       = [0, 3, 6, 7, 8, 9, 10, 13]
    lh_bone_idx          = [10, 11, 12, 7]
    rh_bone_idx          = [13, 14, 15, 7]

    bone_pairs_symmetric = [[('Hips', 'RightUpLeg'), ('Hips', 'LeftUpLeg')],
                            [('RightUpLeg', 'RightLeg'), ('LeftUpLeg', 'LeftLeg')],
                            [('RightLeg', 'RightFoot'), ('LeftLeg', 'LeftFoot')],
                            [('Neck', 'LeftArm'), ('Neck', 'RightArm')],
                            [('LeftArm', 'LeftForeArm'), ('RightArm', 'RightForeArm')],
                            [('LeftForeArm', 'LeftHand'), ('RightForeArm', 'RightHand')]
                            ]

    for bone in bones:
        bone_start = bone[0]; bone_start_idx = joints_names.index(bone_start)
        bone_end = bone[1]; bone_end_idx = joints_names.index(bone_end)
        bones_pairs.append([bone_start_idx, bone_end_idx])

    bone_pairs_symmetric_indexes = []
    for bone_pair_sym in bone_pairs_symmetric:
        right_bone, left_bone = bone_pair_sym[0], bone_pair_sym[1]
        right_bone_start      = right_bone[0]
        right_bone_end        = right_bone[1]
        left_bone_start       = left_bone[0]
        left_bone_end         = left_bone[1]
        index                 = ([joints_names.index(right_bone_start), joints_names.index(right_bone_end)],
                                 [joints_names.index(left_bone_start), joints_names.index(left_bone_end)])
        bone_pairs_symmetric_indexes.append(index)

    number_of_joints = len(joints_names)
    lhip_idx         = joints_names.index('LeftUpLeg')
    rhip_idx         = joints_names.index('RightUpLeg')
    neck_idx         = joints_names.index('Neck')
    pelvis_idx       = joints_names.index('Hips')
    head_idx         = joints_names.index('Head')
    return number_of_joints, bones_pairs, rhip_idx, lhip_idx, neck_idx, pelvis_idx, bone_pairs_symmetric_indexes, \
           ll_bone_idx, rl_bone_idx, lh_bone_idx, rh_bone_idx, torso_bone_idx, head_idx

"""
In the paper of PoseAug: A Differentiable Pose Augmentation Framework for 3D Human Pose Estimation

# J0 - Pelvis
# J1 - Right Hip
# J2 - Right Knee
# J3 - Right Ankle
# J4 - Left Hip
# J5 - Left Knee
# J6 - Left Ankle
# J7 - Spine
# J8 - Neck
# J9 - Head
# J10 - Left Shoulder
# J11 - Left Elbow
# J12 - Left Wrist
# J13 - Right Shoulder
# J14 - Right Elbow
# J15 - Right Wrist

Ct = torch.Tensor([[1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B0:  J0 J1 -- Right Hip Bone
                   [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B1:  J1 J2 -- Right Knee Bone
                   [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B2:  J2 J3 -- Right Ankle Bone
                   [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B3:  J0 J4 -- Left Hip Bone
                   [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B4:  J4 J5 -- Left Knee Bone
                   [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B5:  J5 J6 -- Left Ankle Bone
                   [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # B6:  J0 J7 -- Pelvis-Spine Bone
                   [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],  # B7:  J7 J8 -- Spine-Neck Bone
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],  # B8:  J8 J9 -- Neck-Head Bone
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0],  # B9:  J8 J10 -- Neck-Left Shoulder Bone
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],  # B10: J10 J11 -- Left Arm Bone
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],  # B11: J11 J12 -- Left Hand Bone
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0],  # B12: J8 J13 -- Neck- Right Shoulder Bone
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],  # B13: J13 J14 -- Right Arm Bone
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],  # B14: J14 J15 -- Right Hand Bone
                   ])

# Right Hand - B7 (J7, J8) Spine-Neck --> B12 (J8, J13) --> B13 (J13, J14) --> B14 (J14, J15)
# Left Hand  - B7 (J7, J8) Spine-Neck --> B9 (J8, J10) --> B10 (J10, J11) --> B11 (J11, J12)
# Right Leg  - B0 (J0, J1) --> B1 (J1, J2)  --> B2 (J2, J3) --> B3 (J0, J4) Left Hip Bone  --> B6 (J0, J7) Pelvis-Spine Bone
# Left Leg   - B0 (J0, J1) Right Hip Bone --> B3 (J0, J4)  --> B4 (J4, J5) --> B5 (J5, J6) --> B6 (J0, J7) Pelvis-Spine Bone

# Torso      - B0 (J0, J1) Right Hip Bone --> B3 (J0, J4) Left Hip Bone --> B6 (J0, J7) Pelvis-Spine Bone --> B7 (J7, J8) Spine-Neck Bone
#               --> B8 (J8, J9) Neck-Head Bone --> B9 (J8 J10) Neck-Left Shoulder Bone --> B12 (J8, J13) Neck- Right Shoulder Bone


"""
