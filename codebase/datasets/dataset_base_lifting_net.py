import os.path

import numpy as np
import random
import torch
import cv2
from tqdm             import tqdm
from torch.utils.data import Dataset
from utils            import (MPJPE, NMPJPE, PMPJPE, key_val_2d_info, area, function_to_extend_dim, im_to_torch,
                              compute_intersection)
from torchvision      import transforms
from operator         import itemgetter


which_keys_for_lifting_net = ['bboxes', 'keypoints_det', 'keypoints_det_norm', 'pelvis_cam_z', 'labels', 'target_pose_3d',
                              'triangulated_pose_3d', 'R', 't', 'K', 'dist', 'consider', 'target_pose_2d', 'target_pose_2d_norm',
                              'target_root_rel_depth', 'triangulated_root_rel_depth', 'lifting_net_images',
                              'target_pose_3d_camera_coordinate' , 'triangulated_pose_3d_camera_coordinate']
all_keys_lifting_net       = ['bboxes', 'keypoints_det', 'keypoints_det_norm', 'pelvis_cam_z', 'labels',
                              'target_pose_3d', 'triangulated_pose_3d', 'R', 't', 'K', 'dist', 'consider',
                              'target_pose_2d', 'target_pose_2d_norm', 'image_paths', 'camera_ids', 'subject_ids',
                              'action_ids', 'frame_ids', 'target_root_rel_depth', 'triangulated_root_rel_depth',
                              'lifting_net_images', 'target_pose_3d_camera_coordinate' , 'triangulated_pose_3d_camera_coordinate']


class Dataloader_for_encoder_training(Dataset):
    def return_keys_in_dataloader(self):
        return self.keys_to_return, self.graph_key

    def get_label_windows_pred_frame(self):
        return self.label_windows_pred_frame

    def _get_samples(self):
        return self.samples_present

    def _get_labels(self):
        return self.labels_present

    def _obtain_subjects(self):
        return self.subject_ids

    def __init__(self,
                 calibration_folder : str,
                 dataset,
                 phase : str,
                 num_joints : int,
                 overfit : bool,
                 only_annotations : bool,
                 use_annotations : bool,
                 predictions_data,
                 randomize : bool,
                 num_views : int,
                 dataset_folder : str,
                 dataset_name : str,
                 every_nth_frame : int,
                 every_nth_frame_train_annotated : int,
                 every_nth_frame_train_unannotated : int,
                 ten_percent_3d_from_all: bool,
                 random_seed_for_ten_percent_3d_from_all: float,
                 inp_lifting_net_is_images : bool,
                 pelvis_idx: int,
                 neck_idx: int,
                 lhip_idx: int,
                 rhip_idx: int,
                 crop_size: tuple,
                 use_2D_GT_poses_directly : bool,
                 use_2D_mocap_poses_directly : bool,
                 joints_order : list,
                 json_file_3dv : bool,
                 test_set_in_world_coordinates : bool
                 ):
        """
        A Base Dataset for Loading the Predictions of a 2D Pose Estimator Model and Training of a Single View 2D to 3D Pose Lifter Network.
        :param calibration_folder                : The Calibration Folder.
        :param dataset                           : The Dataset used in the experiment.
        :param phase                             : The current phase of Learning, i.e. train or test/validation.
        :param num_joints                        : The Number of Joints used in the experiment.
        :param overfit                           : If True, we will be training only on one sample of dataset to overfit on it.
        :param only_annotations                  : If True, we will only be using the annotated samples.
        :param use_annotations                   : If True, the dataset will consist of both annotated and unannotated samples. If False, all the samples will be unannotated.
        :param predictions_data                  : The 2D predictions obtained after training the 2D pose Estimator Model.
        :param randomize                         : If True, we will randomize the dataset.
        :param num_views                         : The number of Views used in the experiment. It should be 1.
        :param dataset_folder                    : The folder containing the dataset.
        :param dataset_name                      : The Name of the Dataset used in the experiment.
        :param every_nth_frame                   : The Sampling Rate for the test/validation images.
        :param every_nth_frame_train_annotated   : The Sampling Rate for Annotated Training Samples.
        :param every_nth_frame_train_unannotated : The Sampling Rate for UnAnnotated Training Samples.
        """
        self.dataset                                 = dataset
        self.phase                                   = phase.lower()
        self.num_joints                              = num_joints
        self.randomize                               = randomize
        self.overfit                                 = overfit
        self.only_annotations                        = only_annotations
        self.use_annotations                         = use_annotations
        self.predictions_data                        = predictions_data
        self.calibration_folder                      = calibration_folder
        self.pelvis_idx                              = pelvis_idx
        self.neck_idx                                = neck_idx
        self.lhip_idx                                = lhip_idx
        self.rhip_idx                                = rhip_idx
        self.area_overlap_threshold                  = 0.7
        self.shapes                                  = None
        self.labels                                  = None
        self.ground_truth_3D                         = None
        self.DLT_3D                                  = None
        self.using_graphs                            = False
        self.samples_present                         = None
        self.subject_ids                             = None
        self.data                                    = [] # TODO CHECK
        self.num_windows                             = 0
        self.dataset_name                            = dataset_name
        self.num_views                               = num_views
        self.dataset_folder                          = dataset_folder
        self.file_paths_all                          = []
        self.inp_lifting_net_is_images               = inp_lifting_net_is_images
        self.key_val_2d_info                         = key_val_2d_info
        self.number_of_cameras_considered            = 0
        self.keys_to_return                          = []
        self.graph_key                               = ''
        self.label_windows_pred_frame                = []
        self.every_nth_frame                         = every_nth_frame
        self.every_nth_frame_train_annotated         = every_nth_frame_train_annotated
        self.every_nth_frame_train_unannotated       = every_nth_frame_train_unannotated
        self.ten_percent_3d_from_all                 = ten_percent_3d_from_all
        self.random_seed_for_ten_percent_3d_from_all = random_seed_for_ten_percent_3d_from_all
        self.crop_size                               = crop_size
        self.use_2D_mocap_poses_directly             = use_2D_mocap_poses_directly
        self.use_2D_GT_poses_directly                = use_2D_GT_poses_directly
        self.camera_ids                              = []
        self.json_file_3dv                           = json_file_3dv
        self.test_set_in_world_coordinates           = test_set_in_world_coordinates

        if self.json_file_3dv:
            print("We are using the JSON FILES of the 3DV submissions in {} phase.".format(self.phase))
        else:
            print("We are using the JSON FILES obtained currently in {} phase.".format(self.phase))

        if self.use_2D_GT_poses_directly:
            assert self.use_2D_mocap_poses_directly is False
            print("The input to the Lifting Net are the 2D Ground Truth Keypoints.")
        elif self.use_2D_mocap_poses_directly:
            assert self.use_2D_GT_poses_directly is False
            print("The input to the Lifting Net are the 2D Keypoints captured by Mocap systems.")
        else:
            print("The input to the Lifting Net are the 2D Detected Keypoints.")

        self.joints_order = joints_order

        if self.phase.lower() in ['training', 'train']:
            print("Obtaining the Transformations for the Train Phase for Lifting network only.")
            self.transforms = transforms.Compose([transforms.Resize(size=self.crop_size),
                                                  transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                         hue=0.07),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                  ])
        else:
            print("Obtaining the Transformations for the Test/Validation Phase for Lifting network only.")
            self.transforms = transforms.Compose([transforms.Resize(size=self.crop_size),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                  ])
            
    def calculate_3d_mpjpe(self):
        GT_3D  = np.array(self.ground_truth_3D)
        TRI_3D = np.asarray(self.DLT_3D)
        preds  = []; targets = []
        N      = len(GT_3D)
        for i in range(N):
            pred_i = TRI_3D[i]
            tar_i  = GT_3D[i]
            pred_i = pred_i[~np.isnan(tar_i).any(axis=1)]
            tar_i  = tar_i[~np.isnan(tar_i).any(axis=1)]
            preds.append(pred_i)
            targets.append(tar_i)

        score_3d_mpjpe  = MPJPE(preds_=preds, targets_=targets)
        score_3d_mpjpe  = np.mean(score_3d_mpjpe) * 100.0
        print("MPJPE = {:.4f}, ".format(score_3d_mpjpe), end='  ')
        score_3d_nmpjpe = NMPJPE(preds_=preds, targets_=targets)
        score_3d_nmpjpe = np.mean(score_3d_nmpjpe) * 100.0
        print("NMPJPE = {:.4f}, ".format(score_3d_nmpjpe), end='  ')
        score_3d_pmpjpe = PMPJPE(preds_=preds, targets_=targets)
        score_3d_pmpjpe = np.mean(score_3d_pmpjpe) * 100.0
        print("PMPJPE = {:.4f}, ".format(score_3d_pmpjpe))
        print("DONE")

    def return_labels(self):
        return self.labels_present

    def get_norm_2d_pose(self, pose_2d, bbox: list):
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        crop_width   = (xmax - xmin)# + 1e-5
        crop_height  = (ymax - ymin)# + 1e-5
        area_A       = (xmax - xmin) * (ymax - ymin)
        if area_A == 0:
            pose_2d_norm = np.ones((self.num_joints, 2)).astype(pose_2d.dtype)
        else:
            crop_size    = np.array([crop_width, crop_height])
            crop_shift   = np.array([xmin, ymin])
            pose_2d_norm = ((pose_2d - crop_shift) / crop_size) * 2 - 1.0
        return pose_2d_norm

    def __initialize__(self, prediction_keys_present, player_names_present, frame_nums_present, labels_present, actions_present):
        print("The Number of Prediction Keys present is {}".format(len(prediction_keys_present)))
        print("The Number of Players present is {}".format(len(player_names_present)))
        print("The Number of Frames present is {}".format(len(frame_nums_present)))
        print("The Number of Labels present is {}".format(len(labels_present)))
        print("The Number of Actions present is {}".format(len(actions_present)))

        if self.phase.lower() == 'train' and self.only_annotations:
            assert self.use_annotations is True
            print("The model will be pretrained only on the subjects of the annotated samples.")
            idx                     = np.where(np.array(labels_present) == 1)[0]
            prediction_keys_present = list(itemgetter(*idx)(prediction_keys_present))
            player_names_present    = list(itemgetter(*idx)(player_names_present))
            frame_nums_present      = list(itemgetter(*idx)(frame_nums_present))
            labels_present          = list(itemgetter(*idx)(labels_present))
            actions_present         = list(itemgetter(*idx)(actions_present))

        if self.randomize:
            print("Randomizing the {} dataset ".format(self.phase))
            c = list(zip(player_names_present, frame_nums_present, prediction_keys_present, labels_present, actions_present))
            random.shuffle(c)
            player_names_present, frame_nums_present, prediction_keys_present, labels_present, actions_present = zip(*c)

        if self.overfit:
            player_names_present    = player_names_present[0:1] * 10000
            frame_nums_present      = frame_nums_present[0:1] * 10000
            prediction_keys_present = prediction_keys_present[0:1] * 10000
            labels_present          = labels_present[0] * 10000
            actions_present         = actions_present[0] * 10000

        self.prediction_keys_present = prediction_keys_present
        self.labels_present          = labels_present
        self.player_names_present    = player_names_present
        self.frame_nums_present      = frame_nums_present
        self.actions_present         = actions_present

    def __len__(self):
        if not self.using_graphs:
            num_keys_present = len(self.prediction_keys_present)
            return num_keys_present
        else:
            return self.num_windows

    def func_labels(self):
        idx_anno   = np.where(np.array(self.labels_present) == 1)[0]
        idx_unanno = np.where(np.array(self.labels_present) == 0)[0]
        num_anno   = idx_anno.size
        num_unanno = idx_unanno.size
        num_labels = len(self.labels_present)
        if self.phase.lower() == 'train':
            if self.use_annotations:
                print("Number of Annotated Samples in the ##TRAIN## phase are {}".format(num_anno))
                if self.only_annotations:
                    assert num_anno == num_labels
                    assert num_unanno == 0
                    print("No UnAnnotated Samples are present in the ##TRAIN## Phase.")
                else:
                    print("Number of UnAnnotated Samples in the ##TRAIN## phase are {}".format(num_unanno))
            else:
                print("All the samples in the ##TRAIN## set will be considered as UnAnnotated")
        else:
            assert num_unanno == 0
            assert num_anno == num_labels
            print("All samples in the ##TEST## Phase are Annotated.")
        assert (num_unanno + num_anno) == num_labels
        print("Total Number of subjects for {} are {}".format(self.phase, num_labels))

    def get_consider_value(self, bboxA, bboxB):
        xmin, ymin, xmax, ymax = int(bboxA[0]), int(bboxA[1]), int(bboxA[2]), int(bboxA[3])
        area_A                 = (xmax - xmin) * (ymax - ymin)
        if area_A == 0:
            return 0
        else:
            area_overlapped        = float(compute_intersection(bboxA, [bboxB])[0]) #area(boxA=bboxA, boxB=bboxB) / float(area_A)
            if area_overlapped < self.area_overlap_threshold:
                return 0
            else:
                return 1

    @staticmethod
    def get_pose_2d_gt_from_pose_3d(pose_3d, rvec, t, K, dist):
        pose_2d = cv2.projectPoints(pose_3d, rvec, t, K, dist)[0].reshape(-1, 2)
        return pose_2d

    def get_pose_2d_mocap(self, subject, action, camera_name, idx):
        pass

    def get_shape(self, subject, seq, cam_id):
        return ()

    def get_image_file_path(self, subject_id, action_id, cam_id, frame_idx):
        return ''

    def __getitem__old(self, index):
        label_index       = self.labels_present[index]
        key_val_index     = self.prediction_keys_present[index]
        predictions_index = self.predictions_data[key_val_index]
        subject_index     = self.player_names_present[index]
        action_index      = self.actions_present[index]
        frame_index       = self.frame_nums_present[index]
        tar_3d_index      = np.array(predictions_index['3d-point-anno']).astype(float)
        dlt_3d_index      = np.array(predictions_index['3d-point-dlt']).astype(float)

        if np.any(np.isnan(dlt_3d_index)) or np.any(np.isinf(dlt_3d_index)):
            dlt_3d_index = np.ones_like(tar_3d_index).astype(float) - 100.0

        inp_images_lifting_net = []
        bboxes                 = []
        keypoints_det          = []
        keypoints_det_norm     = []
        target_pose_3d         = []
        triangulated_pose_3d   = []
        pelvis_cam_z           = []
        R, t, K, dist          = [], [], [], []
        target_pose_2d         = []
        target_pose_2d_norm    = []
        consider               = []
        image_paths                            = []
        camera_ids                             = []
        subject_ids                            = []
        action_ids                             = []
        frame_ids                              = []
        labels                                 = []
        target_root_rel_depth                  = []
        triangulated_root_rel_depth            = []
        triangulated_pose_3d_camera_coordinate = []
        target_pose_3d_camera_coordinate       = []

        for camera in self.camera_ids:
            camera_idx                       = self.camera_ids.index(camera)
            vid                              = 'ace_{}'.format(camera_idx)
            predictions_2d_info_index_camera = predictions_index[vid]
            calibration_index_camera         = self.get_calibration(cam_id=camera, action_id=action_index,
                                                                    frame_id=frame_index, subject_id=subject_index)
            R_index_camera    = np.array(calibration_index_camera['R']).astype(float)
            t_index_camera    = np.array(calibration_index_camera['t']).astype(float)
            K_index_camera    = np.array(calibration_index_camera['K']).astype(float)
            dist_index_camera = np.array(calibration_index_camera['dist']).astype(float)
            rvec_index_camera = cv2.Rodrigues(R_index_camera)[0]
            if self.use_2D_GT_poses_directly:
                det_2d_index_camera = self.get_pose_2d_gt_from_pose_3d(pose_3d=tar_3d_index, rvec=rvec_index_camera,
                                                                       t=t_index_camera, K=K_index_camera,
                                                                       dist=dist_index_camera)
            elif self.use_2D_mocap_poses_directly:
                det_2d_index_camera = self.get_pose_2d_mocap(subject=subject_index, action=action_index,
                                                             idx=frame_index, camera_name=camera)
            else:
                det_2d_index_camera = np.array(predictions_2d_info_index_camera["dist-keypoints-det"])

            tar_2d_index_camera      = self.get_pose_2d_gt_from_pose_3d(pose_3d=tar_3d_index, rvec=rvec_index_camera,
                                                                        t=t_index_camera, K=K_index_camera,
                                                                        dist=dist_index_camera)  # np.array(predictions_2d_info_index_camera['Tar_2D'])
            bbox_index_camera        = predictions_2d_info_index_camera['bbox']
            det_2d_norm_index_camera = self.get_norm_2d_pose(pose_2d=det_2d_index_camera, bbox=bbox_index_camera)
            tar_2d_norm_index_camera = self.get_norm_2d_pose(pose_2d=tar_2d_index_camera, bbox=bbox_index_camera)

            shape_camera           = self.get_shape(subject=subject_index, seq=action_index, cam_id=camera)  # self.shapes[camera]
            h_camera, w_camera     = shape_camera[0], shape_camera[1]
            xmin, ymin, xmax, ymax = int(bbox_index_camera[0]), int(bbox_index_camera[1]), int(bbox_index_camera[2]), int(bbox_index_camera[3])

            if self.inp_lifting_net_is_images:
                image_path_idx_camera = self.get_image_file_path(subject_id=subject_index, action_id=action_index,
                                                                 cam_id=camera, frame_idx=frame_index)
                if not os.path.exists(image_path_idx_camera):
                    image                 = torch.zeros(3, self.crop_size[0], self.crop_size[1])
                    consider_index_camera = 0
                else:
                    pil_image = cv2.cvtColor(cv2.imread(image_path_idx_camera), cv2.COLOR_BGR2RGB)
                    pil_image = pil_image[ymin:ymax, xmin:xmax]
                    pil_image = pil_image[ymin:ymax, xmin:xmax]
                    if pil_image.size != 0:
                        pil_image             = im_to_torch(pil_image)
                        image                 = self.transforms(pil_image)
                        consider_index_camera = self.get_consider_value(bboxA=bbox_index_camera,
                                                                        bboxB=[0, 0, h_camera, w_camera])
                    else:
                        image                 = torch.zeros(3, self.crop_size[0], self.crop_size[1])
                        consider_index_camera = 0
                inp_images_lifting_net.append(image)
            else:
                consider_index_camera = self.get_consider_value(bboxA=bbox_index_camera,
                                                                bboxB=[0, 0, h_camera, w_camera])
                image_path_idx_camera = ''
            tar_pose_3d_cam = np.dot(R_index_camera, tar_3d_index.T).T + t_index_camera.reshape(1, 3)
            dlt_pose_3d_cam = np.dot(R_index_camera, dlt_3d_index.T).T + t_index_camera.reshape(1, 3)
            if self.pelvis_idx != -1:
                pelvis_cam = tar_pose_3d_cam[self.pelvis_idx]
            else:
                pelvis_cam = np.mean([tar_pose_3d_cam[self.lhip_idx], tar_pose_3d_cam[self.rhip_idx]], axis=0)
            pelvis_cam_z_cam        = pelvis_cam[2]
            tar_root_relative_depth = tar_pose_3d_cam[:, 2] - pelvis_cam_z_cam
            dlt_root_relative_depth = dlt_pose_3d_cam[:, 2] - pelvis_cam_z_cam

            target_root_rel_depth.append(tar_root_relative_depth)
            triangulated_root_rel_depth.append(dlt_root_relative_depth)
            pelvis_cam_z.append(pelvis_cam_z_cam)
            target_pose_2d.append(tar_2d_index_camera)
            target_pose_2d_norm.append(tar_2d_norm_index_camera)
            keypoints_det.append(det_2d_index_camera)
            keypoints_det_norm.append(det_2d_norm_index_camera)
            bboxes.append(np.array(bbox_index_camera))
            R.append(R_index_camera)
            t.append(t_index_camera)
            K.append(K_index_camera)
            dist.append(dist_index_camera)
            consider.append(consider_index_camera)
            target_pose_3d.append(tar_3d_index)
            triangulated_pose_3d.append(dlt_3d_index)
            labels.append(label_index)
            triangulated_pose_3d_camera_coordinate.append(dlt_pose_3d_cam)
            target_pose_3d_camera_coordinate.append(tar_pose_3d_cam)

            image_paths.append(image_path_idx_camera)
            camera_ids.append(camera)
            subject_ids.append(subject_index)
            action_ids.append(action_index)
            frame_ids.append(frame_index)

        target_pose_2d       = torch.from_numpy(np.array(target_pose_2d)).float()
        target_pose_2d_norm  = torch.from_numpy(np.array(target_pose_2d_norm)).float()
        keypoints_det        = torch.from_numpy(np.array(keypoints_det)).float()
        keypoints_det_norm   = torch.from_numpy(np.array(keypoints_det_norm)).float()
        bboxes               = torch.from_numpy(np.array(bboxes)).float()
        target_pose_3d       = torch.from_numpy(np.array(target_pose_3d)).float()
        triangulated_pose_3d = torch.from_numpy(np.array(triangulated_pose_3d)).float()
        R, t, K, dist        = torch.from_numpy(np.array(R)).float(), torch.from_numpy(np.array(t)).float(), \
                               torch.from_numpy(np.array(K)).float(), torch.from_numpy(np.array(dist)).float()
        consider                                = torch.from_numpy(np.array(consider)).float()
        labels                                 = torch.from_numpy(np.array(labels)).float()
        pelvis_cam_z                           = torch.from_numpy(np.array(pelvis_cam_z)).float()
        target_root_rel_depth                  = torch.from_numpy(np.array(target_root_rel_depth)).float()
        triangulated_root_rel_depth            = torch.from_numpy(np.array(triangulated_root_rel_depth)).float()
        target_pose_3d_camera_coordinate       = torch.from_numpy(np.array(target_pose_3d_camera_coordinate)).float()
        triangulated_pose_3d_camera_coordinate = torch.from_numpy(np.array(triangulated_pose_3d_camera_coordinate)).float()
        ret_val = {'bboxes'                                 : bboxes,
                   'keypoints_det'                          : keypoints_det,
                   'keypoints_det_norm'                     : keypoints_det_norm,
                   'pelvis_cam_z'                           : pelvis_cam_z,
                   'labels'                                 : labels,
                   'target_pose_3d'                         : target_pose_3d,
                   'triangulated_pose_3d'                   : triangulated_pose_3d,
                   'R': R, 't': t, 'K': K, 'dist'           : dist,
                   'consider'                               : consider,
                   'target_pose_2d'                         : target_pose_2d,
                   'target_pose_2d_norm'                    : target_pose_2d_norm,
                   'image_paths'                            : image_paths,
                   'camera_ids'                             : camera_ids,
                   'subject_ids'                            : subject_ids,
                   'action_ids'                             : action_ids,
                   'frame_ids'                              : frame_ids,
                   'target_root_rel_depth'                  : target_root_rel_depth,
                   'triangulated_root_rel_depth'            : triangulated_root_rel_depth,
                   'target_pose_3d_camera_coordinate'       : target_pose_3d_camera_coordinate,
                   'triangulated_pose_3d_camera_coordinate' : triangulated_pose_3d_camera_coordinate
                   }
        if self.inp_lifting_net_is_images:
            inp_images_lifting_net        = torch.stack(inp_images_lifting_net).float()
            ret_val['lifting_net_images'] = inp_images_lifting_net.float()
        ret_val = function_to_extend_dim(data=ret_val, which_dim=1, which_keys=which_keys_for_lifting_net)
        return ret_val

    def __getitem__(self, index):
        if self.json_file_3dv:
            ret_val = self.__getitem__old(index=index)
            return ret_val

        label_index               = self.labels_present[index]
        key_val_index             = self.prediction_keys_present[index]
        predictions_index         = self.predictions_data[key_val_index]
        predictions_2d_info_index = predictions_index[self.key_val_2d_info]

        subject_index = self.player_names_present[index]
        action_index  = self.actions_present[index]
        frame_index   = self.frame_nums_present[index]
        tar_3d_index  = np.array(predictions_index['tar_3d'])
        dlt_3d_index  = np.array(predictions_index['dlt_3d'])

        assert subject_index == predictions_index['subject']
        assert action_index == predictions_index['action']
        assert frame_index == predictions_index['frame_id']

        inp_images_lifting_net      = []
        bboxes                      = []
        keypoints_det               = []
        keypoints_det_norm          = []
        target_pose_3d              = []
        triangulated_pose_3d        = []
        pelvis_cam_z                = []
        R, t, K, dist               = [], [], [], []
        target_pose_2d              = []
        target_pose_2d_norm         = []
        consider                    = []
        cameras                     = list(predictions_2d_info_index)

        image_paths                            = []
        camera_ids                             = []
        subject_ids                            = []
        action_ids                             = []
        frame_ids                              = []
        labels                                 = []
        target_root_rel_depth                  = []
        triangulated_root_rel_depth            = []
        triangulated_pose_3d_camera_coordinate = []
        target_pose_3d_camera_coordinate       = []

        for camera in cameras:
            predictions_2d_info_index_camera = predictions_2d_info_index[camera]
            calibration_index_camera         = self.get_calibration(cam_id=camera, action_id=action_index,
                                                                    frame_id=frame_index, subject_id=subject_index)
            R_index_camera                   = np.array(calibration_index_camera['R'])
            t_index_camera                   = np.array(calibration_index_camera['t'])
            K_index_camera                   = np.array(calibration_index_camera['K'])
            dist_index_camera                = np.array(calibration_index_camera['dist'])
            rvec_index_camera                = cv2.Rodrigues(R_index_camera)[0]
            if self.use_2D_GT_poses_directly:
                det_2d_index_camera = self.get_pose_2d_gt_from_pose_3d(pose_3d=tar_3d_index, rvec=rvec_index_camera,
                                                                       t=t_index_camera, K=K_index_camera,
                                                                       dist=dist_index_camera)
            elif self.use_2D_mocap_poses_directly:
                det_2d_index_camera = self.get_pose_2d_mocap(subject=subject_index, action=action_index,
                                                             idx=frame_index, camera_name=camera)
            else:
                det_2d_index_camera = np.array(predictions_2d_info_index_camera['Det_2D'])

            tar_2d_index_camera      = np.array(predictions_2d_info_index_camera['Tar_2D'])
            bbox_index_camera        = predictions_2d_info_index_camera['bbox']
            image_path_idx_camera    = predictions_2d_info_index_camera['Image_Path_2D']
            det_2d_norm_index_camera = self.get_norm_2d_pose(pose_2d=det_2d_index_camera, bbox=bbox_index_camera)
            tar_2d_norm_index_camera = self.get_norm_2d_pose(pose_2d=tar_2d_index_camera, bbox=bbox_index_camera)
            assert camera == predictions_2d_info_index_camera['Camera_ID']
            shape_camera             = self.get_shape(subject=subject_index, seq=action_index, cam_id=camera) # self.shapes[camera]
            h_camera, w_camera       = shape_camera[0], shape_camera[1]
            xmin, ymin, xmax, ymax   = int(bbox_index_camera[0]), int(bbox_index_camera[1]), int(bbox_index_camera[2]), int(bbox_index_camera[3])

            if self.inp_lifting_net_is_images:
                pil_image = cv2.cvtColor(cv2.imread(image_path_idx_camera), cv2.COLOR_BGR2RGB)
                pil_image = pil_image[ymin:ymax, xmin:xmax]
                pil_image = pil_image[ymin:ymax, xmin:xmax]
                if pil_image.size != 0:
                    pil_image             = im_to_torch(pil_image)
                    image                 = self.transforms(pil_image)
                    consider_index_camera = self.get_consider_value(bboxA=bbox_index_camera, bboxB=[0, 0, h_camera, w_camera])
                else:
                    image = torch.zeros(3, self.crop_size[0], self.crop_size[1])
                    consider_index_camera = 0
                inp_images_lifting_net.append(image)
            else:
                consider_index_camera = self.get_consider_value(bboxA=bbox_index_camera, bboxB=[0, 0, h_camera, w_camera])

            tar_pose_3d_cam = np.dot(R_index_camera, tar_3d_index.T).T + t_index_camera.reshape(1, 3)
            dlt_pose_3d_cam = np.dot(R_index_camera, dlt_3d_index.T).T + t_index_camera.reshape(1, 3)
            if self.pelvis_idx != -1:
                pelvis_cam = tar_pose_3d_cam[self.pelvis_idx]
            else:
                pelvis_cam = np.mean([tar_pose_3d_cam[self.lhip_idx], tar_pose_3d_cam[self.rhip_idx]], axis=0)
            pelvis_cam_z_cam        = pelvis_cam[2]
            tar_root_relative_depth = tar_pose_3d_cam[:, 2] - pelvis_cam_z_cam
            dlt_root_relative_depth = dlt_pose_3d_cam[:, 2] - pelvis_cam_z_cam

            target_root_rel_depth.append(tar_root_relative_depth)
            triangulated_root_rel_depth.append(dlt_root_relative_depth)
            pelvis_cam_z.append(pelvis_cam_z_cam)
            target_pose_2d.append(tar_2d_index_camera)
            target_pose_2d_norm.append(tar_2d_norm_index_camera)
            keypoints_det.append(det_2d_index_camera)
            keypoints_det_norm.append(det_2d_norm_index_camera)
            bboxes.append(np.array(bbox_index_camera))
            R.append(R_index_camera)
            t.append(t_index_camera)
            K.append(K_index_camera)
            dist.append(dist_index_camera)
            consider.append(consider_index_camera)
            target_pose_3d.append(tar_3d_index)
            triangulated_pose_3d.append(dlt_3d_index)
            labels.append(label_index)
            triangulated_pose_3d_camera_coordinate.append(dlt_pose_3d_cam)
            target_pose_3d_camera_coordinate.append(tar_pose_3d_cam)

            image_paths.append(image_path_idx_camera)
            camera_ids.append(camera)
            subject_ids.append(subject_index)
            action_ids.append(action_index)
            frame_ids.append(frame_index)

        target_pose_2d                         = torch.from_numpy(np.array(target_pose_2d)).float()
        target_pose_2d_norm                    = torch.from_numpy(np.array(target_pose_2d_norm)).float()
        keypoints_det                          = torch.from_numpy(np.array(keypoints_det)).float()
        keypoints_det_norm                     = torch.from_numpy(np.array(keypoints_det_norm)).float()
        bboxes                                 = torch.from_numpy(np.array(bboxes)).float()
        target_pose_3d                         = torch.from_numpy(np.array(target_pose_3d)).float()
        triangulated_pose_3d                   = torch.from_numpy(np.array(triangulated_pose_3d)).float()
        R, t, K, dist                          = torch.from_numpy(np.array(R)).float(), torch.from_numpy(np.array(t)).float(), \
                                                 torch.from_numpy(np.array(K)).float(), torch.from_numpy(np.array(dist)).float()
        consider                               = torch.from_numpy(np.array(consider)).float()
        labels                                 = torch.from_numpy(np.array(labels)).float()
        pelvis_cam_z                           = torch.from_numpy(np.array(pelvis_cam_z)).float()
        target_root_rel_depth                  = torch.from_numpy(np.array(target_root_rel_depth)).float()
        triangulated_root_rel_depth            = torch.from_numpy(np.array(triangulated_root_rel_depth)).float()
        target_pose_3d_camera_coordinate       = torch.from_numpy(np.array(target_pose_3d_camera_coordinate)).float()
        triangulated_pose_3d_camera_coordinate = torch.from_numpy(np.array(triangulated_pose_3d_camera_coordinate)).float()
        ret_val = {'bboxes'                                 : bboxes,
                   'keypoints_det'                          : keypoints_det,
                   'keypoints_det_norm'                     : keypoints_det_norm,
                   'pelvis_cam_z'                           : pelvis_cam_z,
                   'labels'                                 : labels,
                   'target_pose_3d'                         : target_pose_3d,
                   'triangulated_pose_3d'                   : triangulated_pose_3d,
                   'R': R, 't': t, 'K': K, 'dist'           : dist,
                   'consider'                               : consider,
                   'target_pose_2d'                         : target_pose_2d,
                   'target_pose_2d_norm'                    : target_pose_2d_norm,
                   'image_paths'                            : image_paths,
                   'camera_ids'                             : camera_ids,
                   'subject_ids'                            : subject_ids,
                   'action_ids'                             : action_ids,
                   'frame_ids'                              : frame_ids,
                   'target_root_rel_depth'                  : target_root_rel_depth,
                   'triangulated_root_rel_depth'            : triangulated_root_rel_depth,
                   'target_pose_3d_camera_coordinate'       : target_pose_3d_camera_coordinate,
                   'triangulated_pose_3d_camera_coordinate' : triangulated_pose_3d_camera_coordinate
                   }
        if self.inp_lifting_net_is_images:
            inp_images_lifting_net        = torch.stack(inp_images_lifting_net).float()
            ret_val['lifting_net_images'] = inp_images_lifting_net.float()
        ret_val = function_to_extend_dim(data=ret_val, which_dim=1, which_keys=which_keys_for_lifting_net)
        return ret_val


    def obtain_mv_samples_sequences(self, subject_idx: str, seq_idx: str, cameras : list,
                                    frames_window: list, labels_window: list, target_frame_window,
                                    time_pred):
        key_val = "{}-{}-{}"
        keys_all_present_within_window = True
        for frame_idx, label_idx in zip(frames_window, labels_window):
            key_val_frame = key_val.format(subject_idx, seq_idx, frame_idx)
            if key_val_frame not in self.prediction_keys_present:
                keys_all_present_within_window = False
                break

        if not keys_all_present_within_window:
            return None

        labels                                 = []
        inp_images_lifting_net                 = []
        bboxes                                 = []
        keypoints_det                          = []
        keypoints_det_norm                     = []
        target_pose_3d                         = []
        triangulated_pose_3d                   = []
        pelvis_cam_z                           = []
        R, t, K, dist                          = [], [], [], []
        target_pose_2d                         = []
        target_pose_2d_norm                    = []
        consider                               = []
        target_root_rel_depth                  = []
        triangulated_root_rel_depth            = []
        target_pose_3d_camera_coordinate       = []
        triangulated_pose_3d_camera_coordinate = []

        id_predicted_frame                     = []
        image_paths_predicted_frame            = []
        camera_ids_predicted_frame             = []
        action_predicted_frame                 = []
        subject_predicted_frame                = []
        count_frames                           = -1

        for frame_idx, label_idx in zip(frames_window, labels_window):
            count_frames                += 1  # This counting starts from 0 here.
            key_val_frame                = key_val.format(subject_idx, seq_idx, frame_idx)
            predictions_frame            = self.predictions_data[key_val_frame]
            tar_3d_frame                 = np.array(predictions_frame['tar_3d'])
            dlt_3d_frame                 = np.array(predictions_frame['dlt_3d'])
            predictions_2d_info_frame    = predictions_frame[self.key_val_2d_info]
            inp_images_lifting_net_frame = []
            bboxes_frame                 = []
            keypoints_det_frame          = []
            keypoints_det_norm_frame     = []
            target_pose_3d_frame         = []
            triangulated_pose_3d_frame   = []
            pelvis_cam_z_frame           = []
            R_frame, t_frame, K_frame, dist_frame        = [], [], [], []
            target_pose_2d_frame                         = []
            target_pose_2d_norm_frame                    = []
            consider_frame                               = []
            labels_frame                                 = []
            target_root_rel_depth_frame                  = []
            triangulated_root_rel_depth_frame            = []
            target_pose_3d_camera_coordinate_frame       = []
            triangulated_pose_3d_camera_coordinate_frame = []

            for camera in cameras:
                predictions_2d_info_frame_camera = predictions_2d_info_frame[camera]
                calibration_frame_camera         = self.get_calibration(cam_id=camera, action_id=seq_idx, frame_id=frame_idx,
                                                                        subject_id=subject_idx)
                R_frame_camera                   = np.array(calibration_frame_camera['R'])
                t_frame_camera                   = np.array(calibration_frame_camera['t'])
                K_frame_camera                   = np.array(calibration_frame_camera['K'])
                dist_frame_camera                = np.array(calibration_frame_camera['dist'])
                rvec_frame_camera                = cv2.Rodrigues(R_frame_camera)[0]
                if self.use_2D_GT_poses_directly:
                    det_2d_frame_camera = self.get_pose_2d_gt_from_pose_3d(pose_3d=tar_3d_frame, rvec=rvec_frame_camera,
                                                                           t=t_frame_camera, K=K_frame_camera,
                                                                           dist=dist_frame_camera)
                elif self.use_2D_mocap_poses_directly:
                    det_2d_frame_camera = self.get_pose_2d_mocap(subject=subject_idx, action=seq_idx, idx=frame_idx, camera_name=camera)
                else:
                    det_2d_frame_camera = np.array(predictions_2d_info_frame_camera['Det_2D'])


                tar_2d_frame_camera              = np.array(predictions_2d_info_frame_camera['Tar_2D'])
                bbox_frame_camera                = predictions_2d_info_frame_camera['bbox']
                image_path_frame_camera          = predictions_2d_info_frame_camera['Image_Path_2D']
                det_2d_norm_frame_camera         = self.get_norm_2d_pose(pose_2d=det_2d_frame_camera, bbox=bbox_frame_camera)
                tar_2d_norm_frame_camera         = self.get_norm_2d_pose(pose_2d=tar_2d_frame_camera, bbox=bbox_frame_camera)
                assert camera == predictions_2d_info_frame_camera['Camera_ID']
                shape_camera           = self.get_shape(subject=subject_idx, seq=seq_idx, cam_id=camera) #self.shapes[camera]
                h_camera, w_camera     = shape_camera[0], shape_camera[1]
                xmin, ymin, xmax, ymax = int(bbox_frame_camera[0]), int(bbox_frame_camera[1]), int(bbox_frame_camera[2]), int(bbox_frame_camera[3])

                if self.inp_lifting_net_is_images:
                    pil_frame_image = cv2.cvtColor(cv2.imread(image_path_frame_camera), cv2.COLOR_BGR2RGB)
                    pil_frame_image = pil_frame_image[ymin:ymax, xmin:xmax]
                    pil_frame_image = pil_frame_image[ymin:ymax, xmin:xmax]
                    if pil_frame_image.size != 0:
                        pil_frame_image       = im_to_torch(pil_frame_image)
                        frame_image           = self.transforms(pil_frame_image)
                        consider_frame_camera = self.get_consider_value(bboxA=bbox_frame_camera, bboxB=[0, 0, h_camera, w_camera])
                    else:
                        frame_image           = torch.zeros(3, self.crop_size[0], self.crop_size[1])
                        consider_frame_camera = 0
                    inp_images_lifting_net_frame.append(frame_image)
                else:
                    consider_frame_camera = self.get_consider_value(bboxA=bbox_frame_camera, bboxB=[0, 0, h_camera, w_camera])

                tar_pose_3d_frame_camera = np.dot(R_frame_camera, tar_3d_frame.T).T + t_frame_camera.reshape(1, 3)
                dlt_pose_3d_frame_camera = np.dot(R_frame_camera, dlt_3d_frame.T).T + t_frame_camera.reshape(1, 3)
                if self.pelvis_idx != -1:
                    pelvis_frame_camera = tar_pose_3d_frame_camera[self.pelvis_idx]
                else:
                    pelvis_frame_camera = np.mean([tar_pose_3d_frame_camera[self.lhip_idx], tar_pose_3d_frame_camera[self.rhip_idx]],
                                                  axis=0)
                pelvis_cam_z_frame_camera            = pelvis_frame_camera[2]
                tar_root_relative_depth_frame_camera = tar_pose_3d_frame_camera[:, 2] - pelvis_cam_z_frame_camera
                dlt_root_relative_depth_frame_camera = dlt_pose_3d_frame_camera[:, 2] - pelvis_cam_z_frame_camera


                if count_frames == time_pred : #count_frames in time_pred: # :  TODO-CHECK
                    assert (frame_idx == target_frame_window)
                    image_paths_predicted_frame.append(image_path_frame_camera)
                    camera_ids_predicted_frame.append(camera)
                    action_predicted_frame.append(seq_idx)
                    subject_predicted_frame.append(subject_idx)
                    id_predicted_frame.append(frame_idx)

                bboxes_frame.append(bbox_frame_camera)
                keypoints_det_frame.append(det_2d_frame_camera)
                keypoints_det_norm_frame.append(det_2d_norm_frame_camera)
                target_pose_3d_frame.append(tar_3d_frame)
                triangulated_pose_3d_frame.append(dlt_3d_frame)
                pelvis_cam_z_frame.append(pelvis_cam_z_frame_camera)
                R_frame.append(R_frame_camera)
                t_frame.append(t_frame_camera)
                K_frame.append(K_frame_camera)
                dist_frame.append(dist_frame_camera)
                target_pose_2d_frame.append(tar_2d_frame_camera)
                target_pose_2d_norm_frame.append(tar_2d_norm_frame_camera)
                consider_frame.append(consider_frame_camera)
                labels_frame.append(label_idx)
                target_root_rel_depth_frame.append(tar_root_relative_depth_frame_camera)
                triangulated_root_rel_depth_frame.append(dlt_root_relative_depth_frame_camera)

                target_pose_3d_camera_coordinate_frame.append(tar_pose_3d_frame_camera)
                triangulated_pose_3d_camera_coordinate_frame.append(dlt_pose_3d_frame_camera)

            if self.inp_lifting_net_is_images:
                inp_images_lifting_net_frame = torch.stack(inp_images_lifting_net_frame)
            inp_images_lifting_net.append(inp_images_lifting_net_frame)
            bboxes.append(bboxes_frame)
            keypoints_det.append(keypoints_det_frame)
            keypoints_det_norm.append(keypoints_det_norm_frame)
            target_pose_3d.append(target_pose_3d_frame)
            triangulated_pose_3d.append(triangulated_pose_3d_frame)
            pelvis_cam_z.append(pelvis_cam_z_frame)
            R.append(R_frame)
            t.append(t_frame)
            K.append(K_frame)
            dist.append(dist_frame)
            target_pose_2d.append(target_pose_2d_frame)
            target_pose_2d_norm.append(target_pose_2d_norm_frame)
            consider.append(consider_frame)
            target_root_rel_depth.append(target_root_rel_depth_frame)
            triangulated_root_rel_depth.append(triangulated_root_rel_depth_frame)
            labels.append(labels_frame)

            target_pose_3d_camera_coordinate.append(target_pose_3d_camera_coordinate_frame)
            triangulated_pose_3d_camera_coordinate.append(triangulated_pose_3d_camera_coordinate_frame)

        bboxes               = torch.from_numpy(np.array(bboxes)).float().transpose(0, 1)
        keypoints_det        = torch.from_numpy(np.array(keypoints_det)).float().transpose(0, 1)
        keypoints_det_norm   = torch.from_numpy(np.array(keypoints_det_norm)).float().transpose(0, 1)
        pelvis_cam_z         = torch.from_numpy(np.array(pelvis_cam_z)).float().transpose(0, 1)
        labels               = torch.from_numpy(np.array(labels)).float().transpose(0, 1)
        target_pose_3d       = torch.from_numpy(np.array(target_pose_3d)).float().transpose(0, 1)
        triangulated_pose_3d = torch.from_numpy(np.array(triangulated_pose_3d)).float().transpose(0, 1)
        R, t, K, dist        = torch.from_numpy(np.array(R)).float().transpose(0, 1), torch.from_numpy(np.array(t)).float().transpose(0, 1), \
                               torch.from_numpy(np.array(K)).float().transpose(0, 1), torch.from_numpy(np.array(dist)).float().transpose(0, 1)

        consider                               = torch.from_numpy(np.array(consider)).float().transpose(0, 1)
        target_pose_2d                         = torch.from_numpy(np.array(target_pose_2d)).float().transpose(0, 1)
        target_pose_2d_norm                    = torch.from_numpy(np.array(target_pose_2d_norm)).float().transpose(0, 1)
        target_root_rel_depth                  = torch.from_numpy(np.array(target_root_rel_depth)).float().transpose(0, 1)
        triangulated_root_rel_depth            = torch.from_numpy(np.array(triangulated_root_rel_depth)).float().transpose(0, 1)
        target_pose_3d_camera_coordinate       = torch.from_numpy(np.array(target_pose_3d_camera_coordinate)).float().transpose(0, 1)
        triangulated_pose_3d_camera_coordinate = torch.from_numpy(np.array(triangulated_pose_3d_camera_coordinate)).float().transpose(0, 1)
        ret_val                                = {'bboxes'                                 : bboxes,
                                                  'keypoints_det'                          : keypoints_det,
                                                  'keypoints_det_norm'                     : keypoints_det_norm,
                                                  'pelvis_cam_z'                           : pelvis_cam_z,
                                                  'labels'                                 : labels,
                                                  'target_pose_3d'                         : target_pose_3d,
                                                  'triangulated_pose_3d'                   : triangulated_pose_3d,
                                                  'R': R, 't': t, 'K': K, 'dist'           : dist,
                                                  'consider'                               : consider,
                                                  'target_pose_2d'                         : target_pose_2d,
                                                  'target_pose_2d_norm'                    : target_pose_2d_norm,
                                                  'image_paths'                            : image_paths_predicted_frame,
                                                  'camera_ids'                             : camera_ids_predicted_frame,
                                                  'subject_ids'                            : subject_predicted_frame,
                                                  'action_ids'                             : action_predicted_frame,
                                                  'frame_ids'                              : id_predicted_frame,
                                                  'target_root_rel_depth'                  : target_root_rel_depth,
                                                  'triangulated_root_rel_depth'            : triangulated_root_rel_depth,
                                                  'target_pose_3d_camera_coordinate'       : target_pose_3d_camera_coordinate,
                                                  'triangulated_pose_3d_camera_coordinate' : triangulated_pose_3d_camera_coordinate
                                                  }
        if self.inp_lifting_net_is_images:
            inp_images_lifting_net        = torch.stack(inp_images_lifting_net)
            inp_images_lifting_net        = inp_images_lifting_net.transpose(0, 1)
            ret_val['lifting_net_images'] = inp_images_lifting_net
        return ret_val


    @staticmethod
    def get_calibration(cam_id, action_id, frame_id, subject_id):
        return {'R' : np.array([]), 't' : np.array([]), 'K' : np.array([]), 'dist' : np.array([])}

    def return_number_of_cameras(self):
        return self.number_of_cameras_considered

    def obtain_prediction_data_stats(self, samples_, labels_, frame_idx_in_sample):
        N                       = len(labels_)
        keys_prediction_data    = list(self.predictions_data.keys())
        print("The number of keys present in the prediction data - {}, and for the samples is {}".format(len(keys_prediction_data), N))
        labels_present          = []
        prediction_keys_present = []
        frame_nums_present      = []
        player_names_present    = []
        actions_present         = []
        samples_present         = []
        keys_present, keys_absent = [], []
        for index in tqdm(range(N)): #range(N):
            sample_idx                         = samples_[index]
            subject_idx, action_idx, frame_idx = sample_idx[0], sample_idx[1], sample_idx[frame_idx_in_sample]
            label_idx                          = labels_[index]
            key_val_idx                        = ['{}'.format(frame_idx), '{}'.format(subject_idx), '{}'.format(action_idx)] \
                if self.json_file_3dv else ['{}'.format(subject_idx), '{}'.format(action_idx), '{}'.format(frame_idx)]
            key_val_idx                        = "-".join(key_val_idx)
            if key_val_idx in keys_prediction_data:
                prediction_keys_present.append(key_val_idx)
                labels_present.append(label_idx)
                frame_nums_present.append(frame_idx)
                player_names_present.append(subject_idx)
                actions_present.append(action_idx)
                samples_present.append(sample_idx)
                if key_val_idx not in keys_present:
                    keys_present.append(key_val_idx)
            else:
                if key_val_idx not in keys_present:
                    keys_absent.append(key_val_idx)
        return prediction_keys_present, player_names_present, frame_nums_present, labels_present, actions_present, samples_present, N


"""
class collate_func_lifting_net(object):
    def __init__(self, N_cameras, all_keys):
        self.N_cameras = N_cameras
        self.all_keys  = all_keys

    def __call__(self, batch):
        new_batch   = []
        for batch_item in batch:
            batch_item_keys = list(batch_item.keys())
            new_batch_      = [[]]*self.N_cameras
            for i in range(self.N_cameras):
                new_batch_[i] = {}
                for key in self.all_keys:
                    if key in batch_item_keys:
                        batch_item_key_val = batch_item[key]
                        new_batch_[i][key] = batch_item_key_val[i]
            new_batch.extend(new_batch_)
        return default_collate(new_batch)
"""
