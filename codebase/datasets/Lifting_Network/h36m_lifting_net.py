import os

from datasets.dataset_base_lifting_net import Dataloader_for_encoder_training
from datasets.h36m_basics              import (get_labeled_samples, image_shapes, H36M_dataset, camera_names,
                                               joint_idxs_32to17_learnable, joint_idxs_32to17)
from torch.utils.data                  import DataLoader
from utils                             import json_read
from datasets.dataset_base             import get_sampler_train


class H36M_dataloader_for_lifting_network_single_frame(Dataloader_for_encoder_training):
    def get_pose_2d_mocap(self, subject, action, camera_name, idx):
        indexes = self.data[subject][action]['idx_frames']
        
        if idx not in indexes:
            return None
        
        pose_2d     = self.data[subject][action]['poses_2d'][camera_name][indexes.index(idx)]
        pose_2d     = pose_2d[self.joints_order]
        
        return pose_2d

    def get_shape(self, subject, seq, cam_id):
        return self.shapes[cam_id]

    def __init__(self,
                 pelvis_idx: int,
                 neck_idx: int,
                 lhip_idx: int,
                 rhip_idx: int,
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
                 ten_percent_3d_from_all : bool,
                 random_seed_for_ten_percent_3d_from_all: float,
                 inp_lifting_net_is_images: bool,
                 use_2D_GT_poses_directly: bool,
                 use_2D_mocap_poses_directly : bool,
                 joints_order: list,
                 json_file_3dv: bool
                 ):

        super(H36M_dataloader_for_lifting_network_single_frame, self).__init__(calibration_folder=calibration_folder,
                                                                               dataset=dataset, phase=phase,
                                                                               num_joints=num_joints,
                                                                               overfit=overfit,
                                                                               only_annotations=only_annotations,
                                                                               use_annotations=use_annotations,
                                                                               predictions_data=predictions_data,
                                                                               randomize=randomize,
                                                                               num_views=num_views,
                                                                               dataset_folder=dataset_folder,
                                                                               dataset_name=dataset_name,
                                                                               every_nth_frame=every_nth_frame,
                                                                               every_nth_frame_train_annotated=every_nth_frame_train_annotated,
                                                                               every_nth_frame_train_unannotated=every_nth_frame_train_unannotated,
                                                                               ten_percent_3d_from_all=ten_percent_3d_from_all,
                                                                               random_seed_for_ten_percent_3d_from_all=random_seed_for_ten_percent_3d_from_all,
                                                                               inp_lifting_net_is_images=inp_lifting_net_is_images,
                                                                               pelvis_idx=pelvis_idx, neck_idx=neck_idx,
                                                                               lhip_idx=lhip_idx, rhip_idx=rhip_idx,
                                                                               crop_size=(256, 256),
                                                                               use_2D_GT_poses_directly=use_2D_GT_poses_directly,
                                                                               use_2D_mocap_poses_directly=use_2D_mocap_poses_directly,
                                                                               joints_order=joints_order,
                                                                               json_file_3dv=json_file_3dv,
                                                                               test_set_in_world_coordinates=True)
        calibration_phase, data_phase, subject_ids_phase, samples_phase, \
            actions_phase, labels_phase = self.dataset.get_values(phase=phase.lower())
        self.data         = data_phase
        self.actions      = actions_phase
        self.shapes       = image_shapes
        self.camera_ids   = camera_names
        assert len(samples_phase) == len(labels_phase)
        samples_, labels_ = get_labeled_samples(samples_phase=samples_phase, labels_phase=labels_phase, phase=phase,
                                                every_nth_frame=self.every_nth_frame, overfit=False,
                                                ten_percent_3d_from_all=self.ten_percent_3d_from_all,
                                                every_nth_frame_train_annotated=self.every_nth_frame_train_annotated,
                                                every_nth_frame_train_unannotated=self.every_nth_frame_train_unannotated,
                                                only_annotations=False, randomize=False)
        prediction_keys_present, player_names_present, frame_nums_present, \
            labels_present, actions_present, samples_present, N = self.obtain_prediction_data_stats(samples_=samples_,
                                                                                                    labels_=labels_,
                                                                                                    frame_idx_in_sample=2)
        print("Total number of predictions found is {}/{}.".format(len(labels_present), N))
        self.__initialize__(prediction_keys_present=prediction_keys_present, player_names_present=player_names_present,
                            frame_nums_present=frame_nums_present, labels_present=labels_present,
                            actions_present=actions_present)
        self.func_labels()
        self.samples_present              = samples_present
        self.calibrations                 = calibration_phase
        self.number_of_cameras_considered = len(self.shapes.keys())
        self.subject_ids                  = subject_ids_phase
        self.image_path                   = '{}/{}/{}/imageSequence/{}/img_{:06d}.jpg'
        # '/cvlabsrc1/cvlab/dataset_H36M/h36m_orig_sampled/S1/Eating-1/imageSequence/54138969/'
        # <self.dataset_folder>/<subject_id>/<Action_id>/imageSequence/<cam_id>/img_{:06d}.jpg

    def get_image_file_path(self, subject_id, action_id, cam_id, frame_idx):
        image_file_path = self.image_path.format(self.dataset_folder, subject_id, action_id, cam_id, frame_idx)
        return image_file_path

    def get_calibration(self, cam_id, action_id, frame_id, subject_id):
        calib_values = self.calibrations[subject_id][cam_id]
        return calib_values


def get_train_set_h36_single_frame(dict_vals_train_h36m_single_view):
    """
    :param dict_vals_train_h36m_single_view: A dictionary of various values needed to create the Train Set of H36M for the training of Lifting Networks (ONLY)
    without any graphs (Thus the name single_frame).
    :return:
    """
    train_set_h36m_single_frame = H36M_dataloader_for_lifting_network_single_frame(**dict_vals_train_h36m_single_view)
    return train_set_h36m_single_frame



def get_test_set_h36m_single_frame(dict_vals_test_h36m_single_view):
    """
    :param dict_vals_test_h36m_single_view: A dictionary of various values needed to create the Test Set of H36M for the training of Lifting Networks (ONLY)
    without any graphs (Thus the name single_frame).
    :return:
    """

    test_set_h36m_single_frame = H36M_dataloader_for_lifting_network_single_frame(**dict_vals_test_h36m_single_view)
    return test_set_h36m_single_frame



def get_h36m_data_loaders_single_frame(config):
    """
    :param config: The Configuration File
    :return:
    """
    dataset_folder                          = '/cvlabsrc1/cvlab/dataset_H36M/h36m_orig_sampled/'
    calibration_folder                      = '/cvlabsrc1/cvlab/dataset_H36M/h36m_orig_sampled' if config.calibration_folder in ['', 'none', 'None', None] else config.calibration_folder
    use_annotations                         = True if config.experimental_setup in ['semi', 'fully', 'weakly'] else False
    only_annotations                        = True if (config.train_with_annotations_only is True and config.perform_test is False) else False
    # only_annotations                        = True if (config.train_with_annotations_only is True or config.pretraining_with_annotated_2D is True) else False
    path_cache                              = '/cvlabdata2/home/citraro/code/hpose/hpose/datasets' if config.path_cache_h36m in ['none', 'None', None, ''] else config.path_cache_h36m
    annotated_subjects                      = config.annotated_subjects
    unannotated_subjects                    = config.unannotated_subjects
    load_from_cache                         = config.load_from_cache
    shuffle                                 = config.shuffle
    randomize                               = config.randomize
    pelvis_idx                              = config.pelvis_idx
    neck_idx                                = config.neck_idx
    lhip_idx                                = config.lhip_idx
    rhip_idx                                = config.rhip_idx
    num_joints                              = config.n_joints
    overfit                                 = config.overfit_dataset
    num_views                               = 4
    dataset_name                            = config.dataset_name
    random_seed_for_ten_percent_3d_from_all = config.random_seed_for_ten_percent_3d_from_all
    ten_percent_3d_from_all                 = config.ten_percent_3d_from_all
    use_2D_GT_poses_directly                = config.use_2D_GT_poses_directly
    use_2D_mocap_poses_directly             = config.use_2D_mocap_poses_directly

    inp_lifting_net_is_images               = config.inp_lifting_net_is_images
    every_nth_frame                         = max(config.every_nth_frame_validation, config.every_nth_frame_test)
    every_nth_frame_train_annotated         = config.every_nth_frame_train_annotated
    every_nth_frame_train_unannotated       = config.every_nth_frame_train_unannotated
    training_subjects                       = annotated_subjects + unannotated_subjects
    training_subjects                       = list(set(training_subjects))
    
    assert config.type_of_2d_pose_model is not None
    
    resnet152_backbone                      = True if 'resnet152' in config.type_of_2d_pose_model else False
    
    if resnet152_backbone:
        print("Generating the 17 Keypoints according to the ResNet152 backbone by Iskakov paper.")
        joints_order = joint_idxs_32to17_learnable
    else:
        print("Generating the 17 Keypoints according to our method.")
        joints_order = joint_idxs_32to17

    training_subjects.sort()
    
    dataset = H36M_dataset(dataset_folder=dataset_folder, path_cache=path_cache, use_annotations=use_annotations,
                           calibration_folder=calibration_folder, use_annotations_only=use_annotations,
                           annotated_subjects=annotated_subjects, load_from_cache=load_from_cache,
                           training_subjects=training_subjects)

    # if not config.perform_test or (config.perform_test and (config.get_json_files_train_set or config.get_json_files_test_set
    #                                                         or config.plot_keypoints_from_learnt_model or config.plot_train_keypoints)):
    load_preds_data_train = (
        (not config.perform_test and not config.create_stats_dataset)
        or (config.perform_test and config.test_on_training_set)
        or (config.create_stats_dataset and not config.stats_dataset_from_test_set))
    
    if load_preds_data_train:
        print("Reading the Predictions of the Train Set of the H36M.")
        assert os.path.exists(config.predictions_data_train_file)
        predictions_data_train = json_read(config.predictions_data_train_file)
    else:
        print("Will not be reading the the Train Set for H36M dataset.")
        predictions_data_train = {}
    
    json_file_3dv_train              = "/cvlabdata2/home/soumava/codes/Step1/FN/preds-eccv-22" in config.predictions_data_train_file
    dict_vals_train_h36m_single_view = {'pelvis_idx'                              : pelvis_idx,
                                        'neck_idx'                                : neck_idx,
                                        'lhip_idx'                                : lhip_idx,
                                        'rhip_idx'                                : rhip_idx,
                                        'calibration_folder'                      : calibration_folder,
                                        'dataset'                                 : dataset ,
                                        'phase'                                   : 'train',
                                        'num_joints'                              : num_joints,
                                        'overfit'                                 : overfit,
                                        'only_annotations'                        : only_annotations,
                                        'use_annotations'                         : use_annotations,
                                        'predictions_data'                        : predictions_data_train,
                                        'randomize'                               : randomize,
                                        'num_views'                               : num_views,
                                        'dataset_folder'                          : dataset_folder,
                                        'dataset_name'                            : dataset_name,
                                        'every_nth_frame'                         : 0,
                                        'every_nth_frame_train_annotated'         : every_nth_frame_train_annotated,
                                        'every_nth_frame_train_unannotated'       : every_nth_frame_train_unannotated,
                                        'ten_percent_3d_from_all'                 : ten_percent_3d_from_all,
                                        'random_seed_for_ten_percent_3d_from_all' : random_seed_for_ten_percent_3d_from_all,
                                        'inp_lifting_net_is_images'               : inp_lifting_net_is_images,
                                        'use_2D_GT_poses_directly'                : use_2D_GT_poses_directly,
                                        'use_2D_mocap_poses_directly'             : use_2D_mocap_poses_directly,
                                        'joints_order'                            : joints_order,
                                        'json_file_3dv'                           : json_file_3dv_train
                                        }

    # if not config.perform_test or (
    #    config.perform_test and (config.get_json_files_train_set or config.get_json_files_test_set
    #    or config.plot_keypoints_from_learnt_model or config.plot_train_keypoints)):
    get_train_loader = (
        (not config.perform_test and not config.create_stats_dataset)
        or (config.perform_test and config.test_on_training_set)
        or (config.create_stats_dataset and not config.stats_dataset_from_test_set))
    if get_train_loader:
        print("Will be loading Training Set.")
                
        train_dataset                = get_train_set_h36_single_frame(dict_vals_train_h36m_single_view=dict_vals_train_h36m_single_view)
        num_samples_train            = len(train_dataset)
        
        train_labels                 = train_dataset.return_labels()
        num_cameras_considered_train = train_dataset.return_number_of_cameras()
        
        print("The number of Views Considered for Train is {}".format(num_cameras_considered_train))
        
        sampler_train, shuffle_train = get_sampler_train(use_annotations=use_annotations, shuffle=shuffle, labels=train_labels,
                                                         only_annotations=only_annotations, batch_size=config.batch_size,
                                                         randomize=randomize, extend_last_batch_graphs=False,
                                                         num_anno_samples_per_batch=config.num_anno_samples_per_batch,
                                                         num_samples=num_samples_train)
        
        train_loader                 = DataLoader(dataset=train_dataset, batch_size=config.batch_size, sampler=sampler_train,
                                                  shuffle=shuffle_train, num_workers=config.num_workers) 
        train_loader_wo_shuffle      = DataLoader(dataset=train_dataset, batch_size=config.batch_size, sampler=None, #sampler_train,
                                                  shuffle=False, num_workers=config.num_workers)
    else:
        print("Will not be loading Training Set.")
        train_loader = None
        train_loader_wo_shuffle = None

    assert os.path.exists(config.predictions_data_test_file)
    
    print("Reading the Predictions of the Test Set of the H36M.")
    
    predictions_data_test           = json_read(config.predictions_data_test_file)
    json_file_3dv_test              = "/cvlabdata2/home/soumava/codes/Step1/FN/preds-eccv-22" in config.predictions_data_test_file
    
    dict_vals_test_h36m_single_view = {'pelvis_idx'                              : pelvis_idx,
                                       'neck_idx'                                : neck_idx,
                                       'lhip_idx'                                : lhip_idx,
                                       'rhip_idx'                                : rhip_idx,
                                       'calibration_folder'                      : calibration_folder,
                                       'dataset'                                 : dataset ,
                                       'phase'                                   : 'test',
                                       'num_joints'                              : num_joints,
                                       'overfit'                                 : False,
                                       'only_annotations'                        : True,
                                       'use_annotations'                         : True,
                                       'predictions_data'                        : predictions_data_test,
                                       'randomize'                               : False,
                                       'num_views'                               : num_views,
                                       'dataset_folder'                          : dataset_folder,
                                       'dataset_name'                            : dataset_name,
                                       'every_nth_frame'                         : every_nth_frame,
                                       'every_nth_frame_train_annotated'         : 0 ,
                                       'every_nth_frame_train_unannotated'       : 0,
                                       'ten_percent_3d_from_all'                 : False,
                                       'random_seed_for_ten_percent_3d_from_all' : 0,
                                       'inp_lifting_net_is_images'               : inp_lifting_net_is_images,
                                       'use_2D_GT_poses_directly'                : use_2D_GT_poses_directly,
                                       'use_2D_mocap_poses_directly'             : use_2D_mocap_poses_directly,
                                       'joints_order'                            : joints_order,
                                       'json_file_3dv'                           : json_file_3dv_test
                                       }
    
    validation_dataset          = get_test_set_h36m_single_frame(dict_vals_test_h36m_single_view=dict_vals_test_h36m_single_view)
    num_cameras_considered_test = validation_dataset.return_number_of_cameras()
    
    print("The number of Views Considered for Test is {}".format(num_cameras_considered_test))
    
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=config.batch_size_test, sampler=None,
                                   shuffle=False, num_workers=config.num_workers)
    
    test_loader       = DataLoader(dataset=validation_dataset, batch_size=config.batch_size_test,
                                   sampler=None, shuffle=False, num_workers=config.num_workers)
    
    if train_loader is not None:
        print("The Batch Size of the Training Set is {}".format(config.batch_size))
        print("The number of training batches is {}.".format(len(train_loader)))

    if validation_loader is not None:
        print("The Batch Size of the Validation/Test set is {}".format(config.batch_size_test))
        print("The number of test/validation batches is {}.".format(len(validation_loader)))

    mpjpe_poses_in_camera_coordinates = False
    
    return train_loader, validation_loader, test_loader, train_loader_wo_shuffle, mpjpe_poses_in_camera_coordinates
