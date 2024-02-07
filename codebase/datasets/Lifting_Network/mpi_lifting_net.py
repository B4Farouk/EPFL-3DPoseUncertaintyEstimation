import os
import h5py
import numpy as np
import cv2
import multiprocessing


from datasets.dataset_base_lifting_net import Dataloader_for_encoder_training
from torch.utils.data                  import DataLoader
from utils                             import json_read, bbox_from_points, compute_intersection
from datasets.dataset_base             import get_sampler_train
from datasets.mpi_basics               import (intrinsics_test_sequences, parse_camera_calibration, training_subjects_all,
                                               joint_idxs_28to17ours_learnable_h36m_train,
                                               joint_idxs_28to17_original_train, joint_idxs_28to17_according_to_h36m_train,
                                               joint_idxs_17to17_according_to_h36m_test, joint_idxs_17to17_original_test,
                                               joint_idxs_17to17ours_learnable_h36m_test, function_to_calculate_new_intrinsics)
from datasets.h36m_basics               import get_labeled_samples
from datasets.Pose_Estimator.mpi_simple import _prepare_poses_train_parallel


class MPI_Dataset(Dataloader_for_encoder_training):
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
        return pose_3d

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
                return None # cam_id, action_id, frame_id, subject_id
            R, t, K, dist = self.get_calibration(subject_id=subject, action_id=seq, cam_id=cam_id, frame_id=frame_idx,
                                                 return_dict=False)
            rvec          = cv2.Rodrigues(R)[0]
            pose_2d       = cv2.projectPoints(pose_3d, rvec, t, K, dist)[0].reshape(-1, 2)
        else:
            indexes = self.data[subject][seq]['idx_frames']
            if frame_idx not in indexes:
                return None
            pose_2d = self.data[subject][seq]['poses_2d'][cam_id][indexes.index(frame_idx)]
        return pose_2d

    def get_pose_2d_mocap(self, subject, action, camera_name, idx):
        camera_name = int(camera_name)
        cam_id      = camera_name
        seq         = action
        frame_idx   = idx
        indexes     = self.data[subject][seq]['idx_frames']
        if idx not in indexes:
            return None
        pose_2d = self.data[subject][seq]['poses_2d'][cam_id][indexes.index(frame_idx)]
        pose_2d = pose_2d[self.joints_order]
        return pose_2d

    def visibility(self, subject, seq, view, idx):
        """
        :param subject:
        :param seq:
        :param view:
        :param idx:
        :return:
        """
        pose                = self.get_pose_2d(subject, seq, view, idx, from_3d=True) #TODO
        bbox_for_visibility = bbox_from_points(pose, pb=0.05)
        shape               = self.get_shape(subject, seq, view)
        bbox_img            = [0, 0, shape[1], shape[0]]
        return float(compute_intersection(bbox_for_visibility, [bbox_img])[0])

    def get_shape(self, subject, seq, cam_id):
        cam_id      = int(cam_id)
        calibration = self.calibrations[subject][seq][cam_id]
        shape       = (int(calibration['height']), int(calibration['width']))
        return shape

    def __init__(self, base: str,
                 base_images : str,
                 min_visibility : float,
                 dict_vals_base):
        super(MPI_Dataset, self).__init__(**dict_vals_base)
        self.base           = base
        self.base_images    = base_images
        self.min_visibility = min_visibility
        self.calibrations   = {}
        self.data           = {}


    def get_calibration(self, cam_id, action_id, frame_id, subject_id, return_dict=True):
        """
        :param cam_id:
        :param action_id: This is same as seq_idx.
        :param frame_id:
        :param subject_id:
        :param return_dict:
        :return:
        """
        cam_id      = int(cam_id)
        calibration = self.calibrations[subject_id][action_id][cam_id]
        if return_dict :
            return calibration
        r           = np.array(calibration['R'])
        t           = np.array(calibration['t']).ravel()
        K           = np.array(calibration['K'])
        dist        = np.array(calibration['dist'])
        return r, t, K, dist



class MPI_INF_3DHP_train_lifting_net_single_frame(MPI_Dataset):
    def get_samples_mpi_train_set_lifting_net(self):
        if self.load_from_cache and os.path.isfile(self.filename_cached_samples):
            print("Loading samples from the cache.")
            samples_loaded = json_read(self.filename_cached_samples)
            print("Done Loading")
            samples_phase = []; key_vals_loaded = []; N_samples_loaded = len(samples_loaded); counter = 0; labels_phase = []
            reading_sample = 0
            for sample in samples_loaded:
                reading_sample += 1
                subject_idx, seq_idx, cam_idx, frame_idx = sample
                """
                key_val  = 'Sub{}-Seq{}-Frame{:05d}-{}'.format(subject_idx, seq_idx, frame_idx, cam_idx)
                counter += 1
                if subject_idx in self.annotated_subjects:
                    labels_phase.append(1)
                else:
                    labels_phase.append(0)

                if key_val not in key_vals_loaded:
                    key_vals_loaded.append(key_val)
                    sample = (subject_idx, seq_idx, frame_idx, cam_idx)  # This will be action_idx in function self.obtain_prediction_data_stats
                    samples_phase.append(sample)
                
                if counter % 10000 == 0 or counter == 1:
                    print("{}-{}".format(counter, N_samples_loaded))
                """
                key_val = 'Sub{}-Seq{}-Frame{:05d}'.format(subject_idx, seq_idx, frame_idx)
                if key_val not in key_vals_loaded:
                    if subject_idx in self.annotated_subjects:
                        labels_phase.append(1)
                    else:
                        labels_phase.append(0)
                    counter += 1
                    if counter % 10000 == 0 or counter == 1:
                        print("\n{} Key Loaded {}-{} \n".format("##"*10, counter, N_samples_loaded))

                    key_vals_loaded.append(key_val)
                    sample = (subject_idx, seq_idx, frame_idx, cam_idx) # This will be action_idx in function self.obtain_prediction_data_stats
                    samples_phase.append(sample)
                # """

                if reading_sample % 1000 == 0 or reading_sample == 1:
                    print("Have read {} samples.".format(reading_sample))

        else:
            raise ValueError("Not allowed")

        print("Before Sampling Total Samples loaded are {}".format(len(samples_phase)))
        assert len(samples_phase) == len(labels_phase)
        samples_train, labels_train = get_labeled_samples(samples_phase=samples_phase, labels_phase=labels_phase, phase='train',
                                                          every_nth_frame=self.every_nth_frame, overfit=False,
                                                          ten_percent_3d_from_all=self.ten_percent_3d_from_all,
                                                          every_nth_frame_train_annotated=self.every_nth_frame_train_annotated,
                                                          every_nth_frame_train_unannotated=self.every_nth_frame_train_unannotated,
                                                          only_annotations=False, randomize=False)
        return samples_train, labels_train


    def __init__(self,
                 base: str,
                 base_images: str,
                 min_visibility: float,
                 calibration_folder: str,
                 phase: str,
                 num_joints: int,
                 overfit: bool,
                 only_annotations: bool,
                 use_annotations: bool,
                 predictions_data,
                 randomize: bool,
                 num_views: int,
                 dataset_folder: str,
                 dataset_name: str,
                 every_nth_frame: int,
                 every_nth_frame_train_annotated: int,
                 every_nth_frame_train_unannotated: int,
                 ten_percent_3d_from_all: bool,
                 random_seed_for_ten_percent_3d_from_all: float,
                 inp_lifting_net_is_images: bool,
                 pelvis_idx: int,
                 neck_idx: int,
                 lhip_idx: int,
                 rhip_idx: int,
                 multiview_sample: bool,
                 annotated_subjects : list,
                 subjects : list,
                 only_chest_height_cameras : bool,
                 load_from_cache: bool,
                 path_cache: str,
                 use_2D_GT_poses_directly: bool,
                 use_2D_mocap_poses_directly: bool,
                 joints_order : list,
                 json_file_3dv : bool,
                 calculate_K   : bool
                 ):

        dict_vals_base = {'calibration_folder'                      : calibration_folder,
                          'dataset'                                 : None,
                          'phase'                                   : phase,
                          'num_joints'                              : num_joints,
                          'overfit'                                 : overfit,
                          'only_annotations'                        : only_annotations,
                          'use_annotations'                         : use_annotations,
                          'predictions_data'                        : predictions_data,
                          'randomize'                               : randomize,
                          'num_views'                               : num_views,
                          'dataset_folder'                          : dataset_folder,
                          'dataset_name'                            : dataset_name,
                          'every_nth_frame'                         : every_nth_frame,
                          'every_nth_frame_train_annotated'         : every_nth_frame_train_annotated,
                          'every_nth_frame_train_unannotated'       : every_nth_frame_train_unannotated,
                          'ten_percent_3d_from_all'                 : ten_percent_3d_from_all,
                          'random_seed_for_ten_percent_3d_from_all' : random_seed_for_ten_percent_3d_from_all,
                          'inp_lifting_net_is_images'               : inp_lifting_net_is_images,
                          'pelvis_idx'                              : pelvis_idx,
                          'neck_idx'                                : neck_idx,
                          'lhip_idx'                                : lhip_idx,
                          'rhip_idx'                                : rhip_idx,
                          'crop_size'                               : (256, 256),
                          'use_2D_mocap_poses_directly'             : use_2D_mocap_poses_directly,
                          'use_2D_GT_poses_directly'                : use_2D_GT_poses_directly,
                          'joints_order'                            : joints_order,
                          'json_file_3dv'                           : json_file_3dv,
                          'test_set_in_world_coordinates'           : True
                          }
        dict_vals_mpi = {'base': base, 'base_images': base_images, 'min_visibility': min_visibility,
                         'dict_vals_base': dict_vals_base}
        super(MPI_INF_3DHP_train_lifting_net_single_frame, self).__init__(**dict_vals_mpi)

        if len(subjects) == 0 or subjects is None:
            self.subjects = ['S{}'.format(i) for i in range(1, 9)]
        else:
            self.subjects = subjects

        self.annotated_subjects = [] if annotated_subjects is None else annotated_subjects
        print("The following subjects are used for training {}.".format(', '.join(self.subjects)))
        if annotated_subjects is not None:
            print("The following subjects are annotated {}".format(', '.join(self.annotated_subjects)))

        self.sequences   = ['Seq1', 'Seq2']
        print("The following sequences are used {}".format(', '.join(self.sequences)))
        self.multiview_sample            = multiview_sample
        self.every_nth_frame_annotated   = every_nth_frame_train_annotated
        self.every_nth_frame_unannotated = every_nth_frame_train_unannotated
        self.only_chest_height_cameras   = only_chest_height_cameras

        if self.only_chest_height_cameras:
            self.camera_ids = [0, 2, 4, 7, 8]
        else:
            self.camera_ids = [0, 1, 2, 4, 5, 6, 7, 8]

        self.number_of_cameras_considered = len(self.camera_ids)
        self.calibrations                 = {}
        self.calculate_K                  = calculate_K

        for subject in self.subjects:
            self.calibrations[subject] = {}
            for seq in self.sequences:
                with open(os.path.join(self.base, subject, seq, "camera.calibration"), "r") as f:
                    calibration = parse_camera_calibration(f)
                    if self.calculate_K:
                        print("Calculating the New K for {} in {} for the Lifting Network in the Train Set.".format(seq, subject))
                        calibration = function_to_calculate_new_intrinsics(seq=seq, subject=subject, base=self.base, old_calibration=calibration)
                    self.calibrations[subject][seq] = calibration

        self.load_from_cache         = load_from_cache
        filename_cached_samples      = os.path.join(path_cache, "MPI_INF_3DHP_train_samples_{}.json".format(training_subjects_all))
        self.filename_cached_samples = filename_cached_samples
        samples_, labels_            = self.get_samples_mpi_train_set_lifting_net()
        prediction_keys_present, player_names_present, frame_nums_present, \
            labels_present, actions_present, samples_present, N = self.obtain_prediction_data_stats(samples_=samples_,
                                                                                                    labels_=labels_,
                                                                                                    frame_idx_in_sample=2)
        self.subject_ids = self.subjects
        print("Total number of predictions found is {}/{}.".format(len(labels_present), N))
        self.__initialize__(prediction_keys_present=prediction_keys_present, player_names_present=player_names_present,
                            frame_nums_present=frame_nums_present, labels_present=labels_present,
                            actions_present=actions_present)
        self.func_labels()
        data            = self._prepare_data_train()
        self.data       = data
        self.image_path = '{}/{}/{}/imageSequence/video_{}/frames/frame_{:05d}.jpg'
        # '/cvlabsrc1/cvlab/dataset_mpi_inf_3dhp/S1/Seq1/imageSequence/video_0/frames'
        # <self.dataset_folder>/<subject_id>/<Action_id>/imageSequence/video_<cam_id>/frames/frame_{:05d}.jpg

    def get_image_file_path(self, subject_id, action_id, cam_id, frame_idx):
        image_file_path = self.image_path.format(self.dataset_folder, subject_id, action_id, cam_id, frame_idx)
        return image_file_path

    def _prepare_data_train(self):
        # self.data = self._prepare_data_train()
        print("Preparing the Training Dataset.")
        threads = 16
        inputs  = []
        for subject in self.subjects:
            for seq in self.sequences:
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



class MPI_INF_3DHP_test_lifting_net_single_frame(MPI_Dataset):

    def _prepare_data_test(self):
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
            data[subject] = {self.sequence: f(subject)}
            print("Loaded {} frames for subject:{}.".format(len(data[subject][self.sequence]['idx_frames']), subject))
        return data

    def __init__(self,
                 base: str,
                 base_images : str,
                 min_visibility : float,
                 calibration_folder: str,
                 phase: str,
                 num_joints : int,
                 overfit : bool,
                 only_annotations: bool,
                 use_annotations: bool,
                 predictions_data,
                 randomize: bool,
                 dataset_folder: str,
                 dataset_name: str,
                 every_nth_frame: int,
                 every_nth_frame_train_annotated: int,
                 every_nth_frame_train_unannotated: int,
                 ten_percent_3d_from_all: bool,
                 random_seed_for_ten_percent_3d_from_all: float,
                 inp_lifting_net_is_images: bool,
                 pelvis_idx: int,
                 neck_idx: int,
                 lhip_idx: int,
                 rhip_idx: int,
                 use_2D_GT_poses_directly: bool,
                 use_2D_mocap_poses_directly: bool,
                 joints_order : list,
                 frame_idx_in_sample_test: int,
                 json_file_3dv: bool,
                 calculate_K: bool
                 ):
        dict_vals_base = {'calibration_folder'                      : calibration_folder,
                          'dataset'                                 : None,
                          'phase'                                   : phase,
                          'num_joints'                              : num_joints,
                          'overfit'                                 : overfit,
                          'only_annotations'                        : only_annotations,
                          'use_annotations'                         : use_annotations,
                          'predictions_data'                        : predictions_data,
                          'randomize'                               : randomize,
                          'num_views'                               : 1,
                          'dataset_folder'                          : dataset_folder,
                          'dataset_name'                            : dataset_name,
                          'every_nth_frame'                         : every_nth_frame,
                          'every_nth_frame_train_annotated'         : every_nth_frame_train_annotated,
                          'every_nth_frame_train_unannotated'       : every_nth_frame_train_unannotated,
                          'ten_percent_3d_from_all'                 : ten_percent_3d_from_all,
                          'random_seed_for_ten_percent_3d_from_all' : random_seed_for_ten_percent_3d_from_all,
                          'inp_lifting_net_is_images'               : inp_lifting_net_is_images,
                          'pelvis_idx'                              : pelvis_idx,
                          'neck_idx'                                : neck_idx,
                          'lhip_idx'                                : lhip_idx,
                          'rhip_idx'                                : rhip_idx,
                          'crop_size'                               : (256, 256),
                          'use_2D_mocap_poses_directly'             : use_2D_mocap_poses_directly,
                          'use_2D_GT_poses_directly'                : use_2D_GT_poses_directly,
                          'joints_order'                            : joints_order,
                          'json_file_3dv'                           : json_file_3dv,
                          'test_set_in_world_coordinates'           : False
                          }

        dict_vals_mpi  = {'base' : base, 'base_images' : base_images, 'min_visibility' : min_visibility,
                          'dict_vals_base' : dict_vals_base}
        super(MPI_INF_3DHP_test_lifting_net_single_frame, self).__init__(**dict_vals_mpi)
        self.subjects                     = ['TS{}'.format(i) for i in range(1, 7)]
        self.sequence                     = 'seq'
        self.camera_ids                   = [0]
        calibrations                      = {}
        self.number_of_cameras_considered = len(self.camera_ids)
        for subject in self.subjects:
            K, dist, R, t, h, w                  = intrinsics_test_sequences(base=self.base, calculate_K=calculate_K, seq=subject)
            calibrations[subject]                = {}
            calibrations[subject][self.sequence] = {}
            calibrations[subject][self.sequence][self.camera_ids[0]] = {'R': R, 't': t, 'K': K, 'dist': dist, 'height': h, 'width': w}

        data              = self._prepare_data_test()
        samples_          = []
        self.data         = data
        self.calibrations = calibrations
        self.subject_ids  = self.subjects

        count_missing = 0
        for subject in self.subjects:
            print("subject - {}".format(subject))
            idx_frames = data[subject][self.sequence]['idx_frames']
            for idx in idx_frames:
                if self.visibility(subject=subject, seq=self.sequence, view=self.camera_ids[0], idx=idx) > min_visibility:
                    samples_.append((subject, self.sequence, self.camera_ids[0], idx))
                else:
                    count_missing += 1

        samples_ = samples_[::self.every_nth_frame]
        N        = len(samples_)
        labels_  = [1] * N
        print("Original {} samples are being loaded in {} phase".format(N, phase))
        prediction_keys_present, player_names_present, frame_nums_present, \
            labels_present, actions_present, samples_present, N = self.obtain_prediction_data_stats(samples_=samples_,
                                                                                                    labels_=labels_,
                                                                                                    frame_idx_in_sample=frame_idx_in_sample_test)#3
        print("Total number of predictions found is {}/{}.".format(len(labels_present), N))
        self.__initialize__(prediction_keys_present=prediction_keys_present, player_names_present=player_names_present,
                            frame_nums_present=frame_nums_present, labels_present=labels_present,
                            actions_present=actions_present)
        self.samples_present = samples_present
        self.image_path      = '{}/{}/imageSequence/img_{:06d}.jpg'
        # /cvlabsrc1/cvlab/MPI_INF_3D/mpi_inf_3dhp/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set/TS1/imageSequence
        # <self.dataset_folder>/<subject_id>/imageSequence/frame_{:05d}.jpg


    def get_image_file_path(self, subject_id, action_id, cam_id, frame_idx):
        image_file_path = self.image_path.format(self.dataset_folder, subject_id, frame_idx)
        return image_file_path



def get_train_set_mpi_single_frame(dict_vals_train_h36m_single_view):
    """
    :param dict_vals_train_h36m_single_view: A dictionary of various values needed to create the Train Set of MPI for the training of Lifting Networks (ONLY)
    without any graphs (Thus the name single_frame).
    :return:
    """
    train_set_mpi_single_frame = MPI_INF_3DHP_train_lifting_net_single_frame(**dict_vals_train_h36m_single_view)
    return train_set_mpi_single_frame



def get_test_set_mpi_single_frame(dict_vals_test_h36m_single_view):
    """
    :param dict_vals_test_h36m_single_view: A dictionary of various values needed to create the Test Set of MPI for the training of Lifting Networks (ONLY)
    without any graphs (Thus the name single_frame).
    :return:
    """

    test_set_mpi_single_frame = MPI_INF_3DHP_test_lifting_net_single_frame(**dict_vals_test_h36m_single_view)
    return test_set_mpi_single_frame



def get_mpi_data_loaders_single_frame(config):
    """
    :param config: The Configuration File
    :return:
    """
    print("\n\n")
    print('#####' * 20)

    subjects                    = config.annotated_subjects + config.unannotated_subjects
    only_chest_height_cameras   = config.only_chest_height_cameras
    multiview_sample            = True #config.multiview_sample
    min_visibility              = config.min_visibility
    annotated_subjects          = config.annotated_subjects
    load_from_cache             = config.load_from_cache
    use_annotations             = True if config.experimental_setup in ['semi', 'fully', 'weakly'] else False
    only_annotations            = True if (config.train_with_annotations_only is True and config.perform_test is False) else False
    # only_annotations            = True if (config.train_with_annotations_only is True or config.pretraining_with_annotated_2D is True) else False

    shuffle                           = config.shuffle
    randomize                         = config.randomize
    pelvis_idx                        = config.pelvis_idx
    neck_idx                          = config.neck_idx
    lhip_idx                          = config.lhip_idx
    rhip_idx                          = config.rhip_idx
    num_joints                        = config.number_of_joints
    overfit                           = config.overfit_dataset
    every_nth_frame                   = max(config.every_nth_frame_validation, config.every_nth_frame_test)
    every_nth_frame_train_annotated   = config.every_nth_frame_train_annotated
    every_nth_frame_train_unannotated = config.every_nth_frame_train_unannotated
    use_2D_GT_poses_directly          = config.use_2D_GT_poses_directly
    use_2D_mocap_poses_directly       = config.use_2D_mocap_poses_directly
    path_cache                        = '/cvlabdata2/home/citraro/code/hpose/hpose/datasets'
    base_train                        = '/cvlabsrc1/cvlab/MPI_INF_3D/mpi_inf_3dhp'
    base_train_images                 = "/cvlabsrc1/cvlab/dataset_mpi_inf_3dhp/"
    dataset_folder                    = base_train
    calibration_folder                = base_train if config.calibration_folder in ['', 'none', 'None', None] else config.calibration_folder
    calculate_K                       = config.calculate_K
    assert config.type_of_2d_pose_model is not None
    resnet152_backbone                = True if 'resnet152' in config.type_of_2d_pose_model else False
    if resnet152_backbone:
        print("Converting the 28 joints to 17 using the learnable triangulation paper of Iskakov.")
        joint_indexes_train = joint_idxs_28to17ours_learnable_h36m_train
        joint_indexes_test  = joint_idxs_17to17ours_learnable_h36m_test
    else:
        assert config.learning_use_h36m_model is True
        if config.learning_use_h36m_model:
            print("Converting the 28 joints to 17 using the crowd pose notations.")
            joint_indexes_train = joint_idxs_28to17_according_to_h36m_train
            joint_indexes_test  = joint_idxs_17to17_according_to_h36m_test # joint_indexes_train
        else:
            print("Converting the 28 joints to 17 using the original notations.")
            joint_indexes_train = joint_idxs_28to17_original_train
            joint_indexes_test  = joint_idxs_17to17_original_test

    if only_chest_height_cameras:
        num_views = 5
    else:
        num_views = 8

    inp_lifting_net_is_images               = config.inp_lifting_net_is_images
    dataset_name                            = config.dataset_name
    random_seed_for_ten_percent_3d_from_all = config.random_seed_for_ten_percent_3d_from_all
    ten_percent_3d_from_all                 = config.ten_percent_3d_from_all
    if not config.perform_test or (config.perform_test and (config.get_json_files_train_set or config.get_json_files_test_set
                                                            or config.plot_keypoints_from_learnt_model or config.plot_train_keypoints)):
        print("Reading the Predictions of the Train Set of the MPI.")
        assert os.path.exists(config.predictions_data_train_file)
        predictions_data_train = json_read(config.predictions_data_train_file)
    else:
        print("Will not be reading the the Train Set for MPI dataset.")
        predictions_data_train = {}

    json_file_3dv_train             = "/cvlabdata2/home/soumava/codes/Step1/FN/preds-eccv-22" in config.predictions_data_train_file
    dict_vals_train_mpi_single_view = {'base'                                    : base_train,
                                       'base_images'                             : base_train_images,
                                       'min_visibility'                          : min_visibility,
                                       'calibration_folder'                      : calibration_folder,
                                       'phase'                                   : 'train',
                                       'num_joints'                              : num_joints,
                                       'overfit'                                 : overfit,
                                       'only_annotations'                        : only_annotations,
                                       'use_annotations'                         : use_annotations,
                                       'predictions_data'                        : predictions_data_train  ,
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
                                       'pelvis_idx'                              : pelvis_idx,
                                       'neck_idx'                                : neck_idx,
                                       'lhip_idx'                                : lhip_idx,
                                       'rhip_idx'                                : rhip_idx,
                                       'annotated_subjects'                      : annotated_subjects,
                                       'subjects'                                : subjects,
                                       'only_chest_height_cameras'               : only_chest_height_cameras,
                                       'load_from_cache'                         : load_from_cache,
                                       'path_cache'                              : path_cache,
                                       'multiview_sample'                        : multiview_sample,
                                       'use_2D_GT_poses_directly'                : use_2D_GT_poses_directly,
                                       'use_2D_mocap_poses_directly'             : use_2D_mocap_poses_directly,
                                       'joints_order'                            : joint_indexes_train,
                                       'json_file_3dv'                           : json_file_3dv_train,
                                       'calculate_K'                             : calculate_K,
                                       }
    if not config.perform_test or (config.perform_test and (config.get_json_files_train_set or config.get_json_files_test_set
                                                            or config.plot_keypoints_from_learnt_model or config.plot_train_keypoints)):
        train_dataset                = get_train_set_mpi_single_frame(dict_vals_train_h36m_single_view=dict_vals_train_mpi_single_view)
        train_labels                 = train_dataset.return_labels()
        num_cameras_considered_train = train_dataset.return_number_of_cameras()
        num_samples_train            = len(train_dataset)
        print("The number of Views Considered for TRAIN is {}".format(num_cameras_considered_train))
        sampler_train, shuffle_train = get_sampler_train(use_annotations=use_annotations, shuffle=shuffle,
                                                         labels=train_labels, only_annotations=only_annotations,
                                                         batch_size=config.batch_size, randomize=randomize,
                                                         extend_last_batch_graphs=False, num_samples=num_samples_train,
                                                         num_anno_samples_per_batch=config.num_anno_samples_per_batch)
        train_loader                 = DataLoader(dataset=train_dataset, batch_size=config.batch_size, sampler=sampler_train,
                                                  shuffle=shuffle_train, num_workers=config.num_workers)
        train_loader_wo_shuffle      = DataLoader(dataset=train_dataset, batch_size=config.batch_size, sampler=None, #sampler_train,
                                                  shuffle=False, num_workers=config.num_workers)
    else:
        print("Will not be loading Training Set.")
        train_loader = train_loader_wo_shuffle = None

    assert os.path.exists(config.predictions_data_test_file)
    print("Reading the Predictions of the Test Set of the MPI.")
    base_test                       = '/cvlabsrc1/cvlab/MPI_INF_3D/mpi_inf_3dhp/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set'
    base_test_images                = '/cvlabsrc1/cvlab/MPI_INF_3D/mpi_inf_3dhp/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set'
    dataset_folder_test             = base_test_images
    calibration_folder_test         = '' # TODO - CHECK
    frame_idx_in_sample_test        = config.frame_idx_in_sample_test
    assert frame_idx_in_sample_test in [2, 3]
    predictions_data_test           = json_read(config.predictions_data_test_file)
    json_file_3dv_test              = "/cvlabdata2/home/soumava/codes/Step1/FN/preds-eccv-22" in config.predictions_data_test_file
    dict_vals_test_mpi_single_view  = {'base'                                    : base_test,
                                       'base_images'                             : base_test_images,
                                       'min_visibility'                          : min_visibility,
                                       'calibration_folder'                      : calibration_folder_test,
                                       'phase'                                   : 'test',
                                       'num_joints'                              : num_joints,
                                       'overfit'                                 : False,
                                       'only_annotations'                        : True,
                                       'use_annotations'                         : True,
                                       'predictions_data'                        : predictions_data_test,
                                       'randomize'                               : False,
                                       'dataset_folder'                          : dataset_folder_test,
                                       'dataset_name'                            : dataset_name,
                                       'every_nth_frame'                         : every_nth_frame,
                                       'every_nth_frame_train_annotated'         : 0,
                                       'every_nth_frame_train_unannotated'       : 0,
                                       'ten_percent_3d_from_all'                 : False,
                                       'random_seed_for_ten_percent_3d_from_all' : 0,
                                       'inp_lifting_net_is_images'               : inp_lifting_net_is_images,
                                       'pelvis_idx'                              : pelvis_idx,
                                       'neck_idx'                                : neck_idx,
                                       'lhip_idx'                                : lhip_idx,
                                       'rhip_idx'                                : rhip_idx,
                                       'use_2D_GT_poses_directly'                : use_2D_GT_poses_directly,
                                       'use_2D_mocap_poses_directly'             : use_2D_mocap_poses_directly,
                                       'joints_order'                            : joint_indexes_test,
                                       'frame_idx_in_sample_test'                : frame_idx_in_sample_test,
                                       'json_file_3dv'                           : json_file_3dv_test,
                                       'calculate_K'                             : calculate_K,
                                       }
    validation_dataset          = get_test_set_mpi_single_frame(dict_vals_test_h36m_single_view=dict_vals_test_mpi_single_view)
    num_cameras_considered_test = 1
    print("The number of Views Considered for Test is {}".format(num_cameras_considered_test))
    validation_loader                = DataLoader(dataset=validation_dataset, batch_size=config.batch_size_test,
                                                  sampler=None, shuffle=False, num_workers=config.num_workers)
    test_loader                      = DataLoader(dataset=validation_dataset, batch_size=config.batch_size_test,
                                                  sampler=None, shuffle=False, num_workers=config.num_workers)
    mpjpe_poses_in_camera_coordinates = True
    if train_loader is not None:
        print("The Batch Size of the Training Set is {}".format(config.batch_size))
        print("The number of training batches is {}.".format(len(train_loader)))

    if validation_loader is not None:
        print("The Batch Size of the Validation/Test set is {}".format(config.batch_size_test))
        print("The number of test/validation batches is {}.".format(len(validation_loader)))

    return train_loader, validation_loader, test_loader, train_loader_wo_shuffle, mpjpe_poses_in_camera_coordinates
