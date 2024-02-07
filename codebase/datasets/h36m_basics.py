import numpy as np
import os
import glob
import h5py
import multiprocessing
import random

from itertools  import repeat
from utils      import json_read
from operator   import itemgetter



cameras_mapping = {'54138969' : 0, '55011271' : 1, '58860488' : 2, '60457274': 3}

image_shapes   = {'54138969': (1002, 1000), '55011271': (1000, 1000), '58860488': (1000, 1000), '60457274': (1002, 1000)}
subjects       = {'training': ['S1', 'S5', 'S6', 'S7', 'S8'], 'testing': ['S2', 'S3', 'S4'], 'validation': ['S9', 'S11']}
camera_names   = ['54138969', '55011271', '58860488', '60457274']
all_subjects   = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
blacklist      = {'S9': ('Greeting-2', 'SittingDown-2', 'Waiting-1')}

joint_names_all   = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Site',
                     'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'Site',
                     'Spine', 'Spine1', 'Neck', 'Head', 'Site-head',
                     'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb', 'Site',
                     'L_Wrist_End', 'Site', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb',
                     'Site', 'R_Wrist_End', 'Site']
joint_names_17    = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot',
                     'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Spine1', 'Neck', 'Head', 'Site-head',
                     'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand']
joint_idxs_32to17 = [joint_names_all.index(key) for key in joint_names_17]


joint_idxs_32to17_learnable = [3, 2, 1, 6, 7, 8, 0, 12, 13, 15, 27, 26, 25, 17, 18, 19, 14]
joint_names_17_learnable    = [joint_names_all[x] for x in joint_idxs_32to17_learnable]



def _prepare_data_action_subject(action, subject, calibration, base):
    """
    :param action:
    :param subject:
    :param calibration:
    :param base:
    :return:
    """
    data  = h5py.File(os.path.join(base, subject, action, "annot.h5"), "r")
    data2 = {'idx_frames': list(sorted(set(data['frame'])))}
    idxs  = np.array_split(np.arange(len(data['frame'])), 4)
    # sanity check
    for idxs_cam in idxs[1:]:
        assert len(idxs_cam) == len(idxs[0])

    # As the 3d poses here are in camera coordinate we have to project them to world
    # To do so we take the 3d pose of camera 1 only as they should be the same
    poses_cam0        = np.array_split(np.array(data['pose']['3d']), 4)[0]
    camera_name       = str(data['camera'][0])
    R                 = np.array(calibration['extrinsics'][subject][camera_name]['R'])
    t                 = np.array(calibration['extrinsics'][subject][camera_name]['t'])
    poses_world       = np.tensordot(R.T, poses_cam0.T-t.ravel()[:, None, None], axes=([1], [0])).T
    data2['poses_3d'] = poses_world/1000.
    data2['poses_2d'] = {}
    for idxs_cam in idxs:
        camera_name                    = str(data['camera'][idxs_cam[0]])
        data2['poses_2d'][camera_name] = np.array(data['pose']['2d'])[idxs_cam]
    return data2


def get_action_names(base, subject):
    """

    :param base:
    :param subject:
    :return:
    """
    names = [os.path.basename(x) for x in glob.glob(os.path.join(base, subject, "*"))]
    if subject in blacklist:
        for blacklist_action in blacklist[subject]:
            if blacklist_action in names:
                names.remove(blacklist_action)
                print("Removing action {} for subject {}.".format(blacklist_action, subject))
            else:
                print("Cannot remove action {} for subject {} because not present!? {}".format(blacklist_action, subject, names))
    return names



class H36M_dataset(object):
    def get_data(self, mode):
        """

        :param mode:
        :return:
        """
        if self.load_from_cache:
            print("Load cache from {} for mode {}.".format(self.path_cache, mode))
            data = np.load(os.path.join(self.path_cache, "cached_poses_{}.npz".format(mode)), allow_pickle=True).get('arr_0').item()
        else:
            data = self._prepare_data(mode, threads=16)
            np.savez_compressed(os.path.join(self.path_cache, "cached_poses_{}".format(mode)), data)

        samples = []; labels = []; a = 0
        for subject, x1 in data.items():
            if mode.lower() == 'training':
                if self.use_annotations and subject in self.annotated_subjects:
                    label = 1
                else:
                    label = 0
            else:
                label = 1
            counter = 0
            if (mode.lower() == 'training' and subject in self.subject_ids_train) or mode.lower() != 'training':
                print("{}-{}".format(subject, mode), end='\t')
                for action, x2 in x1.items():
                    for idx in x2['idx_frames']:
                        samples.append((subject, action, idx, label))
                        labels.append(label)
                        counter += 1
                a += counter
                print(label, counter)
        print("Total Samples loaded for {} is {}".format(mode, a))
        assert (len(labels) == len(samples))
        N = len(samples)
        for i in range(N):
            assert labels[i] == samples[i][-1]
        return samples, data, labels

    def get_calib_data(self, phase):
        if phase.lower() == 'train':
            return self.calibration_train
        elif phase.lower() == 'validation':
            return self.calibration_validation
        else:
            return self.calibration_test

    def get_values(self, phase):
        """

        :param phase:
        :return:
        """
        if phase.lower() == 'train':
            return self.calibration_train, self.data_train, self.subject_ids_train, self.samples_train, \
                   self.actions_train, self.labels_train
        elif phase.lower() == 'validation':
            return self.calibration_validation, self.data_validation, self.subject_ids_validation, \
                   self.samples_validation, self.actions_validation, self.labels_validation
        else:
            return self.calibration_test, self.data_test, self.subject_ids_test, self.samples_test,\
                   self.actions_test, self.labels_test

    def _prepare_data(self, mode, threads=16):
        """

        :param mode:
        :param threads:
        :return:
        """
        if mode == 'testing':
            raise NotImplementedError()
        data = {}
        for subject in subjects[mode]:
            data[subject] = {}
            with multiprocessing.Pool(threads) as pool:
                res = pool.starmap(_prepare_data_action_subject, zip(self.actions[subject], repeat(subject),
                                                                     repeat(self.calibration), repeat(self.dataset_folder)))
            for action, data2 in zip(self.actions[subject], res):
                data[subject][action] = data2
                print("Loaded {} frames for subject:{} action:{}.".format(len(data2['idx_frames']), subject, action))
        return data

    def load_calibration(self):
        """

        :return:
        """
        calibration_      = json_read(os.path.join(self.calibration_folder, "h36m_calibration.json"))
        calibration_train = {}; calibration_test = {}; calibration_validation = {}; calibration = {}
        intrinsics        = calibration_['intrinsics']
        for subject in all_subjects:
            ext_sub  = calibration_['extrinsics'][subject]
            if subject in self.subject_ids_train:
                calibration_train[subject] = {}
            elif subject in self.subject_ids_validation:
                calibration_validation[subject] = {}
            elif subject in self.subject_ids_test:
                calibration_test[subject] = {}
            calibration[subject] = {}
            for camera_name in camera_names:
                R          = np.array(ext_sub[camera_name]['R'])
                t          = np.array(ext_sub[camera_name]['t']) / 1000
                K          = np.array(intrinsics[camera_name]['K'])
                dist       = np.array(intrinsics[camera_name]['dist']) # ace_cam_id = cam_idx_ss_ace[camera_name]
                calib_val  = {'R': R, 't': t, 'K': K, 'dist': dist}
                if subject in self.subject_ids_train:
                    calibration_train[subject][camera_name] = calib_val
                elif subject in self.subject_ids_validation:
                    calibration_validation[subject][camera_name] = calib_val
                elif subject in self.subject_ids_test:
                    calibration_test[subject][camera_name] = calib_val
                calibration[subject][camera_name] = calib_val
        return calibration_train, calibration_validation, calibration_test, calibration

    def __init__(self,
                 dataset_folder,
                 path_cache,
                 use_annotations,
                 calibration_folder,
                 use_annotations_only,
                 training_subjects,
                 annotated_subjects=None,
                 load_from_cache=True):
        """

        :param dataset_folder:
        :param path_cache:
        :param use_annotations:
        :param calibration_folder:
        :param use_annotations_only:
        :param training_subjects:
        :param annotated_subjects:
        :param load_from_cache:
        """

        self.dataset_folder       = dataset_folder
        self.use_annotations      = use_annotations
        self.use_annotations_only = use_annotations_only
        self.calibration_folder   = calibration_folder
        if len(training_subjects) == 0:
            self.subject_ids_train = subjects['training']
        else:
            self.subject_ids_train = training_subjects

        print("The Subjects considered for Training are {}".format(', '.join(self.subject_ids_train)))
        if use_annotations:
            assert annotated_subjects is not None
            for annotated_subject in annotated_subjects:
                assert annotated_subject in self.subject_ids_train
            self.annotated_subjects = annotated_subjects
            print("The Annotated subjects considered for Training are {}".format(', '.join(self.annotated_subjects)))
        else:
            self.annotated_subjects = []
        val_mode  = 'validation'
        test_mode = 'validation'
        self.subject_ids_validation = subjects[val_mode]
        print("The Subjects considered for Validation are {}".format(', '.join(self.subject_ids_validation)))
        self.subject_ids_test = subjects[test_mode]
        print("The Subjects considered for Test are {}".format(', '.join(self.subject_ids_test)))
        self.subjects         = all_subjects
        print("Loading the Calibration parameters for the Train and Test Set.")

        calibration_train, calibration_validation, calibration_test, calibration = self.load_calibration()
        self.calibration_train      = calibration_train
        self.calibration_validation = calibration_validation
        if test_mode.lower() in ['test', 'testing']:
            self.calibration_test = calibration_test
        else:
            assert test_mode.lower() == 'validation'
            self.calibration_test = calibration_validation

        self.calibration     = calibration
        self.path_cache      = path_cache
        self.load_from_cache = load_from_cache

        samples_train, data_train, labels_train                = self.get_data(mode='training')
        samples_validation, data_validation, labels_validation = self.get_data(mode='validation')
        samples_test, data_test, labels_test                   = self.get_data(mode=test_mode)

        self.data_train      = data_train
        self.data_validation = data_validation
        self.data_test       = data_test

        self.samples_test       = samples_test
        self.samples_train      = samples_train
        self.samples_validation = samples_validation

        self.labels_train       = labels_train
        self.labels_validation  = labels_validation
        self.labels_test        = labels_test

        print("Loading the actions for all the subjects.")
        self.actions            = {subject: get_action_names(self.dataset_folder, subject) for subject in self.subjects}
        print("Loading the actions for all the subjects in Training.")
        self.actions_train      = {subject: get_action_names(self.dataset_folder, subject) for subject in self.subject_ids_train}
        print("Loading the actions for all the subjects in Validation.")
        self.actions_validation = {subject: get_action_names(self.dataset_folder, subject) for subject in self.subject_ids_validation}
        print("Loading the actions for all the subjects in Testing.")
        self.actions_test       = {subject: get_action_names(self.dataset_folder, subject) for subject in self.subject_ids_test}
        self.shapes             = image_shapes


def get_labeled_samples(samples_phase, labels_phase, phase, every_nth_frame, ten_percent_3d_from_all,
                        every_nth_frame_train_annotated, every_nth_frame_train_unannotated, only_annotations,
                        randomize, overfit):
    if phase.lower() in ['test', 'validation']:
        assert every_nth_frame is not None
        print("Sampling Every {} frame for {} mode.".format(every_nth_frame, phase))
        samples = samples_phase[::every_nth_frame]
        labels  = labels_phase[::every_nth_frame]
        assert sum(labels) == len(samples)
    else:
        print("\n")
        print("----------" * 20)
        if ten_percent_3d_from_all:
            N_labels = len(labels_phase)
            print("Performing 10% of all {} Samples to be Annotated.. Rest all are UnAnnotated.".format(N_labels))
            labels_phase = [0] * N_labels
            label        = 1
            ii           = 0
            for index in range(0, N_labels, 10):
                # s                    = samples_phase[index]
                # s                    = list(s)
                # s[-1]                = label
                # s                    = tuple(s)
                # samples_phase[index] = s
                labels_phase[index]  = label
                ii += 1
            assert every_nth_frame_train_annotated == 1
            print("After performing 10% sampling of all samples.. we have {} samples annotated and rest all are unannotated.".format(ii))
            print("----------" * 20)
        assert every_nth_frame_train_annotated is not None
        assert every_nth_frame_train_unannotated is not None
        idx_labeled = np.where(np.array(labels_phase) == 1)[0]
        print("Before Sampling -- The number of Annotated Samples are {}".format(len(idx_labeled)))
        if len(idx_labeled) > 0:
            print("Sampling Every {} frame for Labeled samples for {} mode.".format(every_nth_frame_train_annotated, phase))
            samples_labeled = list(itemgetter(*idx_labeled)(samples_phase))
            labels_labeled  = list(itemgetter(*idx_labeled)(labels_phase))
            samples_labeled = samples_labeled[::every_nth_frame_train_annotated]
            labels_labeled  = labels_labeled[::every_nth_frame_train_annotated]
            print("After Sampling .. Total Number of Annotated Samples are {}".format(len(labels_labeled)))
        else:
            samples_labeled = []; labels_labeled = []

        idx_unlabeled = np.where(np.array(labels_phase) == 0)[0]
        print("Before Sampling -- The number of UnAnnotated Samples are {}".format(len(idx_unlabeled)))
        if len(idx_unlabeled) > 0:
            print("Sampling Every {} frame for UnLabeled samples for {} mode.".format(every_nth_frame_train_unannotated, phase))
            samples_unlabeled = list(itemgetter(*idx_unlabeled)(samples_phase))
            labels_unlabeled  = list(itemgetter(*idx_unlabeled)(labels_phase))
            samples_unlabeled = samples_unlabeled[::every_nth_frame_train_unannotated]
            labels_unlabeled  = labels_unlabeled[::every_nth_frame_train_unannotated]
            print("After Sampling .. Total Number of UnAnnotated Samples are {}".format(len(labels_unlabeled)))
        else:
            samples_unlabeled = []; labels_unlabeled = []
        samples = samples_labeled + samples_unlabeled
        labels  = labels_labeled + labels_unlabeled

    if only_annotations:
        assert (phase.lower() == 'train')  # and (self.use_annotations is True) TODO CHECK THIS.
        print("The model will be pretrained only on the subjects of the annotated samples.")
        idx     = np.where(np.array(labels) == 1)[0]
        samples = list(itemgetter(*idx)(samples))
        labels  = list(itemgetter(*idx)(labels))

    if randomize:
        print("Randomizing the {} dataset ".format(phase))
        c = list(zip(samples, labels))
        random.shuffle(c)
        samples, labels = zip(*c)

    if overfit:
        samples = samples[0:1] * 10000
        labels  = labels[0:1] * 10000
    return samples, labels
