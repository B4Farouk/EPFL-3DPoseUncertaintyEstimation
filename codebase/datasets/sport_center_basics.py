import numpy as np
import os
from utils import json_read
import random


frame_ids_with_annotations = [21380, 21383, 21386, 21389, 21392, 21395, 21398, 21401, 21404, 21407, 21410, 21413, 21416, 21419, 21422, 21425, 21428, 21431, 21434, 21437]\
                             + list(np.arange(30000, 30050))
subject_ids = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
shapes      = {'ace_0': (992, 1376, 3), 'ace_1': (896, 1536, 3), 'ace_2': (896, 1536, 3), 'ace_3': (1024, 1328, 3), 'ace_4': (928, 1392, 3), 'ace_5': (896, 1360, 3)}

joint_names_17_ours = ['nose', 'LEye', 'REye', 'LEar', 'REar', 'Lshoulder', 'Rshoulder', 'Lelbow', "Relbow",
                       'Lwrist', 'Rwrist', 'Lhip', 'Rhip', 'Lknee', 'Rknee', 'Lankle', 'Rankle']
joint_names_13_ours = ['nose', 'Lshoulder', 'Rshoulder', 'Lelbow', "Relbow", 'Lwrist', 'Rwrist',
                       'Lhip', 'Rhip', 'Lknee', 'Rknee', 'Lankle', 'Rankle']
joint_idxs_17       = [joint_names_17_ours.index(x) for x in joint_names_17_ours]
joint_idxs_13       = [joint_names_17_ours.index(x) for x in joint_names_13_ours]
joint_names_17_learnable      = ['RightFoot', 'RightLeg', 'RightUpLeg', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
                                 'Hips', 'Spine1', 'Neck', 'Site-head', 'RightHand', 'RightForeArm', 'RightArm',
                                 'LeftArm', 'LeftForeArm', 'LeftHand', 'Head']
joints_idx_17_to_13_learnable = [16, 13, 12, 14, 11, 15, 10, 3, 2, 4, 1, 5, 0]
joint_names_13_learnable      = [joint_names_17_learnable[x] for x in joints_idx_17_to_13_learnable]


def get_sport_center_dataset_characteristics(config):
    resnet152_backbone = True if 'resnet152' in config.type_of_2d_pose_model else False  # False if 'crowd_pose' in config.type_of_2d_pose_model else True
    if not resnet152_backbone:
        joints_names = joint_names_13_ours

    else:
        joints_names = joint_names_13_learnable

    bones_pairs = []
    bones       = [('nose', 'Lshoulder'), ('Lshoulder', 'Lelbow'), ('Lelbow', 'Lwrist'),
                   ('nose', 'Rshoulder'), ('Rshoulder', 'Relbow'), ('Relbow', 'Rwrist'),
                   ('Lhip', 'Lknee'), ('Lknee', 'Lankle'),
                   ('Rhip', 'Rknee'), ('Rknee', 'Rankle')]

    for bone in bones:
        bone_start = bone[0]; bone_start_idx = joints_names.index(bone_start)
        bone_end = bone[1]; bone_end_idx = joints_names.index(bone_end)
        bones_pairs.append([bone_start_idx, bone_end_idx])

    bone_pairs_symmetric = [[('nose', 'Lshoulder'),   ('nose', 'Rshoulder')],
                            [('Lshoulder', 'Lelbow'), ('Rshoulder', 'Relbow')],
                            [('Lelbow', 'Lwrist'), ('Relbow', 'Rwrist')],
                            [('Lhip', 'Lknee'), ('Rhip', 'Rknee')],
                            [('Lknee', 'Lankle'), ('Rknee', 'Rankle')]
                            ]
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
    lhip_idx         = joints_names.index('Lhip')
    rhip_idx         = joints_names.index('Rhip')
    neck_idx         = joints_names.index('nose')
    pelvis_idx       = -1 #[lhip_idx, rhip_idx]
    head_idx         = -1 #joints_names.index('Head')
    ll_bone_idx      = None # TODO - FOR DISCRIMINATOR
    rl_bone_idx      = None # TODO - FOR DISCRIMINATOR
    lh_bone_idx      = None # TODO - FOR DISCRIMINATOR
    rh_bone_idx      = None # TODO - FOR DISCRIMINATOR
    torso_bone_idx   = None # TODO - FOR DISCRIMINATOR

    return number_of_joints, bones_pairs, rhip_idx, lhip_idx, neck_idx, pelvis_idx, bone_pairs_symmetric_indexes, \
           ll_bone_idx, rl_bone_idx, lh_bone_idx, rh_bone_idx, torso_bone_idx, head_idx


def get_files_with_annotations(num_samples, start_point, trajs_txt, every_nth_frame, include_all_samples=False):
    image_files_train_1 = ['None'] * 10000
    image_files_train_2 = ['None'] * 9000
    image_files_test_1  = ['None'] * 3000
    image_files_test_2  = ['None'] * 3000
    with open(trajs_txt) as file_in:
        next(file_in)
        for line in file_in:
            image_subjects = line.split(' ')
            image_idx = int(image_subjects[-1].split('\n')[0])
            if (image_idx >= 20001) and (image_idx <= 30000):
                image_files_train_1[image_idx - 20001] = line
            elif (image_idx >= 41001) and (image_idx <= 50000):
                image_files_train_2[image_idx - 41001] = line
            elif (image_idx >= 30001) and (image_idx <= 33000):
                image_files_test_1[image_idx - 30001] = line
            elif (image_idx >= 50001) and (image_idx <= 53000):
                image_files_test_2[image_idx - 50001] = line


    step_size = every_nth_frame
    if not include_all_samples:
        print("The starting point is {} and number of samples is {}".format(start_point, num_samples))
        a = []; b = []
        x = image_files_train_1[start_point:start_point + num_samples: step_size]
        y = image_files_train_2[start_point:start_point + num_samples: step_size]
        for frame_id in frame_ids_with_annotations:
            if frame_id <= 30000:
                offset = 20001
                yyy    = image_files_train_1[frame_id - offset]
            else:
                offset = 30001
                yyy    = image_files_test_1[frame_id - offset]

            if yyy not in x:
                x.append(yyy)

            if yyy not in a:
                a.append(yyy)
    else:
        x = image_files_train_1[0:10000:step_size]
        y = image_files_train_2[0:9001:step_size]
        a = image_files_test_1[0:3001:step_size]
        b = image_files_test_2[0:3001:step_size]

    image_files_train = x + y
    image_files_test  = a + b
    print("Number of Frames Considered for Training are {}".format(len(image_files_train)))
    print("Number of Frames Considered for Test are {}".format(len(image_files_test)))
    return image_files_train, image_files_test


def get_vals_keys(subject_ids, frame_ids_cons):
    vals = {}; keys = []; gt_folder = '/cvlabsrc1/cvlab/dataset_sportcenter/20170915/gt_human_poses/'
    for folder_ in ['21380_21440_3', '30000_30050_1']:
        folder = os.path.join(gt_folder, folder_)
        for subject_id in subject_ids:
            file_name = '{}/pose_subject{}.json'.format(folder, subject_id)
            if os.path.exists(file_name):
                print("Loading the 3D annotations from {}".format(file_name))
                data       = json_read(file_name)
                vid_frames = data['3d']['idx_frame']
                for frame_id in frame_ids_cons:
                    if frame_id not in vid_frames:
                        continue
                    else:
                        vid_frame_idx = vid_frames.index(frame_id)
                        vals['{}-S{}'.format(frame_id, subject_id)] = np.asarray(data['3d']['pose'][vid_frame_idx])
                        keys.append('{}-S{}'.format(frame_id, subject_id))
    return vals, keys


def get_annotated_3d_data(subject_ids):
    vals, keys = get_vals_keys(subject_ids=subject_ids, frame_ids_cons=frame_ids_with_annotations)
    return vals, keys



def generate_subjects_image_list_with_labels(lst, values, values_not_consider):
    good_idx = []; subject_idx = []; num_files = len(lst)
    for i in range(0, num_files):
        image_subjects = lst[i].split(' ')
        image_idx      = image_subjects[-1].split('\n')[0]
        count_subject  = 0
        for subject_presence in image_subjects[:-1]:
            if subject_presence == 'True':
                sid = image_idx + '-S' + str(count_subject)
                if sid not in values_not_consider:
                    subject_idx.append(sid)
                    if sid in values:
                        good_idx.append(1)
                    else:
                        good_idx.append(0)
            count_subject += 1
    return subject_idx, good_idx


def get_hard_test_set():
    print("Obtaining the DIFFICULT TEST SET.")
    hard_test_dict  = {'S7': [21395, 21434, 21392, 21413, 21431, 21428, 30004, 30005, 21425, 21416, 21419, 21422, 30006, 30007, 21410],
                       'S12': [21437, 21434, 21431, 30005, 30006, 30007, 30004, 21428, 21425, 30008, 30003, 21386, 21419, 21422, 21389]}
    subject_tst_idx = []; labels_test_candidates = []; small_test_dict_keys = list(hard_test_dict.keys())
    for small_test_dict_key in small_test_dict_keys:
        frames = hard_test_dict[small_test_dict_key]
        for frame in frames:
            subject_tst_idx.append('{}-{}'.format(frame, small_test_dict_key))
            labels_test_candidates.append(1)
    return subject_tst_idx, labels_test_candidates


def get_easy_test_set():
    print("Obtaining the DIFFICULT TEST SET.")
    hard_test_dict  = {'S7': [21395, 21434, 21392, 21413, 21431, 21428, 30004, 30005, 21425, 21416, 21419, 21422, 30006, 30007, 21410],
                       'S12': [21437, 21434, 21431, 30005, 30006, 30007, 30004, 21428, 21425, 30008, 30003, 21386, 21419, 21422, 21389]}
    easy_test_dict = {}
    for key in hard_test_dict:
        hard_samples_key = hard_test_dict[key]
        easy_samples_key = []
        for frame_id in frame_ids_with_annotations:
            if frame_id not in hard_samples_key:
                easy_samples_key.append(frame_id)
        easy_test_dict[key] = easy_samples_key

    subject_tst_idx = []; labels_test_candidates = []; small_test_dict_keys = list(easy_test_dict.keys())
    for small_test_dict_key in small_test_dict_keys:
        frames = easy_test_dict[small_test_dict_key]
        for frame in frames:
            subject_tst_idx.append('{}-{}'.format(frame, small_test_dict_key))
            labels_test_candidates.append(1)
    return subject_tst_idx, labels_test_candidates



class Basketball_dataset(object):
    def function_to_return_cameras(self):
        return self.cameras

    def function_to_return_calibration(self):
        return self.calibration

    def function_to_return_trajectories(self):
        return self.trajectories

    def __init__(self,
                 num_samples: int,
                 trajs_txt_folder: str,
                 every_nth_frame: int,
                 consider_six_views : bool,
                 include_all_samples : bool,
                 start_point : int,
                 hard_test_set : bool,
                 easy_test_set : bool,
                 subject_ids_train : list,
                 subject_ids_test : list,
                 randomize : bool,
                 calibration_folder: str,
                 trajectories_folder: str,
                 num_joints: int
                 ):
        super(Basketball_dataset, self).__init__()

        self.trajs_txt_folder    = trajs_txt_folder
        self.every_nth_frame     = every_nth_frame
        self.trajs_txt           = os.path.join(self.trajs_txt_folder, 'trajs.txt')
        self.consider_six_views  = consider_six_views
        self.hard_test_set       = hard_test_set
        self.easy_test_set       = easy_test_set
        self.randomize           = randomize
        self.include_all_samples = include_all_samples
        self.start_point         = start_point
        self.subject_ids_test    = subject_ids_test
        self.subject_ids_train   = subject_ids_train
        self.num_samples         = num_samples
        self.calibration_folder  = calibration_folder
        self.trajectories_folder = trajectories_folder
        self.num_joints          = num_joints


        if self.consider_six_views:
            print("Will be considering only 6 views.")
            cameras = ['ace_{}'.format(cam_idx) for cam_idx in range(6)]
        else:
            print("Will be considering only 4 views.")
            cameras = ['ace_1', 'ace_2', 'ace_4', 'ace_5']
        self.cameras = cameras

        print("Obtaining the image files for training and testing respectively.")
        image_files_train, image_files_test = get_files_with_annotations(trajs_txt=self.trajs_txt,
                                                                         every_nth_frame=self.every_nth_frame,
                                                                         include_all_samples=include_all_samples,
                                                                         num_samples=self.num_samples,
                                                                         start_point=self.start_point)
        pose_3d_data_annotated_subject_frame_train,\
            annotated_subject_frame_train_keys = get_annotated_3d_data(subject_ids=subject_ids_train)
        pose_3d_data_annotated_subject_frame_test,  \
            subject_frame_test_keys            = get_annotated_3d_data(subject_ids=subject_ids_test)
        subject_frame_train_idx,\
                       labels_train_candidates = generate_subjects_image_list_with_labels(lst=image_files_train, values=annotated_subject_frame_train_keys,
                                                                                          values_not_consider=subject_frame_test_keys)
        if randomize:
            random.shuffle(image_files_train)
            random.shuffle(image_files_test)

        if self.hard_test_set:
            subject_frame_test_idx, labels_test_candidates = get_hard_test_set()
        elif self.easy_test_set:
            subject_frame_test_idx, labels_test_candidates = get_easy_test_set()
        else:
            subject_frame_test_idx = subject_frame_test_keys; labels_test_candidates = [1] * len(subject_frame_test_keys)

        s1 = set(subject_frame_train_idx)
        s2 = set(subject_frame_test_idx)
        s3 = list(s1.intersection(s2))
        s4 = list(s2.intersection(s1))
        assert (len(s3) == 0)
        assert (len(s4) == 0)

        pose_3d_data_subject_frame_train = {}; labels_train = []
        for ind, key in enumerate(subject_frame_train_idx):
            if key in annotated_subject_frame_train_keys:
                pose_3d_data_subject_frame_train[key] = pose_3d_data_annotated_subject_frame_train[key]
                labels_train.append(1)
            else:
                y = np.zeros((self.num_joints, 3))
                y.fill(np.inf)
                pose_3d_data_subject_frame_train[key] = y
                labels_train.append(0)

        self.subject_train_idx       = subject_frame_train_idx
        self.labels_train_candidates = labels_train
        self.pose_3d_data_train      = pose_3d_data_subject_frame_train
        self.subject_tst_idx         = subject_frame_test_idx
        self.labels_test_candidates  = labels_test_candidates
        self.pose_3d_data_test       = pose_3d_data_annotated_subject_frame_test

        print("Total Number of Samples being considered for training is {}".format(len(self.subject_train_idx)))
        print("Total Number of Samples being considered for test is {}".format(len(self.subject_tst_idx)))

        print("Loading the Calibration parameters.")
        self.calibration = json_read(os.path.join(self.calibration_folder, "global_poses.json"))
        print("Loading the Trajectories parameters.")
        self.trajectories = json_read(os.path.join(self.trajectories_folder, "trajectories.json"))

    def return_images(self, phase):
        if phase.lower() == 'train':
            return self.subject_train_idx, self.labels_train_candidates, self.pose_3d_data_train
        elif phase.lower() == 'validation' or phase.lower() == 'test':
            return self.subject_tst_idx, self.labels_test_candidates, self.pose_3d_data_test
        else:
            NotImplementedError("We have not implemented this for this phase.")