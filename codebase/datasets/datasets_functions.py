
from datasets.Pose_Estimator.h36m_simple       import get_h36m_dataset_characteristics, get_h36m_simple_data_loaders
from datasets.Pose_Estimator.mpi_simple        import get_mpi_dataset_characteristics, get_mpi_simple_data_loaders
from datasets.Lifting_Network.h36m_lifting_net import get_h36m_data_loaders_single_frame
from datasets.Lifting_Network.mpi_lifting_net  import get_mpi_data_loaders_single_frame

def get_dataset_characteristics(config):
    """
    Function to obtain certain characteristics of the datasets which we will be using.
    :param config: The Configuration File consisting of the arguments of training.
    :return: number_of_joints --> The number of joints used in our framework.
             bone_pairs       --> The pairs of bones between two joints.
             rhip_idx         --> The index of the Right Hip.
             lhip_idx         --> The index of the Left Hip.
             neck_idx         --> The index of the Neck.
             pelvis_idx       --> The index of the Pelvis.
    """

    if config.dataset_name.lower() == 'h36m':
        function_to_obtain_dataset_characteristics = get_h36m_dataset_characteristics
        num_cameras_multiview = 4

    elif config.dataset_name.lower() == 'mpi':
        function_to_obtain_dataset_characteristics = get_mpi_dataset_characteristics
        if config.only_chest_height_cameras:
            num_cameras_multiview = 5
        else:
            num_cameras_multiview = 8

    else:
        raise ValueError("Wrong Dataset has been selected.")

    number_of_joints, bone_pairs, rhip_idx, lhip_idx, neck_idx, pelvis_idx, bones_pairs_for_equal_length, \
        ll_bone_idx, rl_bone_idx, lh_bone_idx, rh_bone_idx, torso_bone_idx, head_idx = function_to_obtain_dataset_characteristics(config=config)

    return number_of_joints, bone_pairs, rhip_idx, lhip_idx, neck_idx, pelvis_idx, bones_pairs_for_equal_length, \
           num_cameras_multiview, ll_bone_idx, rl_bone_idx, lh_bone_idx, rh_bone_idx, torso_bone_idx, head_idx


def get_datasets_simple(config, training_pose_lifting_net_without_graphs):
    """
    Function to obtain the Datasets and DataLoaders of various stages of learning.
    :param config: The Configuration File consisting of the arguments of training.
    :param training_pose_lifting_net_without_graphs: If True, we will make number of frames = 1.
                                                    This means we are training the Pose Estimator Model + Lifting Net (No Graphs).
                                                    If False, we will be training Pose Estimator (No Graphs)
    :return: train_loader            --> A Dataloader for the Training Set.
             validation_loader       --> A Dataloader for the Validation Set.
             test_loader             --> A Dataloader for the Test Set.
             train_loader_wo_shuffle --> A Dataloader for the Training Set without Shuffling. This is for evaluating the learnt models by plotting the keypoints.
    """
    if 'alphapose' in config.type_of_2d_pose_model:
        pose_model_type = 1
    elif 'crowd_pose' in config.type_of_2d_pose_model:
        pose_model_type = 2
    elif 'resnet152' in config.type_of_2d_pose_model:
        pose_model_type = 3
    else:
        raise NotImplementedError

    print("Obtaining the Training, Validation and Test datasets for {}".format(config.dataset_name.upper()), end=' ')
    print("For training the 2D Pose Estimator Model.")
    if config.dataset_name.lower() == 'h36m':
        print("Pose Estimator H36M " * 5)
        function_to_load_data = get_h36m_simple_data_loaders

    elif config.dataset_name.lower() == 'mpi':
        print("Pose Estimator MPI " * 5)
        function_to_load_data = get_mpi_simple_data_loaders

    else:
        raise ValueError("Wrong Dataset has been selected.")

    train_loader, validation_loader, test_loader, train_loader_wo_shuffle, \
        mpjpe_poses_in_camera_coordinates = function_to_load_data(config=config, pose_model_type=pose_model_type,
                                                                  training_pose_lifting_net_without_graphs=training_pose_lifting_net_without_graphs)
    return train_loader, validation_loader, test_loader, train_loader_wo_shuffle, mpjpe_poses_in_camera_coordinates


def get_datasets_lifting_net_only(config, with_graphs : bool):
    """
    :param config: The Configuration File consisting of the arguments of training.
    :param with_graphs : A Boolean Variable to denote if the 2D-3D Lifting Network is trained with graphs or not.
    :return:
    """
    print("Obtaining the Training, Validation and Test datasets for {}".format(config.dataset_name.upper()), end=' ')
    print("For training the 2D-3D Lifting Network Only.")
    if not with_graphs:
        print("There will be no Graphs used for training the 2D-3D lifting Network..")
    else:
        print("There will be Graphs used for training the 2D-3D lifting Network.")

    if config.dataset_name.lower() == 'h36m':
        print("H36M " * 5)
        function_to_load_data = get_h36m_data_loaders_single_frame
            # print("CPN Detections.")
            # function_to_load_data = get_h36m_data_loaders_single_frame_cpn_detections
        
    elif config.dataset_name.lower() == 'mpi':
        print("MPI " * 5)
        print("No Graphs " * 5)
        function_to_load_data = get_mpi_data_loaders_single_frame
        
    else:
        raise ValueError("Wrong Dataset has been selected.")

    train_loader, validation_loader, test_loader, train_loader_wo_shuffle, \
        mpjpe_poses_in_camera_coordinates = function_to_load_data(config=config)
    return train_loader, validation_loader, test_loader, train_loader_wo_shuffle, mpjpe_poses_in_camera_coordinates
