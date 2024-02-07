import numpy as np
import re
import os
import h5py
import scipy



training_subjects_all = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']

joint_names_all_28               = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis', # 5
                                    'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow', # 11
                                    'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', #17
                                    'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', #23
                                    'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe']
joint_names_all_17_original      = ['head_top', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
                                    'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee', 'right_ankle',
                                    'left_hip', 'left_knee', 'left_ankle', 'pelvis', 'spine', 'head']
joint_idxs_28to17_original_train = [joint_names_all_28.index(key) for key in joint_names_all_17_original]



joint_names_all_17_according_to_h36m      = ['pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee',
                                             'left_ankle', 'spine', 'neck', 'head', 'head_top', 'left_shoulder',
                                             'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist']
joint_idxs_28to17_according_to_h36m_train = [joint_names_all_28.index(key) for key in joint_names_all_17_according_to_h36m]
joint_idxs_17to17_according_to_h36m_test  = [joint_names_all_17_original.index(joint) for joint in joint_names_all_17_according_to_h36m]
joint_idxs_17to17_original_test           = [joint_names_all_17_original.index(joint) for joint in joint_names_all_17_original]

all_subjects          = ['S{}'.format(i) for i in range(1, 9)]


# FOR TRAIN USING RESNET152
joint_names_17_learnable                   = ['right_ankle', 'right_knee', 'right_hip', 'left_hip', 'left_knee', 'left_ankle',
                                              'pelvis', 'spine', 'neck', 'head_top', 'right_wrist', 'right_elbow',
                                              'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist', 'head']
joint_idxs_28to17ours_learnable_h36m_train = [joint_names_all_28.index(key) for key in joint_names_17_learnable]
joint_idxs_17to17ours_learnable_h36m_test  = [joint_names_all_17_original.index(key) for key in joint_names_17_learnable]




def intrinsics_test_sequences(base: str, calculate_K: bool, seq='TS1'):
    """
    :param base
    :param calculate_K:
    :param seq:
    :return:
    """
    shapes = {'TS1': (2048, 2048, 3), 'TS2': (2048, 2048, 3), 'TS3': (2048, 2048, 3),
              'TS4': (2048, 2048, 3), 'TS5': (1080, 1920, 3), 'TS6': (1080, 1920, 3)
              }
    if seq in ['TS1', 'TS2', 'TS3', 'TS4']:
        # camera 8 indoor
        sensorSize_mm   = np.array([10, 10])
        focalLength_mm  = 7.325058937
        centerOffset_mm = np.array([-0.032288432, 0.092929602])
        pixelAspect     = 1.000442863
        t               = np.array([3427.28, 1387.86, 309.42]) / 1000
        dist            = np.zeros((5,))
        up              = np.array([-0.208215, 0.976233, 0.06014])
        right           = np.array([0.000575281, 0.0616098, -0.9981])
    elif seq in ['TS5', 'TS6']:
        # camera 5 outdoor (pablo) (One of the extrinsics is correct, but for intrinsics it should be fine)
        sensorSize_mm   = np.array([10, 5.625000000])
        focalLength_mm  = 8.770747185  #8.786903381 #
        centerOffset_mm = np.array([-0.104908645, 0.104899704])  #np.array([-0.119419098, -0.135618150]) #
        pixelAspect     = 0.993236423  #0.992438614 #

        t     = np.array([-2104.3074, 1038.6707, -4596.6367]) / 1000
        dist  = np.array([-0.276859611, 0.131125256, -0.000360494, -0.001149441, -0.049318332])
        up    = np.array([0.025272345, 0.995038509, 0.096227370])
        right = np.array([-0.939647257, -0.009210289, 0.342020929])
    else:
        raise ValueError("Unrecognized seq '{}'".format(seq))

    h, w, _ = shapes[seq]
    # K       = np.eye(3, 3)
    # # pixelAspect = 1
    # K[0, 0] = w * focalLength_mm / sensorSize_mm[0] * pixelAspect
    # K[1, 1] = h * focalLength_mm * pixelAspect / sensorSize_mm[1] * pixelAspect
    # K[0, 2] = w * (0.5 + centerOffset_mm[0] / sensorSize_mm[0]) * pixelAspect
    # K[1, 2] = h * (0.5 + centerOffset_mm[1] / sensorSize_mm[1]) * pixelAspect
    #
    # R = np.row_stack([right, -up, np.cross(up, right)])
    # R = np.eye(3)
    # t = np.zeros((3,))
    if calculate_K is False:
        K = np.eye(3, 3)
        # pixelAspect = 1
        K[0, 0] = w * focalLength_mm / sensorSize_mm[0] * pixelAspect
        K[1, 1] = h * focalLength_mm * pixelAspect / sensorSize_mm[1] * pixelAspect
        K[0, 2] = w * (0.5 + centerOffset_mm[0] / sensorSize_mm[0]) * pixelAspect
        K[1, 2] = h * (0.5 + centerOffset_mm[1] / sensorSize_mm[1]) * pixelAspect
    else:
        print("Calculating the K matrix for the Test Sequence {}.".format(seq))
        mat      = h5py.File(os.path.join(base, seq, 'annot_data.mat'), 'r')
        poses_2d = np.array(mat['annot2'])[:, 0]
        poses_3d = np.array(mat['annot3'])[:, 0] / 1000
        poses_2d = poses_2d.reshape((-1, 17, 2))
        poses_3d = poses_3d.reshape((-1, 17, 3))
        fx, cx = np.linalg.lstsq(poses_3d[:, :, [0, 2]].reshape((-1, 2)),
                                 (poses_2d[:, :, 0] * poses_3d[:, :, 2]).reshape(-1, 1),
                                 rcond=None)[0].flatten()
        fy, cy = np.linalg.lstsq(poses_3d[:, :, [1, 2]].reshape((-1, 2)),
                                 (poses_2d[:, :, 1] * poses_3d[:, :, 2]).reshape(-1, 1),
                                 rcond=None)[0].flatten()

        K       = np.eye(3, 3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

    R = np.row_stack([right, -up, np.cross(up, right)])
    R = np.eye(3)
    t = np.zeros((3,))
    return K, dist, R, t, h, w



def parse_camera_calibration(f, y_z_swapped=True):
    """

    :param f:
    :param y_z_swapped:
    :return:
    """
    line_re = re.compile(r'(\w+)\s+(.+)')
    types   = {'name': 'int', 'sensor': 'vec2', 'size': 'vec2', 'animated': 'int', 'intrinsic': 'mat4',
               'extrinsic': 'mat4', 'radial': 'int'}
    f.readline()
    camera_properties = {}
    props             = None
    for line in f.readlines():
        line_match = line_re.fullmatch(line.strip())
        if line_match:
            key, value = line_match.groups()
            values     = value.split(' ')
            value_type = types[key]
            if value_type == 'int':
                assert len(values) == 1
                parsed_value = int(values[0])
            elif value_type == 'vec2':
                assert len(values) == 2
                parsed_value = np.array([float(v) for v in values])
            elif value_type == 'mat4':
                assert len(values) == 4 * 4
                parsed_value = np.array([float(v) for v in values]).reshape((4, 4))
            else:
                print('Skipping unrecognized camera calibration field:', key)
                continue

            if key == 'name':
                props = {}
                camera_properties[parsed_value] = props
            else:
                props[key] = parsed_value

    cameras = {}
    for i, props in camera_properties.items():
        R = props['extrinsic'][:3, :3]
        t = props['extrinsic'][:3, -1]
        if y_z_swapped:
            R = R[:, [0, 2, 1]]*np.array([1, -1, 1])[None]
            t = t
        cameras[i] = {'K': props['intrinsic'][:3, :3], 'dist': np.zeros((5,)), 'R': R, 't': t/1000,
                      'width': props['size'][0], 'height': props['size'][1]}
    return cameras



def function_to_calculate_new_intrinsics(base, subject, seq, old_calibration):
    mat      = scipy.io.loadmat(os.path.join(base, subject, seq, 'annot.mat'))
    arr      = np.stack(mat['annot2'].flatten())
    poses_2d = arr.reshape((arr.shape[0], arr.shape[1], 28, 2))
    arr      = np.stack(mat['annot3'].flatten())
    poses_3d = arr.reshape((arr.shape[0], arr.shape[1], 28, 3))
    poses_3d = poses_3d / 1000
    num_cams = arr.shape[0]

    assert (num_cams == len(old_calibration.keys()))
    calib_keys = list(old_calibration.keys())
    for calib_key in calib_keys:
        calib_key_idx  = calib_keys.index(calib_key)
        poses_2d_i     = poses_2d[calib_key_idx].reshape((-1, 28, 2))
        poses_3d_i     = poses_3d[calib_key_idx].reshape((-1, 28, 3))
        fx_i, cx_i     = np.linalg.lstsq(poses_3d_i[:, :, [0, 2]].reshape((-1, 2)),
                                         (poses_2d_i[:, :, 0] * poses_3d_i[:, :, 2]).reshape(-1, 1),
                                         rcond=None)[0].flatten()
        fy_i, cy_i     = np.linalg.lstsq(poses_3d_i[:, :, [1, 2]].reshape((-1, 2)),
                                         (poses_2d_i[:, :, 1] * poses_3d_i[:, :, 2]).reshape(-1, 1),
                                         rcond=None)[0].flatten()
        new_K_i        = np.eye(3, 3)
        new_K_i[0, 0]  = fx_i
        new_K_i[1, 1]  = fy_i
        new_K_i[0, 2]  = cx_i
        new_K_i[1, 2]  = cy_i
        assert 'K' in list(old_calibration[calib_key].keys())
        old_calibration[calib_key]['K'] = new_K_i
    return old_calibration
