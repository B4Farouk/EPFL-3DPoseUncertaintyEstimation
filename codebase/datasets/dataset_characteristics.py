def h36m_characteristics():
    joint_names_17    = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot',  'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Spine1', 'Neck', 'Head', 'Site-head',
                         'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand']
    joints_ordering   = {'Hips'         : 0, 
                         'RightUpLeg'   : 1,
                         'RightLeg'     : 2,
                         'RightFoot'    : 3,
                         'LeftUpLeg'    : 4,
                         'LeftLeg'      : 5,
                         'LeftFoot'     : 6,
                         'Spine1'       : 7,
                         'Neck'         : 8,
                         'Head'         : 9,
                         'Site-head'    : 10,
                         'LeftArm'      : 11,
                         'LeftForeArm'  : 12,
                         'LeftHand'     : 13,
                         'RightArm'     : 14,
                         'RightForeArm' : 15,
                         'RightHand'    : 16}

    bones_pairs       = []
    bones             = [('Hips', 'RightUpLeg'), ('RightUpLeg', 'RightLeg'), ('RightLeg', 'RightFoot'),    # Right Leg is done
                         ('Hips', 'LeftUpLeg'), ('LeftUpLeg', 'LeftLeg'), ('LeftLeg', 'LeftFoot'),         # Left Leg is done
                         ('Hips', 'Spine1'), ('Spine1', 'Neck'), ('Neck', 'Head'), ('Head', 'Site-head'),  # Spine is done
                         ('Neck', 'LeftArm'), ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'),     # Left Arm is done
                         ('Neck', 'RightArm'), ('RightArm', 'RightForeArm'), ('RightForeArm', 'RightHand') # Right Arm is done
                        ]
    bones_ordering       = {('Hips', 'RightUpLeg')        : 0 , # Pelvis     <--> Right Hip
                            ('RightUpLeg', 'RightLeg')    : 1,  # Right Hip  <--> Right Knee
                            ('RightLeg', 'RightFoot')     : 2,  # Right Knee <---> Right Foot
                            ('Hips', 'LeftUpLeg')         : 3,  # Pelvis     <--> Left Hip
                            ('LeftUpLeg', 'LeftLeg')      : 4,  # Left Hip   <--> Left Knee
                            ('LeftLeg', 'LeftFoot')       : 5,  # Left Knee  <---> Left Foot
                            ('Hips', 'Spine1')            : 6,  # Pelvis     <---> Spine
                            ('Spine1', 'Neck')            : 7,  # Spine      <---> Neck
                            ('Neck', 'Head')              : 8,  # Neck       <---> Forehead
                            ('Head', 'Site-head')         : 9,  # Forehead   <---> Head
                            ('Neck', 'LeftArm')           : 10, # Neck           <---> Left Shoulder
                            ('LeftArm', 'LeftForeArm')    : 11, # Left Shoulder  <---> Left Elbow
                            ('LeftForeArm', 'LeftHand')   : 12, # Left Elbow     <---> Left Hand
                            ('Neck', 'RightArm')          : 13, # Neck           <---> Right Shoulder
                            ('RightArm', 'RightForeArm')  : 14, # Right Shoulder <---> Right Elbow
                            ('RightForeArm', 'RightHand') : 15  # Right Elbow    <---> Right Hand
                            }    

    bone_pairs_symmetric = [[('Hips', 'RightUpLeg'), ('Hips', 'LeftUpLeg')],
                            [('RightUpLeg', 'RightLeg'), ('LeftUpLeg', 'LeftLeg')],
                            [('RightLeg', 'RightFoot'), ('LeftLeg', 'LeftFoot')],
                            [('Neck', 'LeftArm'), ('Neck', 'RightArm')],
                            [('LeftArm', 'LeftForeArm'), ('RightArm', 'RightForeArm')],
                            [('LeftForeArm', 'LeftHand'), ('RightForeArm', 'RightHand')]
                            ]

    for bone in bones:
        bone_start = bone[0]; bone_start_idx = joint_names_17.index(bone_start)
        bone_end   = bone[1]; bone_end_idx   = joint_names_17.index(bone_end)
        bones_pairs.append([bone_start_idx, bone_end_idx])

    bone_pairs_symmetric_indexes = []
    for bone_pair_sym in bone_pairs_symmetric:
        right_bone, left_bone = bone_pair_sym[0], bone_pair_sym[1]
        right_bone_start      = right_bone[0]
        right_bone_end        = right_bone[1]
        left_bone_start       = left_bone[0]
        left_bone_end         = left_bone[1]
        index                 = ([joint_names_17.index(right_bone_start), joint_names_17.index(right_bone_end)],
                                 [joint_names_17.index(left_bone_start), joint_names_17.index(left_bone_end)])
        bone_pairs_symmetric_indexes.append(index)

    number_of_joints = len(joint_names_17)
    lhip_idx         = joint_names_17.index('LeftUpLeg')
    rhip_idx         = joint_names_17.index('RightUpLeg')
    neck_idx         = joint_names_17.index('Neck')
    pelvis_idx       = joint_names_17.index('Hips')
    head_idx         = joint_names_17.index('Head')
    ll_bone_idx      = [0, 1, 2, 6] # This is according to the values in <bones>. 
    rl_bone_idx      = [3, 4, 5, 6] # This is according to the values in <bones>.
    lh_bone_idx      = [7, 10, 11, 12] # This is according to the values in <bones>.
    rh_bone_idx      = [7, 13, 14, 15] # This is according to the values in <bones>.
    torso_bone_idx   = [0, 3, 6, 7, 8, 9, 10, 13] # This is according to the values in <bones>.
    body_partitions  = {"ll"    : ll_bone_idx,
                        "rl"    : rl_bone_idx,
                        "torso" : torso_bone_idx,
                        "lh"    : lh_bone_idx,
                        "rh"    : rh_bone_idx
                        }
    
    ret_vals_dict    = {'joint_names_17' : joint_names_17, 'bones' : bones, 'number_of_joints' : number_of_joints, 'lhip_idx' : lhip_idx, 'rhip_idx' : rhip_idx,
                        'neck_idx' : neck_idx, 'pelvis_idx' : pelvis_idx, 'head_idx' : head_idx, 'bone_pairs_symmetric' : bone_pairs_symmetric_indexes,
                        # 'll_bone_idx' : ll_bone_idx, 'rl_bone_idx' : rl_bone_idx, 'lh_bone_idx' : lh_bone_idx, 'rh_bone_idx' : rh_bone_idx, 'torso_bone_idx' : torso_bone_idx, 
                        'joints_ordering' : joints_ordering, 'bones_ordering' : bones_ordering, 'body_partitions' : body_partitions, 'bones_pairs' : bones_pairs
                        }
    return ret_vals_dict




def mpi_characteristics():
    """
    TODO
    """

def get_dataset_characteristics(dataset_name:str):

    if dataset_name.lower() == 'h36m':
        func = h36m_characteristics
    elif dataset_name.lower() == 'mpi':
        func = mpi_characteristics
    else:
        raise NotImplementedError("We have not implemented this for other datasets.")
    
    ret_vals_dict = func()
    return ret_vals_dict