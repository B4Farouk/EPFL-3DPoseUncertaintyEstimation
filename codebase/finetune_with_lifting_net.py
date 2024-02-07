import copy
#  from Models.pose_estimator_models          import obtain_2D_Pose_Estimator_Model TODO: UNCOMMENT THIS WHEN TRAINING 2D POSE ESTIMATOR
from utils                                 import obtain_parameters_to_optimize, get_optimizer_scheduler
from losses                                import get_losses_module_pose_lifting_net_together, Equalize_bone_pairs
from Trainers.trainer_only_lifting_net     import Trainer_lifting_net
from Trainers.trainer_lifting_pose_net     import Trainer_pose_estimator_lifting_net
from Trainers.trainer_only_lifting_net_cpn import Trainer_lifting_net_cpn_keypoints
from datasets.cpn_data                     import get_opt_cpn, get_cpn_dataset_train, get_cpn_dataset_test
from datasets.datasets_functions           import (get_dataset_characteristics, get_datasets_simple, get_datasets_lifting_net_only)
from Models.lifting_networks               import obtain_Lifting_Network_with_Uncertainty

# seeding
SEED = 12
import numpy
numpy.random.seed(SEED)
import torch
torch.manual_seed(SEED)

def finetune_with_lifting_net(config, device, use_cuda):
    opt_cpn_train, opt_cpn_test = get_opt_cpn(config=config)
    if config.use_2d_pose_estimator_with_lifting_net is True:
        print("[FINETUNING-MODULE]: Obtaining the 2D pose Estimator Model which will also be learnt along with the 2D-3D Lifting Network (with / without graphs).")
        # model_2d_pose_estimator, input_size_alphapose = obtain_2D_Pose_Estimator_Model(config=config) TODO: UNCOMMENT THIS WHEN TRAINING 2D POSE ESTIMATOR
    else:
        model_2d_pose_estimator = None
        input_size_alphapose    = (0, 0)

    n_joints, bone_pairs, rhip_idx, lhip_idx, neck_idx, pelvis_idx, bone_pairs_symmetric, \
        num_cameras_multiview, ll_bone_idx, rl_bone_idx, lh_bone_idx, rh_bone_idx, \
               torso_bone_idx, head_idx = get_dataset_characteristics(config)
    config.n_joints                     = n_joints
    config.rhip_idx                     = rhip_idx
    config.lhip_idx                     = lhip_idx
    config.neck_idx                     = neck_idx
    config.pelvis_idx                   = pelvis_idx
    config.input_size_alphapose         = input_size_alphapose
    config.num_samples_batch_for_graphs = config.batch_size * num_cameras_multiview
    config.bone_pairs_symmetric         = bone_pairs_symmetric
    config.bones_pairs                  = bone_pairs
    config.ll_bone_idx                  = ll_bone_idx
    config.rl_bone_idx                  = rl_bone_idx
    config.lh_bone_idx                  = lh_bone_idx
    config.rh_bone_idx                  = rh_bone_idx
    config.torso_bone_idx               = torso_bone_idx
    config.head_idx                     = head_idx

    if config.use_cpn_detections:
        print("[FINETUNING-MODULE]: Using the CPN detected keypoints for {} dataset".format(config.dataset_name.upper()))
        opt_cpn_train, \
             opt_cpn_test = get_opt_cpn(config=config)
        dataset_cpn_train = get_cpn_dataset_train(opt=opt_cpn_train)
        dataset_cpn_test  = get_cpn_dataset_test(opt=opt_cpn_test)
    else:
        dataset_cpn_train = None
        dataset_cpn_test  = None

    lifting_network = obtain_Lifting_Network_with_Uncertainty(config=config)
    use_graphs      = False
    if model_2d_pose_estimator is not None:
        train_loader, validation_loader, test_loader, train_loader_simple, \
                    mpjpe_poses_in_camera_coordinates = get_datasets_simple(config=config, training_pose_lifting_net_without_graphs=True)
    else:
        train_loader, validation_loader, test_loader, train_loader_simple, \
                    mpjpe_poses_in_camera_coordinates = get_datasets_lifting_net_only(config=config, with_graphs=use_graphs)
        
    if config.symmetric_bones:
        print("[FINETUNING-MODULE]: We will make sure that the Bones predicted in 3D have symmetric length in left and right part of the body.")
        symmetric_bones_module = Equalize_bone_pairs(number_of_joints=n_joints, bone_pairs=bone_pairs_symmetric)
    else:
        symmetric_bones_module = None

    if use_cuda:
        print("[FINETUNING-MODULE]: Converting the 2D-3D Lifting Network to Cuda.")
        lifting_network = lifting_network.to(device=device)
        if model_2d_pose_estimator is not None:
            print("[FINETUNING-MODULE]: Converting the 2D pose Estimator Model to Cuda.")
            model_2d_pose_estimator = model_2d_pose_estimator.to(device=device)
            
    if not(config.perform_test) and not(config.create_stats_dataset):
        print("[FINETUNING-MODULE]: We are in the Training mode. So need to obtain the optimizer and scheduler.")
        assert config.pretraining_with_annotated_2D is False
        params_to_optimize = obtain_parameters_to_optimize(
            config=config,
            model_2d_pose_estimator=model_2d_pose_estimator,
            lifting_network=lifting_network,
            embedding_network=None)
        
        optimizer, scheduler = get_optimizer_scheduler(
            params=params_to_optimize,
            optimizer_type=config.optimizer,
            scheduler_type=config.scheduler,
            step_val=config.scheduler_tau,
            gamma_val=config.scheduler_gamma)
        
        # assert (config.calculate_loss_supervised_3d or config.calculate_loss_supervised_2d) is True
        assert config.calculate_uncertainty_loss is True
        total_loss_key          = 'total_loss'
        losses_module, losses_keys, \
                print_loss_keys = get_losses_module_pose_lifting_net_together(config=config, device=device, total_loss_key=total_loss_key)
        individual_losses_names = list(losses_module.keys())

    else:
        print("[FINETUNING-MODULE]: We are in the Evaluation / Test mode. So there is NO need to obtain the optimizer and scheduler.")
        optimizer, scheduler    = None, None
        total_loss_key, losses_module, losses_keys, print_loss_keys = '', {}, {}, {}
        individual_losses_names = []


    max_num_joints_to_alter = 3
    if config.type_lifting_network == 'resnet' or config.type_lifting_network == 'modulated_gcn': 
        inp_lifting_net_is_images = False
    elif config.type_lifting_network == 'resnet18' or config.type_lifting_network == 'resnet50':
        raise NotImplementedError("This {} type of Lifting Network has not been coded into.".format(config.type_lifting_network))
        inp_lifting_net_is_images = True
    else:
        raise ValueError("This {} type of Lifting Network has not been coded into.")
    

    config.inp_lifting_net_is_images = inp_lifting_net_is_images
    lifting_network_orig             = copy.deepcopy(lifting_network)
    unsup_loss_in_2d                 = False #config.not_predict_depth # config.unsup_loss_in_2d
    
    if config.test_on_training_set:
        assert config.perform_test, "If you want to test on the training set, you need to set perform_test to True."
    
    if config.stats_dataset_from_test_set:
        assert config.create_stats_dataset,\
            "If you want to create the stats dataset from the test set, you need to set create_stats_dataset to True."
    
    use_train_loader_as_validation_and_test_loader = (
        (config.test_on_training_set and config.perform_test)
        or (config.create_stats_dataset and not config.stats_dataset_from_test_set)
    )
    
    dict_vals_lifting_net            = {'individual_losses_names'             : individual_losses_names,
                                        'losses_keys'                         : losses_keys,
                                        'save_dir'                            : config.save_dir,
                                        'device'                              : device,
                                        'train_loader'                        : train_loader,
                                        'validation_loader'                   : (
                                            train_loader if use_train_loader_as_validation_and_test_loader else validation_loader),
                                        'test_loader'                         : (
                                            train_loader if use_train_loader_as_validation_and_test_loader else test_loader),
                                        'debugging'                           : config.debugging,
                                        'save_plot_freq'                      : config.save_plot_freq,
                                        'save_model_freq'                     : config.save_model_freq,
                                        'use_cuda'                            : use_cuda,
                                        'print_freq'                          : config.print_freq,
                                        'tb_logs_folder'                      : config.tb_logs_folder,
                                        'dataset_name'                        : config.dataset_name.lower(),
                                        'eval_freq'                           : config.eval_freq,
                                        'json_file_name'                      : config.json_file_name,
                                        'save_json_file_with_save_dir'        : config.save_json_file_with_save_dir,
                                        'lifting_network_orig'                : lifting_network_orig,
                                        'n_joints'                            : n_joints,
                                        'bone_pairs'                          : bone_pairs,
                                        'rhip_idx'                            : rhip_idx,
                                        'lhip_idx'                            : lhip_idx,
                                        'neck_idx'                            : neck_idx,
                                        'pelvis_idx'                          : pelvis_idx,
                                        'head_idx'                            : head_idx,
                                        'use_view_info_lifting_net'           : config.use_view_info_lifting_net,
                                        'print_loss_keys'                     : print_loss_keys,
                                        'num_views'                           : 1,
                                        'loss_modules'                        : losses_module,
                                        'patience_early_stopping'             : config.patience_early_stopping,
                                        'delta_early_stopping'                : config.delta_early_stopping,
                                        'save_file_name'                      : config.save_file_name,
                                        'plot_keypoints'                      : config.plot_keypoints,
                                        'train_loader_simple'                 : train_loader_simple,
                                        'number_of_batches_to_plot'           : config.number_of_batches_to_plot,
                                        'total_loss_key'                      : total_loss_key,
                                        'max_num_joints_to_alter'             : max_num_joints_to_alter,
                                        'mpjpe_poses_in_camera_coordinates'   : mpjpe_poses_in_camera_coordinates,
                                        'symmetric_bones'                     : config.symmetric_bones,
                                        'symmetric_bones_module'              : symmetric_bones_module,
                                        'inp_lifting_net_is_images'           : inp_lifting_net_is_images,
                                        'calculate_early_stopping'            : config.calculate_early_stopping,
                                        'detach_gradient_target'              : config.detach_gradient_target,
                                        'use_other_samples_in_loss'           : config.use_other_samples_in_loss,
                                        'lambda_other_samples'                : config.lambda_other_samples,
                                        'swap_inp_tar_unsup'                  : config.swap_inp_tar_unsup,
                                        'max_train_iterations'                : config.max_train_iterations,
                                        'clip_grad_by_norm'                   : config.clip_grad_by_norm,
                                        'clip_grad_by_norm_val'               : config.clip_grad_by_norm_val,
                                        'clip_grad_by_val'                    : config.clip_grad_by_val,
                                        'clip_grad_by_val_val'                : config.clip_grad_by_val_val,
                                        'unsup_loss_in_2d'                    : unsup_loss_in_2d,
                                        'pose_model_name'                     : config.type_of_2d_pose_model,
                                        'experimental_setup'                  : config.experimental_setup,
                                        'inp_lifting_det_keypoints'           : config.inp_det_keypoints,
                                        'loss_in_camera_coordinates'          : config.loss_in_camera_coordinates,
                                        'lifting_use_gt_2d_keypoints'         : config.lifting_use_gt_2d_keypoints,
                                        'joints_masking_indices'              : config.joints_masking_indices,
                                        'joints_masking_type'                 : config.joints_masking_type
                                        }
    
    project_lifting_3d = False #config.project_lifting_3d
    if not config.use_2d_pose_estimator_with_lifting_net:
        if not config.use_cpn_detections:
            Trainer = Trainer_lifting_net(**dict_vals_lifting_net) # This is with/without graphs.
        else:
            Trainer = Trainer_lifting_net_cpn_keypoints(dict_vals_base=dict_vals_lifting_net,
                                                            crop_uv=opt_cpn_train.crop_uv, out_channels=opt_cpn_train.out_channels,
                                                            in_channels=opt_cpn_train.in_channels, pad=opt_cpn_train.pad,
                                                            use_pose_refine_net_output_modulated_gcn=config.use_pose_refine_net_output_modulated_gcn)
    
    else:
        
        dict_vals_extra = {'use_2D_GT_poses_directly'           : config.use_2D_GT_poses_directly,
                           'use_2D_DET_poses_directly'          : config.use_2D_DET_poses_directly,
                           'temp_softmax'                       : config.temp_softmax,
                           'dlt_version'                        : config.dlt_version,
                           'svd_version'                        : config.svd_version,
                           'weighted_dlt'                       : config.weighted_dlt,
                           'final_weights'                      : config.final_weights,
                           'calc_weights'                       : config.DLT_with_weights,
                           'minimum_pairs_needed'               : config.minimum_pairs_needed,

                           'method_of_geometric_median'         : config.method_of_geometric_median,
                           'fixed_cov_matrix'                   : config.fixed_cov_matrix,
                           'fix_cov_val'                        : config.fix_cov_val,
                           'add_epsilon_to_weights'             : config.add_epsilon_to_weights,
                           'epsilon_value'                      : config.epsilon_value,
                           'remove_bad_joints'                  : config.remove_bad_joints,
                           'use_valid_cameras_only'             : config.use_valid_cameras_only,
                           'threshold_valid_cameras'            : config.threshold_valid_cameras,
                           'normalize_confidences_by_sum'       : config.normalize_confidences_by_sum,
                           'minimum_valid_cameras'              : config.minimum_valid_cameras,

                           'calculate_loss_in_2d'               : config.calculate_loss_supervised_2d,
                           'use_weights_in_2d_loss'             : config.use_weights_in_2d_loss,
                           'use_dets_for_labeled_in_loss'       : config.use_dets_for_labeled_in_loss,

                           # This is valid only for labeled samples.
                           'not_use_norm_pose_2d_in_loss'       : config.not_use_norm_pose_2d_in_loss,
                           'project_lifting_3d'                 : project_lifting_3d,
                           }
        
        Trainer = Trainer_pose_estimator_lifting_net(**dict_vals_extra,  # This is with/without graphs.
                                                     dict_vals_base=dict_vals_lifting_net)
        

    Trainer.main_process(model_2d_pose_estimator=model_2d_pose_estimator,
                         lifting_network=lifting_network, optimizer=optimizer, scheduler=scheduler, config=config)