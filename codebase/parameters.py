import os
from datasets.sport_center_basics import subject_ids


# def dataset_parameters(parser):
#     # print("Obtaining the Parameters of the Dataset.")
#     # parser.add_argument('--dataset',  type=str, default='h36m',  choices=['h36m', 'mpi', 'sport_center'], 
#     #                     help='Different Datasets to use for training the model.')
#     # parser.add_argument('--subjects_train', type=str, default='S1,S5,S6,S7,S8', help='The Training Ids.')
#     # parser.add_argument('--subjects_test',  type=str, default='S9,S11',         help='The Testing Ids.' )
#     # ## These are for CPN.
#     # parser.add_argument('--use_cpn_detections', action='store_true', default=False,
#     #                     help='If True, use the CPN detections.')
#     # parser.add_argument('--reverse_augmentation_cpn', action='store_true', default=False,
#     #                     help='If True, we will have reverse augmentation for CPN detected keypoints.')
#     # parser.add_argument('--data_augmentation_cpn', action='store_true', default=False,
#     #                     help='If True, we will apply data augmentation for the CPN detected keypoints.')
#     # parser.add_argument('--test_augmentation_cpn', action='store_true', default=False,
#     #                     help='If True, we will apply augmentation for the CPN detected keypoints in the Test Phase.')
#     # parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)
#     # parser.add_argument('--experimental_setup',                 type=str, default='fully',  choices=['semi', 'weakly', 'fully', 'un'],
#     #                 help='The various SEMI/WEAKLY/FULLY/UN supervised experimental setups.')

#     '''
#     Dataset and experimental setup.
#     '''
#     parser.add_argument('--dataset_name',                       type=str, default='h36m',  choices=['h36m', 'mpi', 'ski', 'none', 'h36_mpi', 'mpi_h36',
#                                                                                                     'sport_center'],
#                         help='Different Datasets to use for training the model.')
#     parser.add_argument('--use_cpn_detections', action='store_true', default=False,
#                         help='If True, use the CPN detections.')
#     parser.add_argument('--reverse_augmentation_cpn', action='store_true', default=False,
#                         help='If True, we will have reverse augmentation for CPN detected keypoints.')
#     parser.add_argument('--data_augmentation_cpn', action='store_true', default=False,
#                         help='If True, we will apply data augmentation for the CPN detected keypoints.')
#     parser.add_argument('--test_augmentation_cpn', action='store_true', default=False,
#                         help='If True, we will apply augmentation for the CPN detected keypoints in the Test Phase.')

#     parser.add_argument('--experimental_setup',                 type=str, default='semi',  choices=['semi', 'weakly', 'fully', 'un'],
#                         help='The various SEMI/WEAKLY/FULLY/UN supervised experimental setups.')
#     parser.add_argument('--every_nth_frame_train_annotated',    type=int, default=1,
#                         help='Consider every Nth Frame of the Annotated Samples for training.')
#     parser.add_argument('--every_nth_frame_train_unannotated',  type=int, default=1,
#                         help='Consider every Nth Frame of the UnAnnotated Samples for training.')
#     parser.add_argument('--every_nth_frame_validation',         type=int, default=1,
#                         help='Consider every Nth Frame for Validation.')
#     parser.add_argument('--every_nth_frame_test',               type=int, default=1,
#                         help='Consider every Nth Frame for Test.')
#     parser.add_argument('--batch_size',                         type=int, default=15,
#                         help='The batch size for training the model.')
#     parser.add_argument('--batch_size_test',                    type=int, default=15,
#                         help='The batch size for the validation/test phase.')
#     parser.add_argument('--num_anno_samples_per_batch',         type=int, default=0,
#                         help='The Number of Annotated Samples per Batch in a Semi-Supervised Learning Setup. '
#                             'If 0, batches can have arbitrary number of annotated samples in a batch.')
#     parser.add_argument('--only_chest_height_cameras',          action='store_true', default=False,
#                         help='Only Consider the Chest Height Cameras. Only valid for MPI dataset.')
#     parser.add_argument('--ignore_sub8_seq2',                   action='store_true', default=False,
#                         help='if True, we will remove the Sequence 2 of Subject 8 from the training set '
#                             'and use it in the Test Set for the Multiview Setup.')
#     parser.add_argument('--learning_use_h36m_model',            action='store_true', default=False,
#                         help='If True, we are basically running on MPI dataset using the model trained on H36M dataset. '
#                             'So the joints ordering should be similar to H36M.')
#     parser.add_argument('--min_visibility',                     type=float,          default=0.75,
#                         help='The Minimum Visibility needed to be considered as a poses. Only valid for MPI dataset.')
#     parser.add_argument('--multiview_sample', action='store_true', default=True, help='Only Valid for Mpi Data.')
#     parser.add_argument('--frame_idx_in_sample_test', type=int, default=2, choices=[2, 3],
#                         help='Index of the Frame index in the Test Set of MPI.')

#     parser.add_argument('--annotated_subjects',                       type=str, nargs='+', default=[],
#                         help='The Annotated Subject(s) for training the Model.')
#     parser.add_argument('--unannotated_subjects',                     type=str, nargs='+', default=[],
#                         help='The UnAnnotated Subject(s) for training the Model.')
#     parser.add_argument('--ten_percent_3d_from_all',                  action='store_true', default=False,
#                         help='If True, we will use 10% of all the images as the Annotated Set.')
#     parser.add_argument('--use_augmentations',                        action='store_true', default=False,
#                         help='If True, we will use Augmentations')
#     parser.add_argument('--consider_unused_annotated_as_unannotated', action='store_true', default=False,
#                         help='If True, we will also consider the unused samples of the annotated subject(s) in the unannotated set.')
#     parser.add_argument('--train_with_annotations_only',              action='store_true', default=False,
#                         help='If True, we will be training using only the annotated samples which will be fully supervised.')
#     parser.add_argument('--randomize',                                action='store_true', default=False,
#                         help='If True, we will be randomizing the samples which will be the output of the dataloader.')
#     parser.add_argument('--overfit_dataset',                          action='store_true', default=False,
#                         help='If True, we will be verifying our learning setup by training on 1 sample and '
#                             'thereby over-fitting on it. This is just done as a verification step.')
#     parser.add_argument('--shuffle',                                  action='store_true', default=False,
#                         help='If True, we will be shuffling the batches.')
#     parser.add_argument('--load_from_cache',                          action='store_true', default=False,
#                         help='If True, we will be loading the images from the cache.')
#     parser.add_argument('--minimum_views_needed_for_triangulation',   type=int, default=3,
#                         help='The Minimum Number of Views needed for Triangulation.')
#     parser.add_argument('--random_seed_for_ten_percent_3d_from_all',  type=int, default=2)
#     parser.add_argument('--evaluate_on_S8_seq2_mpi',     action='store_true', default=False,
#                         help='If True, we will evaluate the learnt model on the Sequence 2 of Subject 8.')
#     parser.add_argument('--calculate_K', action='store_true', default=False,
#                         help='If True, we will calculate the intrinsics for every camera for the training set.')

#     # ARGUMENTS for Loading the SPORT-CENTER DATASETS.
#     parser.add_argument('--test_subjects_sport_center', type=int, nargs='+', default=[7, 12], choices=subject_ids,
#                         help='The test subjects for the Sport Center Dataset.')
#     parser.add_argument('--num_samples_train_sport_center', type=int,  default=1000,
#                         help='The number of frames to be considered for generating the training set.')
#     parser.add_argument('--every_nth_frame_sport_center', type=int, default=3,
#                         help='The sampling rate for the frames comprising of the training set.')
#     parser.add_argument('--consider_six_views_sport_center', action='store_true',  default=True,
#                         help='If True, we will always consider the Six multi-view cameras, else we will consider the 4 side cameras.')
#     parser.add_argument('--include_all_samples_sport_center', action='store_true',  default=False,
#                         help='If True, we will consider all the samples for training the model.')
#     parser.add_argument('--start_point_sport_center', type=int, default=1000,
#                         help='The starting frame from where we begin training our model after 20k frames (with shutter open)'
#                             ' and 40k frames (with shutter closed).')
#     parser.add_argument('--hard_test_set_sport_center', action='store_true',  default=False,
#                         help='If True, we will evaluate our model on the Hard Test set of the Sport Center Dataset.')
#     parser.add_argument('--easy_test_set_sport_center', action='store_true',  default=False,
#                         help='If True, we will evaluate our model on the Easy Test set of the Sport Center Dataset.')
#     parser.add_argument('--radius_sport_center', type=float, default=0.5)

#     return parser


# def basic_parameters(parser):
#     print("Obtaining the Basic Parameters for Training.")
#     parser.add_argument('--batch_size',                         type=int, default=15,
#                     help='The batch size for training the model.')
#     parser.add_argument('--batch_size_test',                    type=int, default=15,
#                         help='The batch size for the validation/test phase.')
#     parser.add_argument('--epochs',                     type=int,  default=50,
#                         help='The number of epochs to train the model.')
#     parser.add_argument('--num_workers',                type=int,  default=1,
#                         help='The number of workers for the dataloader.')
#     parser.add_argument('--number_of_batches_to_plot',  type=int,  default=2,
#                         help='The number of Batches to plot the 2D detections.')
#     parser.add_argument('--max_train_iterations',       type=int, default=0,
#                         help='The Maximum Number of Training Iterations')

#     parser.add_argument('--eval_freq',       type=int,   default=5000,
#                         help='The Number of Training Iterations after which we will evaluate the model.')
#     parser.add_argument('--print_freq',      type=int,   default=100,
#                         help='The Number of Iterations after which we will print out the logs.')
#     parser.add_argument('--save_plot_freq',  type=int,   default=5000,
#                         help='The Number of Training Iterations after which we will save the plots.')
#     parser.add_argument('--save_model_freq', type=int,   default=5,
#                         help='The number of Epochs after which we will save the model. '
#                             'If > 1, we will not be storing the model after epoch of training.')
#     parser.add_argument('--save_dir',        type=str,   default='RES',
#                         help='The Directory to Store the Results.')
#     parser.add_argument('--normalize_range', type=float, default=1,
#                         help='Extend the Range of the input to loss from [-1, 1] to [-x to x]')

#     parser.add_argument('--small_test_set', action='store_true', default=False,
#                         help='If True, we will be using a smaller test set. This is only valid if the dataset is Sports Center.')
#     parser.add_argument('--debugging',      action='store_true', default=False,
#                         help='If True, we will be training the model in a debugging mode only for 2 epochs '
#                             'and 5 iterations to validate the overall working of the model.')
#     parser.add_argument('--plot_keypoints', action='store_true', default=False,
#                         help='If True, we will be plotting the detected 2D keypoints '
#                             'and the projection of the predicted 3D by the Refine Net.')

#     parser.add_argument('--save_file_name', default='TRIAL', type=str, help='Folder to Save the Models.')

#     return parser

# def general_optimizer_parameters(parser):
#     print("Obtaining the Parameters of the Optimizers.")
#     # General Optimizer:
#     parser.add_argument('--optimizer',       type=str,   default='adam',
#                         choices=['adam', 'rmsprop', 'sgd', 'adamw', 'rmsprop_added'],
#                         help='The various optimizers to use to optimize the learning parameters.')
#     parser.add_argument('--scheduler',       type=str,   default='none',
#                         choices=['none', None, 'step', 'exp'], help='The various schedulers to use to schedule the learning rate.')
#     parser.add_argument('--scheduler_gamma', type=float, default=0.3,
#                         help='Learning rate reduction after tau epochs. Should be close to 1 for exponential scheduling.')
#     parser.add_argument('--scheduler_tau',   type=int,   default=[80], nargs='+',
#                         help='Step size(s) before reducing learning rate.')
#     parser.add_argument('--use_different_optimizers', action='store_true', default=False,
#                         help='If True, we will use different optimizers for different models.')
#     return parser

# def training_hyperparameters(parser):
#     print("Obtaining the Parameters for learning the parameters of the lifting and (or) refinement network.")
#     parser.add_argument('--train_2d_pose_estimator',   action='store_true', default=False,
#                     help='If True, we will train the 2D pose estimator.')
#     parser.add_argument('--train_embedding_network', action='store_true', default=False,
#                     help='If True, we will train the 3D Pose --> Embedding --> 2D Pose.')                    
#     ## Defining the Optimizer for  the Lifting Network:
#     parser.add_argument('--train_lifting_net', action='store_true',  default=False,
#                         help='If True, we will train the Lifting Net.')
#     parser.add_argument('--lr_lifting_net',              type=float, default=0.0001,
#                         help='The Learning Rate to train the Lifting Net not based on ResNets.')
#     parser.add_argument('--wd_lifting_net',              type=float, default=0.0001,
#                         help='The Weight Decay to train the Lifting Net not based on ResNets.')
#     parser.add_argument('--lifting_lr_resnets_backbone', type=float, default=0.0001,
#                         help='The Learning Rate of the ResNet based BackBone in ResNets based 2D to 3D Lifting Network.')
#     parser.add_argument('--lifting_wd_resnets_backbone', type=float, default=0.0001,
#                         help='The Weight Decay of the ResNet based BackBone in ResNets based 2D to 3D Lifting Network.')
#     parser.add_argument('--lifting_lr_resnets_lifter',   type=float, default=0.0001,
#                         help='The Learning Rate of the Lifting Net Module in ResNets based 2D to 3D Lifting Network.')
#     parser.add_argument('--lifting_wd_resnets_lifter',   type=float, default=0.0001,
#                         help='The Weight Decay of the Lifting Net Module in ResNets based 2D to 3D Lifting Network.')
#     parser.add_argument('--optimizer_lifting_network', type=str,   default='adam',
#                         choices=['adam', 'rmsprop', 'sgd', 'adamw', 'rmsprop_added'],
#                         help='Separate optimizer for training the 2D-3D Lifting Net.')
#     parser.add_argument('--use_residual_lifting_in_resnet', action='store_true', default=False,
#                         help='Will be using the Residual Network to output the 3D pose.')
#     parser.add_argument('--unsup_loss_in_2d', action='store_true', default=False,
#                         help='If True, we will calculate the 2D loss between the projection of the 3D pose estimated by the '
#                              'lifting or refine network and the 2D detections.')

#     # Defining the Optimizers for the Refinement Network:
#     parser.add_argument('--train_refine_net', action='store_true', default=False,
#                         help='If True, we will train the Refine Net based on GCNs.')
#     parser.add_argument('--lr_refine_net',    type=float,          default=0.0001,
#                         help='The Learning Rate to train the Refine Net.')
#     parser.add_argument('--wd_refine_net',    type=float,          default=0.0001,
#                         help='The Weight Decay to train the Refine Net.')
#     parser.add_argument('--optimizer_refine_network', type=str,   default='adam',
#                         choices=['adam', 'rmsprop', 'sgd', 'adamw', 'rmsprop_added'],
#                         help='Separate optimizer for training the GCN Based Refine Net.')
#     return parser


# def loss_parameters(parser):
#     print("Obtaining the Parameters of the Losses.")
#     # Losses need to train the Lifting Network and the Refinement Network:
#     parser.add_argument('--calculate_loss_supervised_3d', action='store_true', default=False,
#                         help='If True, we will calculate the supervised 3D loss between the GT 3D poses '
#                             'and the poses predicted by the Refine Net.')
#     parser.add_argument('--lambda_loss_supervised_3d',    type=float,          default=0.1,
#                         help='The weight of the supervised 3D loss.')
#     parser.add_argument('--loss_supervised_3d',           type=str,            default='none',
#                         choices=['l2', 'l1', 'nse', 'mse', 'smooth_l1', 'none', None],
#                         help='The various loss functions that we can use to calculate the supervised 3D loss.')
#     parser.add_argument('--loss_supervised_3d_lifting', action='store_true', default=False,
#                         help='If True, the Input to the Supervised 3D Loss will be the output of the Lifting Network. Only for ILYA.')
#     parser.add_argument('--use_embedding_to_3d_as_inp_sup_3d', action='store_true', default=False,
#                         help='. Only for ILYA.') # TODO


#     # Ensuring Symmetric Bones in the predicted 3D poses.
#     parser.add_argument('--symmetric_bones',    action='store_true', default=False,
#                         help='If True, we will use bone length loss such that the predicted bones in 3D have symmetric length '
#                             'in left and right part of body.')
#     # parser.add_argument('--lambda_bone_length_loss', type=str,            default=1.0, help='The weight of the Bone Length Loss.')
#     parser.add_argument('--symmetric_bones_after_gcn', action='store_true', default=False,
#                         help='If True, we will use the bone length loss after the GCN based Refine Net, '
#                             'else it will be used on each of the 3D poses present in the window.')

#     # Other parameters of the losses.
#     parser.add_argument('--size_average', action='store_true', default=False,
#                         help='If True, we will be averaging the loss value by its size.')
#     parser.add_argument('--reduce',       action='store_true', default=False,
#                         help='If True, we will be applying a reduction function to the loss value.')
#     parser.add_argument('--norm',         action='store_true', default=False,
#                         help='If True, we will be normalizing the poses before calculating the loss.')
#     parser.add_argument('--reduction',    type=str,            default='none',
#                         choices=['none', 'mean', 'sum'],
#                         help='The various reduction techniques to be used for calculating the loss.')
#     parser.add_argument('--detach_gradient_target', action='store_true', default=False,
#                         help='If True, we will be detaching the target from the backpropagation graph.')
#     parser.add_argument('--print_stats_of_data',  action='store_true', default=False)
#     parser.add_argument('--not_use_norm_pose_2d_in_loss', action='store_true', default=False,
#                         help='If True, the 2D poses in the loss calculation will not be normalized using the bounding boxes, '
#                             'else we will be using the normalized 2D poses in the 2D loss calculation.')
#     parser.add_argument('--swap_inp_tar_unsup',  action='store_true', default=False,
#                         help='If False, the detected keypoints will be target and the projected keypoints will be the input, '
#                             'else it will be reversed.')
#     parser.add_argument('--writer_file_name', type=str, default='Runs')

#     parser.add_argument('--calculate_loss_supervised_2d',        action='store_true', default=False,
#                     help='If True, we will be calculating the supervised 2D loss using GT 2D.')

#     parser.add_argument('--calculate_loss_3d', action='store_true', default=False,
#                     help='If True, we will calculate the loss between the 3D predicted by the Refine Net '
#                          'and the 3D output from the Embedding.')
    
#     parser.add_argument('--calculate_loss_contrastive', action='store_true', default=False,
#                     help='If True, we will calculate the Contrastive loss between the embeddings belonging to same / different poses.')                         

#     parser.add_argument('--calculate_loss_2d', action='store_true', default=False,
#                     help='If True we will calculate the 2D loss between the output of the 2D pose estimator '
#                          'and the predicted 2D poses from the Refine Net.')

#     parser.add_argument('--calculate_temporal_consistency_loss', action='store_true', default=False,
#                     help='If True, we will be predicting consistent 3D motion across time within a given window.')

#     parser.add_argument('--calculate_multi_view_consistency_3D_loss', action='store_true', default=False,
#                         help='If True, we will be imposing a consistency loss between the 3D poses predicted for all the cameras for a given sample.')

#     return parser



# def early_stopping_conds(parser):
#     print("Obtaining the Early Stopping Criterions.")
#     parser.add_argument('--early_stopping',                  action='store_true', default=False,
#                         help='If True, we will perform Early Stopping.')
#     parser.add_argument('--patience_early_stopping',         type=int,            default=20,
#                         help='The number of steps before early stopping is performed.')
#     parser.add_argument('--delta_early_stopping',            type=float,          default=0.001,
#                         help='The delta after which early stopping is to be performed.')
#     parser.add_argument('--which_criterion_early_stopping',  type=str,            default='singleview_mpjpe',
#                         choices=['multiview_mpjpe', 'singleview_mpjpe', 'val_loss'],
#                         help='Various scores to consider for Early Stopping.')
#     parser.add_argument('--calculate_early_stopping',        action='store_true', default=False,
#                         help='If True, we will be performing Early Stopping.')
#     return parser



# def grad_clip_norm(parser):
#     print("Obtaining the Parameters to Clip the Gradient by Value or Norm.")
#     # GRADIENTS CLIPPING BY NORM or BY VALUE
#     parser.add_argument('--clip_grad_by_norm', action='store_true',  default=False,
#                         help='If True, we will clip the Norm of the Gradients of the Models using the <clip_grad_by_norm_val> below')
#     parser.add_argument('--clip_grad_by_norm_val', type=float, default=1.0,
#                         help='The maximum Value of the norm of the Gradients.')
#     parser.add_argument('--clip_grad_by_val', action='store_true',  default=False,
#                         help='If True, we will clip the gradients of the models in the range of +- <clip_grad_by_val_val>.')
#     parser.add_argument('--clip_grad_by_val_val', type=float, default=1.0,
#                         help='The range of the clipping the gradients by value.')
#     return parser


# def func_to_load_checkpoints(parser):
#     print("ARGUMENTS to LOAD THE MODEL FROM A CHECKPOINT")
#     parser.add_argument('--checkpoint_path',                     type=str,            default='',
#                         help='The path of the checkpoint where the weights of the 2D POse Estimator are stored.')
#     parser.add_argument('--load_from_checkpoint',                action='store_true', default=False,
#                         help='If True, Weights of the 2D pose Estimator will be loaded from stored checkpoint path.')
#     parser.add_argument('--load_optimizer_checkpoint',           action='store_true', default=False,
#                         help='If True, the parameters of the optimizer for the 2D pose estimator '
#                             'will also be loaded if they are stored.')
#     parser.add_argument('--load_scheduler_checkpoint',           action='store_true', default=False,
#                         help='If True, the parameters of the scheduler for the 2D pose estimator '
#                             'will also be loaded if they are stored.')
#     parser.add_argument('--evaluate_learnt_model',               action='store_true', default=False,
#                         help='If True, we will be evaluating the model by calculating MPJPE only without performing any training.')
#     parser.add_argument('--perform_test',                        action='store_true', default=False,
#                         help='If True, we will be performing various testing protocols.')
#     parser.add_argument('--get_json_files_train_set',            action='store_true', default=False,
#                         help='If True, we will be obtaining the Json files containing the 2D detected/ground-truth keypoints '
#                             'for every view, along with the predicted/ground-truth 3D of the Train Set.')
#     parser.add_argument('--get_json_files_test_set',             action='store_true', default=False,
#                         help='If True, we will be obtaining the Json files containing the 2D detected/ground-truth keypoints '
#                             'for every view, along with the predicted/ground-truth 3D of the Test Set.')
#     parser.add_argument('--json_file_name',                type=str,            default='json_file_lifting_net',
#                         help='The Json file where we will store the outputs of the process.')
#     parser.add_argument('--save_json_file_with_save_dir',  action='store_true', default=False,
#                         help='If True, The Json Files will be stored in the Directory of <config.save_dir>')
#     parser.add_argument('--plot_keypoints_from_learnt_model',    action='store_true', default=False,
#                         help='If True, we will be plotting the predicted and the target 2D Keypoints.')
#     parser.add_argument('--plot_train_keypoints',    action='store_true', default=False,
#                         help='If True, we will be plotting the predicted and the target 2D Keypoints for the TRAIN Set Only.')
#     parser.add_argument('--plot_test_keypoints',     action='store_true', default=False,
#                         help='If True, we will be plotting the predicted and the target 2D Keypoints for the TEST Set Only.')
#     parser.add_argument('--not_calculate_mpjpe_at_start_of_training', action='store_true', default=False,
#                         help='If True, we will not calculate MPJPE at the beginning of the training.')
#     return parser


# def cpn_data_params(parser):
#     print("Obtaining the parameters of CPN datasets - 1.")
#     parser.add_argument('--data_augmentation',    type=bool,  default=False)
#     parser.add_argument('--reverse_augmentation', type=bool,  default=False)
#     parser.add_argument('--test_augmentation',    type=bool,  default=False)
#     parser.add_argument('--crop_uv',              type=int,   default=0)
#     parser.add_argument('--root_path',            type=str,   default='/Users/soumava/Desktop/GCN' if os.path.exists('/Users/soumava/Desktop/GCN') else '/cvlabdata2/cvlab/datasets_soumava')
#     parser.add_argument('--actions',              type=str,   default='*')
#     parser.add_argument('--downsample',           type=int,   default=1)
#     parser.add_argument('--subset',               type=float, default=1)
#     parser.add_argument('--stride',               type=int,   default=1)
#     parser.add_argument('--frames',               type=int,   default=31)
    
#     parser.add_argument('--n_joints',     type=int, default=17)
#     parser.add_argument('--out_joints',   type=int, default=17)
#     parser.add_argument('--out_all',      type=int, default=1)
#     parser.add_argument('--out_all_test', type=int, default=0)
#     parser.add_argument('--in_channels',  type=int, default=2)
#     parser.add_argument('--out_channels', type=int, default=3)
#     parser.add_argument('--manualSeed',   type=int, default=1)
#     parser.add_argument('--MAE',          action='store_true')
#     parser.add_argument('-tds', '--t_downsample', type=int, default=1)
#     return parser


# def obtain_lifting_network(parser):
#     print("Obtaining the parameters of loading the Lifting Network.")
#     parser.add_argument('--type_lifting_network',    type=str,            default='mlp',
#                         choices=['mlp', 'residual_lifting', 'resnet50', 'resnet18', 'temporal_Pavllo', 'modulated_gcn'],
#                         help='The Type of 2D to 3D Pose Lifting Network to be used.')
#     parser.add_argument('--use_batch_norm_mlp', action='store_true', default=False,
#                         help='If True, we will have 1D Batch Norm layers for MLP')
#     parser.add_argument('--num_residual_layers', type=int, default=4,
#                         help='The number of Residual Layers in the Residual Lifter as Developed.')
#     parser.add_argument('--loss_in_camera_coordinates', action='store_true', default=False,
#                         help='IF True, we will calculate the loss in Camera Coordinates.')
#     parser.add_argument('--use_pose_refine_net_output_modulated_gcn', action='store_true', default=False,
#                         help='IF True, ')
#     parser.add_argument('--not_predict_depth',         action='store_true', default=False,
#                         help='If True, We will be directly regressing the XYZ world coordinates of the joints. '
#                             'Otherwise, we will be predicting the Depth Directly.')

#     parser.add_argument('--inp_det_keypoints', action='store_true', default=False,
#                         help='If True, the input to Lifting Network are 2D poses in image coordinates, else it will be in normalized image coordinates.')
#     parser.add_argument('--use_view_info_lifting_net', action='store_true', default=False,
#                         help='If True, we will also add the view info in the input to the lifting network.')
#     parser.add_argument('--remove_head_view_info',     action='store_true', default=False,
#                         help='If True, we will remove the head information from the input to the Lifting Network.')
#     parser.add_argument('--load_pretrained_weights', action='store_true', default=False,
#                         help='If True, we will upload the pretrained weights of the MLP trained by Leo.')
#     parser.add_argument('--encoder_dropout',         type=float,          default=0.1,
#                         help='The Dropout Rate to be used for the Lifting Network.')
#     parser.add_argument('--embedding_layer_size',    type=int, nargs='+', default=[128],
#                         help='The size of the Embedding Layer for ResNets based 2D to 3D Lifting Network.')
#     parser.add_argument('--batch_norm_eval_imagenet_backbone', action='store_true', default=False,
#                         help='If True, the Batch Normalization Layer will be in the eval model '
#                             'for the ResNets based 2D to 3D Lifting Network.')
#     parser.add_argument('--finetune_with_lifting_net',  action='store_true', default=False,
#                         help='True if You want to Finetune only the Encoder or the Lifting Network.')
#     parser.add_argument('--inp_lifting_net_is_images', action='store_true', default=False,
#                         help='If True, the input to the lifting network will be the images, else it will be the normalized 3D poses.')
#     parser.add_argument('--use_2d_pose_estimator_with_lifting_net', action='store_true', default=False,
#                         help='If True, we will be using the 2D Pos Estimator Model with a Lifting Net.')
#     parser.add_argument('--project_lifting_3d', action='store_true', default=False,
#                         help='If True, we will project the 3D poses obtained by the Lifting Net to calculate the 2D Loss,'
#                             'else the 3D pose obtained by triangulation will be projected. '
#                             'If False, the 2D loss will only update the params of the 2D Pose Estimator Model.')
#     return parser


def get_parameters(parser):
    parser.add_argument('--project_refine_net',            action='store_true', default=False,
                    help='If True, We will directly project the output of the Refine Net to 2D.')

    parser.add_argument('--save_file_name', default='models', type=str, help='Folder to Save the Models.')
    parser.add_argument('--calibration_folder', default='', type=str)

    '''
    Dataset and experimental setup.
    '''
    parser.add_argument('--dataset_name',                       type=str, default='h36m',  choices=['h36m', 'mpi', 'ski', 'none', 'h36_mpi', 'mpi_h36',
                                                                                                    'sport_center'],
                        help='Different Datasets to use for training the model.')
    parser.add_argument('--use_cpn_detections', action='store_true', default=False,
                        help='If True, use the CPN detections.')
    parser.add_argument('--reverse_augmentation_cpn', action='store_true', default=False,
                        help='If True, we will have reverse augmentation for CPN detected keypoints.')
    parser.add_argument('--data_augmentation_cpn', action='store_true', default=False,
                        help='If True, we will apply data augmentation for the CPN detected keypoints.')
    parser.add_argument('--test_augmentation_cpn', action='store_true', default=False,
                        help='If True, we will apply augmentation for the CPN detected keypoints in the Test Phase.')

    parser.add_argument('--experimental_setup',                 type=str, default='semi',  choices=['semi', 'weakly', 'fully', 'un'],
                        help='The various SEMI/WEAKLY/FULLY/UN supervised experimental setups.')
    parser.add_argument('--every_nth_frame_train_annotated',    type=int, default=1,
                        help='Consider every Nth Frame of the Annotated Samples for training.')
    parser.add_argument('--every_nth_frame_train_unannotated',  type=int, default=1,
                        help='Consider every Nth Frame of the UnAnnotated Samples for training.')
    parser.add_argument('--every_nth_frame_validation',         type=int, default=1,
                        help='Consider every Nth Frame for Validation.')
    parser.add_argument('--every_nth_frame_test',               type=int, default=1,
                        help='Consider every Nth Frame for Test.')
    parser.add_argument('--batch_size',                         type=int, default=15,
                        help='The batch size for training the model.')
    parser.add_argument('--batch_size_test',                    type=int, default=15,
                        help='The batch size for the validation/test phase.')
    parser.add_argument('--num_anno_samples_per_batch',         type=int, default=0,
                        help='The Number of Annotated Samples per Batch in a Semi-Supervised Learning Setup. '
                            'If 0, batches can have arbitrary number of annotated samples in a batch.')
    parser.add_argument('--only_chest_height_cameras',          action='store_true', default=False,
                        help='Only Consider the Chest Height Cameras. Only valid for MPI dataset.')
    parser.add_argument('--ignore_sub8_seq2',                   action='store_true', default=False,
                        help='if True, we will remove the Sequence 2 of Subject 8 from the training set '
                            'and use it in the Test Set for the Multiview Setup.')
    parser.add_argument('--learning_use_h36m_model',            action='store_true', default=False,
                        help='If True, we are basically running on MPI dataset using the model trained on H36M dataset. '
                            'So the joints ordering should be similar to H36M.')
    parser.add_argument('--min_visibility',                     type=float,          default=0.75,
                        help='The Minimum Visibility needed to be considered as a poses. Only valid for MPI dataset.')
    parser.add_argument('--multiview_sample', action='store_true', default=True, help='Only Valid for Mpi Data.')
    parser.add_argument('--frame_idx_in_sample_test', type=int, default=2, choices=[2, 3],
                        help='Index of the Frame index in the Test Set of MPI.')

    parser.add_argument('--annotated_subjects',                       type=str, nargs='+', default=[],
                        help='The Annotated Subject(s) for training the Model.')
    parser.add_argument('--unannotated_subjects',                     type=str, nargs='+', default=[],
                        help='The UnAnnotated Subject(s) for training the Model.')
    parser.add_argument('--ten_percent_3d_from_all',                  action='store_true', default=False,
                        help='If True, we will use 10% of all the images as the Annotated Set.')
    parser.add_argument('--use_augmentations',                        action='store_true', default=False,
                        help='If True, we will use Augmentations')
    parser.add_argument('--consider_unused_annotated_as_unannotated', action='store_true', default=False,
                        help='If True, we will also consider the unused samples of the annotated subject(s) in the unannotated set.')
    parser.add_argument('--train_with_annotations_only',              action='store_true', default=False,
                        help='If True, we will be training using only the annotated samples which will be fully supervised.')
    parser.add_argument('--randomize',                                action='store_true', default=False,
                        help='If True, we will be randomizing the samples which will be the output of the dataloader.')
    parser.add_argument('--overfit_dataset',                          action='store_true', default=False,
                        help='If True, we will be verifying our learning setup by training on 1 sample and '
                            'thereby over-fitting on it. This is just done as a verification step.')
    parser.add_argument('--shuffle',                                  action='store_true', default=False,
                        help='If True, we will be shuffling the batches.')
    parser.add_argument('--load_from_cache',                          action='store_true', default=False,
                        help='If True, we will be loading the images from the cache.')
    parser.add_argument('--path_cache_h36m', type=str, default = '')
    parser.add_argument('--minimum_views_needed_for_triangulation',   type=int, default=3,
                        help='The Minimum Number of Views needed for Triangulation.')
    parser.add_argument('--random_seed_for_ten_percent_3d_from_all',  type=int, default=2)
    parser.add_argument('--evaluate_on_S8_seq2_mpi',     action='store_true', default=False,
                        help='If True, we will evaluate the learnt model on the Sequence 2 of Subject 8.')
    parser.add_argument('--calculate_K', action='store_true', default=False,
                        help='If True, we will calculate the intrinsics for every camera for the training set.')

    # ARGUMENTS for Loading the SPORT-CENTER DATASETS.
    parser.add_argument('--test_subjects_sport_center', type=int, nargs='+', default=[7, 12], choices=subject_ids,
                        help='The test subjects for the Sport Center Dataset.')
    parser.add_argument('--num_samples_train_sport_center', type=int,  default=1000,
                        help='The number of frames to be considered for generating the training set.')
    parser.add_argument('--every_nth_frame_sport_center', type=int, default=3,
                        help='The sampling rate for the frames comprising of the training set.')
    parser.add_argument('--consider_six_views_sport_center', action='store_true',  default=True,
                        help='If True, we will always consider the Six multi-view cameras, else we will consider the 4 side cameras.')
    parser.add_argument('--include_all_samples_sport_center', action='store_true',  default=False,
                        help='If True, we will consider all the samples for training the model.')
    parser.add_argument('--start_point_sport_center', type=int, default=1000,
                        help='The starting frame from where we begin training our model after 20k frames (with shutter open)'
                            ' and 40k frames (with shutter closed).')
    parser.add_argument('--hard_test_set_sport_center', action='store_true',  default=False,
                        help='If True, we will evaluate our model on the Hard Test set of the Sport Center Dataset.')
    parser.add_argument('--easy_test_set_sport_center', action='store_true',  default=False,
                        help='If True, we will evaluate our model on the Easy Test set of the Sport Center Dataset.')
    parser.add_argument('--radius_sport_center', type=float, default=0.5)

    '''
    ARGUMENTS for TRAINING on H36M and MPI datasets together.
    '''
    # FOR H36M DATASET
    parser.add_argument('--every_nth_frame_train_annotated_h36m',    type=int, default=1,
                        help='Consider every Nth Frame of the Annotated Samples for training in the H36M dataset.')
    parser.add_argument('--every_nth_frame_train_unannotated_h36m',  type=int, default=1,
                        help='Consider every Nth Frame of the UnAnnotated Samples for training in the H36M dataset.')
    parser.add_argument('--annotated_subjects_h36m',                 type=str, nargs='+', default=[],
                        help='The Annotated Subject(s) of the H36M dataset for training the Model.')
    parser.add_argument('--unannotated_subjects_h36m',               type=str, nargs='+', default=[],
                        help='The UnAnnotated Subject(s) of the H36M dataset for training the Model.')
    parser.add_argument('--ten_percent_3d_from_all_h36m',            action='store_true', default=False,
                        help='If True, we will use 10% of all the images of the H36M dataset as the Annotated Set.')
    parser.add_argument('--load_from_cache_h36m',                    action='store_true', default=False,
                        help='If True, we will be loading the images from the cache for the H36M Dataset.')
    parser.add_argument('--minimum_views_needed_for_triangulation_h36m',   type=int, default=3,
                        help='The Minimum Number of Views needed for Triangulation for the H36M Dataset.')

    # FOR MPI DATASET
    parser.add_argument('--every_nth_frame_train_annotated_mpi',    type=int, default=1,
                        help='Consider every Nth Frame of the Annotated Samples for training in the MPI dataset.')
    parser.add_argument('--every_nth_frame_train_unannotated_mpi',  type=int, default=1,
                        help='Consider every Nth Frame of the UnAnnotated Samples for training in the MPI dataset.')
    parser.add_argument('--annotated_subjects_mpi',                 type=str, nargs='+', default=[],
                        help='The Annotated Subject(s) of the MPI dataset for training the Model.')
    parser.add_argument('--unannotated_subjects_mpi',               type=str, nargs='+', default=[],
                        help='The UnAnnotated Subject(s) of the MPI dataset for training the Model.')
    parser.add_argument('--ten_percent_3d_from_all_mpi',            action='store_true', default=False,
                        help='If True, we will use 10% of all the images of the H36M dataset as the Annotated Set.')
    parser.add_argument('--load_from_cache_mpi',                    action='store_true', default=False,
                        help='If True, we will be loading the images from the cache for the MPI Dataset.')
    parser.add_argument('--minimum_views_needed_for_triangulation_mpi',   type=int, default=3,
                        help='The Minimum Number of Views needed for Triangulation for the MPI Dataset.')
    parser.add_argument('--use_validation_test_set', type=str, default='mpi', choices=['mpi', 'h36m', 'both'],
                        help='The Validation set to be used while training with a combination of H36M and MPI dataset.')
    '''
    # Optimizer and Scheduler Setups.
    '''
    parser.add_argument('--train_2d_pose_estimator',   action='store_true', default=False,
                        help='If True, we will train the 2D pose estimator.')
    parser.add_argument('--pose_lr',                 type=float,          default=0.0001,
                        help='The Learning Rate to train the 2D pose estimator.')
    parser.add_argument('--pose_wd',                 type=float,          default=0.0001,
                        help='The Weight Decay to train the 2D pose estimator.')
    parser.add_argument('--temp_softmax',            type=float,          default=0.01,
                        help='The Temperature Parameter of Arg SoftMax to calculate the keypoints from 2D heatmaps.')
    parser.add_argument('--use_2D_GT_poses_directly',  action='store_true', default=False,
                        help='If True, we will use the 2D Ground Truth Keypoints directly as the input for the Triangulation Process '
                            'without training of the 2D Pose Estimator.')
    parser.add_argument('--use_2D_DET_poses_directly', action='store_true', default=False,
                        help='If True, we will use the Detected 2D Keypoints directly as the input for the Triangulation Process '
                            'without training of the 2D Pose Estimator.')
    parser.add_argument('--use_2D_mocap_poses_directly',  action='store_true', default=False,
                        help='' # TODO
                        )

    parser.add_argument('--optimizer_model_2d_pose_estimator', type=str,   default='adam',
                        choices=['adam', 'rmsprop', 'sgd', 'adamw', 'rmsprop_added'],
                        help='Separate optimizer for training the 2D pose Estimator Model.')


    parser.add_argument('--train_lifting_net', action='store_true',  default=False,
                        help='If True, we will train the Lifting Net.')
    parser.add_argument('--lr_lifting_net',              type=float, default=0.0001,
                        help='The Learning Rate to train the Lifting Net not based on ResNets.')
    parser.add_argument('--wd_lifting_net',              type=float, default=0.0001,
                        help='The Weight Decay to train the Lifting Net not based on ResNets.')
    parser.add_argument('--lifting_lr_resnets_backbone', type=float, default=0.0001,
                        help='The Learning Rate of the ResNet based BackBone in ResNets based 2D to 3D Lifting Network.')
    parser.add_argument('--lifting_wd_resnets_backbone', type=float, default=0.0001,
                        help='The Weight Decay of the ResNet based BackBone in ResNets based 2D to 3D Lifting Network.')
    parser.add_argument('--lifting_lr_resnets_lifter',   type=float, default=0.0001,
                        help='The Learning Rate of the Lifting Net Module in ResNets based 2D to 3D Lifting Network.')
    parser.add_argument('--lifting_wd_resnets_lifter',   type=float, default=0.0001,
                        help='The Weight Decay of the Lifting Net Module in ResNets based 2D to 3D Lifting Network.')
    parser.add_argument('--optimizer_lifting_network', type=str,   default='adam',
                        choices=['adam', 'rmsprop', 'sgd', 'adamw', 'rmsprop_added'],
                        help='Separate optimizer for training the 2D-3D Lifting Net.')
    parser.add_argument('--use_residual_lifting_in_resnet', action='store_true', default=False,
                        help='Will be using the Residual Network to output the 3D pose.')
    # parser.add_argument('--unsup_loss_in_2d', action='store_true', default=False,
    #                     help='If True, we will calculate the 2D loss between the projection of the 3D pose estimated by the '
    #                          'lifting or refine network and the 2D detections of the 2D pose estimator model.')


    parser.add_argument('--train_refine_net', action='store_true', default=False,
                        help='If True, we will train the Refine Net based on GCNs.')
    parser.add_argument('--lr_refine_net',    type=float,          default=0.0001,
                        help='The Learning Rate to train the Refine Net.')
    parser.add_argument('--wd_refine_net',    type=float,          default=0.0001,
                        help='The Weight Decay to train the Refine Net.')
    parser.add_argument('--optimizer_refine_network', type=str,   default='adam',
                        choices=['adam', 'rmsprop', 'sgd', 'adamw', 'rmsprop_added'],
                        help='Separate optimizer for training the GCN Based Refine Net.')


    parser.add_argument('--train_embedding_network', action='store_true', default=False,
                        help='If True, we will train the 3D Pose --> Embedding --> 2D Pose.')
    parser.add_argument('--lr_embedding_network', type=float,          default=0.0001,
                        help='The Learning Rate to train the Embedding Network.')
    parser.add_argument('--wd_embedding_network', type=float,          default=0.0001,
                        help='The Weight Decay to train the Embedding Network.')
    parser.add_argument('--optimizer_embedding_network', type=str,   default='adam',
                        choices=['adam', 'rmsprop', 'sgd', 'adamw', 'rmsprop_added'],
                        help='Separate optimizer for training the Embedding Network.')

    parser.add_argument('--use_different_optimizers', action='store_true', default=False,
                        help='If True, we will use different optimizers for different models.')
    parser.add_argument('--optimizer',       type=str,   default='adam',
                        choices=['adam', 'rmsprop', 'sgd', 'adamw', 'rmsprop_added'],
                        help='The various optimizers to use to optimize the learning parameters.')
    parser.add_argument('--scheduler',       type=str,   default='none',
                        choices=['none', None, 'step', 'exp', 'cyclic'], help='The various schedulers to use to schedule the learning rate.')
    parser.add_argument('--scheduler_gamma', type=float, default=0.3,
                        help='Learning rate reduction after tau epochs. Should be close to 1 for exponential scheduling.')
    parser.add_argument('--scheduler_tau',   type=int,   default=[80], nargs='+',
                        help='Step size(s) before reducing learning rate.')
    
    '''
    2D Joints Masking Parameters 
    '''
    
    parser.add_argument("--joints_masking_indices", type=int, nargs='+', default=[],
                        help="The indices of the 2D joints to be masked out during training.") # added by Farouk
    parser.add_argument("--joints_masking_type", type=str, choices=["random", "consistent"], default="random",
                        help="The mask type used to create masks to mask 2D joints during training.") # added by Farouk
    
    '''
    Parameters of Different Losses to be used. 
    '''

    parser.add_argument('--calculate_uncertainty_loss', action='store_true', default=False)
    parser.add_argument('--calculate_depth_only', action='store_true', default=False) # added by Farouk
    

    parser.add_argument('--calculate_loss_3d', action='store_true', default=False,
                        help='If True, we will calculate the loss between the 3D predicted by the Refine Net '
                            'and the 3D output from the Embedding.')
    parser.add_argument('--lambda_loss_3d',    type=float,          default=0.1,
                        help='The weight of the 3D loss.')
    parser.add_argument('--loss_3d',           type=str,            default='none',
                        choices=['l2', 'l1', 'nse', 'mse', 'smooth_l1', 'none', None],
                        help='The various loss functions that we can use to calculate the 3D loss.')


    parser.add_argument('--calculate_loss_supervised_3d', action='store_true', default=False,
                        help='If True, we will calculate the supervised 3D loss between the GT 3D poses '
                            'and the poses predicted by the Refine Net.')
    parser.add_argument('--lambda_loss_supervised_3d',    type=float,          default=0.1,
                        help='The weight of the supervised 3D loss.')
    parser.add_argument('--loss_supervised_3d',           type=str,            default='none',
                        choices=['l2', 'l1', 'nse', 'mse', 'smooth_l1', 'none', None],
                        help='The various loss functions that we can use to calculate the supervised 3D loss.')
    parser.add_argument('--loss_supervised_3d_lifting', action='store_true', default=False,
                        help='If True, the Input to the Supervised 3D Loss will be the output of the Lifting Network. Only for ILYA.')
    parser.add_argument('--use_embedding_to_3d_as_inp_sup_3d', action='store_true', default=False,
                        help='. Only for ILYA.') # TODO


    parser.add_argument('--calculate_loss_supervised_2d',        action='store_true', default=False,
                        help='If True, we will be calculating the supervised 2D loss using GT 2D.')
    parser.add_argument('--loss_supervised_2d_after_projection', action='store_true', default=False,
                        help='If True, the predictions for the supervised 2D loss will be the Projection '
                            'of the predicted 3D pose or it will be the detections of the 2D pose Estimator.')
    parser.add_argument('--loss_supervised_2d_detections',       action='store_true', default=False,
                        help='If True, we will be using 2D Detections of the Pose Estimator Model in the Loss Calculation, '
                            'else we will use the Embedding to 2D Projection for the loss calculation. '
                            'This is valid only for Ilya Solution as of now.')
    parser.add_argument('--lambda_loss_supervised_2d',           type=float,          default=1.0,
                        help='The weight of the supervised 2D loss.')
    parser.add_argument('--loss_supervised_2d',                  type=str,            default='none',
                        choices=['l2', 'l1', 'nse', 'mse', 'smooth_l1', 'none', None],
                        help='The various loss functions that we can use to calculate the supervised 2D loss.')

    parser.add_argument('--calculate_loss_contrastive', action='store_true', default=False,
                        help='If True, we will calculate the Contrastive loss between the embeddings belonging to same / different poses.')
    parser.add_argument('--lambda_loss_contrastive',    type=float,          default=0.1,
                        help='The weight of the Contrastive loss.')
    parser.add_argument('--temperature_contrastive',    type=float,          default=0.1,
                        help='The Temperature Factor of the Contrastive Loss.')

    parser.add_argument('--calculate_loss_2d', action='store_true', default=False,
                        help='If True we will calculate the 2D loss between the output of the 2D pose estimator '
                            'and the predicted 2D poses from the Refine Net.')
    parser.add_argument('--lambda_loss_2d',    type=float,          default=0.1,
                        help='The weight of the 2D Loss.')
    parser.add_argument('--loss_2d',           type=str,            default='none',
                        choices=['l2', 'l1', 'nse', 'mse', 'smooth_l1', 'none', None],
                        help='The various loss dlt_functions that we can use to calculate the 2D loss.')


    parser.add_argument('--symmetric_bones',    action='store_true', default=False,
                        help='If True, we will use bone length loss such that the predicted bones in 3D have symmetric length '
                            'in left and right part of body.')
    # parser.add_argument('--lambda_bone_length_loss', type=str,            default=1.0, help='The weight of the Bone Length Loss.')
    parser.add_argument('--symmetric_bones_after_gcn', action='store_true', default=False,
                        help='If True, we will use the bone length loss after the GCN based Refine Net, '
                            'else it will be used on each of the 3D poses present in the window.')

    parser.add_argument('--calculate_temporal_consistency_loss', action='store_true', default=False,
                        help='If True, we will be predicting consistent 3D motion across time within a given window.')
    parser.add_argument('--lambda_time_consistency_loss', type=str,            default=1.0,
                        help='The weight of the Time Consistency Loss.')
    parser.add_argument('--time_consistency_loss', type=str,            default='none',
                        choices=['l2', 'l1', 'nse', 'mse', 'smooth_l1', 'none', None],
                        help='The various loss dlt_functions that we can use to calculate the 2D loss.')
    parser.add_argument('--temporal_consistency_margin', type=float, default=0.1,
                        help='The Margin for checking if the poses are varying within a limit or not.')
    parser.add_argument('--temporal_loss_overall', action='store_true', default=False,
                        help='If True, the temporal loss will be for the entire sequence of frames, else it will be calculated between 2 consecutive frames, '
                            'followed by averaging of the loss.')



    parser.add_argument('--calculate_multi_view_consistency_3D_loss', action='store_true', default=False,
                        help='If True, we will be imposing a consistency loss between the 3D poses predicted for all the cameras for a given sample.')
    parser.add_argument('--lambda_multi_view_consistency_3D_loss', type=str,            default=1.0,
                        help='The weight of the Time Consistency Loss.')
    parser.add_argument('--multi_view_consistency_3D_loss', type=str,            default='none',
                        choices=['l2', 'l1', 'nse', 'mse', 'smooth_l1', 'none', None],
                        help='The various loss dlt_functions that we can use to calculate the 2D loss.')



    parser.add_argument('--size_average', action='store_true', default=False,
                        help='If True, we will be averaging the loss value by its size.')
    parser.add_argument('--reduce',       action='store_true', default=False,
                        help='If True, we will be applying a reduction function to the loss value.')
    parser.add_argument('--norm',         action='store_true', default=False,
                        help='If True, we will be normalizing the poses before calculating the loss.')
    parser.add_argument('--reduction',    type=str,            default='none',
                        choices=['none', 'mean', 'sum'],
                        help='The various reduction techniques to be used for calculating the loss.')
    parser.add_argument('--detach_gradient_target', action='store_true', default=False,
                        help='If True, we will be detaching the target from the backpropagation graph.')
    parser.add_argument('--print_stats_of_data',  action='store_true', default=False)
    parser.add_argument('--not_use_norm_pose_2d_in_loss', action='store_true', default=False,
                        help='If True, the 2D poses in the loss calculation will not be normalized using the bounding boxes, '
                            'else we will be using the normalized 2D poses in the 2D loss calculation.')
    parser.add_argument('--swap_inp_tar_unsup',  action='store_true', default=False,
                        help='If False, the detected keypoints will be target and the projected keypoints will be the input, '
                            'else it will be reversed.')


    '''
    DLT Versions.
    '''
    parser.add_argument('--with_DLT',               action='store_true', default=False,
                        help='If True, we will run our setup with DLT.') # parser.add_argument('--with_DLT', type=str, default='none', choices=['pre', 'ilya', 'dlt', 'dlt-graphs', 'none'])

    parser.add_argument('--DLT_with_weights',       action='store_true', default=False,
                        help='If True, we will run DLT with weights.')
    parser.add_argument('--use_weights_in_2d_loss', action='store_true', default=False,
                        help='If True, We will be using the weights calculated by in the 2D loss of the learning framework.')
    parser.add_argument('--dlt_version',            type=int,  default=0,         choices=[0, 1],
                        help='If 0, the DLT version of SVD will be used, else I will be using the DLT of EDO.')
    parser.add_argument('--svd_version',            type=int , default=0,         choices=[0, 1],
                        help='If 0, the standard SVD will be used in DLT else we will use the versions of Dang Zheng.')
    parser.add_argument('--weighted_dlt',           type=str,  default='geo-med',
                        choices=['geo-med-diff', 'geo-med', 'dbscan', 'med', 'med-dist', 'leo', 'from_net'],
                        help='The various methods to generate the weights. Only valid if <with_DLT> and <DLT_with_weights> are set to True.')
    parser.add_argument('--final_weights',          type=str,  default='med',     choices=['geo-mean', 'med'],
                        help='The various methods to calculate the final weights for every camera in a pair. Only valid if <with_DLT> and <DLT_with_weights> are set to True.')
    parser.add_argument('--minimum_pairs_needed',   type=int,  default=4,
                        help='The minimum number of pairs needed to calculate the weights when <weighted_dlt> is set to geo-med or leo. '
                            'Only valid if <with_DLT> and <DLT_with_weights> are set to True.')
    parser.add_argument('--fixed_cov_matrix',                  action='store_true', default=False,
                        help='If True, we will be using a fixed Covariance Matrix in the Weights Calculation by Geometric Median Method.')
    parser.add_argument('--fix_cov_val',                       type=float,          default=1.0,
                        help='The value of Covariance for a fixed Covariance Matrix in the Weights Calculation by Geometric Median Method.')
    parser.add_argument('--add_epsilon_to_weights',            action='store_true', default=False,
                        help='If True, we will be adding epsilon value to the weights to avoid NaNs in the SVD backward for stability.')
    parser.add_argument('--epsilon_value',                     type=float,          default=1.0,
                        help='The Value of Epsilon which we will add to the weights.')
    parser.add_argument('--normalize_confidences_by_sum',      action='store_true', default=False,
                        help='If True, we will normalize the confidences by their sum.')
    parser.add_argument('--method_of_geometric_median', type=str, default='prev', choices=['now', 'prev'],
                        help='If Previous, we will use the formulation of our ECCV submission else we will be using the our current formulation.')

    parser.add_argument('--remove_bad_joints',       action='store_true', default=False,
                        help='If True, we will remove the joints whose number of valid (or good) cameras are less than <minimum_valid_cameras>.')
    parser.add_argument('--use_valid_cameras_only',  action='store_true', default=False,
                        help='If True, we will only use the valid cameras for Triangulation and Projection. '
                            'Otherwise we will use the invalid cameras for Triangulation and Projection too which seems not to be correct.')
    parser.add_argument('--threshold_valid_cameras', type=float,          default=0.3,
                        help='The Value of Threshold for thresholding the confidences to obtain valid cameras for every joint.')
    parser.add_argument('--minimum_valid_cameras',   type=int,            default=3,
                        help='If number of valid cameras that must be present for a joint to be considered '
                            'in the computational loop.')

    '''
    DLT with Graphs.
    '''
    parser.add_argument('--use_graphs_in_DLT',       action='store_true', default=False,
                        help='If True, we will run our setup with DLT with Graphs.')


    '''
    Parameters for 2D Pose Estimator.
    '''
    parser.add_argument('--type_of_2d_pose_model',       type=str,            default='none',
                        choices=['crowd_pose', 'resnet152', 'none', 'crowd_pose_conf', 'resnet152_conf', 'alphapose'],
                        help='The Various 2D pose Estimators that we can use.')
    parser.add_argument('--alphapose_model_type', type=str, default='fast_pose_res50',
                        choices=['fast_pose_res50', 'fast_pose_res50_duc', 'fast_pose_res50_dcn'],
                        help='The various type of Alphapose Pose 2D Estimators we will be using in our code.')

    parser.add_argument('--fast_inference',              action='store_true', default=True,
                        help='Only Valid if <type_of_2d_pose_model> is set to <crowd_pose>.')
    parser.add_argument('--generate_conf_using_nn',      action='store_true', default=False,
                        help='If True, we will use a different neural network to generate the confidence values, '
                            'otherwise it will be generated using our proposed ECCV approach. '
                            'Remember that <with_DLT> should also be True if this is True')
    parser.add_argument('--position_to_gen_confidences', type=str,            default='none',
                        choices=['layer_1', 'layer_2', 'layer_3', 'layer_4', 'none'],
                        help='Specifies the layer from which the confidences will be generated. '
                            'Valid only if <generate_conf_using_nn> is True.')
    parser.add_argument('--resnet152_path',   type=str, default='/cvlabdata2/home/soumava/codes/Step1/FN/models/resnet152_preCOCO.pth',
                        help='The path where the pretrained weights for ResNet152 is stored.')
    parser.add_argument('--checkpoint_halpe',               type=str, default='/cvlabdata2/home/soumava/codes/Step1/Final/models/Halpe/halpe26_fast_res50_256x192.pth')
    parser.add_argument('--batch_norm_eval_pose_estimator', action='store_true', default=False,
                        help='If True, the Batch Normalization Layer will be in the eval model '
                            'for the 2D Pose Estimator Network.')


    '''
    ARGUMENTS to LOAD THE MODEL FROM A CHECKPOINT.
    '''
    parser.add_argument('--checkpoint_path',                     type=str,            default='',
                        help='The path of the checkpoint where the weights of the 2D POse Estimator are stored.')
    parser.add_argument('--load_from_checkpoint',                action='store_true', default=False,
                        help='If True, Weights of the 2D pose Estimator will be loaded from stored checkpoint path.')
    parser.add_argument('--load_optimizer_checkpoint',           action='store_true', default=False,
                        help='If True, the parameters of the optimizer for the 2D pose estimator '
                            'will also be loaded if they are stored.')
    parser.add_argument('--load_scheduler_checkpoint',           action='store_true', default=False,
                        help='If True, the parameters of the scheduler for the 2D pose estimator '
                            'will also be loaded if they are stored.')
    parser.add_argument('--evaluate_learnt_model',               action='store_true', default=False,
                        help='If True, we will be evaluating the model by calculating MPJPE only without performing any training.')
    parser.add_argument('--perform_test',                        action='store_true', default=False,
                        help='If True, we will be performing various testing protocols.')
    parser.add_argument('--get_json_files_train_set',            action='store_true', default=False,
                        help='If True, we will be obtaining the Json files containing the 2D detected/ground-truth keypoints '
                            'for every view, along with the predicted/ground-truth 3D of the Train Set.')
    parser.add_argument('--get_json_files_test_set',             action='store_true', default=False,
                        help='If True, we will be obtaining the Json files containing the 2D detected/ground-truth keypoints '
                            'for every view, along with the predicted/ground-truth 3D of the Test Set.')
    parser.add_argument('--json_file_name',                type=str,            default='json_file_lifting_net',
                        help='The Json file where we will store the outputs of the process.')
    parser.add_argument('--save_json_file_with_save_dir',  action='store_true', default=False,
                        help='If True, The Json Files will be stored in the Directory of <config.save_dir>')
    parser.add_argument('--plot_keypoints_from_learnt_model',    action='store_true', default=False,
                        help='If True, we will be plotting the predicted and the target 2D Keypoints.')
    parser.add_argument('--plot_train_keypoints',    action='store_true', default=False,
                        help='If True, we will be plotting the predicted and the target 2D Keypoints for the TRAIN Set Only.')
    parser.add_argument('--plot_test_keypoints',     action='store_true', default=False,
                        help='If True, we will be plotting the predicted and the target 2D Keypoints for the TEST Set Only.')
    parser.add_argument('--not_calculate_mpjpe_at_start_of_training', action='store_true', default=False,
                        help='If True, we will not calculate MPJPE at the beginning of the training.')
    '''
    --- EARLY STOPPING CONDITIONS
    '''
    parser.add_argument('--early_stopping',                  action='store_true', default=False,
                        help='If True, we will perform Early Stopping.')
    parser.add_argument('--patience_early_stopping',         type=int,            default=20,
                        help='The number of steps before early stopping is performed.')
    parser.add_argument('--delta_early_stopping',            type=float,          default=0.001,
                        help='The delta after which early stopping is to be performed.')
    parser.add_argument('--which_criterion_early_stopping',  type=str,            default='singleview_mpjpe',
                        choices=['multiview_mpjpe', 'singleview_mpjpe', 'val_loss'],
                        help='Various scores to consider for Early Stopping.')
    parser.add_argument('--calculate_early_stopping',        action='store_true', default=False,
                        help='If True, we will be performing Early Stopping.')

    '''
    OTHER IMPORTANT ARGUMENTS.
    '''
    parser.add_argument('--epochs',                     type=int,  default=50,
                        help='The number of epochs to train the model.')
    parser.add_argument('--num_workers',                type=int,  default=4,
                        help='The number of workers for the dataloader.')
    parser.add_argument('--number_of_batches_to_plot',  type=int,  default=2,
                        help='The number of Batches to plot the 2D detections.')
    parser.add_argument('--max_train_iterations',       type=int, default=0,
                        help='The Maximum Number of Training Iterations')

    parser.add_argument('--eval_freq',       type=int,   default=5000,
                        help='The Number of Training Iterations after which we will evaluate the model.')
    parser.add_argument('--print_freq',      type=int,   default=100,
                        help='The Number of Iterations after which we will print out the logs.')
    parser.add_argument('--save_plot_freq',  type=int,   default=5000,
                        help='The Number of Training Iterations after which we will save the plots.')
    parser.add_argument('--save_model_freq', type=int,   default=5,
                        help='The number of Epochs after which we will save the model. '
                            'If > 1, we will not be storing the model after epoch of training.')
    parser.add_argument('--save_dir',        type=str,   default='RES',
                        help='The Directory to Store the Results.')
    parser.add_argument('--normalize_range', type=float, default=1,
                        help='Extend the Range of the input to loss from [-1, 1] to [-x to x]')


    parser.add_argument('--small_test_set', action='store_true', default=False,
                        help='If True, we will be using a smaller test set. This is only valid if the dataset is Sports Center.')
    parser.add_argument('--debugging',      action='store_true', default=False,
                        help='If True, we will be training the model in a debugging mode only for 2 epochs '
                            'and 5 iterations to validate the overall working of the model.')
    parser.add_argument('--plot_keypoints', action='store_true', default=False,
                        help='If True, we will be plotting the detected 2D keypoints '
                            'and the projection of the predicted 3D by the Refine Net.')


    '''
    ARGUMENTS TO DEFINE THE LIFTING NETWORK
    '''
    parser.add_argument('--type_lifting_network',    type=str,            default='mlp',
                        choices=['mlp', 'resnet', 'resnet50', 'resnet18', 'temporal_Pavllo', 'modulated_gcn'],
                        help='The Type of 2D to 3D Pose Lifting Network to be used.')
    parser.add_argument('--use_batch_norm_mlp', action='store_true', default=False,
                        help='If True, we will have 1D Batch Norm layers for MLP')
    parser.add_argument('--loss_in_camera_coordinates', action='store_true', default=False,
                        help='IF True, we will calculate the loss in Camera Coordinates.')
    parser.add_argument('--use_pose_refine_net_output_modulated_gcn', action='store_true', default=False,
                        help='IF True, ')
    parser.add_argument('--not_predict_depth',         action='store_true', default=False,
                        help='If True, We will be directly regressing the XYZ world coordinates of the joints. '
                            'Otherwise, we will be predicting the Depth Directly.')
    parser.add_argument('--num_residual_layers', type=int, default=4,
                        help='The number of Residual Layers in the Residual Lifter as Developed.')

    parser.add_argument('--inp_det_keypoints', action='store_true', default=False,
                        help='If True, the input to Lifting Network are 2D poses in image coordinates, else it will be in normalized image coordinates.')
    parser.add_argument('--use_view_info_lifting_net', action='store_true', default=False,
                        help='If True, we will also add the view info in the input to the lifting network.')
    # parser.add_argument('--remove_head_view_info',     action='store_true', default=False,
    #                    help='If True, we will remove the head information from the input to the Lifting Network.')
    parser.add_argument('--load_pretrained_weights', action='store_true', default=False,
                        help='If True, we will upload the pretrained weights of the MLP trained by Leo.')
    parser.add_argument('--encoder_dropout',         type=float,          default=0.1,
                        help='The Dropout Rate to be used for the Lifting Network.')
    parser.add_argument('--embedding_layer_size',    type=int, nargs='+', default=[128],
                        help='The size of the Embedding Layer for ResNets based 2D to 3D Lifting Network.')
    parser.add_argument('--batch_norm_eval_imagenet_backbone', action='store_true', default=False,
                        help='If True, the Batch Normalization Layer will be in the eval model '
                            'for the ResNets based 2D to 3D Lifting Network.')
    parser.add_argument('--finetune_with_lifting_net',  action='store_true', default=False,
                        help='True if You want to Finetune only the Encoder or the Lifting Network.')
    parser.add_argument('--inp_lifting_net_is_images', action='store_true', default=False,
                        help='If True, the input to the lifting network will be the images, else it will be the normalized 3D poses.')
    parser.add_argument('--use_2d_pose_estimator_with_lifting_net', action='store_true', default=False,
                        help='If True, we will be using the 2D Pos Estimator Model with a Lifting Net.')
    parser.add_argument('--project_lifting_3d', action='store_true', default=False,
                        help='If True, we will project the 3D poses obtained by the Lifting Net to calculate the 2D Loss,'
                            'else the 3D pose obtained by triangulation will be projected. '
                            'If False, the 2D loss will only update the params of the 2D Pose Estimator Model.')

    # Parameters of the 3D Pose to the Embedding Network.
    parser.add_argument('--embedding_network',               type=str,            default='mlp', choices=['mlp', 'none'],
                        help='Type of the Embedding Network to be used after the GCN based Refine Network.')
    parser.add_argument('--embedding_sizes_mlp',             type=int,            default=[64],  nargs='+',
                        help='A list of the hidden layers of the MLP based embedding network to be used to embed'
                            ' the 3D pose obtained from the GCN based Refine Network.')
    parser.add_argument('--projection_layers_mlp',           type=int,            default=[],    nargs='+',
                        help='A list of the projection layers that will be used to project the embedding to a 2D pose of all the cameras.'
                            'If the length of this list is 0, we will directly project the 3D pose obtained '
                            'by the refine network using the camera intrinsic and extrinsic parameters.')
    parser.add_argument('--dropout_rate_embedding_network',  type=float,          default=0.1,
                        help='The drop out rate to be used in the Embedding Network.')
    parser.add_argument('--add_relu_embedding_network',      action='store_true', default=False,
                        help='If True, we will add ReLU in the Embedding Network.')
    parser.add_argument('--add_dropout_embedding_network',   action='store_true', default=False,
                        help='If True, we will add Dropout in the Embedding Network.')

    # Arguments for Refine Network based on Graph Conv
    parser.add_argument('--type_of_gcn',        type=str,            default='default', choices=['default', 'basic'],
                        help='The different type of Graph based Refine Network to be used in our implementation. '
                            'By default, we we will use the Graph Conv Layer predefined in DGL Library.')
    parser.add_argument('--predict_depth_gcn',  action='store_true', default=False,
                        help='If True, the GCN will also predict the depth of the 3D poses after using the Graphs.')
    parser.add_argument('--predict_delta_depth_gcn',  action='store_true', default=False,
                        help='If True, the GCN will also predict the increment of depth of the 3D poses after using the Graphs.')

    parser.add_argument('--inp_depth_gcn', action='store_true', default=False,
                        help='If True, the input will be depth of the 3D poses predicted by the Lifting Network or obtained by DLT.')
    parser.add_argument('--num_cameras_predict_depth', type=int,  default=1,
                        help='The number of cameras for which we are predicting the depth.')
    parser.add_argument('--hidden_sizes_gcn',   type=int, nargs='+', default=[32, 64, 128],
                        help='The various hidden sizes to be used for every relation in the Graph Based Refine Network.')
    parser.add_argument('--temporal_relations_gcn',        type=int, nargs='+', default=[1],
                        help='The various strides that must be considered to represent a relation in the temporal '
                            'to define the Graph Based Refine Network.')
    parser.add_argument('--aggregate_func_gcn', type=str,            default='mean', choices=['mean', 'max', 'sum', 'min', 'avg'],
                        help='The various aggregate functions that can be considered for every relation and every hidden size.')
    parser.add_argument('--inp_size_gcn',       type=int, default=0,
                        help='The input feature size for every node(~=joint) in the Graph based Refine Network. '
                            'If 0, it will be the XYZ world coordinates of every joint.')
    parser.add_argument('--use_relu_gcn',       action='store_true', default=False,
                        help='If True, we will use Relu as an activation function in the Graph based Refine Network.')
    parser.add_argument('--graph_nodes_name',   type=str,            default='joints',
                        help='A string to denote the name of each node in the Graph. '
                            'We only have one type of nodes which will be represented/denoted by this value.')
    parser.add_argument('--type_activation_gcn', type=str, default='relu', choices=['relu', 'leaky_relu', 'none'],
                        help='The Various Activation Functions to be used for defining the GCNs.')
    parser.add_argument('--use_batch_norm_gcn', action='store_true', default=False,
                        help='If True, we will be using Batch Norm in the GCN.')
    parser.add_argument('--output_dim_gcn', type=int, default=64,
                        help='The Output Dimension of the node features after the Graph Convolution operation.')
    parser.add_argument('--hidden_dims_gcn_relation', action='append',
                        help='To specify the hidden dimensions of the GCNs for various relations. '
                            'It must be done as follow --hidden_dims_gcn_relation <comma separated values without spaces>. '
                            'As an Example --hidden_dims_gcn_relation 32,64 --hidden_dims_gcn_relation 16,32 specifies the hidden dimensions of the GCNs for two relations.'
                            'ALWAYS remember that its length should be one more than the number of strides specified in <--temporal_relations_gcn> '
                            'to accommodate the spatial relation which is not explicitly defined.')
    parser.add_argument('--edge_weights', type=str, default='ones', choices=['ones', 'structured', 'random', 'salient'],
                        help='Different Types of Edges that can be present in the learning with Graphs.')
    parser.add_argument('--drop_edges',   action='store_true', default=False,
                        help='If True, we will be dropping a certain number of edges per relation. '
                            'The dropout rate is specified by <--edges_drop_rate> defined below.')
    parser.add_argument('--edges_drop_rate', type=float, default=0.1,
                        help='The Dropout Rate to drop a % of edges for every relation.')
    parser.add_argument('--use_other_samples_in_loss', action='store_true', default=False,
                        help='If True, the other samples of the windows will also be considered in the calculation of the loss.') # TODO EXPLANATION.
    parser.add_argument('--lambda_other_samples', default=1.0, type=float,
                        help='Weight of the other samples in the Loss. By Default, it is set to 1.0.')

    parser.add_argument("--actions_to_consider", type=str, nargs='+', default=[],
                        help='The various actions to be considered for training.')
    parser.add_argument('--delta_t_0',           type=int,            default=25,
                        help='The sampling rate of center of the every window.')
    parser.add_argument('--half_window_size',    type=int,            default=5,
                        help='The number of samples in one half of the window.')
    parser.add_argument('--time_pred',           type=int, nargs='+', default=[5],
                        help='The instance of time for predicting the pose.')
    parser.add_argument('--sampling_rate',       type=int,            default=5,
                        help='The sampling rate within each window.')
    parser.add_argument('--extend_last',         action='store_true', default=False,
                        help='If True, we will extend the last window to its complete size by repeating the last sample.')
    parser.add_argument('--mask_type',           type=str,            default='none', choices=['none', 'mask_ensemble_graph', 'mask_ensemble_joints'],
                        help='Choices of Various Edge Masking Technique.')
    parser.add_argument('--mask_gen_factor',     type=int,            default=4,
                        help='It controls the number of masks to be generated for the entire batch, which is equal to batch_size // mask_gen_factor.')
    parser.add_argument('--mask_scale',             type=float,       default=2.0,
                        help='The amount of overlap between the generated masks.')
    parser.add_argument('--num_cameras_per_window', type=int,         default=0,
                        help='The Number of cameras to be considered for each frame in the training of the lifting net with graphs.')
    parser.add_argument('--evaluate_after_gcn', action='store_true', default=False,
                        help='If True, we will be calculating the MPJPE after GCN, else it will be evaluated at the output of 2D-3D Lifting Network.')

    # OTHER IMPORTANT ARGUMENTS.
    parser.add_argument('--pretraining_with_annotated_2D', action='store_true', default=False,
                        help='If True, we will be pretraining with the Annotated 2D.')
    parser.add_argument('--use_dets_for_labeled_in_loss', action='store_true', default=False,
                        help='If False, we will be projecting the triangulated 3D for the supervised training samples')
    parser.add_argument('--tb_logs_folder', type=str, default='tb_logs')


    parser.add_argument('--predictions_data_train_file', default='', type=str,
                        help='The File which contains the necessary data for training the Lifting Net. '
                            'Basically it needs to created after training the 2D pose Estimator Model.')
    parser.add_argument('--predictions_data_test_file',  default='', type=str,
                        help='The File which contains the necessary data for evaluating the Lifting Net. '
                            'Basically it needs to created after training the 2D pose Estimator Model.')


    """
    Using a Discriminator
    """
    parser.add_argument('--use_3d_discriminator',  action='store_true',      default=False,
                        help='If True, we will be using a discriminator for the 3D poses predicted by the Lifting net / Refine Net / 2D Pose Estimator.')
    parser.add_argument('--type_3d_discriminator',      type=str,            default='kcs',
                        choices=['basic', 'kcs', 'kcs_xent_sep', 'kcs_xent_together', 'mehdi_kcs'],
                        help='Different type of 3D Discriminators that can be used.')
    parser.add_argument('--activation_mehdi', type=str, default='leakyrelu', choices=['leakyrelu', 'relu', 'sigmoid', 'sigmoid'],
                        help='The Activation function to be used in the KCS Discriminator prepared by Mehdi. ')
    parser.add_argument('--mehdi_channel',      type=int, default=1000, help='The value of channel for mehdi\'s KSC discriminator.')
    parser.add_argument('--mehdi_mid_channel',  type=int, default=100,  help='The value of mid_channel for mehdi\'s KSC discriminator.')
    parser.add_argument('--lr_3d_discriminator',        type=float,          default=0.0001,
                        help='The Learning Rate to train the Refine Net.')
    parser.add_argument('--wd_3d_discriminator',        type=float,          default=0.0001,
                        help='The Weight Decay to train the Refine Net.')
    parser.add_argument('--optimizer_3d_discriminator', type=str,   default='adam',
                        choices=['adam', 'rmsprop', 'sgd', 'adamw', 'rmsprop_added'],
                        help='Separate optimizer for training the GCN Based Refine Net.')
    parser.add_argument('--discriminator_3d_for_lifting_output', action='store_true', default=False,
                        help='If True, we will use a discriminator for 3D poses predicted by the Lifting net.')
    parser.add_argument('--discriminator_3d_for_refine_output',  action='store_true', default=False,
                        help='If True, we will use a discriminator for 3D poses predicted by the GCN Based Refine net.')
    parser.add_argument('--discriminator_3d_for_pose_output',   action='store_true',  default=False,
                        help='If True, we will use a discriminator for 3D poses obtained by triangulating the predictions of the 2D Pose Estimator.')
    parser.add_argument('--criterion_3d_discriminator', type=str, default='l2',
                        choices=['l2', 'l1', 'nse', 'mse', 'smooth_l1', 'none', 'xent', 'cross_entropy', 'bce',
                                'binary_cross_entropy', None],
                        help='The various loss functions that we can use to calculate the loss for training the discriminator.')
    parser.add_argument('--lambda_3d_discriminator', default=1.0, type=float,
                        help='Weight of the Loss obtained for the 3D discriminator. By Default, it is set to 1.0.')

    # GRADIENTS CLIPPING BY NORM or BY VALUE
    parser.add_argument('--clip_grad_by_norm', action='store_true',  default=False,
                        help='If True, we will clip the Norm of the Gradients of the Models using the <clip_grad_by_norm_val> below')
    parser.add_argument('--clip_grad_by_norm_val', type=float, default=1.0,
                        help='The maximum Value of the norm of the Gradients.')
    parser.add_argument('--clip_grad_by_val', action='store_true',  default=False,
                        help='If True, we will clip the gradients of the models in the range of +- <clip_grad_by_val_val>.')
    parser.add_argument('--clip_grad_by_val_val', type=float, default=1.0,
                        help='The range of the clipping the gradients by value.')
    
    # MIXED LOSS
    parser.add_argument("--mixed_loss", action="store_true", default=False,
                        help="If true, uses a mixed loss for the training.")
    parser.add_argument("--mixed_loss_lambda", type=float, default=1.0,
                        help="The lambda value for the mixed loss.")
    
    # STATISTICS DATASET CREATION
    parser.add_argument("--create_stats_dataset", action="store_true", default=False,
                        help="If true, creates the statistics dataset for the training set.")
    parser.add_argument("--stats_dataset_from_test_set", action="store_true", default=False,
                        help="If true, creates the statistics dataset from the validation set instead of the training set."
                        "Requires the --create_stats_dataset flag.")
    parser.add_argument("--stats_dataset_savepath", required=False, type=str, default=None,
                        help="The path to save the statistics dataset. If None, it will be saved in the same directory as the training set."
                        "Requires the --create_stats_dataset flag.")
    
    # UNCERTAINTY PLOTS AT TEST TIME
    parser.add_argument("--uncertainty_plots", action="store_true", default=False,
                        help="If true, plots and saves the uncertainty plots, and possibly associated batch data. Requires the --perform_test flag.")
    parser.add_argument("--uncertainty_plots_dim", choices=["3d", "2d"], type=str, default="3d",
                        help="The dimension of the uncertainty plots. Only valid if --uncertainty_plots is True.")
    parser.add_argument("--uncertainty_plots_match_pred_to_gt_keypoints", action="store_true", default=False,
                        help="If true, matches the predicted keypoints to the ground truth keypoints on the plot. Only valid if --uncertainty_plots is True.")
    parser.add_argument("--uncertainty_plots_normalize_coords", action="store_true", default=False,
                        help="If true, normalizes the coordinates of the 3D keypoints. Only valid if --uncertainty_plots is True.")
    
    # USE GT 2D POSES FOR TRAINING
    parser.add_argument("--lifting_use_gt_2d_keypoints", action="store_true", default=False,
                        help="If true, uses the ground truth 2D keypoints for training the lifting network. Only valid if --perform_test is False.")
    
    # TEST ON TRAINING SET
    parser.add_argument("--test_on_training_set", action="store_true", default=False,
                        help="If true, tests on the training set. Only valid if --perform_test is True.")
    
    # BACKBONE NORMALIZATION
    parser.add_argument("--lifting_backbone_normalization", default=None, type=str, choices=["batch", "layer"],
                        help="The type of normalization to be used in the backbone of the lifting network.")
    
    return parser
