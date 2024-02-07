import os
import torch
import time
import copy
import datetime


import torch.nn as nn
import numpy    as np

from torch.utils.tensorboard.writer import SummaryWriter
from utils                          import (transfer_partial_weights, get_state_dict_from_multiple_gpu_to_single, calculate_mpjpe_, mkdir_if_missing, 
                                            json_write, calculate_pck)



class Trainer_lifting_net(object):
    def __init__(self,
                 individual_losses_names,
                 losses_keys,
                 save_dir,
                 device,
                 train_loader,
                 validation_loader,
                 test_loader,
                 train_loader_simple,
                 debugging,
                 save_plot_freq,
                 save_model_freq,
                 use_cuda,
                 print_freq,
                 writer_file_name,
                 dataset_name,
                 eval_freq,
                 json_file_name,
                 save_json_file_with_save_dir,
                 lifting_network_orig,
                 
                 discriminator_3d_net_orig,
                 
                 number_of_joints,
                 bone_pairs,
                 rhip_idx,
                 lhip_idx,
                 neck_idx,
                 pelvis_idx,
                 head_idx,
                 print_loss_keys,
                 num_views,
                 loss_modules,
                 patience_early_stopping,
                 delta_early_stopping,
                 save_file_name,
                 plot_keypoints,
                 number_of_batches_to_plot,
                 total_loss_key,

                 
                 max_num_joints_to_alter,
                 symmetric_bones,
                 symmetric_bones_module,
                 
                 inp_lifting_net_is_images,
                 
                 
                 calculate_early_stopping            : bool,
                 detach_gradient_target              : bool,
                 use_other_samples_in_loss           : bool,
                 lambda_other_samples                : float,
                 discriminator_3d_for_lifting_output : bool,
                 
                 type_3d_discriminator      : str,
                 swap_inp_tar_unsup         : bool,
                 use_view_info_lifting_net  : bool,
                 
                 max_train_iterations   : bool,
                 clip_grad_by_norm      : bool,
                 clip_grad_by_norm_val  : float,
                 clip_grad_by_val       : bool,
                 clip_grad_by_val_val   : float,
                 unsup_loss_in_2d       : bool,
                 
                 experimental_setup         : str,
                 inp_lifting_det_keypoints  : bool,
                 project_lifting_3d         : bool,
                 out_all_test
                 ):

        self.loss_modules       = loss_modules
        self.losses_keys        = losses_keys
        self.save_dir           = save_dir
        self.device             = device
        self.train_loader       = train_loader
        self.dataset_name       = dataset_name
        self.eval_freq          = eval_freq
        self.count_iterations   = 0
        self.plot_keypoints     = plot_keypoints
        self.validation_loader  = validation_loader
        self.test_loader        = test_loader
        self.debugging          = debugging
        self.save_plot_freq     = save_plot_freq
        self.save_model_freq    = save_model_freq
        self.use_cuda           = use_cuda
        self.print_freq         = print_freq
        self.writer_file_name   = writer_file_name
        self.number_of_joints   = number_of_joints
        self.json_file_name     = json_file_name
        self.bone_pairs         = bone_pairs
        self.rhip_idx           = rhip_idx
        self.lhip_idx           = lhip_idx
        self.neck_idx           = neck_idx
        self.pelvis_idx         = pelvis_idx
        self.head_idx           = head_idx
        self.print_loss_keys    = print_loss_keys
        self.num_views          = num_views
        self.save_file_name     = save_file_name
        self.total_loss_key     = total_loss_key
        self.m1, self.m2        = 0.02, 0.05
        self.out_all_test       = out_all_test
        
        
        self.swap_inp_tar_unsup = swap_inp_tar_unsup
        
        self.unsup_loss_in_2d   = unsup_loss_in_2d
        self.project_lifting_3d = project_lifting_3d

        
        self.inp_lifting_det_keypoints       = inp_lifting_det_keypoints
        self.discriminator_3d_net_orig       = discriminator_3d_net_orig
        self.use_other_samples_in_loss       = use_other_samples_in_loss
        self.lambda_other_samples            = lambda_other_samples
        self.detach_gradient_target          = detach_gradient_target
        self.calculate_early_stopping        = calculate_early_stopping
        self.inp_lifting_net_is_images       = inp_lifting_net_is_images
        self.number_of_batches_to_plot       = number_of_batches_to_plot
        self.train_loader_simple             = train_loader_simple
        self.best_mpjpe_sv                   = float(np.inf)
        self.best_epoch_based_on_mpjpe_sv    = 0
        self.best_iters_based_on_mpjpe_sv    = 0
        self.best_mpjpe_obtained_sv          = False
        self.best_lifting_network_sv         = None
        self.best_model_2d_pose_estimator_sv = None
        self.best_refine_network_sv          = None
        self.best_discriminator_sv           = None
        self.symmetric_bones_after_gcn       = False
        self.evaluate_after_gcn              = False
        
        self.final_loss_values               = None
        self.individual_losses_names         = individual_losses_names
        self.lifting_network_orig            = lifting_network_orig
        
        self.state_dict_keys                 = {'optimizer'            : 'state_dict_optimizer',
                                                'scheduler'            : 'state_dict_scheduler',
                                                'lifting_network'      : 'state_dict_lifting_network',
                                                'refine_network'       : 'state_dict_refine_network',
                                                'pose_2d_estimator'    : 'state_dict_pose_model',
                                                'discriminator_3d_net' : 'discriminator_3d_network'
                                                }
        self.patience_early_stopping      = patience_early_stopping
        self.delta_early_stopping         = delta_early_stopping
        self.counter_early_stopping       = 0
        self.best_score_early_stopping    = None
        self.early_stop_flag              = False
        self.val_loss_min_early_stopping  = np.Inf
        self.save_json_file_with_save_dir = save_json_file_with_save_dir
        self.pck_thresholds               = [0.1, 0.15, 0.2]
        self.use_view_info_lifting_net    = use_view_info_lifting_net
        

        
        self.max_num_joints_to_alter             = max_num_joints_to_alter

        self.discriminator_3d_for_refine_output  = False
        self.discriminator_3d_for_lifting_output = discriminator_3d_for_lifting_output
        self.using_discriminator_3d              = self.discriminator_3d_for_lifting_output or self.discriminator_3d_for_refine_output \
                                                   
        self.type_3d_discriminator               = type_3d_discriminator
        self.inp_3d_discriminator                = 'world' if self.type_3d_discriminator == 'basic' else 'camera'


        self.symmetric_bones               = symmetric_bones
        self.symmetric_bones_module        = symmetric_bones_module
        self.lifting_preds_key             = 'lifting_3d'
        self.lifting_targets_key           = 'target_3d'
        self.preds_triangulation_3d_key    = 'pred_3d_dlt'
        self.targets_triangulation_3d_key  = 'target_3d_dlt'
        self.pose_2d_key                   = 'pose_2d'
        self.target_2d_key                 = 'target_2d'
        self.weights_2d_key                = 'weights_2d_key'
        self.poses_2d_proj_norm_key        = 'poses_2d_proj_norm'
        self.poses_2d_det_norm_key         = 'poses_2d_det_norm'
        self.weights_2d_pose_key           = 'weights_2d_pose'
        self.target_pose_2d_norm_key       = 'target_pose_2d_norm'
        self.labels_loss_2d_key            = 'labels_loss_2d'
        self.pose_2d_proj_loss_key         = 'pose_2d_proj_loss'
        self.target_pose_2d_loss_key       = 'target_pose_2d_loss'
        self.keypoints_det_dist_loss_key   = 'keypoints_det_dist_loss'
        self.pred_3d_key                   = 'pred_3d'
        self.target_3d_key                 = 'target_3d'
        self.temporal_3d_key               = 'temporal_3d'
        self.weights_3d_key                = 'weights_3d'


        self.max_train_iterations          = max_train_iterations
        self.max_train_iterations_obtained = False
        self.clip_grad_by_norm             = clip_grad_by_norm
        self.clip_grad_by_norm_val         = clip_grad_by_norm_val
        self.clip_grad_by_val              = clip_grad_by_val
        self.clip_grad_by_val_val          = clip_grad_by_val_val
        self.experimental_setup            = experimental_setup

        self.reset_loss_params()



    def get_writer_extension(self):
        """
        :param self: The Trainer Class.
        :return: The filename to store the parameters of TensorBoard.
        """
        savepath = os.path.join(self.save_dir)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        write_file_name = os.path.join(savepath, self.writer_file_name)
        return write_file_name

    def perform_early_stopping(self, score):
        """
        Function to perform Early Stopping of Learning due to divergence or over-fitting.
        :param score: The score to be used for checking the divergence or over-fitting of the model.
        """
        if self.best_score_early_stopping is None:
            self.best_score_early_stopping = score

        elif score > (self.best_score_early_stopping + self.delta_early_stopping):
            self.counter_early_stopping += 1
            print(f'EarlyStopping counter: {self.counter_early_stopping} out of {self.patience_early_stopping}')
            if self.counter_early_stopping >= self.patience_early_stopping:
                self.early_stop_flag = True
                print("Training is to be Stopped Now.")
        else:
            self.best_score_early_stopping = score
            self.counter_early_stopping    = 0

    def reset_loss_params(self):
        """
        Function to get reset the loss value params for printing.
        :return: None
        """
        Losses = {}
        for loss_name in self.individual_losses_names:
            Losses[loss_name]       = []
        Losses[self.total_loss_key] = []
        self.final_loss_values      = Losses

    def load_from_checkpoint(self, load_file_name, optimizer, scheduler, load_optimizer_checkpoint,
                             load_scheduler_checkpoint, lifting_network, refine_network, discriminator_3d_net, model_2d_pose_estimator=None):

        """
        Function to Load the weights of the Lifting Network and other modules such as Optimizer/Scheduler if Needed.
        :param load_file_name: The Checkpoint File to load the Pretrained Weights for all the modules.
        :param optimizer: The Optimizer used for optimization.
        :param scheduler: The Scheduler used for Optimization.
        :param load_optimizer_checkpoint: If True, optimizer will also be loaded with its stored state in the checkpoint file.
        :param load_scheduler_checkpoint: If True, scheduler will also be loaded with its stored state in the checkpoint file.
        :param lifting_network: The 2D to 3D Lifting Network.
        :param refine_network: The GCN based Refine Network.
        :param discriminator_3d_net : The 3D discriminator used.
        :return: The start epoch, the optimizer, the scheduler (if present) and the Lifting Network.
        """

        print("Loading from {}".format(load_file_name))
        if not os.path.exists(load_file_name):
            raise ValueError("Wrong checkpoint file.")
        checkpoint      = torch.load(load_file_name)
        start_epoch     = checkpoint['epoch']
        keys_checkpoint = list(checkpoint.keys())
        
        if model_2d_pose_estimator is not None and self.state_dict_keys['pose_2d_estimator'] in keys_checkpoint:
            print("Loading the Weights of the 2D Pose Estimator.")
            pose_model_old_state_dict = checkpoint[self.state_dict_keys['pose_2d_estimator']]
            pose_model_new_state_dict = get_state_dict_from_multiple_gpu_to_single(old_state_dict=pose_model_old_state_dict)
            model_2d_pose_estimator   = transfer_partial_weights(model=model_2d_pose_estimator, pretrained_state_dict=pose_model_new_state_dict)
            print("Done Loading the Weights of the 2D Pose Estimator.")

        if lifting_network is not None and self.state_dict_keys['lifting_network'] in keys_checkpoint:
            print("Loading the Weights of the 2D to 3D Lifting Network.")
            lifting_network_old_state_dict = checkpoint[self.state_dict_keys['lifting_network']]
            lifting_network_new_state_dict = get_state_dict_from_multiple_gpu_to_single(old_state_dict=lifting_network_old_state_dict)
            lifting_network                = transfer_partial_weights(model=lifting_network, pretrained_state_dict=lifting_network_new_state_dict)
            print("Done Loading the Weights of the 2D to 3D Lifting Network.")

        if refine_network is not None and self.state_dict_keys['refine_network'] in keys_checkpoint:
            print("Loading the Weights of the GCN based Refine Network on 3D poses.")
            refine_network_old_state_dict = checkpoint[self.state_dict_keys['refine_network']]
            refine_network_new_state_dict = get_state_dict_from_multiple_gpu_to_single(old_state_dict=refine_network_old_state_dict)
            refine_network                = transfer_partial_weights(model=refine_network, pretrained_state_dict=refine_network_new_state_dict)
            print("Done Loading the Weights of the GCN based Refine Network.")


        if discriminator_3d_net is not None and self.state_dict_keys['discriminator_3d_net'] is keys_checkpoint:
            print("Loading the Weights of the 3D Discriminator Network.")
            discriminator_3d_net_old_state_dict = checkpoint[self.state_dict_keys['discriminator_3d_net']]
            discriminator_3d_net_new_state_dict = get_state_dict_from_multiple_gpu_to_single(old_state_dict=discriminator_3d_net_old_state_dict)
            discriminator_3d_net                = transfer_partial_weights(model=discriminator_3d_net, pretrained_state_dict=discriminator_3d_net_new_state_dict)

        if load_optimizer_checkpoint and optimizer is not None and self.state_dict_keys['optimizer'] in keys_checkpoint:
            print("Loading the Stored State of the Optimizer.")
            optimizer_pose_old_state_dict = checkpoint[self.state_dict_keys['optimizer']]
            # if isinstance(optimizer, MultipleInstances):
            #     optimizer.load_state_dict(state_dict=optimizer_pose_old_state_dict)
            # else:
            optimizer_pose_new_state_dict = get_state_dict_from_multiple_gpu_to_single(old_state_dict=optimizer_pose_old_state_dict)
            optimizer.load_state_dict(optimizer_pose_new_state_dict)
            print("Done Loading the Stored State of the Optimizer.")

        if load_scheduler_checkpoint and scheduler is not None and self.state_dict_keys['scheduler'] in keys_checkpoint:
            print("Loading the Stored State of the Scheduler.")
            scheduler_pose_old_state_dict = checkpoint[self.state_dict_keys['scheduler']]
            # if isinstance(scheduler, MultipleInstances):
            #     scheduler.load_state_dict(state_dict=scheduler_pose_old_state_dict)
            # else:
            scheduler_pose_new_state_dict = get_state_dict_from_multiple_gpu_to_single(scheduler_pose_old_state_dict)
            scheduler.load_state_dict(scheduler_pose_new_state_dict)
            print("Done Loading the Stored State of the Scheduler.")

        ret_vals_dict = {'start_epoch' : start_epoch, 'optimizer' : optimizer, 'scheduler': scheduler,  'lifting_network' : lifting_network, 
                         'refine_network': refine_network, 'model_2d_pose_estimator' : model_2d_pose_estimator, 'discriminator_3d_net' : discriminator_3d_net}
        return ret_vals_dict


    def save_model(self, lifting_network, refine_network, discriminator_3d_net, optimizer,
                   scheduler, epoch, iterations, saving_best, model_2d_pose_estimator=None, suffix=None, save_with_iters: bool = False):
        """
        Function to save the necessary parameters of the Networks and Optimizer/Scheduler if needed.
        :param lifting_network          : The 2D to 3D Lifting Network.
        :param refine_network           : The GCN based Refine Network.
        :param optimizer                : The optimizer used to learn the model params.
        :param scheduler                : The optimizer used to learn the model params.
        :param epoch                    : The current epoch of saving the models.
        :param iterations               : The current iteration of saving the models.
        :param saving_best              : If True, we will store the best models.
        :param suffix                   : The string which will be added to save filename.
        :param model_2d_pose_estimator  : The 2D pose Estimator Model.
        :param discriminator_3d_net     : The 3D discriminator used.
        :param save_with_iters          : If True, we will also add the iteration number in the save filename.
        """

        print("Saving the Model after epoch {} and Total Iterations {} of Training.".format(epoch, iterations), end="\t")
        save_file_name = self.save_file_name
        savepath       = os.path.join(self.save_dir, save_file_name)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        save_params    = {'epoch': epoch, 'niters': iterations}

        if model_2d_pose_estimator is not None:
            # Saving the Parameters of the 2D Pose Estimator Model.
            save_params[self.state_dict_keys['pose_2d_estimator']] = model_2d_pose_estimator.state_dict()

        if lifting_network is not None:
            # Saving the parameters of the 2D-3D Lifting Network.
            save_params[self.state_dict_keys['lifting_network']]   = lifting_network.state_dict()

        if refine_network is not None:
            save_params[self.state_dict_keys['refine_network']]    = refine_network.state_dict()

        if discriminator_3d_net is not None:
            save_params[self.state_dict_keys['discriminator_3d_net']] = discriminator_3d_net.state_dict()

        if optimizer is not None:
            # Saving the parameters of the optimizer
            # if isinstance(optimizer, MultipleInstances):
            #     save_params[self.state_dict_keys['optimizer']] = optimizer.get_state_dict()
            # else:
            save_params[self.state_dict_keys['optimizer']] = optimizer.state_dict()

        if scheduler is not None:
            # if isinstance(scheduler, MultipleInstances):
            #     save_params[self.state_dict_keys['scheduler']] = scheduler.get_state_dict()
            # else:  # Saving the parameters of the scheduler
            save_params[self.state_dict_keys['scheduler']] = scheduler.state_dict()

        if saving_best:
            assert suffix is not None
            file_path = savepath + os.sep + 'model_best-{}.pth.tar'.format(suffix)
        else:
            extra = [str(epoch)]
            if save_with_iters:
                extra.append(str(iterations))
            if suffix is not None:
                extra.append(suffix)
            extra     = '-'.join(extra)

            file_path = savepath + os.sep + 'model-{}.pth.tar'.format(extra)

        print("We are storing the parameters of models in {}.".format(file_path))
        torch.save(save_params, file_path)


    def print_loss(self, loss_values_per_iteration_to_print: dict):
        """
        Function to print the loss values.
        :param loss_values_per_iteration_to_print: A dictionary containing the various losses obtained in a iteration.
        :return The String to be print after the iteration.
        """
        string_to_print = []
        losses_present  = list(loss_values_per_iteration_to_print.keys())
        for loss_name in losses_present:
            print_str = self.print_loss_keys[loss_name]
            iter_val  = loss_values_per_iteration_to_print[loss_name][0]
            avg_val   = loss_values_per_iteration_to_print[loss_name][1]
            string_to_print.append(print_str.format(iter_val, avg_val))
        string_to_print = ' , '.join(string_to_print)
        return string_to_print

    
    def function_to_calculate_loss_per_iteration(self, vals: dict, batch_idx: int):
        """
        :param vals: A dictionary consisting of the various values needed to calculate the loss of a given iteration.
        :param batch_idx: The current batch index.
        :return: loss_values_per_iteration_to_print --> A dictionary containing the loss values of different loss modules
                                                        along with the overall loss for Every Iteration for Printing.
                flag_explode ---> A flag which indicates whether the losses have exploded. If True, I will store all the checkpoints immediately.
        """
        loss_values_per_iteration_to_print = {}; total_loss = []; flag_explode = False; vals_keys = list(vals.keys())
        for loss_name in self.individual_losses_names:
            loss_keys      = self.losses_keys[loss_name]
            pred_key       = loss_keys[0]
            tar_key        = loss_keys[1]
            weight_key     = loss_keys[2]
            pred_val       = vals[pred_key]
            tar_val        = vals[tar_key]
            if tar_val is not None and self.detach_gradient_target and loss_name != 'loss_3d_pose_discriminator':
                tar_val = tar_val.detach()

            weight_val     = None if weight_key not in vals_keys else vals[weight_key]
            loss_module    = self.loss_modules[loss_name]
            loss_vals_dict = {'pred': pred_val, 'target': tar_val, 'weights': weight_val}
            loss_val       = loss_module(**loss_vals_dict)
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                # print("Loss is Nan.")
                continue
            else:
                total_loss.append(loss_val)
                xxx = np.mean(self.final_loss_values[loss_name])
                self.final_loss_values[loss_name].append(loss_val.item())
                yyy = np.mean(self.final_loss_values[loss_name])
                loss_values_per_iteration_to_print[loss_name] = (loss_val.item(), yyy)
                if batch_idx > 0 and len(self.final_loss_values[loss_name]) > 0 and (yyy > 3 * xxx):
                    print("{} has exploded.".format(loss_name.upper()))
                    flag_explode = True

        if len(total_loss) >= 1:
            total_loss = sum(total_loss)
            self.final_loss_values[self.total_loss_key].append(total_loss.item())
            yyy1       = np.mean(self.final_loss_values[self.total_loss_key])
            loss_values_per_iteration_to_print[self.total_loss_key] = (total_loss.item(), yyy1)
        else:
            total_loss = None

        return total_loss, loss_values_per_iteration_to_print, flag_explode


    def main_process(self, lifting_network, config, optimizer, scheduler, discriminator_3d_net, model_2d_pose_estimator, refine_network):
        """
        Function to Perform the main process of the experiment of Lifting 2D poses to 3D.
        :param model_2d_pose_estimator  : The 2D Pose Estimator Model.
        :param lifting_network          :  The 2D to 3D Lifting Network.
        :param config                   : The Configuration File consisting of all the parameters needed to run the experiment.
        :param optimizer                : The optimizer used to learn the learnable parameters.
        :param refine_network           : The GCN based Refine Network.
        :param discriminator_3d_net     : The 3D discriminator Network.
        :param scheduler                : The scheduler used to anneal the learning rate of the learnable parameters.
        """
        load_optimizer_checkpoint = False if config.evaluate_learnt_model else config.load_optimizer_checkpoint
        load_scheduler_checkpoint = False if config.evaluate_learnt_model else config.load_scheduler_checkpoint
        # ret_vals_dict = {'start_epoch' : start_epoch, 'optimizer' : optimizer, 'scheduler': scheduler,  'lifting_network' : lifting_network, 
                        #  'refine_network': refine_network, 'model_2d_pose_estimator' : model_2d_pose_estimator, 'discriminator_3d_net' : discriminator_3d_net}
        if config.load_from_checkpoint:
            print("Evaluating the Learnt Model stored in the Checkpoint {}.".format(config.checkpoint_path))
            loaded_checkpoint_dict = self.load_from_checkpoint(lifting_network=lifting_network, scheduler=scheduler,
                                                                 load_file_name=config.checkpoint_path, optimizer=optimizer,
                                                                 load_optimizer_checkpoint=load_optimizer_checkpoint,
                                                                 load_scheduler_checkpoint=load_scheduler_checkpoint,
                                                                 refine_network=refine_network,
                                                                 model_2d_pose_estimator=model_2d_pose_estimator,
                                                                 discriminator_3d_net=discriminator_3d_net)
            start_epoch             = loaded_checkpoint_dict['start_epoch']
            optimizer               = loaded_checkpoint_dict['optimizer']
            scheduler               = loaded_checkpoint_dict['scheduler']
            lifting_network         = loaded_checkpoint_dict['lifting_network']
            refine_network          = loaded_checkpoint_dict['refine_network']
            model_2d_pose_estimator = loaded_checkpoint_dict['model_2d_pose_estimator']
            discriminator_3d_net    = loaded_checkpoint_dict['discriminator_3d_net']
        else:
            start_epoch = 0

        if config.perform_test:
            save_best                = False
            calculate_early_stopping = False
            calculate_best           = False

            if config.evaluate_learnt_model:
                obtain_json = False # We will calculate the MPJPE.
                print("Will be Evaluating the Model using Single View MPJPE.")
                self.evaluate_models(lifting_network=lifting_network, refine_network=refine_network, epoch=start_epoch, phase='test', n_mpjpe=True, p_mpjpe=True, 
                                     obtain_json=obtain_json, calculate_best=calculate_best, save_best=save_best,
                                     calculate_early_stopping=calculate_early_stopping, model_2d_pose_estimator=model_2d_pose_estimator, 
                                     discriminator_3d_net=discriminator_3d_net)

            if config.get_json_files_train_set:
                obtain_json = True
                print("Will be Obtaining the Necessary Predictions for the Training of the Encoder On the Train Set.")
                self.evaluate_models(lifting_network=lifting_network, refine_network=refine_network, epoch=start_epoch, phase='train', n_mpjpe=False, p_mpjpe=False, 
                                     obtain_json=obtain_json, calculate_best=calculate_best, save_best=save_best, calculate_early_stopping=calculate_early_stopping,
                                     model_2d_pose_estimator=model_2d_pose_estimator, discriminator_3d_net=discriminator_3d_net)

            if config.get_json_files_test_set:
                obtain_json = True
                print("Will be Obtaining the Necessary Predictions for the Training of the Encoder On the Test Set.")
                self.evaluate_models(lifting_network=lifting_network, refine_network=refine_network, epoch=start_epoch, phase='test', n_mpjpe=False, p_mpjpe=False, 
                                     obtain_json=obtain_json, calculate_best=calculate_best, save_best=save_best, calculate_early_stopping=calculate_early_stopping,
                                     model_2d_pose_estimator=model_2d_pose_estimator, discriminator_3d_net=discriminator_3d_net)

            if config.plot_train_keypoints:
                print("We will plot the 2D projections of the Target 3D pose and the 3D Pose predicted by the network for the Train Set.")
                self.evaluate_models_by_plotting(suffix='Train', lifting_network=lifting_network, epoch=start_epoch,
                                                 refine_network=refine_network, return_train=False, phase='train',
                                                 model_2d_pose_estimator=model_2d_pose_estimator,
                                                 discriminator_3d_net=discriminator_3d_net)
            if config.plot_test_keypoints:
                print("We will plot the 2D projections of the Target 3D pose and the 3D Pose predicted by the network for the Test Set.")
                self.evaluate_models_by_plotting(suffix='Test', lifting_network=lifting_network, epoch=start_epoch,
                                                 refine_network=refine_network, return_train=True, phase='test',
                                                 model_2d_pose_estimator=model_2d_pose_estimator,
                                                 discriminator_3d_net=discriminator_3d_net)
        
        else:
            # No need to save the best model for training as they are the original models.
            # No need to perform Early Stopping
            # But We need to calculate the Best Values of the Evaluation Metrics.
            if not config.not_calculate_mpjpe_at_start_of_training:
                print("Will be Evaluating the Model using Single View MPJPE before the start of the Training.")
                self.evaluate_models(lifting_network=lifting_network, refine_network=refine_network, epoch=start_epoch,
                                     phase='test', n_mpjpe=True, p_mpjpe=True, save_best=False,
                                     calculate_best=True, obtain_json=False, calculate_early_stopping=False,
                                     model_2d_pose_estimator=model_2d_pose_estimator, discriminator_3d_net=discriminator_3d_net)

            print("-------" * 20)
            print("Will be performing the Training Loop.")
            self.training_loop(config=config, lifting_network=lifting_network, optimizer=optimizer, scheduler=scheduler,
                               start_epoch=start_epoch, refine_network=refine_network,
                               model_2d_pose_estimator=model_2d_pose_estimator, discriminator_3d_net=discriminator_3d_net)   

    def training_loop(self, config, lifting_network, optimizer, scheduler, start_epoch, refine_network,
                      model_2d_pose_estimator, discriminator_3d_net):
        """
        Function to perform the entire training and the learning process.
        :param config                   : The Configuration File consisting of all the parameters needed to run the experiment.
        :param lifting_network          : The 2D to 3D Lifting Network.
        :param refine_network           : The GCN based Refine Network.
        :param discriminator_3d_net     : The 3D pose discriminator
        :param optimizer                : The optimizer used to optimize the parameters.
        :param scheduler                : The scheduler used to anneal the learning rate of the learnable parameters.
        :param start_epoch              : The current epoch to start the training process.
        :param model_2d_pose_estimator  : The 2D Pose Estimator Model.
        """
        write_file_name = self.get_writer_extension()
        writer          = SummaryWriter(log_dir=write_file_name)
        train_time      = 0
        start_time      = time.time()
        
        print("The Start Epoch is set to {}".format(start_epoch + 1))
        print("Beginning the Training Process.")
        for epoch in range(start_epoch + 1, start_epoch + config.epochs + 1):
            start_train_time = time.time()
            if epoch == start_epoch + config.epochs:
                norm_mpjpe = True
            else:
                norm_mpjpe = False
            
            self.reset_loss_params()
            self.perform_one_epoch_of_training(epoch=epoch, lifting_network=lifting_network, optimizer=optimizer, scheduler=scheduler, norm_mpjpe=norm_mpjpe, 
                                               refine_network=refine_network, model_2d_pose_estimator=model_2d_pose_estimator, discriminator_3d_net=discriminator_3d_net)
            
            train_time += round(time.time() - start_train_time)
            for loss_name in self.individual_losses_names:
                writer.add_scalar(loss_name.upper(), np.mean(self.final_loss_values[loss_name]), epoch)

            epoch_time = round(time.time() - start_train_time)
            epoch_time = str(datetime.timedelta(seconds=epoch_time))
            print("Total elapsed epoch:: {} time ::(h:m:s): {}".format(epoch, epoch_time))
            if (epoch == 1) or (epoch % self.save_model_freq == 0) or (epoch == (start_epoch + config.epochs)):
                print("Saving the model after {} epochs of Training.".format(epoch))
                self.save_model(epoch=epoch, lifting_network=lifting_network, optimizer=optimizer, scheduler=scheduler,
                                iterations=self.count_iterations, saving_best=False, refine_network=refine_network,
                                model_2d_pose_estimator=model_2d_pose_estimator, discriminator_3d_net=discriminator_3d_net)
            print('\n-----\n')
            if config.debugging:
                if epoch == 2:
                    break

            if self.early_stop_flag:
                print("Early Stopping Criterion is met after training for {} epochs.".format(epoch))
                break

            if self.max_train_iterations_obtained:
                print("Maximum Number of Training Iterations Already Obtained after {} epochs.".format(epoch))
                break
        
        print("Training has Finished.")
        elapsed    = round(time.time() - start_time)
        elapsed    = str(datetime.timedelta(seconds=elapsed))
        train_time = str(datetime.timedelta(seconds=train_time))
        print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
        writer.flush()
        writer.close()
        norm_mpjpe = True
        if self.best_mpjpe_obtained_sv:
            # No need to calculate the best evaluation metrics.
            # No need to perform early stopping
            # No need to Save the Best model as it has already been saved before during the training process.
            print("SINGLE-VIEW  Evaluating the Best 2D Pose Estimator Model and 3D Refine Network on the Test Set after the Entire Training Process is complete.")
            self.evaluate_models(lifting_network=self.best_lifting_network_sv, obtain_json=False,
                                 epoch=self.best_epoch_based_on_mpjpe_sv, phase='test', n_mpjpe=norm_mpjpe, p_mpjpe=norm_mpjpe,
                                 calculate_best=False, refine_network=self.best_refine_network_sv, calculate_early_stopping=False,
                                 save_best=False, model_2d_pose_estimator=self.best_model_2d_pose_estimator_sv,
                                 discriminator_3d_net=discriminator_3d_net)
        else:
            print("WARNING WARNING " * 5)
            print("We didn't obtain any better MPJPE during the Training Process.")
        print("Done Training")


    def perform_one_epoch_of_training(self, epoch, lifting_network, optimizer, scheduler, norm_mpjpe, refine_network,
                                      model_2d_pose_estimator, discriminator_3d_net):
        """
        Function to Perform One epoch of Training.
        :param model_2d_pose_estimator: The 2D Pose Estimator Model.
        :param epoch                : The current epoch of training.
        :param lifting_network      : The 2D to the #D Lifting Network.
        :param refine_network       : The GCN based Refine Network.
        :param optimizer            : The optimizer used to learn the learnable parameters.
        :param scheduler            : The scheduler used to anneal the learning rate of the learnable parameters.
        :param discriminator_3d_net : The 3D Pose Discriminator Model.
        :param norm_mpjpe           : if True, we will be calculating the Normalized and the 'Procrustes' Aligned MPJPE also.
        :return: The Overall loss values.
        """

        lifting_network.train()
        if model_2d_pose_estimator is not None:
            model_2d_pose_estimator.train()
        
        if refine_network is not None:
            refine_network.train()
        
        if discriminator_3d_net is not None:
            discriminator_3d_net.train()

        train_dataloader         = self.train_loader
        suffix                   = 'Train'
        suffix_1                 = 'Tr-Val'
        len_loader               = len(train_dataloader)
        stop_training            = False
        train_flag               = True
        save_best                = True                          # In the Training Process, we need to save the best models.
        calculate_early_stopping = self.calculate_early_stopping # In the Training Process, we need to perform Early Stopping.
        calculate_best           = True                          # In the Training Process, we need to calculate the best Evaluation Metrics.
        print("Trainer for Lifting Network {} for Epoch {}".format(suffix, epoch))
        for batch_idx, data_train in enumerate(train_dataloader):
            if (self.count_iterations % self.save_plot_freq == 0) or (self.count_iterations == 0) \
                    and ((batch_idx + 1) != len_loader):
                save_image = 1
            else:
                save_image = 0
            vals = self.forward_pass(lifting_network=lifting_network, data=data_train, refine_network=refine_network, for_mpjpe=False, train_flag=train_flag, 
                                     model_2d_pose_estimator=model_2d_pose_estimator, discriminator_3d_net=discriminator_3d_net)
            if vals is None:
                print(" in <perform_one_epoch_of_training> function - {}".format(batch_idx + 1))
                continue

            if save_image > 0 and self.plot_keypoints:
                print("Will be plotting the 2D keypoints predicted by the network, the projection of the Refined 3D obtained by GCN Refine Network"
                      "and the 2D target keypoints.")
                self.evaluate_models_by_plotting(suffix=suffix, lifting_network=lifting_network, epoch=epoch, refine_network=refine_network, return_train=False, phase='train',
                                                 model_2d_pose_estimator=model_2d_pose_estimator, discriminator_3d_net=discriminator_3d_net)
                self.evaluate_models_by_plotting(suffix=suffix_1, lifting_network=lifting_network, epoch=epoch, refine_network=refine_network, return_train=True, phase='test',
                                                 model_2d_pose_estimator=model_2d_pose_estimator, discriminator_3d_net=discriminator_3d_net)
            
            overall_loss_iteration, loss_values_per_iteration_to_print, \
                flag_explode = self.function_to_calculate_loss_per_iteration(vals=vals, batch_idx=batch_idx)

            lifting_network.zero_grad()
            if refine_network is not None:
                refine_network.zero_grad()
            if discriminator_3d_net is not None:
                discriminator_3d_net.zero_grad()
            if model_2d_pose_estimator is not None:
                model_2d_pose_estimator.zero_grad()

            if overall_loss_iteration is None:
                print("Not Back Propagating at Epoch {} in the Iteration {}".format(epoch, self.count_iterations))
                self.count_iterations += 1
                continue
            else:
                optimizer.zero_grad()
                overall_loss_iteration.backward()
                if self.clip_grad_by_norm:
                    if model_2d_pose_estimator is not None:
                        nn.utils.clip_grad_norm_(model_2d_pose_estimator.parameters(), self.clip_grad_by_norm_val)
                    if lifting_network is not None:
                        nn.utils.clip_grad_norm_(lifting_network.parameters(), self.clip_grad_by_norm_val)
                    if refine_network is not None:
                        nn.utils.clip_grad_norm_(refine_network.parameters(), self.clip_grad_by_norm_val)
                    if discriminator_3d_net is not None:
                        nn.utils.clip_grad_norm_(discriminator_3d_net.parameters(), self.clip_grad_by_norm_val)

                if self.clip_grad_by_val:
                    if model_2d_pose_estimator is not None:
                        nn.utils.clip_grad_value_(model_2d_pose_estimator.parameters(), self.clip_grad_by_val_val)
                    if lifting_network is not None:
                        nn.utils.clip_grad_norm_(lifting_network.parameters(), self.clip_grad_by_val_val)
                    if refine_network is not None:
                        nn.utils.clip_grad_value_(refine_network.parameters(), self.clip_grad_by_val_val)
                    if discriminator_3d_net is not None:
                        nn.utils.clip_grad_norm_(discriminator_3d_net.parameters(), self.clip_grad_by_val_val)

                optimizer.step()
                self.count_iterations += 1  # One Iteration of Training has been performed

            if (batch_idx + 1) % self.print_freq == 0 or (batch_idx + 1) == 1 or (batch_idx + 1 == len_loader) or flag_explode:
                print("{} :::: Epoch {} -- Iteration {} / {}"
                      "  (Total Iterations = {}) ".format(suffix, epoch, batch_idx + 1, len_loader, self.count_iterations) +
                      self.print_loss(loss_values_per_iteration_to_print=loss_values_per_iteration_to_print))
                    
            if self.count_iterations % self.eval_freq == 0:
                print("Saving the Models after {} iterations of training achieved in the {} epoch.".format(self.count_iterations, epoch))
                self.save_model(epoch=epoch, lifting_network=lifting_network, optimizer=optimizer, scheduler=scheduler, iterations=self.count_iterations, saving_best=False, 
                                refine_network=refine_network, model_2d_pose_estimator=model_2d_pose_estimator, discriminator_3d_net=discriminator_3d_net, save_with_iters=True)
                
                print("Calculating the MPJPE of the learnt 2D-3D Lifting Network after {} iterations of "
                      "training achieved in the {} epoch.".format(self.count_iterations, epoch))
                self.evaluate_models(lifting_network=lifting_network, epoch=epoch, n_mpjpe=norm_mpjpe, p_mpjpe=norm_mpjpe, phase='test', obtain_json=False, 
                                     refine_network=refine_network, save_best=save_best, calculate_early_stopping=calculate_early_stopping, calculate_best=calculate_best,
                                     model_2d_pose_estimator=model_2d_pose_estimator, discriminator_3d_net=discriminator_3d_net)

                if self.early_stop_flag:
                    print("Early Stopping Criterion is Met After Training for {} Iterations in the Epoch {}.".format(self.count_iterations, epoch))
                    print("Training is to be stopped now.")
                    stop_training = True
                    break

            if (self.max_train_iterations > 0) and (self.count_iterations > self.max_train_iterations):
                print("Maximum Number of Training Iterations of {} Reached. "
                      "We are now stopping the training process of our network.".format(self.max_train_iterations))
                stop_training                      = True
                self.max_train_iterations_obtained = True
                break

            if self.debugging:
                if batch_idx + 1 == 5:
                    break
        
        if not stop_training or self.max_train_iterations_obtained:
            n_mpjpe = True
            print("Calculating the MPJPE of the learnt 2D-3D Lifting Network after {}th epoch of training.".format(epoch))
            self.evaluate_models(lifting_network=lifting_network, phase='test', obtain_json=False, n_mpjpe=n_mpjpe, p_mpjpe=n_mpjpe, calculate_best=calculate_best, epoch=epoch,
                                 refine_network=refine_network, save_best=save_best, calculate_early_stopping=calculate_early_stopping,
                                 model_2d_pose_estimator=model_2d_pose_estimator, discriminator_3d_net=discriminator_3d_net)
            if scheduler is not None:
                scheduler.step()

    def get_json_file(self):
        """
        TODO
        """


    
    def evaluate_models(self, model_2d_pose_estimator, lifting_network, refine_network, discriminator_3d_net, epoch: int, phase: str,
                        n_mpjpe: bool, p_mpjpe: bool, calculate_best: bool, obtain_json: bool, calculate_early_stopping: bool,
                        save_best: bool, iterations=None):
        """
        Function to Calculate the MPJPE of the single view 2D to 3D Lifting Network.
        :param model_2d_pose_estimator  : The 2D Pose Estimator Model.
        :param lifting_network          : The 2D to 3D Pose Lifting Network.
        :param refine_network           : The GCN based Refine Network.
        :param discriminator_3d_net     : The Discriminator for 3D poses.
        :param epoch                    : The current epoch at which we are evaluating the MPJPE or obtaining the Json File.
        :param phase                    :  Determines which Phase we are currently in. For MPJPE Calculation, it should be Test.
        :param n_mpjpe                  : If True, we will also calculate the normalized MPJPE.
        :param p_mpjpe                  : If True, we will also calculate the PMPJPE.
        :param calculate_best           : If True, we will be storing and saving the best model obtained so far.
        :param obtain_json              : If True, we will be obtaining the JSON Files for Visualization.
        :param iterations               : The current Iteration at which we are computing the MPJPE.
        :param save_best                : If True, we will be saving the best models.
        :param calculate_early_stopping : If True, we will calculate the Early Stopping Criteria. It should be done only during the training of the networks.
        :return:
        """
        if obtain_json is False:
            assert phase.lower() == 'test'
            print("This is is valid only for the Test Phase.")
            print("We are calculating the MPJPE between the predicted 3D pose and the Target Pose in Single View at Epoch {} ".format(epoch), end='')
        else:
            print("We will be storing the keypoints for visualization at Epoch {} ".format(epoch), end='')

        if iterations is not None:
            print(" and {} iterations.".format(self.count_iterations), end='')

        if phase.lower() == 'test':
            print(" in the Test Phase.")
            dataloader = self.test_loader
        else:
            print(" in the Training Phase.")
            assert phase.lower() in ['train', 'training']
            dataloader = self.train_loader

        len_dataloader = len(dataloader)
        lifting_network.eval()
        if model_2d_pose_estimator is not None:
            model_2d_pose_estimator.eval()
        if refine_network is not None:
            refine_network.eval()
        if discriminator_3d_net is not None:
            discriminator_3d_net.eval()

        with torch.no_grad():
            preds_mpjpe_sv = []; targets_mpjpe_sv = []; json_data_phase = {}
            for batch_idx, data_test in enumerate(dataloader):
                batch_idx_1 = batch_idx + 1
                # if batch_idx_1 == 1 or batch_idx_1 % self.print_freq == 0 or batch_idx_1 == len_dataloader:
                #     print("Processing batch {} out of {}".format(batch_idx+1, len_dataloader))
                vals = self.forward_pass(lifting_network=lifting_network, data=data_test, for_mpjpe=not obtain_json, refine_network=refine_network, train_flag=False,
                                         model_2d_pose_estimator=model_2d_pose_estimator, discriminator_3d_net=discriminator_3d_net)

                if vals is None:
                    print(" in <evaluate_models> function {} of the trainer_only_lifting_net_cpn_keypoints class".format(batch_idx + 1))
                    continue
                
                if obtain_json:
                    json_data_phase = self.get_json_file(vals=vals, json_data_phase=json_data_phase)
                else:
                    predictions_lifting = vals[self.lifting_preds_key]
                    targets_lifting     = vals[self.lifting_targets_key]
                    preds_mpjpe_sv.extend(predictions_lifting)
                    targets_mpjpe_sv.extend(targets_lifting)

                if self.debugging:
                    if batch_idx_1 == 5:
                        break
        
        if not obtain_json:
            print("Calculating the MPJPE of the learnt model after {}.".format(epoch))
            preds_mpjpe_sv   = torch.stack(preds_mpjpe_sv,   dim=0)
            targets_mpjpe_sv = torch.stack(targets_mpjpe_sv, dim=0)
            mpjpe_3d_sv      = calculate_mpjpe_(predictions=preds_mpjpe_sv, targets=targets_mpjpe_sv, n_mpjpe=n_mpjpe, p_mpjpe=p_mpjpe,
                                                print_string='of the Single View 2D to 3D Lifting Network' if self.evaluate_after_gcn is False else
                                                'of the Graphs Based Refine Network')

            if self.dataset_name.lower() == 'mpi':
                    calculate_pck(preds=preds_mpjpe_sv, targets=targets_mpjpe_sv) # TODO

            if calculate_best:
                if mpjpe_3d_sv < self.best_mpjpe_sv:
                    self.best_mpjpe_sv                   = mpjpe_3d_sv
                    self.best_lifting_network_sv         = copy.deepcopy(lifting_network)         if lifting_network is not None else None
                    self.best_refine_network_sv          = copy.deepcopy(refine_network)          if refine_network is not None else None
                    self.best_model_2d_pose_estimator_sv = copy.deepcopy(model_2d_pose_estimator) if model_2d_pose_estimator is not None else None
                    self.best_discriminator_sv           = copy.deepcopy(discriminator_3d_net) if model_2d_pose_estimator is not None else None
                    self.best_epoch_based_on_mpjpe_sv    = epoch
                    self.best_iters_based_on_mpjpe_sv    = self.count_iterations
                    self.best_mpjpe_obtained_sv          = True
                    print("We have obtained the Best MPJPE at Epoch {} after {} Iterations.".format(self.best_epoch_based_on_mpjpe_sv, self.best_iters_based_on_mpjpe_sv))
                    print("and its value is {:.4f} for the Network.".format(self.best_mpjpe_sv * 100.0))
                    if save_best:
                        print("Saving the Best Model Obtained so far.")
                        self.save_model(lifting_network=self.best_lifting_network_sv, epoch=epoch, saving_best=True, suffix='3D-pose-single-view', iterations=None, optimizer=None, 
                                        scheduler=None, refine_network=self.best_refine_network_sv, model_2d_pose_estimator=self.best_model_2d_pose_estimator_sv,
                                        discriminator_3d_net=self.best_discriminator_sv)
                    print("Done Storing the Best Model obtained so far.")
            
            if calculate_early_stopping:
                self.perform_early_stopping(score=mpjpe_3d_sv)    
        
        else:
            json_file_name_save = '{}-{}.json'.format(self.json_file_name, phase)
            if self.save_json_file_with_save_dir:
                json_file_name_save = os.path.join(self.save_dir, 'JSON_FILE', json_file_name_save)
                mkdir_if_missing(json_file_name_save)
            json_write(filename=json_file_name_save, data=json_data_phase)
            print("Done saving the Json Data in {} for the Phase {}".format(json_file_name_save, phase.upper()))


    def forward_pass(self, lifting_network, refine_network, data, for_mpjpe, train_flag, model_2d_pose_estimator, discriminator_3d_net):
        """
        :param lifting_network           : The 2D to the #D Lifting Network.
        :param refine_network            : The GCN based Refine Network.
        :param train_flag                : If True, we will pass the data through the entire set of models (as expected in the Training Phase).
                                          If False we will pass the data only through the <lifting_network>.
        :param model_2d_pose_estimator  : The 2D Pose Estimator Model.
        :param discriminator_3d_net     : The 3D Pose Discriminator Network.
        :param data:
        :param for_mpjpe:
        :return:
        """
        
        if ['None'] == list(data.keys()):
            print("The Entire Batch is not Fit for calculating the Triangulated 3D Pose", end=' ')
            return None

        inp_lifting_net = data['input_2D_update'] # This has to be of size N x F x J x 2.
        target          = data['gt_3D'] # This has to be size N x F x J x 3 for train and N x 1 x J x 3 for the test.
        scale           = data['scale'] # This is of size N x 1.
        images          = None # No input images.
        if train_flag :
            split = 'train'
        else:
            split = 'test'

        if self.use_cuda: 
            inp_lifting_net = inp_lifting_net.to(self.device)
            target          = target.to(self.device)
            scale           = scale.to(self.device)
            images          = images.to(self.device) if images is not None else None
            
        batch_size, num_frames, num_joints, inp_joints_feat_size = inp_lifting_net.size()
        
        # inp_joints_feat_size should be 2. assert inp_joints_feat_size == 2
        assert model_2d_pose_estimator is None
        assert refine_network          is None

        if train_flag:
            n_joint_to_alter          = np.random.poisson(self.max_num_joints_to_alter)
            joints_to_alter           = np.random.randint(0, self.number_of_joints, n_joint_to_alter)
            joints_to_alter           = torch.from_numpy(joints_to_alter).type_as(inp_lifting_net).long()
            inp_lifting_net           = inp_lifting_net + torch.randn_like(inp_lifting_net) * self.m1
            added_noise               = torch.randn(batch_size, num_frames, n_joint_to_alter, 2) * self.m2
            inp_lifting_net[:,
              :, joints_to_alter, :] += added_noise.type_as(inp_lifting_net)

        inp_vals_dict_lifting_net = {'input_2D' : inp_lifting_net, 'gt_3D' : target, 'images' : images, 'split' : split, 'scale' : scale}
        out_lifting_net_dict      = lifting_network(inp_vals_dict_lifting_net)
        out_lifting_net           = out_lifting_net_dict['pred_out']   # This should batch_size X num_frames X num_joints X 3.
        target_lifting_net        = out_lifting_net_dict['out_target'] # This should batch_size X num_frames X num_joints X 3.
        if self.symmetric_bones:
            out_lifting_net = self.symmetric_bones_module(out_lifting_net)

        if not train_flag:
            if for_mpjpe:
                if not self.out_all_test:
                    central_frame_test = num_frames // 2
                    out_lifting_net    = out_lifting_net[:, [central_frame_test], ...] # This is something I need to change for the test set.
                # out_lifting_net    = out_lifting_net.reshape(-1, num_joints, 3)
                # target_lifting_net = target_lifting_net.reshape(-1, num_joints, 3)
                ret_vals           = {self.lifting_preds_key: out_lifting_net, self.lifting_targets_key: target_lifting_net}
            else:
                # TODO
                """
                # This for JSON FILE
                """
        
        else: # ('fake_3d_pose', 'real_3d_pose', None)
            forward_more          = True  if discriminator_3d_net else False
            out_lifting_net       = out_lifting_net.reshape(-1,    num_joints, 3)
            target_lifting_net    = target_lifting_net.reshape(-1, num_joints, 3)
            ret_vals              = {self.pred_3d_key: out_lifting_net, self.target_3d_key : target_lifting_net}
            if forward_more:
                out_disc_fake_samples = discriminator_3d_net(out_lifting_net)
                out_disc_real_samples = discriminator_3d_net(target_lifting_net)
                ret_vals_dis          = {'real_3d_pose': out_disc_real_samples, 'fake_3d_pose': out_disc_fake_samples}
                ret_vals              = {**ret_vals, **ret_vals_dis}        
        return ret_vals
