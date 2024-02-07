import cv2
import os
import torch
import numpy as np
import time
import copy
import datetime
import torch.nn as nn
from plotting_poses                    import plot_poses_2D_3D
from torch.utils.tensorboard.writer    import SummaryWriter
from utils                             import (get_state_dict_from_multiple_gpu_to_single,
                                               transfer_partial_weights, mkdir_if_missing, json_write, calculate_mpjpe)

from typing import Optional, Dict, Union, List, Any

from validation.metrics import depth_nll, depth_mse, std_scaled_depth_mse

# seeding
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)


class Trainer_lifting_net(object):


    def __init__(self, 
                 individual_losses_names: list, 
                 losses_keys: list, 
                 save_dir: str, 
                 device: str, 
                 train_loader, validation_loader, test_loader, 
                 debugging: bool, 
                 save_plot_freq: int,
                 save_model_freq: int, 
                 use_cuda: str, 
                 print_freq: int, 
                 tb_logs_folder: str, 
                 dataset_name: str, 
                 eval_freq: int, 
                 json_file_name: str, 
                 save_json_file_with_save_dir: str,
                 lifting_network_orig, 
                 n_joints: int, 
                 bone_pairs: list, 
                 rhip_idx: int, 
                 lhip_idx: int, 
                 neck_idx: int, 
                 pelvis_idx: int, 
                 head_idx: int,
                 use_view_info_lifting_net: bool, 
                 print_loss_keys: str, 
                 num_views: int, 
                 loss_modules: list, 
                 patience_early_stopping: int, 
                 delta_early_stopping: float, 
                 save_file_name: str,
                 plot_keypoints: bool, 
                 train_loader_simple, 
                 number_of_batches_to_plot: int, 
                 total_loss_key: str, 
                 max_num_joints_to_alter: int, 
                 mpjpe_poses_in_camera_coordinates: bool,
                 symmetric_bones: bool, 
                 symmetric_bones_module, 
                 inp_lifting_net_is_images: bool, 
                 calculate_early_stopping: bool, 
                 detach_gradient_target: bool, 
                 use_other_samples_in_loss: bool, 
                 lambda_other_samples: float, 
                 swap_inp_tar_unsup: bool, 
                 max_train_iterations: int, 
                 clip_grad_by_norm: bool, 
                 clip_grad_by_norm_val: float, 
                 clip_grad_by_val: bool, 
                 clip_grad_by_val_val: float,
                 unsup_loss_in_2d: bool, 
                 pose_model_name: str, 
                 experimental_setup: str, 
                 inp_lifting_det_keypoints: bool, 
                 loss_in_camera_coordinates: bool,
                 lifting_use_gt_2d_keypoints: bool,
                 joints_masking_indices: List[str],
                 joints_masking_type: str):
        
        self.individual_losses_names           = individual_losses_names
        self.losses_keys                       = losses_keys
        self.save_dir                          = save_dir
        self.device                            = device
        self.train_loader                      = train_loader
        self.validation_loader                 = validation_loader
        self.test_loader                       = test_loader
        self.debugging                         = debugging
        self.save_plot_freq                    = save_plot_freq
        self.save_model_freq                   = save_model_freq
        self.use_cuda                          = use_cuda
        self.print_freq                        = print_freq
        self.tb_logs_folder                    = tb_logs_folder
        self.dataset_name                      = dataset_name
        self.eval_freq                         = eval_freq
        self.json_file_name                    = json_file_name
        self.save_json_file_with_save_dir      = save_json_file_with_save_dir
        self.lifting_network_orig              = lifting_network_orig
        self.n_joints                          = n_joints
        self.bone_pairs                        = bone_pairs
        self.rhip_idx                          = rhip_idx
        self.lhip_idx                          = lhip_idx
        self.neck_idx                          = neck_idx
        self.pelvis_idx                        = pelvis_idx
        self.head_idx                          = head_idx
        self.use_view_info_lifting_net         = use_view_info_lifting_net
        self.print_loss_keys                   = print_loss_keys
        self.num_views                         = num_views
        self.loss_modules                      = loss_modules
        self.patience_early_stopping           = patience_early_stopping
        self.delta_early_stopping              = delta_early_stopping
        self.save_file_name                    = save_file_name
        self.plot_keypoints                    = plot_keypoints
        self.train_loader_simple               = train_loader_simple
        self.number_of_batches_to_plot         = number_of_batches_to_plot
        self.total_loss_key                    = total_loss_key
        self.max_num_joints_to_alter           = max_num_joints_to_alter
        self.mpjpe_poses_in_camera_coordinates = mpjpe_poses_in_camera_coordinates
        self.symmetric_bones                   = symmetric_bones
        self.symmetric_bones_module            = symmetric_bones_module
        self.inp_lifting_net_is_images         = inp_lifting_net_is_images
        self.calculate_early_stopping          = calculate_early_stopping
        self.detach_gradient_target            = detach_gradient_target
        self.use_other_samples_in_loss         = use_other_samples_in_loss
        self.lambda_other_samples              = lambda_other_samples
        self.swap_inp_tar_unsup                = swap_inp_tar_unsup
        self.max_train_iterations              = max_train_iterations
        self.clip_grad_by_norm                 = clip_grad_by_norm
        self.clip_grad_by_norm_val             = clip_grad_by_norm_val
        self.clip_grad_by_val                  = clip_grad_by_val
        self.clip_grad_by_val_val              = clip_grad_by_val_val
        self.unsup_loss_in_2d                  = unsup_loss_in_2d
        self.pose_model_name                   = pose_model_name
        self.experimental_setup                = experimental_setup
        self.inp_lifting_det_keypoints         = inp_lifting_det_keypoints
        self.loss_in_camera_coordinates        = loss_in_camera_coordinates

        self.lifting_preds_key                 = 'lifting_3d'
        self.lifting_targets_key               = 'target_3d'
        self.individual_losses_names           = individual_losses_names
        self.lifting_network_orig              = lifting_network_orig
        self.preds_key                         = "pred_rel_depth_and_uncertainty"
        self.targets_key                       = "target_rel_depth"
        self.weights_3d_key                    = "weights"
        self.state_dict_keys                   = {'optimizer'            : 'state_dict_optimizer',
                                                  'scheduler'            : 'state_dict_scheduler',
                                                  'lifting_network'      : 'state_dict_lifting_network',
                                                  'pose_2d_estimator'    : 'state_dict_pose_model',                                                  
                                                }
        self.m1, self.m2                       = 0.02, 0.05
        self.counter_early_stopping            = 0
        self.best_score_early_stopping         = None
        self.early_stop_flag                   = False
        self.pck_thresholds                    = [0.1, 0.15, 0.2]
        self.max_train_iterations_obtained     = False
        
        self.masking_config = {
            "joint_indices": joints_masking_indices,
            "masking_type": joints_masking_type
        }
        
        self.count_iterations  = 0
        self.validation_step   = 0
        
        self.best_uncertainty_loss = float(np.inf)
        self.best_iters_based_on_uncertainty_loss = 0
        self.best_epoch_based_on_uncertainty_loss = 0
        self.best_lifting_network_uncertainty_loss = None
        self.best_model_2d_pose_estimator_uncertainty_loss = None
        
        self.best_mpjpe_sv                        = float(np.inf)
        self.best_epoch_based_on_mpjpe_sv         = 0
        self.best_iters_based_on_mpjpe_sv         = 0
        self.best_lifting_network_sv              = None
        self.best_model_2d_pose_estimator_sv      = None
        
        self.lifting_use_gt_2d_keypoints = lifting_use_gt_2d_keypoints
        

    def create_tb_writer_folder(self) -> str:
        """
        :param self: The Trainer Class.
        :return: The filename to store the parameters of TensorBoard.
        """
        savepath = os.path.join(self.save_dir)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        tb_logs_folder = os.path.join(savepath, self.tb_logs_folder)
        return tb_logs_folder
    
    
    # TODO: I do not think this is needed for now.
    """
    def perform_early_stopping(self, score):
        "
        Function to perform Early Stopping of Learning due to divergence or over-fitting.
        :param score: The score to be used for checking the divergence or over-fitting of the model.
        "
        if self.best_score_early_stopping is None:
            self.best_score_early_stopping = score

        elif score > (self.best_score_early_stopping + self.delta_early_stopping):
            self.counter_early_stopping += 1
            print(f'EarlyStopping counter: {self.counter_early_stopping} out of {self.patience_early_stopping}')
            if self.counter_early_stopping >= self.patience_early_stopping:
                self.early_stop_flag = True
                print("[TRAINER]: Training is to be Stopped Now.")
        else:
            self.best_score_early_stopping = score
            self.counter_early_stopping    = 0
    """

    def main_process(self, model_2d_pose_estimator, lifting_network, config, optimizer, scheduler):
        """
        Function to Perform the main process of the experiment of Lifting 2D poses to 3D.
        :param model_2d_pose_estimator  : The 2D Pose Estimator Model.
        :param lifting_network          :  The 2D to 3D Lifting Network.
        :param config                   : The Configuration File consisting of all the parameters needed to run the experiment.
        :param optimizer                : The optimizer used to learn the learnable parameters.
        :param scheduler                : The scheduler used to anneal the learning rate of the learnable parameters.
        """
        load_optimizer_checkpoint = False if config.evaluate_learnt_model else config.load_optimizer_checkpoint
        load_scheduler_checkpoint = False if config.evaluate_learnt_model else config.load_scheduler_checkpoint
        
        start_epoch = 0
        if config.load_from_checkpoint:
            print("[TRAINER][MAIN]: Evaluating the Learnt Model stored in the Checkpoint {}.".format(config.checkpoint_path))
            # start_epoch, optimizer, scheduler, \
            _, _, _,\
                lifting_network, model_2d_pose_estimator = self.load_from_checkpoint(lifting_network=lifting_network, scheduler=scheduler,
                                                                 load_file_name=config.checkpoint_path, optimizer=optimizer,
                                                                 load_optimizer_checkpoint=load_optimizer_checkpoint,
                                                                 load_scheduler_checkpoint=load_scheduler_checkpoint,
                                                                 model_2d_pose_estimator=model_2d_pose_estimator)
        
        print("-------" * 20)

        if config.perform_test:
            # Testing the Model
            
            if config.uncertainty_plots:
                print("[TRAINER][MAIN][EVAL]: Will be plotting the Uncertainty Plots.")
                self.uncertainty_plots(
                    model_2d_pose_estimator=model_2d_pose_estimator,
                    lifting_network=lifting_network,
                    dataset_name=self.dataset_name,
                    dataloader=self.test_loader,
                    save_dir=self.save_dir,
                    plot_sample_size=5,
                    projection=config.uncertainty_plots_dim,
                    normalize_coords=config.uncertainty_plots_normalize_coords,
                    gt_only=False,
                    match_pred_to_gt_kpts=config.uncertainty_plots_match_pred_to_gt_keypoints,
                    do_save_batch_data=False,
                    use_multiprocessing=True,
                    loader_type="train" if config.test_on_training_set else "test")
            
            """
            if config.evaluate_learnt_model:
                # We will calculate the MPJPE.
                print("[TRAINER][MAIN][EVAL]: Will be Evaluating the Model using Single View MPJPE.")
                self.evaluate_models(
                    model_2d_pose_estimator=model_2d_pose_estimator,
                    lifting_network=lifting_network,
                    obtain_json=False,
                    eval_phase="test",
                    epoch=start_epoch,
                    n_mpjpe=True, p_mpjpe=True,
                    tb_writer=None)

            if config.get_json_files_train_set:
                print("[TRAINER][MAIN][EVAL]: Will be Obtaining the Necessary Predictions for the Training of the Encoder On the Train Set.")
                self.evaluate_models(
                    model_2d_pose_estimator=model_2d_pose_estimator,
                    lifting_network=lifting_network,
                    obtain_json=True,
                    eval_phase="train",
                    epoch=start_epoch,
                    tb_writer=None)

            if config.get_json_files_test_set:
                print("[TRAINER][MAIN][EVAL]: Will be Obtaining the Necessary Predictions for the Training of the Encoder On the Test Set.")
                self.evaluate_models(
                    model_2d_pose_estimator=model_2d_pose_estimator,
                    lifting_network=lifting_network,
                    obtain_json=True,
                    eval_phase="test",
                    epoch=start_epoch,
                    tb_writer=None)
            
            if config.plot_train_keypoints:
                print("[TRAINER][MAIN][EVAL-PLOT]: We will plot the 2D projections of the Target 3D pose and the 3D Pose predicted by the network for the Train Set.")
                self.evaluate_models_by_plotting(suffix='Train', lifting_network=lifting_network, epoch=start_epoch,
                                                 return_train=True, phase='train', model_2d_pose_estimator=model_2d_pose_estimator)

            if config.plot_test_keypoints:
                print("[TRAINER][MAIN][EVAL-PLOT]: We will plot the 2D projections of the Target 3D pose and the 3D Pose predicted by the network for the Test Set.")
                self.evaluate_models_by_plotting(suffix='Test', lifting_network=lifting_network, epoch=start_epoch,
                                                 return_train=True, phase='test', model_2d_pose_estimator=model_2d_pose_estimator)
            """
        elif config.create_stats_dataset:
            print("[TRAINER][MAIN][CREATE-DATASET]: Will be Creating the Stats Dataset.")
            self.create_stats_dataset(
                model_2d_pose_estimator=model_2d_pose_estimator,
                lifting_network=lifting_network,
                dataloader=self.test_loader,
                save_path=config.stats_dataset_savepath,
                
            )
                
        else:
            # detect any anomalies such as nan gradient
            torch.autograd.set_detect_anomaly(True, check_nan=True)
            
            # Training the Model
            tb_writer = SummaryWriter(
                log_dir=self.create_tb_writer_folder(),
                filename_suffix="_training_logs")
            
            # No need to save the best model for training as they are the original models.
            # But We need to calculate the Best Values of the Evaluation Metrics.
            if not config.not_calculate_mpjpe_at_start_of_training:
                print("[TRAINER][MAIN][EVAL]: Will be Evaluating the Model using Single View MPJPE before the start of the Training.")
                self.evaluate_models(
                    model_2d_pose_estimator=model_2d_pose_estimator,
                    lifting_network=lifting_network,
                    obtain_json=False,
                    eval_phase="test",
                    epoch=start_epoch, iterations=self.count_iterations,
                    update_best=True, save_best=True,
                    n_mpjpe=True, p_mpjpe=True, 
                    tb_writer=tb_writer)
                
            print("[TRAINER][MAIN]: Start of the Training Loop.")
            self.training_loop(
                config=config,
                model_2d_pose_estimator=model_2d_pose_estimator,
                lifting_network=lifting_network,
                optimizer=optimizer, scheduler=scheduler,
                start_epoch=start_epoch,
                tb_writer=tb_writer)
            
            tb_writer.flush()
            tb_writer.close()
       

    def reset_training_metrics_tracking(self):
        """
        Function to get reset the loss value params for printing.
        :return: None
        """
        self.epoch_losses = {
            loss_name:{"mean": 0, "count": 0} for loss_name in self.individual_losses_names}
        self.epoch_losses = {**self.epoch_losses, **{self.total_loss_key: {"mean": 0, "count": 0}}} # adding the total loss
        
        self.train_mpjpe_3d_sv = {"mean": 0, "count": 0, "max": -np.inf, "min": np.inf}
        self.train_depth_mse = {"mean": 0, "count": 0, "max": -np.inf, "min": np.inf}
        self.train_std_scaled_depth_mse = {"mean": 0, "count": 0, "max": -np.inf, "min": np.inf}
        self.train_std = {"mean": 0, "count": 0, "max": -np.inf, "min": np.inf}
     
         
    def training_loop(
        self,
        config: Dict[str, Union[str, int, float, bool]],
        model_2d_pose_estimator,
        lifting_network,
        optimizer, scheduler,
        start_epoch: int,
        tb_writer: Optional[SummaryWriter] =None):
        """
        Function to perform the entire training and the learning process.
        :param config                   : The Configuration File consisting of all the parameters needed to run the experiment.
        :param lifting_network          : The 2D to 3D Lifting Network.
        :param optimizer                : The optimizer used to optimize the parameters.
        :param scheduler                : The scheduler used to anneal the learning rate of the learnable parameters.
        :param start_epoch              : The current epoch to start the training process.
        :param model_2d_pose_estimator  : The 2D Pose Estimator Model.
        """
        print("[TRAINER][LOOP]: Is depth only training? (uncertainty is 1 and not trained): {}".format(config.calculate_depth_only))
        print("[TRAINER][LOOP]: The start epoch is set to {}".format(start_epoch + 1))

        start_time = time.time()
        train_time = 0
        for epoch in range(start_epoch + 1, start_epoch + config.epochs + 1):
            # resetting loss and validation metrics tracking
            self.reset_training_metrics_tracking()
            norm_mpjpe = (epoch == start_epoch + config.epochs)
            
            # epoch
            start_train_time = time.time()
            self.training_epoch(
                model_2d_pose_estimator=model_2d_pose_estimator,
                lifting_network=lifting_network,
                optimizer=optimizer,
                scheduler=scheduler, 
                epoch=epoch, 
                norm_mpjpe=norm_mpjpe,
                tb_writer=tb_writer)
            train_time += round(time.time() - start_train_time)
            epoch_time = round(time.time() - start_train_time)
            epoch_time = str(datetime.timedelta(seconds=epoch_time))
            print("[TRAINER][LOOP][TIME]: Finished epoch ({}) Total Time: {} (h:m:s)".format(epoch, epoch_time))
            
            # log training metrics
            for loss_name in self.individual_losses_names:
                tb_writer.add_scalar("TRAINING_"+loss_name.upper(), self.epoch_losses[loss_name]["mean"], epoch)
            
            tb_writer.add_scalar("TRAINING_TOTAL_LOSS", self.epoch_losses[self.total_loss_key]["mean"], epoch)
            
            tb_writer.add_scalar("TRAINING_MPJPE_SV", self.train_mpjpe_3d_sv["mean"], epoch)
            tb_writer.add_scalar("TRAINING_DEPTH_RMSE_MAX", np.sqrt(self.train_depth_mse["max"]), epoch)
            tb_writer.add_scalar("TRAINING_DEPTH_RMSE_MIN", np.sqrt(self.train_depth_mse["min"]), epoch)
            
            tb_writer.add_scalar("TRAINING_DEPTH_RMSE", np.sqrt(self.train_depth_mse["mean"]), epoch)
            tb_writer.add_scalar("TRAINING_STD_SCALED_DEPTH_RMSE_MAX", np.sqrt(self.train_std_scaled_depth_mse["max"]), epoch)
            tb_writer.add_scalar("TRAINING_STD_SCALED_DEPTH_RMSE_MIN", np.sqrt(self.train_std_scaled_depth_mse["min"]), epoch)
            
            tb_writer.add_scalar("TRAINING_STD_SCALED_DEPTH_RMSE",np.sqrt(self.train_std_scaled_depth_mse["mean"]), epoch)
            tb_writer.add_scalar("TRAINING_STD_SCALED_DEPTH_RMSE_MAX", np.sqrt(self.train_std["max"]), epoch)
            tb_writer.add_scalar("TRAINING_STD_SCALED_DEPTH_RMSE_MIN", np.sqrt(self.train_std["min"]), epoch)
            
            tb_writer.add_scalar("TRAINING_STD", self.train_std["mean"], epoch)
            tb_writer.add_scalar("TRAINING_STD_MAX", self.train_std["max"], epoch)
            tb_writer.add_scalar("TRAINING_STD_MIN", self.train_std["min"], epoch)
            
            tb_writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
            
            if (epoch == 1) or (epoch % self.save_model_freq == 0) or (epoch == (start_epoch + config.epochs)):
                print("[TRAINER][LOOP][SAVING]: Saving the model after ({}) epochs".format(epoch))
                # TODO: need to uncomment the optimizer and scheduler saving if we expect to resume training.
                self.save_model(
                    model_2d_pose_estimator=model_2d_pose_estimator,
                    lifting_network=lifting_network,
                    optimizer=None, # optimizer,
                    scheduler=None, # scheduler,
                    epoch=epoch)
            print("\n-----\n")
            if config.debugging:
                if epoch >= 2:
                    break

            if self.early_stop_flag:
                print("[TRAINER][LOOP]: Early Stopping Criterion is met after training for {} epochs.".format(epoch))
                break

            if self.max_train_iterations_obtained:
                print("[TRAINER][LOOP]: Maximum Number of Training Iterations Already Obtained after {} epochs.".format(epoch))
                break
            
        # Training time
        print("[TRAINER][LOOP]: Training has Finished.")
        elapsed    = round(time.time() - start_time)
        elapsed    = str(datetime.timedelta(seconds=elapsed))
        train_time = str(datetime.timedelta(seconds=train_time))
        print("[TRAINER][LOOP]: Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
        
        if self.best_lifting_network_sv is not None:
            # No need to Save the Best model as it has already been saved before during the training process.
            print("[TRAINER][LOOP][EVAL]: Evaluating the best SV-MPJPE 2D-Pose Estimator Model \
                and 3D Lifting Network on the Test Set after the Entire Training Process is complete.")
            self.evaluate_models(
                model_2d_pose_estimator=self.best_model_2d_pose_estimator_sv,
                lifting_network=self.best_lifting_network_sv,
                obtain_json=False,
                eval_phase="test",
                epoch=self.best_epoch_based_on_mpjpe_sv,
                n_mpjpe=True, p_mpjpe=True,
                tb_writer=None)
        else:
            print("[TRAINER][LOOP][WARNING]: We didn't obtain any better MPJPE during the training process.")
        
        if self.best_lifting_network_uncertainty_loss is not None:
            # No need to Save the Best model as it has already been saved before during the training process.
            print("[TRAINER][LOOP][EVAL]: Evaluating the best Uncertainty Loss 2D-Pose Estimator Model \
                and 3D Lifting Network on the Test Set after the Entire Training Process is complete.")
            self.evaluate_models(
                model_2d_pose_estimator=self.best_model_2d_pose_estimator_uncertainty_loss,
                lifting_network=self.best_lifting_network_uncertainty_loss,
                obtain_json=False,
                eval_phase="test",
                epoch=self.best_epoch_based_on_uncertainty_loss,
                n_mpjpe=False, p_mpjpe=False,
                tb_writer=None)
        else:
            print("[TRAINER][LOOP][WARNING]: We didn't obtain any better Uncertainty Loss during the training process.")
        print("[TRAINER][LOOP]: Done.")
        
        
    def uncertainty_plots(
        self,
        model_2d_pose_estimator,
        lifting_network,
        dataset_name: str,
        dataloader,
        save_dir: str,
        plot_sample_size: int =0,
        projection: str ="3d",
        normalize_coords: bool =False,
        gt_only: bool =False,
        match_pred_to_gt_kpts: bool =False,
        do_save_batch_data: bool =False,
        use_multiprocessing: bool =True,
        loader_type: str ="test"):
        
        assert plot_sample_size >= 0, "The plot sample size should be greater than or equal to 0. Got {}.".format(plot_sample_size)
        assert isinstance(plot_sample_size, int), "The plot sample size should be an integer. Got {}.".format(plot_sample_size)
        assert dataloader is not None
        assert projection in ["3d", "2d"]
        
        def skeleton_plots(
            batch_data: Dict[str, Any],
            projection: str,
            normalize_coords: bool,
            saving_config: Dict[str, str],
            loader_type: str ="test"):
            # assertions
            assert projection in ["3d", "2d"]
            
            # imports
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("darkgrid")
            if projection == "3d":
                from mpl_toolkits.mplot3d import Axes3D
                
            bones = [
                # upper body
                
                (9, 10), # head to site
                (8, 9),  # neck to head
                (7, 8),  # spine1 to neck
                
                (8, 11), # neck to left arm
                (8, 14), # neck to right arm
                
                (11, 12), # left arm to left fore arm
                (12, 13), # left fore arm to left hand
                
                (14, 15), # right arm to right fore arm
                (15, 16), # right fore arm to right hand
                
                # lower body
                
                (0, 7), # hips to spine1
                
                (0, 1), # hips to right up leg
                (1, 2), # right up leg to right leg
                (2, 3), # right leg to right foot
                
                (0, 4), # hips to left up leg
                (4, 5), # left up leg to left leg
                (5, 6) # left leg to left foot
            ]
            
            # data
            xs_pred = batch_data["xs_pred"]
            ys_pred = batch_data["ys_pred"]
            zs_pred = batch_data["zs_pred"]
            pred_stds = batch_data["stds_pred"]
            xs_gt = batch_data["xs_gt"]
            ys_gt = batch_data["ys_gt"]
            zs_gt = batch_data["zs_gt"]
            
            # normalize the coordinates
            if normalize_coords:
                xs_pred = (xs_pred - xs_pred.min(1, keepdims=True)) / (xs_pred.max(1, keepdims=True) - xs_pred.min(1, keepdims=True))
                ys_pred = (ys_pred - ys_pred.min(1, keepdims=True)) / (ys_pred.max(1, keepdims=True) - ys_pred.min(1, keepdims=True))
                zs_pred = (zs_pred - zs_pred.min(1, keepdims=True)) / (zs_pred.max(1, keepdims=True) - zs_pred.min(1, keepdims=True))
                xs_gt = (xs_gt - xs_gt.min(1, keepdims=True)) / (xs_gt.max(1, keepdims=True) - xs_gt.min(1, keepdims=True))
                ys_gt = (ys_gt - ys_gt.min(1, keepdims=True)) / (ys_gt.max(1, keepdims=True) - ys_gt.min(1, keepdims=True))
                zs_gt = (zs_gt - zs_gt.min(1, keepdims=True)) / (zs_gt.max(1, keepdims=True) - zs_gt.min(1, keepdims=True))
            
            # checking the lengths
            n_samples = len(xs_pred)
            assert n_samples == len(ys_pred) == len(zs_pred) == len(pred_stds) \
                == len(xs_gt) == len(ys_gt) == len(zs_gt)
            
            # plots
            for sample_i in range(n_samples):
                fig = plt.figure(figsize=(10, 10))
                
                ax  = None
                match projection:
                    case "2d": ax = fig.add_subplot(111)
                    case "3d": ax = fig.add_subplot(111, projection="3d")
                    case _   : raise ValueError("The projection should be either 2d or 3d. Got {}.".format(projection))
                
                # plot the bones linking the 2D keypoints
                for bone in bones:
                    start_i, end_i = bone[0], bone[1]
                    bone_world_xs = [xs_gt[sample_i][start_i], xs_gt[sample_i][end_i]]
                    bone_world_ys = [ys_gt[sample_i][start_i], ys_gt[sample_i][end_i]]
                    match projection:
                        case "2d": ax.plot(bone_world_xs, bone_world_ys, color="green",
                                           linewidth=5, linestyle="-", alpha=0.5)
                        case "3d":
                            bone_world_zs = [zs_gt[sample_i][start_i], zs_gt[sample_i][end_i]]
                            ax.plot(bone_world_xs, bone_world_zs, bone_world_ys, color="green",
                                           linewidth=5, linestyle="-", alpha=0.5)
                        
                    if not(gt_only):
                        bone_pred_xs = [xs_pred[sample_i][start_i], xs_pred[sample_i][end_i]]
                        bone_pred_ys = [ys_pred[sample_i][start_i], ys_pred[sample_i][end_i]]
                        match projection:
                            case "2d": ax.plot(bone_pred_xs, bone_pred_ys, color="grey",
                                               linewidth=5, linestyle="-", alpha=0.5)
                            case "3d":
                                bone_pred_zs = [zs_pred[sample_i][start_i], zs_pred[sample_i][end_i]]
                                ax.plot(bone_pred_xs, bone_pred_zs, bone_pred_ys, color="grey",
                                               linewidth=5, linestyle="-", alpha=0.5)
                
                # match the predicted and the ground truth keypoints
                if match_pred_to_gt_kpts and not(gt_only):
                    for joint_i, (z_pred, z_world) in enumerate(zip(zs_pred[sample_i], zs_gt[sample_i])):
                        matching_xs = [xs_pred[sample_i][joint_i], xs_gt[sample_i][joint_i]]
                        matching_ys = [ys_pred[sample_i][joint_i], ys_gt[sample_i][joint_i]]
                        match projection:
                            case "2d":
                                color = "blue"
                                ax.plot(matching_xs, matching_ys,
                                               color=color, linewidth=2, linestyle="--", alpha=0.5)
                            case "3d":
                                color = "orange" if z_pred > z_world else "blue"
                                matching_zs = [z_pred, z_world]
                                ax.plot(matching_xs, matching_zs, matching_ys,
                                        color=color, linewidth=2, linestyle="--", alpha=0.5)
                
                cmap_arg = "viridis" 
                match projection:
                    case "2d":
                        # plot the ground truth 2.5D target keypoints
                        ax.scatter(xs_gt[sample_i], ys_gt[sample_i],
                                   c="green", marker="o", s=1, label="target")
                        if not(gt_only):
                            pathcol = ax.scatter(xs_pred[sample_i], ys_pred[sample_i],
                                            c=pred_stds[sample_i], cmap=cmap_arg,
                                            marker="o", s=50, label="predicted")
                            fig.colorbar(pathcol)
                    case "3d":
                        # plot the 2.5D predicted keypoints, using ground turth 2D pose, only depth is predicted
                        ax.scatter(xs_gt[sample_i], zs_gt[sample_i], ys_gt[sample_i],
                                          c="green", marker="o", s=1, label="target")
                        if not(gt_only):
                            pathcol = ax.scatter(xs_pred[sample_i], zs_pred[sample_i], ys_pred[sample_i],
                                                    c=pred_stds[sample_i], cmap=cmap_arg,
                                                    marker="o", s=50, label="predicted")
                            fig.colorbar(pathcol)
                
                ax.set_title("Predicted vs Target 3D-Pose with Estimated Uncertainties")
                ax.set_xlabel("x")
                ax.set_ylabel("z")
                if projection == "3d":
                    ax.set_zlabel("y")
                ax.legend()
                
                if not(os.path.exists(saving_config["dirpath"])):
                    os.makedirs(saving_config["dirpath"])
                
                save_path = os.path.join(
                    saving_config["dirpath"],
                    ("test_" if loader_type == "test" else "train_") + saving_config["fname_format"].format(sample_i, projection))
                fig.savefig(fname=save_path, dpi=300)
                
        def plot_xy(
            metric_x: str,
            metric_y: str,
            metric_x_vals: List[float],
            metric_y_vals: List[float],
            saving_config: Dict[str, str],
            z_score_normalization: bool =True,
            density_based: bool =False,
            loader_type: str ="test",
            remove_outliers: bool =True):
            
            import statsmodels.api as sm
            from scipy.stats import zscore
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("darkgrid")
            
            from scipy.stats import spearmanr, pearsonr
            
            fig = plt.figure(figsize=(5, 5))
            ax = fig.subplots(1, 1)
            
            if remove_outliers:
                max_percentile = 95
                min_percentile = 0
                
                upperbound = np.percentile(metric_x_vals, max_percentile)
                lowerbound = np.percentile(metric_x_vals, min_percentile)
                metric_x_vals = [val for val in metric_x_vals if val < upperbound and val >= lowerbound]
                
                upperbound = np.percentile(metric_y_vals, max_percentile)
                lowerbound = np.percentile(metric_y_vals, min_percentile)
                metric_y_vals = [val for val in metric_y_vals if val < upperbound and val >= lowerbound]
            
            # center
            if z_score_normalization:
                metric_x_vals = zscore(metric_x_vals)
                metric_y_vals = zscore(metric_y_vals)
                        
            # plot the uncertainty loss vs uncertainty
            if density_based:
                sns.kdeplot(x=metric_x_vals, y=metric_y_vals, ax=ax, shade=True, shade_lowest=False)
            else:
                ax.scatter(metric_x_vals, metric_y_vals, color="blue", marker="o", s=20)
            
            # axis limits    
            min_x, max_x = ax.get_xlim()
            min_y, max_y = ax.get_ylim()
            
            # plot the OLS fit
            exog = sm.add_constant(np.array(metric_x_vals).reshape(-1, 1))
            endog = np.array(metric_y_vals).reshape(-1, 1)
            ols_result = sm.OLS(endog=endog, exog=exog, hasconst=True).fit()
            ols_xs_linspace = np.linspace(min_x, max_x, 100)
            exog = sm.add_constant(ols_xs_linspace.reshape(-1, 1))
            ols_xs_linspace_preds = ols_result.predict(exog=exog)
            ax.plot(ols_xs_linspace, ols_xs_linspace_preds, color="red", linewidth=2, label="OLS")
            
            # plot the ideal fit
            ideal_x = np.linspace(min_x, max_x, 100)
            ideal_y = ideal_x
            ideal_x_and_y = zip(ideal_x, ideal_y)
            ideal_x_and_y = [(x, y) for x, y in ideal_x_and_y if y >= min_y and y <= max_y]
            ideal_x, ideal_y = zip(*ideal_x_and_y)
            ax.plot(ideal_x, ideal_y, color="limegreen", linewidth=2, linestyle="--", label="Ideal Direction")
            
            if z_score_normalization:
                x_label = f"{metric_x.upper()} (z-score)"
                y_label = f"{metric_y.upper()} (z-score)"
                
            ax.set_title(f"{metric_y.upper()} vs. {metric_x.upper()}")
            ax.set_xlabel(f"{x_label.upper()}")
            ax.set_ylabel(f"{y_label.upper()}")
            ax.legend()
            
            # compute the spearman correlation
            spearman_corr, spearman_pval = spearmanr(metric_x_vals, metric_y_vals)
            pearson_corr, pearson_pval = pearsonr(metric_x_vals, metric_y_vals)
            print("[CORRELATION]: {} vs. {} ::: Spearman Correlation: {:.2f}, Spearman P-Value: {}".format(
                metric_x, metric_y, spearman_corr, spearman_pval))
            print("[CORRELATION]: {} vs. {} ::: Pearson Correlation: {:.2f}, Pearson P-Value: {}".format(
                metric_x, metric_y, pearson_corr, pearson_pval))
            
            # create the saving directory if it does not exist
            if not(os.path.exists(saving_config["dirpath"])):
                os.makedirs(saving_config["dirpath"])
                
            # save the plot
            save_path = os.path.join(
                saving_config["dirpath"], ("test_" if loader_type == "test" else "train_") + saving_config["fname"])
            fig.savefig(fname=save_path, dpi=300, format="png", bbox_inches="tight")
            
        
        def save_batch_data(batch_data: Dict[str, Any], filepath: str):
            # imports
            import json
            native_batch_data = {key:(value.tolist() if isinstance(value, np.ndarray) else value) for key, value in batch_data.items()}
            
            dir = os.path.dirname(filepath)
            if not(os.path.exists(dir)):
                os.makedirs(dir)
            
            with open(filepath, 'w') as json_file:
                json.dump(native_batch_data, json_file, indent=4)
                        
        best_batch = {
            "uncertainty_loss": float(np.inf),
            "batch_idx": None,
            "xs_pred": None,
            "ys_pred": None,
            "zs_pred": None,
            "stds_pred": None,
            "xs_gt": None,
            "ys_gt": None,
            "zs_gt": None,
        }
        
        worst_batch = {
            "uncertainty_loss": float(-np.inf),
            "batch_idx": None,
            "xs_pred": None,
            "ys_pred": None,
            "zs_pred": None,
            "stds_pred": None,
            "xs_gt": None,
            "ys_gt": None,
            "zs_gt": None,
        }
        
        batchwise_avg_mpjpe_sv = []
        batchwise_avg_depth_mse = []
        batchwise_avg_uncertainties = []
        batchwise_avg_depth_nll = []
        
        if dataset_name.lower() == "h36m":
            print("[UNCERTAINTY-PLOT]: Will be plotting the Uncertainty Plots for the Human3.6M Dataset.")
            print("[UNCERTAINTY-PLOT]: Evaluating the model to find the best and worst batches according to the Uncertainty Loss.")
            for batch_idx, batch in enumerate(dataloader, start=1):                
                with torch.no_grad():
                    vals = self.forward_pass(
                        model_2d_pose_estimator=model_2d_pose_estimator,
                        lifting_network=lifting_network,
                        data=batch,
                        for_mpjpe=True,
                        train_flag=False)
                
                    if vals is None:
                        print("[UNCERTAINTY-PLOT][WARNING]: Forward pass returns None in <evaluate_models> function for batch ({})".format(batch_idx))
                        continue
                
                # compute mpjpe
                pred_depths = vals["depth"].reshape(-1, self.n_joints).to(self.device)
                pred_stds = vals["uncertainty"].reshape(-1, self.n_joints).to(self.device)
                target_depths = batch["target_root_rel_depth"].reshape(-1, self.n_joints).to(self.device)
                uncertainty_loss = depth_nll(
                    pred_depth=pred_depths,
                    target_depth=target_depths,
                    pred_std=pred_stds).item()
                batchwise_avg_depth_nll.append(uncertainty_loss)
                
                depth_mse_val = depth_mse(pred_depth=pred_depths, target_depth=target_depths)
                batchwise_avg_depth_mse.append(depth_mse_val.item())
                
                preds_mpjpe_sv   = vals[self.lifting_preds_key].detach().cpu().numpy().tolist()
                targets_mpjpe_sv = vals[self.lifting_targets_key].detach().cpu().numpy().tolist()
                mpjpe_sv_val, _, _ = calculate_mpjpe(
                        predictions=preds_mpjpe_sv,
                        targets=targets_mpjpe_sv,
                        n_mpjpe=False, p_mpjpe=False)
                batchwise_avg_mpjpe_sv.append(mpjpe_sv_val)
                
                # will use numpy for the remaining operations
                pred_stds = pred_stds.cpu().numpy()
                batchwise_avg_uncertainties.append(pred_stds.mean())
                
                # get the 3D reconstructed (predicted) pose coordinates
                pred_pose_3d = vals[self.lifting_preds_key].squeeze().cpu().numpy()
                xs_pred = pred_pose_3d[..., 0]
                ys_pred = pred_pose_3d[..., 1]
                zs_pred = pred_pose_3d[..., 2]
                
                # get the 3D ground truth pose coordinates
                target_pose_3d = batch["target_pose_3d"]
                target_pose_3d = target_pose_3d.squeeze().cpu().numpy()
                xs_gt = target_pose_3d[..., 0]
                ys_gt = target_pose_3d[..., 1]
                zs_gt = target_pose_3d[..., 2]
                
                # update best batch
                if uncertainty_loss < best_batch["uncertainty_loss"]:
                    best_batch["batch_idx"] = batch_idx
                    best_batch["uncertainty_loss"] = uncertainty_loss
                    best_batch["xs_pred"] = xs_pred
                    best_batch["ys_pred"] = ys_pred
                    best_batch["zs_pred"] = zs_pred
                    best_batch["stds_pred"] = pred_stds
                    best_batch["xs_gt"] = xs_gt
                    best_batch["ys_gt"] = ys_gt
                    best_batch["zs_gt"] = zs_gt
                # update worst batch
                if uncertainty_loss > worst_batch["uncertainty_loss"]:
                    worst_batch["batch_idx"] = batch_idx
                    worst_batch["uncertainty_loss"] = uncertainty_loss
                    worst_batch["xs_pred"] = xs_pred
                    worst_batch["ys_pred"] = ys_pred
                    worst_batch["zs_pred"] = zs_pred
                    worst_batch["stds_pred"] = pred_stds
                    worst_batch["xs_gt"] = xs_gt
                    worst_batch["ys_gt"] = ys_gt
                    worst_batch["zs_gt"] = zs_gt
                             
            print("[UNCERTAINTY-PLOT]: The best Uncertainty Loss is {:.4f} obtained on batch {}".format(
                best_batch["uncertainty_loss"], best_batch["batch_idx"]))
            print("[UNCERTAINTY-PLOT]: The worst Uncertainty Loss is {:.4f} obtained on batch {}".format(
                worst_batch["uncertainty_loss"], worst_batch["batch_idx"]))
            
            if plot_sample_size > 0:
                n_cameras = 4
                plot_sample_size = min(plot_sample_size, dataloader.batch_size * n_cameras)
                print("[UNCERTAINTY-PLOT]: Randomly selecting {} samples from each of"
                      " the best and the worst batch for plotting.".format(plot_sample_size))
                
            n_best = len(best_batch["stds_pred"])
            sampled_best_indices  = np.random.choice(range(n_best), replace=False, size=plot_sample_size)
            best_batch["sampled_indices"] = sampled_best_indices
                
            n_worst = len(worst_batch["stds_pred"])
            sampled_worst_indices = np.random.choice(range(n_worst), replace=False, size=plot_sample_size)
            worst_batch["sampled_indices"] = sampled_worst_indices
                
            if plot_sample_size < dataloader.batch_size:
                print("[UNCERTAINTY-PLOT]: The sampled indices for the best batch are {}".format(sampled_best_indices))
                print("[UNCERTAINTY-PLOT]: The sampled indices for the worst batch are {}".format(sampled_worst_indices))
                
            keys_to_ignore = set(["batch_idx", "uncertainty_loss", "sampled_indices"])
            for key, value in best_batch.items():
                if key in keys_to_ignore:
                    continue
                best_batch[key] = value[sampled_best_indices]
                
            for key, value in worst_batch.items():
                if key in keys_to_ignore:
                    continue
                worst_batch[key] = value[sampled_worst_indices]
            
            print("[UNCERTAINTY-PLOT]: Creating and saving the plots...")
            
            if do_save_batch_data:
                print("[UNCERTAINTY-PLOT]: Will be saving the plotted data.")
            
            batch_name_zip_data = zip(["best", "worst"], [best_batch, worst_batch])
            plot_saving_fname_format = "batch_uncertainty_skeleton_normalized_coords_plot_{}_{}.png"
            
            if use_multiprocessing:
                print("[UNCERTAINTY-PLOT]: Using multiprocessing to plot and save the plots.")
                from multiprocessing import Process
                processes = []
                for batch_name, batch_data in batch_name_zip_data:
                    plot_saving_config = {
                        "dirpath": os.path.join(save_dir, "plots", batch_name),
                        "fname_format": batch_name + '_' + plot_saving_fname_format
                    }
                    process = Process(
                        target=skeleton_plots,
                        args=(batch_data, projection, normalize_coords, plot_saving_config, loader_type))
                    processes.append(process)
                    process.start()
                    if do_save_batch_data:
                        data_saving_filepath = os.path.join(save_dir, "plots_data", batch_name + "_batch_data.json")
                        process = Process(
                            target=save_batch_data,
                            args=(batch_data, data_saving_filepath))
                        processes.append(process)
                        process.start()
                    
                dirpath = os.path.join(save_dir, "plots")
                
                saving_config = {"dirpath": dirpath, "fname": "depth_rmse_vs_std.png"}
                process = Process(target=plot_xy,
                                  args=("STD", "DEPTH RMSE",
                                        batchwise_avg_uncertainties, np.sqrt(batchwise_avg_depth_mse).tolist(), saving_config,
                                        True, True, loader_type))
                process.start()
                processes.append(process)
                
                saving_config = {"dirpath": dirpath, "fname": "mpjpe_vs_std.png"}
                process = Process(target=plot_xy,
                                  args=("STD", "MPJPE",
                                        batchwise_avg_uncertainties, batchwise_avg_mpjpe_sv, saving_config,
                                        True, True, loader_type))
                process.start()
                processes.append(process)
                
                saving_config = {"dirpath": dirpath, "fname": "depthnll_vs_depthrmse.png"}
                process = Process(target=plot_xy,
                                  args=("DEPTH RMSE", "DEPTH NLL",
                                        batchwise_avg_depth_mse, batchwise_avg_depth_nll, saving_config,
                                        True, True, loader_type))                
                process.start()
                processes.append(process)
                
                saving_config = {"dirpath": dirpath, "fname": "depthnll_vs_mpjpe.png"}
                process = Process(target=plot_xy,
                                  args=("MPJPE", "DEPTH NLL",
                                        batchwise_avg_mpjpe_sv, batchwise_avg_depth_nll, saving_config,
                                        True, True, loader_type))                
                process.start()
                processes.append(process)
                        
                for process in processes:
                    process.join()
                    
            else:
                print("[UNCERTAINTY-PLOT]: Not using multiprocessing to create and save the plots.")
                for batch_name, batch_data in batch_name_zip_data:
                    plot_saving_config = {
                        "dirpath": os.path.join(save_dir, "plots", batch_name),
                        "fname_format": batch_name + '_' + plot_saving_fname_format
                    }
                    skeleton_plots(batch_data, projection, normalize_coords, plot_saving_config,
                                   loader_type)
                    if do_save_batch_data:
                        data_saving_filepath = os.path.join(save_dir, "data", batch_name + "_batch_data.json")
                        save_batch_data(batch_data, filepath=data_saving_filepath)
                        
                dirpath = os.path.join(save_dir, "plots") 
                saving_config = {"dirpath": dirpath, "fname": "depth_rmse_vs_std.png"}
                plot_xy("STD", "DEPTH RMSE",
                        batchwise_avg_uncertainties,
                        np.sqrt(batchwise_avg_depth_mse).tolist(), saving_config,
                        True, True, loader_type)
                
                saving_config = {"dirpath": dirpath, "fname": "mpjpe_vs_std.png"}
                plot_xy("STD", "MPJPE", batchwise_avg_uncertainties, batchwise_avg_mpjpe_sv, saving_config,
                        True, True, loader_type)
                
                saving_config = {"dirpath": dirpath, "fname": "mpjpe_vs_depthnll.png"}
                plot_xy("DEPTH RMSE", "DEPTH NLL", batchwise_avg_depth_mse, batchwise_avg_depth_nll, saving_config,
                         True, True, loader_type)
                
                saving_config = {"dirpath": dirpath, "fname": "mpjpe_vs_depthnll.png"}
                plot_xy("MPJPE", "DEPTH NLL", batchwise_avg_mpjpe_sv, batchwise_avg_depth_nll, saving_config,
                         True, True, loader_type)
        else:
            raise ValueError("Unknown dataset name {}".format(dataset_name))
        
        print("[UNCERTAINTY-PLOT]: Done.")
        
    
    def create_stats_dataset(
        self,
        model_2d_pose_estimator,
        lifting_network,
        dataloader,
        save_path: str):
        import jsonlines as jsonl
        
        bones = [
                # upper body
                
                (9, 10), # head to site
                (8, 9),  # neck to head
                (7, 8),  # spine1 to neck
                
                (8, 11), # neck to left arm
                (8, 14), # neck to right arm
                
                (11, 12), # left arm to left fore arm
                (12, 13), # left fore arm to left hand
                
                (14, 15), # right arm to right fore arm
                (15, 16), # right fore arm to right hand
                
                # lower body
                
                (0, 7), # hips to spine1
                
                (0, 1), # hips to right up leg
                (1, 2), # right up leg to right leg
                (2, 3), # right leg to right foot
                
                (0, 4), # hips to left up leg
                (4, 5), # left up leg to left leg
                (5, 6) # left leg to left foot
        ]
        
        adjacency_matrix = torch.zeros((self.n_joints, self.n_joints), dtype=torch.int32)
        for bone in bones:
            start_i, end_i = bone[0], bone[1]
            adjacency_matrix[start_i, end_i] = 1
            adjacency_matrix[end_i, start_i] = 1
            
        save_path = save_path if save_path is not None else os.path.join(self.save_dir, "statistics_datasets", "statistics_dataset.json")
        if os.path.exists(save_path):
            raise ValueError("The save path {} already exists. It would have been overwritten if the process continued.".format(save_path))
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print("[CREATE-STATISTICS-DATASET]: Will be creating the statistics dataset.")
        with jsonl.open(save_path, 'a') as writer:    
            for batch_idx, batch in enumerate(dataloader, start=1):
                vals = None           
                with torch.no_grad():
                    vals = self.forward_pass(
                        model_2d_pose_estimator=model_2d_pose_estimator,
                        lifting_network=lifting_network,
                        data=batch,
                        for_mpjpe=True,
                        train_flag=False)
                
                    if vals is None:
                        print("[CREATE-STATISTICS-DATASET][WARNING]: Forward pass returns None in <create_stats_dataset> function for batch ({})"\
                            .format(batch_idx))
                        continue
                
                frame_ids = torch.cat(batch["frame_ids"], dim=0).tolist()
                batch_preds_3d = vals[self.lifting_preds_key].squeeze()
                batch_labels_3d = vals[self.lifting_targets_key].squeeze()
                for frame_id, sample_preds_3d, sample_labels_3d in zip(frame_ids, batch_preds_3d, batch_labels_3d):
                    l2_errors = torch.linalg.norm(sample_preds_3d - sample_labels_3d, ord=2, dim=1)
                    for joint_i in range(self.n_joints):
                        adj_joints = adjacency_matrix[joint_i] == 1
                        joint_pred_3d = sample_preds_3d[joint_i].reshape(1, 3)
                        adj_joints_preds_3d = sample_preds_3d[adj_joints].reshape(-1, 3)
                        adj_joints_vects = adj_joints_preds_3d - joint_pred_3d
                        entry = {
                            "frame_id": str(frame_id),
                            "joint_num": int(joint_i),
                            "l2_error": float(l2_errors[joint_i].item()),
                            "avg_adj_bone_norms": float(adj_joints_vects.norm(dim=1).mean().item()),
                            "min_adj_bone_norms": float(adj_joints_vects.norm(dim=1).min().item()),
                            "max_adj_bone_norms": float(adj_joints_vects.norm(dim=1).max().item()),
                        }
                        
                        normalized_adj_joints_vects = nn.functional.normalize(adj_joints_vects, p=2, dim=1)
                        normalized_adj_joints_sims = torch.matmul(normalized_adj_joints_vects, normalized_adj_joints_vects.T)
                        identical_vect_mask = (
                            torch.eye(normalized_adj_joints_sims.size(0), dtype=torch.bool)
                            .to(normalized_adj_joints_sims.device)
                        )
                        entry["avg_cosine_sim"] = float(
                            normalized_adj_joints_sims.masked_fill_(identical_vect_mask, 0).mean().item())
                        entry["max_cosine_sim"] = float(normalized_adj_joints_sims.max().item())
                        entry["min_cosine_sim"] = float(normalized_adj_joints_sims.min().item())
                        
                        writer.write(entry)
        
        print("[CREATE-STATISTICS-DATASET]: Done.")

    def training_epoch(self,
                       epoch,
                       lifting_network,
                       optimizer,
                       scheduler,
                       norm_mpjpe,
                       model_2d_pose_estimator,
                       tb_writer: Optional[SummaryWriter] =None):
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

        train_dataloader         = self.train_loader
        suffix                   = 'Train'
        suffix_1                 = 'Tr-Val'
        len_loader               = len(train_dataloader)
        stop_training            = False
        train_flag               = True

        print("[TRAINER][EPOCH]: Trainer for Lifting Network ({}) for epoch ({})".format(suffix, epoch))
        for batch_idx, data_train in enumerate(train_dataloader, start=1):
            if (self.count_iterations % self.save_plot_freq == 0) or (self.count_iterations == 0) \
                    and (batch_idx != len_loader):
                save_image = 1
            else:
                save_image = 0
                
            vals = self.forward_pass(
                lifting_network=lifting_network,
                data=data_train,
                for_mpjpe=True, # In this case, this must be True to track the MPJPE during training.
                train_flag=train_flag,
                model_2d_pose_estimator=model_2d_pose_estimator,
                masking_config=self.masking_config)
                        
            if vals is None:
                print("[TRAINER][EPOCH][WARNING]: Forward pass returns None in <training_epoch> function in batch {}".format(batch_idx))
                continue
            
            if save_image > 0 and self.plot_keypoints:
                print("[TRAINER][EPOCH][EVAL-PLOT]: Will be plotting the 2D keypoints predicted by the network, the projection of the Refined 3D obtained by GCN Refine Network"
                      "and the 2D target keypoints.")
                self.evaluate_models_by_plotting(suffix=suffix, lifting_network=lifting_network, epoch=epoch,
                                                 return_train=False, phase="train", model_2d_pose_estimator=model_2d_pose_estimator)
                self.evaluate_models_by_plotting(suffix=suffix_1, lifting_network=lifting_network, epoch=epoch,
                                                 return_train=True, phase="test", model_2d_pose_estimator=model_2d_pose_estimator)

            overall_loss_iteration, loss_values_per_iteration_to_print, loss_exploded = (
                self.compute_iteration_losses(vals=vals, batch_idx=batch_idx))
            
            self.update_training_metrics(vals=vals, batch=data_train, batch_idx=batch_idx)

            lifting_network.zero_grad()
            if model_2d_pose_estimator is not None:
                model_2d_pose_estimator.zero_grad()

            if overall_loss_iteration is None:
                print("[TRAINER][EPOCH][WARNING]: No Back-Prop at epoch {} in Iteration {}".format(epoch, self.count_iterations))
                self.count_iterations += 1 # one iteration done.
                continue

            else:
                optimizer.zero_grad()
                overall_loss_iteration.backward()

                if self.clip_grad_by_norm:
                    if model_2d_pose_estimator is not None:
                        nn.utils.clip_grad_norm_(model_2d_pose_estimator.parameters(), self.clip_grad_by_norm_val)
                    if lifting_network is not None:
                        nn.utils.clip_grad_norm_(lifting_network.parameters(), self.clip_grad_by_norm_val)
                    
                if self.clip_grad_by_val:
                    if model_2d_pose_estimator is not None:
                        nn.utils.clip_grad_value_(model_2d_pose_estimator.parameters(), self.clip_grad_by_val_val)
                    if lifting_network is not None:
                        nn.utils.clip_grad_norm_(lifting_network.parameters(), self.clip_grad_by_val_val)
                    
                optimizer.step()
                self.count_iterations += 1  # one iteration done.

            if (batch_idx % self.print_freq == 0) or (batch_idx == 1) or (batch_idx == len_loader) or loss_exploded:
                print("[TRAINER][EPOCH]: {} :::: epoch ({}) -- Iteration ({}/{})"
                      "  (Total Iterations = {}) ".format(suffix, epoch, batch_idx, len_loader, self.count_iterations) +
                      self.print_loss(loss_values_per_iteration_to_print=loss_values_per_iteration_to_print))

            if self.count_iterations % self.eval_freq == 0:
                # TODO: no need to save the model after every evaluation.
                """
                print("[TRAINER][EPOCH][SAVING]: Saving the Models after ({}) iterations of training achieved in epoch ({})."\
                    .format(self.count_iterations, epoch))
                self.save_model(
                    model_2d_pose_estimator=model_2d_pose_estimator, lifting_network=lifting_network, 
                    optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, iterations=self.count_iterations,
                    is_best=False)
                """
                
                print("""[TRAINER][EPOCH][EVAL]: Calculating the MPJPE of the learnt 2D-3D Lifting Network \
                    after ({}) iterations of training achieved in epoch ({}).""".format(self.count_iterations, epoch))
                
                self.evaluate_models(
                    model_2d_pose_estimator=model_2d_pose_estimator,
                    lifting_network=lifting_network,
                    obtain_json=False,
                    eval_phase="test",
                    epoch=epoch,
                    update_best=True, save_best=True,
                    n_mpjpe=norm_mpjpe, p_mpjpe=norm_mpjpe,
                    tb_writer=tb_writer)
                    
                if self.early_stop_flag:
                    print("[TRAINER][EPOCH][STOP]: Early Stopping Criterion is Met \
                        After Training for {} Iterations in the epoch {}.".format(self.count_iterations, epoch))
                    print("[TRAINER][EPOCH][STOP]: Training is to be stopped now.")
                    stop_training = True
                    break
            
            if (self.max_train_iterations > 0) and (self.count_iterations > self.max_train_iterations):
                print("""
                      [TRAINER][EPOCH][STOP]: Maximum number of training iterations ({}) is Reached. \
                          We are now stopping the training process of our network.""".format(self.max_train_iterations))
                stop_training = True
                self.max_train_iterations_obtained = True
                break

            if self.debugging:
                if batch_idx == 5:
                    break
        
        if stop_training or self.max_train_iterations_obtained:
            print("[TRAINER][EPOCH][EVAL]: Calculating the MPJPE of the learnt 2D-3D Lifting Network after epoch ({}).".format(epoch))
            self.evaluate_models(
                model_2d_pose_estimator=model_2d_pose_estimator,
                lifting_network=lifting_network,
                obtain_json=False,
                eval_phase="test",
                epoch=epoch,
                update_best=True, save_best=True,
                n_mpjpe=norm_mpjpe, p_mpjpe=norm_mpjpe,
                tb_writer=tb_writer)
            return None
        
        if scheduler is not None:
            scheduler.step()


    @staticmethod 
    def __compute_joints_mask(
        masking_config: Dict[str, Union[str, List[str]]],
        batch_size: int,
        n_joints: int,
        device: torch.device) -> Optional[torch.Tensor]:
        """Joints masking function.
        In the obtained mask, one (1) means the joint is masked and zero (0) means it is not masked (visible to the network).    
        """
        if masking_config is None:
            return None
        
        masked_indices = masking_config["joint_indices"]
        n_masked_indices = len(masked_indices)
        if n_masked_indices == 0:
            return None    
            
        masking_type = masking_config["masking_type"]
        match masking_type:
            case "random":
                mask = torch.zeros((batch_size, n_joints), device=device, dtype=torch.bool)
                submask = (torch.rand(batch_size, n_masked_indices, device=device) >= 0.5)
                submask.type(torch.bool)
                mask.data[:, masked_indices] = submask
            case "consistent":
                mask = torch.zeros((batch_size, n_joints), dtype=torch.bool, device=device)
                mask.data[:, masked_indices] = 1
            case _:
                raise ValueError(f"Unknown masking type {masking_type}")
        
        mask.requires_grad = False
        return mask


    def forward_pass(
        self,
        lifting_network,
        data, for_mpjpe,
        train_flag,
        model_2d_pose_estimator=None,
        masking_config=None):
        """
        :param lifting_network           : The 2D to the #D Lifting Network.
        :param train_flag                : If True, we will pass the data through the entire set of models (as expected in the Training Phase).
                                          If False we will pass the data only through the <lifting_network>.
        :param data:
        :param for_mpjpe:
        :return:
        """

        if ['None'] == list(data.keys()):
            print("[TRAINER][FORWARD]: The Entire Batch is not Fit for calculating the Triangulated 3D Pose", end=' ')
            return None

        data, camera_indexes, R_shape_val = self.collapse_views(data=data)
        bounding_boxes          = data['bboxes']
        # bounding_boxes is of shape Number of Samples x number_of_frames x 4.

        consider               = data['consider']
        # bounding_boxes is of shape Number of Samples x number_of_frames.

        inp_images_lifting_net = None if not self.inp_lifting_net_is_images else data['lifting_net_images']
        # inp_images_lifting_net is of shape Number of Samples x number_of_frames x C x H' x W' if it is not None.

        keypoints_det          = data['keypoints_det'] if not(self.lifting_use_gt_2d_keypoints) else data["target_pose_2d"]
        # keypoints_det is of shape Number of Samples x number_of_frames x number of joints x 2.
        keypoints_det_norm     = data['keypoints_det_norm'] if not(self.lifting_use_gt_2d_keypoints) else data["target_pose_2d_norm"]
        # keypoints_det_norm is of shape Number of Samples x number_of_frames x number of joints x 2.

        camera_indexes         = camera_indexes.type_as(keypoints_det_norm)
        # It is of shape Number of Samples x number_of_frames.

        pelvis_cam_z           = data['pelvis_cam_z'] # It is the z coordinate of the pelvis joint.
        # pelvis_cam_z is of shape Number of Samples x number_of_frames x number of cameras.

        labels                 = data['labels']
        # labels is of shape Number of Samples x number_of_frames

        target_pose_3d         = data['target_pose_3d']
        # target_pose_3d is of shape Number of Samples x number_of_frames x number of joints x 3.

        triangulated_pose_3d = data['triangulated_pose_3d']
        # target_pose_3d is of shape Number of Samples x number_of_frames x number of joints x 3.

        R, t, K, dist = data['R'], data['t'], data['K'], data['dist']
        # R is of    shape Number of Samples x number_of_frames x 3 x 3.
        # t is of    shape Number of Samples x number_of_frames x 3.
        # K is of    shape Number of Samples x number_of_frames x 3 x 3.
        # dist is of shape Number of Samples x number_of_frames x 5.
        phase = 'train' if train_flag else 'test'

        target_root_rel_depth = data['target_root_rel_depth']
        # target_root_rel_depth is of shape Number of Samples x number_of_frames x number of cameras x number of joints.
        # (Only the root relative depth for each joint of the target 3D Pose).
        triangulated_root_rel_depth = data['triangulated_root_rel_depth']
        # triangulated_pose_3d_root_rel_depth is of shape Number of Samples x number_of_frames x number of joints.
        # (Only the root relative depth for each joint of the triangulated 3D Pose).

        batch_size, number_of_frames, n_joints, dim = keypoints_det.size()
        if self.use_cuda:
            bounding_boxes              = bounding_boxes.to(self.device)
            keypoints_det               = keypoints_det.to(self.device)
            pelvis_cam_z                = pelvis_cam_z.to(self.device)
            labels                      = labels.to(self.device)
            target_pose_3d              = target_pose_3d.to(self.device)
            triangulated_pose_3d        = triangulated_pose_3d.to(self.device)
            consider                    = consider.to(self.device)
            R, t, K, dist               = R.to(self.device), t.to(self.device), K.to(self.device), dist.to(self.device)
            keypoints_det_norm          = keypoints_det_norm.to(self.device)
            camera_indexes              = camera_indexes.to(self.device)
            inp_images_lifting_net      = None if inp_images_lifting_net is None else inp_images_lifting_net.to(self.device)
            target_root_rel_depth       = target_root_rel_depth.to(self.device)
            triangulated_root_rel_depth = triangulated_root_rel_depth.to(self.device)

        num_samples           = batch_size * number_of_frames # num_samples is batch_size * number_of_cameras.
        R, t, K, dist         = R.reshape(-1, R_shape_val[0], R_shape_val[1]), t.reshape(-1, 3), K.reshape(-1, 3, 3), dist.reshape(-1, 5)
        inp_lifting_keypoints = (
            keypoints_det_norm.reshape(-1, n_joints, dim) if not self.inp_lifting_det_keypoints else keypoints_det.reshape(-1, n_joints, dim))
        keypoints_det         = keypoints_det.reshape(-1, n_joints, dim)
        pelvis_cam_z          = pelvis_cam_z.reshape(-1, 1)
        
        if train_flag:
            n_joint_to_alter          = np.random.poisson(self.max_num_joints_to_alter)
            joints_to_alter           = np.random.randint(0, self.n_joints, n_joint_to_alter)
            joints_to_alter           = torch.from_numpy(joints_to_alter).type_as(inp_lifting_keypoints).long()
            inp_lifting_keypoints     = inp_lifting_keypoints + torch.randn_like(inp_lifting_keypoints) * self.m1
            added_noise               = torch.randn(num_samples, n_joint_to_alter, 2) * self.m2
            inp_lifting_keypoints[:,
                joints_to_alter, :]  += added_noise.type_as(inp_lifting_keypoints)

        if inp_images_lifting_net is not None:
            c1, h1, w1             = inp_images_lifting_net.size(-3), inp_images_lifting_net.size(-2), inp_images_lifting_net.size(-1)
            inp_images_lifting_net = inp_images_lifting_net.reshape(-1, c1, h1, w1)
            
        # TODO: Useless code. Remove it.
        # if not self.remove_head_view_info:
        #    head_joints           = inp_lifting_keypoints[:, self.head_idx, :].reshape(-1, 1, 2)
        #    inp_lifting_keypoints = torch.cat((inp_lifting_keypoints, head_joints), dim=1)
        # n_joints = n_joints + (0 if self.remove_head_view_info else 1)
        
        val_dict_lifting_net    = {
            'inp_lifting': inp_lifting_keypoints, 
            'inp_images_lifting_net' : inp_images_lifting_net,
            'R': R, 't': t, 'K': K, 'dist': dist,
            'pelvis_cam_z': pelvis_cam_z,
            'det_poses_2d': keypoints_det,
            'view_info' : camera_indexes,
            'phase' : phase, 
            'num_frames' : number_of_frames, 
            'num_joints' : n_joints,
            'joints_mask': Trainer_lifting_net.__compute_joints_mask(
                masking_config=masking_config,
                batch_size=batch_size,
                n_joints=n_joints,
                device=self.device
            )}
        
        # forward pass
        out_lifting_net         = lifting_network(val_dict_lifting_net)
        poses_3d_world_lifting  = out_lifting_net['pose_3d_world']
        poses_3d_camera_lifting = out_lifting_net['pose_3d_cam']
        out_depth               = out_lifting_net['rel_depth']
        out_uncertainty         = out_lifting_net['uncertainty']

        # if (train_flag and self.use_graphs) or (self.evaluate_after_gcn and not train_flag):
        #   time_pred = self.time_pred
        # else:
        #   time_pred = 0
        time_pred = 0
        out_depth_lifting_net = out_depth.reshape(-1, number_of_frames, n_joints)
        out_depth_lifting_net = out_depth_lifting_net[:, time_pred, ...]
        out_depth_lifting_net = out_depth_lifting_net.reshape(-1, n_joints)

        if not train_flag:
            # Test Phase.
            # We are evaluating here after the 3D poses predicted by the Lifting Network.
            ret_vals = self.get_inp_target_for_testing(poses_3d_world=poses_3d_world_lifting,
                                                       poses_3d_camera=poses_3d_camera_lifting, target_pose_3d=target_pose_3d,
                                                       for_mpjpe=for_mpjpe, bounding_boxes=bounding_boxes,
                                                       keypoints_det=keypoints_det, triangulated_pose_3d=triangulated_pose_3d,
                                                       R=R, t=t, K=K, dist=dist, data=data)
            ret_vals["depth"] = out_depth.reshape(-1, self.n_joints)
            ret_vals["uncertainty"] = out_uncertainty.reshape(-1, self.n_joints)
            
        else:
            out_depth       = out_depth.reshape(-1, self.n_joints)
            out_uncertainty = out_uncertainty.reshape(-1, self.n_joints)
            
            pred_rel_depth_and_uncertainty = torch.stack((out_depth, out_uncertainty), dim=-1)
            target_root_rel_depth          = target_root_rel_depth.reshape(-1, self.n_joints)
            
            # TODO: Unused code for nout_depthow.
            # labels_loss = labels.reshape(-1, 1)
            # triangulated_root_rel_depth = triangulated_root_rel_depth.reshape(-1, self.n_joints)
            # consider_3d_loss = consider.reshape(-1, 1) # whether
            # weights_3d = consider_3d_loss
            # targets_unsup = triangulated_root_rel_depth
            # if self.experimental_setup.lower() == 'fully':
            #    targets = targets_sup * labels_loss
            # else:
            #    targets = targets_sup * labels_loss + (1 - labels_loss) * targets_unsup

            ret_vals = {
                self.preds_key          : pred_rel_depth_and_uncertainty,
                self.targets_key        : target_root_rel_depth,
                "depth"                 : out_depth, 
                "uncertainty"           : out_uncertainty,
                self.lifting_preds_key  : poses_3d_world_lifting.reshape(-1, self.n_joints, 3),
                self.lifting_targets_key: target_pose_3d.reshape(-1, self.n_joints, 3)
                # self.weights_3d_key  : None, #weights_3d,
                }
            
        return ret_vals
    
    
    def evaluate_models(
        self,
        model_2d_pose_estimator,
        lifting_network,
        obtain_json: bool,
        eval_phase: str,
        epoch: int, iterations: Optional[int] =None,    
        update_best: bool =False,
        save_best: bool =False,
        n_mpjpe: bool =False, p_mpjpe: bool =False,
        tb_writer: Optional[SummaryWriter] =None,
        print_metrics: bool =True):
        """
        Function to Calculate the MPJPE of the single view 2D to 3D Lifting Network.
        :param model_2d_pose_estimator  : The 2D Pose Estimator Model.
        :param lifting_network          : The 2D to 3D Pose Lifting Network.
        :param epoch                    : The current epoch at which we are evaluating the MPJPE or obtaining the Json File.
        :param phase                    :  Determines which Phase we are currently in. For MPJPE Calculation, it should be Test.
        :param n_mpjpe                  : If True, we will also calculate the normalized MPJPE.
        :param p_mpjpe                  : If True, we will also calculate the PMPJPE.
        :param update_best           : If True, we will be storing and saving the best model obtained so far.
        :param obtain_json              : If True, we will be obtaining the JSON Files for Visualization.
        :param iterations               : The current Iteration at which we are computing the MPJPE.
        :param save_best                : If True, we will be saving the best models.
        :param mpjpe_3d_sv: If True, we will be storing the MPJPE values of the Single View 2D to 3D Lifting Network.
        :return:
        """
        if not(obtain_json):
            assert eval_phase.lower() == 'test'
            print("[TRAINER][EVAL][STORING]: The keypoints will not be stored, and will be used for evaluation directly.")
            print("[TRAINER][EVAL]: We are calculating the MPJPE between the predicted 3D pose and the Target Pose in Single View at epoch {} ".format(epoch), end='')
        else:
            print("[TRAINER][EVAL][STORING]: We will be storing the keypoints for visualization at epoch ({}) ".format(epoch), end='')

        if iterations is not None:
            print("and ({}) iterations, ".format(self.count_iterations), end='')

        if eval_phase.lower() == 'test':
            print("in the Test Phase.")
            dataloader = self.test_loader
        else:
            print("in the Training Phase.")
            assert eval_phase.lower() in ['train', 'training']
            dataloader = self.train_loader

        len_dataloader = len(dataloader)
        lifting_network.eval()
        if model_2d_pose_estimator is not None:
            model_2d_pose_estimator.eval()
            
        mpjpe_3d_sv_mean_dict = {"mean": 0, "count": 0, "max": -np.inf, "min": np.inf}
        depth_nll_mean_dict = {"mean": 0, "count": 0, "max": -np.inf, "min": np.inf}
        depth_mse_dict = {"mean": 0, "count": 0, "max": -np.inf, "min": np.inf}
        std_scaled_depth_mse_dict = {"mean": 0, "count": 0, "max": -np.inf, "min": np.inf}
        std_dict = {"mean": 0, "count": 0, "max": -np.inf, "min": np.inf}
        
        # the following lambda updates the mean of a sequence of samples 
        # given the old mean, the old number of samples, and the new sample
        # it accurately computes the mean without buffering all the elements of the sequence
        update_mean = lambda new_x, old_m, old_n: (new_x + old_m * old_n) / (old_n + 1) 
        
        with torch.no_grad():
            json_data_phase = {}
            for batch_idx, data_test in enumerate(dataloader, start=1):
                if (batch_idx == 1) or (batch_idx % self.print_freq == 0) or (batch_idx == len_dataloader):
                    print("[TRAINER][EVAL]: Processing batch ({}/{})".format(batch_idx, len_dataloader))
                    
                vals = self.forward_pass(
                    lifting_network=lifting_network,
                    data=data_test,
                    for_mpjpe=not(obtain_json),
                    train_flag=False,
                    model_2d_pose_estimator=model_2d_pose_estimator)
                
                if vals is None:
                    print("[TRAINER][EVAL][WARNING]: Forward pass returns None in <evaluate_models> function for batch ({})".format(batch_idx))
                    continue

                if not(obtain_json):            
                    # validation metrics
                    preds_mpjpe_sv   = vals[self.lifting_preds_key].detach().cpu().numpy().tolist()
                    targets_mpjpe_sv = vals[self.lifting_targets_key].detach().cpu().numpy().tolist()
                    mpjpe_sv_val, _, _ = calculate_mpjpe(
                        predictions=preds_mpjpe_sv,
                        targets=targets_mpjpe_sv,
                        n_mpjpe=n_mpjpe, p_mpjpe=p_mpjpe)
                    
                    depths        = vals["depth"].detach().cpu()
                    uncertainties = vals["uncertainty"].detach().cpu()
                    
                    mpjpe_3d_sv_mean_dict["mean"] = update_mean(
                        mpjpe_sv_val, mpjpe_3d_sv_mean_dict["mean"], mpjpe_3d_sv_mean_dict["count"])
                    mpjpe_3d_sv_mean_dict["count"] += 1
                    mpjpe_3d_sv_mean_dict["max"] = max(mpjpe_3d_sv_mean_dict["max"], mpjpe_sv_val)
                    mpjpe_3d_sv_mean_dict["min"] = min(mpjpe_3d_sv_mean_dict["min"], mpjpe_sv_val)
                    
                    target_root_rel_depths = np.array(data_test["target_root_rel_depth"])
                    depth_nll_val = depth_nll(
                        pred_depth=depths,
                        target_depth=target_root_rel_depths,
                        pred_std=uncertainties).item()
                    depth_nll_mean_dict["mean"] = update_mean(
                       depth_nll_val, depth_nll_mean_dict["mean"], depth_nll_mean_dict["count"])
                    depth_nll_mean_dict["count"] += 1
                    depth_nll_mean_dict["max"] = max(depth_nll_mean_dict["max"], depth_nll_val)
                    depth_nll_mean_dict["min"] = min(depth_nll_mean_dict["min"], depth_nll_val)
                    
                    depth_mse_val = depth_mse(
                        pred_depth=depths,
                        target_depth=target_root_rel_depths).item()
                    depth_mse_dict["mean"] = update_mean(
                        depth_mse_val, depth_mse_dict["mean"], depth_mse_dict["count"])
                    depth_mse_dict["count"] += 1
                    depth_mse_dict["max"] = max(depth_mse_dict["max"], depth_mse_val)
                    depth_mse_dict["min"] = min(depth_mse_dict["min"], depth_mse_val)
                    
                    std_scaled_depth_mse_val = std_scaled_depth_mse(
                        pred_depth=depths,
                        target_depth=target_root_rel_depths,
                        pred_std=uncertainties).item()
                    std_scaled_depth_mse_dict["mean"] = update_mean(
                        std_scaled_depth_mse_val, std_scaled_depth_mse_dict["mean"], std_scaled_depth_mse_dict["count"])
                    std_scaled_depth_mse_dict["count"] += 1
                    std_scaled_depth_mse_dict["max"] = max(std_scaled_depth_mse_dict["max"], std_scaled_depth_mse_val)
                    std_scaled_depth_mse_dict["min"] = min(std_scaled_depth_mse_dict["min"], std_scaled_depth_mse_val)
                    
                    std_dict["mean"] = update_mean(
                        uncertainties.mean(), std_dict["mean"], std_dict["count"]).item()
                    std_dict["count"] += 1
                    std_dict["max"] = max(std_dict["max"], uncertainties.max().item())
                    std_dict["min"] = min(std_dict["min"], uncertainties.min().item())
                    
                    """
                    if self.lifting_preds_key in key_vals and self.lifting_targets_key in key_vals and model_2d_pose_estimator is not None:
                        # calculate_mpjpe_mv = True # TODO: Unused for now.
                        predictions_pose_3d_tri = vals[self.preds_triangulation_3d_key]
                        targets_pose_3d_tri     = vals[self.targets_triangulation_3d_key]
                        predictions_pose_3d     = np.array(predictions_pose_3d_tri.detach().cpu() if self.use_cuda is True else predictions_pose_3d_tri.detach())
                        targets_pose_3d         = np.array(targets_pose_3d_tri.detach().cpu() if self.use_cuda is True else targets_pose_3d_tri.detach())
                        preds_mpjpe_mv = predictions_pose_3d.tolist() # TODO: Unused for now.
                        targets_mpjpe_mv = targets_pose_3d.tolist() # TODO: Unused for now.
                    """
                else:
                    raise NotImplemented("JSON File is not implemented yet.")
                    json_data_phase = self.get_json_file(vals=vals, json_data_phase=json_data_phase)
                    
            if not(obtain_json):
                if print_metrics:
                    print("[TRAINER][EVAL]: The MPJPE of the Single View 2D to 3D Lifting Network is {:.4f}.".format(mpjpe_3d_sv_mean_dict["mean"]))
                    print("[TRAINER][EVAL]: The Uncertainty Loss of the Single View 2D to 3D Lifting Network is {:.4f}.".format(
                        depth_nll_mean_dict["mean"]))
                    print("[TRAINER][EVAL]: The Depth RMSE of the Single View 2D to 3D Lifting Network is {:.4f}.".format(
                        np.sqrt(depth_mse_dict["mean"])))
                    print("[TRAINER][EVAL]: The Uncertainty Weighted Depth RMSE of the Single View 2D to 3D Lifting Network is {:.4f}.".format(
                        np.sqrt(std_scaled_depth_mse_dict["mean"])))
                    
                # logging validation metrics
                if tb_writer is not None:
                    tb_writer.add_scalar("VALIDATION_3D_MPJPE_SV", mpjpe_3d_sv_mean_dict["mean"], self.validation_step)
                    tb_writer.add_scalar("VALIDATION_3D_MPJPE_SV_MAX", mpjpe_3d_sv_mean_dict["max"], self.validation_step)
                    tb_writer.add_scalar("VALIDATION_3D_MPJPE_SV_MIN", mpjpe_3d_sv_mean_dict["min"], self.validation_step)                  
                    
                    tb_writer.add_scalar("VALIDATION_DEPTH_NLL", depth_nll_mean_dict["mean"], self.validation_step)
                    tb_writer.add_scalar("VALIDATION_DEPTH_NLL_MAX", depth_nll_mean_dict["max"], self.validation_step)
                    tb_writer.add_scalar("VALIDATION_DEPTH_NLL_MIN", depth_nll_mean_dict["min"], self.validation_step)             
                                        
                    tb_writer.add_scalar("VALIDATION_DEPTH_RMSE", np.sqrt(depth_mse_dict["mean"]), self.validation_step)
                    tb_writer.add_scalar("VALIDATION_DEPTH_RMSE_MAX", np.sqrt(depth_mse_dict["max"]), self.validation_step)
                    tb_writer.add_scalar("VALIDATION_DEPTH_RMSE_MIN", np.sqrt(depth_mse_dict["min"]), self.validation_step)

                    tb_writer.add_scalar("VALIDATION_UNCERTAINTY_WEIGHTED_DEPTH_RMSE",
                                         np.sqrt(std_scaled_depth_mse_dict["mean"]), self.validation_step)
                    tb_writer.add_scalar("VALIDATION_UNCERTAINTY_WEIGHTED_DEPTH_RMSE_MAX",
                                         np.sqrt(std_scaled_depth_mse_dict["max"]), self.validation_step)
                    tb_writer.add_scalar("VALIDATION_UNCERTAINTY_WEIGHTED_DEPTH_RMSE_MIN",
                                         np.sqrt(std_scaled_depth_mse_dict["min"]), self.validation_step)

                    tb_writer.add_scalar("VALIDATION_STD", std_dict["mean"], self.validation_step)
                    tb_writer.add_scalar("VALIDATION_STD_MAX", std_dict["max"], self.validation_step)
                    tb_writer.add_scalar("VALIDATION_STD_MIN", std_dict["min"], self.validation_step)

                
                # TODO: Unused for now.
                # if calculate_mpjpe_mv:
                #   _ = calculate_mpjpe(predictions=preds_mpjpe_mv, targets=targets_mpjpe_mv, n_mpjpe=n_mpjpe,
                #                        p_mpjpe=p_mpjpe, print_string="of the 2D Pose Estimator obtained by Triangulating the 2D Detections")
                # if self.dataset_name.lower() == 'mpi':
                #   calculate_pck(preds=preds_mpjpe_sv, targets=targets_mpjpe_sv)
                
                self.validation_step += 1
                
                current_mpjpe_3d_sv = mpjpe_3d_sv_mean_dict["mean"]
                current_uncertainty_loss = depth_nll_mean_dict["mean"]
                if update_best:
                    if current_uncertainty_loss < self.best_uncertainty_loss:
                        self.best_uncertainty_loss = current_uncertainty_loss
                        self.best_epoch_based_on_uncertainty_loss = epoch
                        self.best_iters_based_on_uncertainty_loss = self.count_iterations
                        
                        self.best_lifting_network_uncertainty_loss = (
                            copy.deepcopy(lifting_network) if lifting_network is not None else None)
                        self.best_model_2d_pose_estimator_uncertainty_loss = (
                            copy.deepcopy(model_2d_pose_estimator) if model_2d_pose_estimator is not None else None)
                        
                        print("[TRAINER][EVAL]: We have obtained the best Uncertainty Loss at epoch ({}) \
                            after ({}) iterations, and its value is {:.4f}".format(
                                self.best_epoch_based_on_uncertainty_loss,
                                self.best_iters_based_on_uncertainty_loss,
                                self.best_uncertainty_loss))
                        
                        if save_best:
                            print("[TRAINER][EVAL][SAVING]: Saving the best model obtained so far based on uncertainty loss.")
                            self.save_model(
                                model_2d_pose_estimator=self.best_model_2d_pose_estimator_sv,
                                lifting_network=self.best_lifting_network_sv,
                                optimizer=None, scheduler=None,
                                epoch=epoch,
                                is_best=True,
                                suffix="uncertainty_loss")
                            print("[TRAINER][EVAL][SAVING]: Done saving the best model.")
                        
                    if current_mpjpe_3d_sv < self.best_mpjpe_sv:
                        self.best_mpjpe_sv = current_mpjpe_3d_sv
                        self.best_epoch_based_on_mpjpe_sv = epoch
                        self.best_iters_based_on_mpjpe_sv = self.count_iterations
                        
                        self.best_lifting_network_sv = copy.deepcopy(lifting_network) if lifting_network is not None else None
                        self.best_model_2d_pose_estimator_sv = copy.deepcopy(model_2d_pose_estimator) if model_2d_pose_estimator is not None else None
                        print("[TRAINER][EVAL]: We have obtained the best MPJPE at epoch ({}) \
                              after ({}) iterations, and its value is {:.4f}.".format(
                                  self.best_epoch_based_on_mpjpe_sv,
                                  self.best_iters_based_on_mpjpe_sv,
                                  self.best_mpjpe_sv))
                        
                        if save_best:
                            print("[TRAINER][EVAL][SAVING]: Saving the best model obtained so far based on SV MPJPE.")
                            self.save_model(
                                model_2d_pose_estimator=self.best_model_2d_pose_estimator_sv,
                                lifting_network=self.best_lifting_network_sv,
                                optimizer=None, scheduler=None,
                                epoch=epoch,
                                is_best=True,
                                suffix="3d_sv_mpjpe")
                            print("[TRAINER][EVAL][SAVING]: Done saving the best model.")
            else:
                json_file_name_save = '{}-{}.json'.format(self.json_file_name, eval_phase)
                if self.save_json_file_with_save_dir:
                    json_file_name_save = os.path.join(self.save_dir, 'JSON_FILE', json_file_name_save)
                    mkdir_if_missing(json_file_name_save)
                json_write(filename=json_file_name_save, data=json_data_phase)
                print("[TRAINER][EVAL][SAVING]: Done saving the JSON Data in {} for the Phase {}".format(json_file_name_save, eval_phase.upper()))


    def load_from_checkpoint(self, load_file_name, optimizer, scheduler, load_optimizer_checkpoint, load_scheduler_checkpoint, model_2d_pose_estimator, lifting_network):
        """
        Function to Load the weights of the Lifting Network and other modules such as Optimizer/Scheduler if Needed.
        :param load_file_name: The Checkpoint File to load the Pretrained Weights for all the modules.
        :param optimizer: The Optimizer used for optimization.
        :param scheduler: The Scheduler used for Optimization.
        :param load_optimizer_checkpoint: If True, optimizer will also be loaded with its stored state in the checkpoint file.
        :param load_scheduler_checkpoint: If True, scheduler will also be loaded with its stored state in the checkpoint file.
        :param lifting_network: The 2D to 3D Lifting Network.
        :param model_2d_pose_estimator: The 2D pose Estimator Model.
        :return: The start epoch, the optimizer, the scheduler (if present) and the Lifting Network.
        """

        print("[TRAINER][LOADING]: Loading from {}".format(load_file_name))
        if not os.path.exists(load_file_name):
            raise ValueError("Wrong checkpoint file.")
        checkpoint      = torch.load(load_file_name)
        start_epoch     = checkpoint['epoch']
        keys_checkpoint = list(checkpoint.keys())

        if model_2d_pose_estimator is not None and self.state_dict_keys['pose_2d_estimator'] in keys_checkpoint:
            print("[TRAINER][LOADING]: Loading the Weights of the 2D Pose Estimator.")
            pose_model_old_state_dict = checkpoint[self.state_dict_keys['pose_2d_estimator']]
            pose_model_new_state_dict = get_state_dict_from_multiple_gpu_to_single(old_state_dict=pose_model_old_state_dict)
            model_2d_pose_estimator   = transfer_partial_weights(model=model_2d_pose_estimator, pretrained_state_dict=pose_model_new_state_dict)
            print("[TRAINER][LOADING]: Done Loading the Weights of the 2D Pose Estimator.")

        if lifting_network is not None and self.state_dict_keys['lifting_network'] in keys_checkpoint:
            print("[TRAINER][LOADING]: Loading the Weights of the 2D to 3D Lifting Network.")
            lifting_network_old_state_dict = checkpoint[self.state_dict_keys['lifting_network']]
            lifting_network_new_state_dict = get_state_dict_from_multiple_gpu_to_single(old_state_dict=lifting_network_old_state_dict)
            lifting_network                = transfer_partial_weights(model=lifting_network, pretrained_state_dict=lifting_network_new_state_dict)
            print("[TRAINER][LOADING]: Done Loading the Weights of the 2D to 3D Lifting Network.")

        if load_optimizer_checkpoint and optimizer is not None and self.state_dict_keys['optimizer'] in keys_checkpoint:
            print("[TRAINER][LOADING]: Loading the Stored State of the Optimizer.")
            optimizer_pose_old_state_dict = checkpoint[self.state_dict_keys['optimizer']]
            optimizer_pose_new_state_dict = get_state_dict_from_multiple_gpu_to_single(old_state_dict=optimizer_pose_old_state_dict)
            optimizer.load_state_dict(optimizer_pose_new_state_dict)
            print("[TRAINER][LOADING]: Done Loading the Stored State of the Optimizer.")

        if load_scheduler_checkpoint and scheduler is not None and self.state_dict_keys['scheduler'] in keys_checkpoint:
            print("[TRAINER][LOADING]: Loading the Stored State of the Scheduler.")
            scheduler_pose_old_state_dict = checkpoint[self.state_dict_keys['scheduler']]
            scheduler_pose_new_state_dict = get_state_dict_from_multiple_gpu_to_single(scheduler_pose_old_state_dict)
            scheduler.load_state_dict(scheduler_pose_new_state_dict)
            print("[TRAINER][LOADING]: Done Loading the Stored State of the Scheduler.")
            
        return start_epoch, optimizer, scheduler, lifting_network, model_2d_pose_estimator


    def save_model(
        self,
        model_2d_pose_estimator, lifting_network,
        optimizer, scheduler,
        epoch: int, iterations: Optional[int] =None,
        is_best: bool =False, suffix: str =None):
        """
        Function to save the necessary parameters of the Networks and Optimizer/Scheduler if needed.
        :param lifting_network          : The 2D to 3D Lifting Network.
        :param optimizer                : The optimizer used to learn the model params.
        :param scheduler                : The optimizer used to learn the model params.
        :param epoch                    : The current epoch of saving the models.
        :param iterations               : The current iteration of saving the models.
        :param saving_best              : If True, we will store the best models.
        :param suffix                   : The string which will be added to save filename.
        :param model_2d_pose_estimator  : The 2D pose Estimator Model.
        :param save_with_iters          : If True, we will also add the iteration number in the save filename.
        """
        print("[TRAINER][SAVING]: Saving the Model after ({}) epochs and ({}) Total Iterations of Training.".format(epoch, iterations), end="\t")
        save_path = os.path.join(self.save_dir, self.save_file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_params    = {'epoch': epoch, 'niters': iterations}
        
        # Saves
        if model_2d_pose_estimator is not None:
            # Saving the Parameters of the 2D Pose Estimator Model.
            save_params[self.state_dict_keys['pose_2d_estimator']] = model_2d_pose_estimator.state_dict()
        if lifting_network is not None:
            # Saving the parameters of the 2D-3D Lifting Network.
            save_params[self.state_dict_keys['lifting_network']]   = lifting_network.state_dict()
        if optimizer is not None:
            # Saving the parameters of the optimizer
            save_params[self.state_dict_keys['optimizer']] = optimizer.state_dict()
        if scheduler is not None:
            save_params[self.state_dict_keys['scheduler']] = scheduler.state_dict()
            
        # Checkpoint name
        if is_best:
            assert suffix is not None
            checkpoint_name = 'model_best-{}.pth.tar'.format(suffix)
        else:
            extra = [str(epoch)]
            if iterations is not None:
                extra.append(str(iterations))
            if suffix is not None:
                extra.append(suffix)
            checkpoint_name = 'model-{}.pth.tar'.format('-'.join(extra))
        checkpoint_path = os.path.join(save_path, checkpoint_name)

        print("[TRAINER][SAVING]: We are storing the parameters of models in {}.".format(checkpoint_path))
        torch.save(save_params, checkpoint_path)


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


    def update_training_metrics(self, vals: dict, batch, batch_idx: int):
        preds_mpjpe_sv =  vals[self.lifting_preds_key].detach().cpu().numpy().tolist()
        targets_mpjpe_sv = vals[self.lifting_targets_key].detach().cpu().numpy().tolist()
        mpjpe_sv_val, _, _ = calculate_mpjpe(
          predictions=preds_mpjpe_sv, targets=targets_mpjpe_sv, n_mpjpe=False, p_mpjpe=False)
                 
        depths        = vals["depth"].detach().cpu()
        uncertainties = vals["uncertainty"].detach().cpu()
        
        # the following lambda updates the mean of a sequence of samples
        # given the old mean, the old number of samples, and the new sample
        # it accurately computes the mean without buffering all the elements of the sequence
        update_mean = lambda new_x, old_m, old_n: (new_x + old_m * old_n) / (old_n + 1)
        
        self.train_mpjpe_3d_sv["mean"] = update_mean(
            mpjpe_sv_val, self.train_mpjpe_3d_sv["mean"], self.train_mpjpe_3d_sv["count"])
        self.train_mpjpe_3d_sv["count"] += 1
        self.train_mpjpe_3d_sv["max"] = max(self.train_mpjpe_3d_sv["max"], mpjpe_sv_val)
        self.train_mpjpe_3d_sv["min"] = min(self.train_mpjpe_3d_sv["min"], mpjpe_sv_val)
       
        target_root_rel_depths = np.array(batch["target_root_rel_depth"])        
        depth_mse_val = depth_mse(
            pred_depth=depths, target_depth=target_root_rel_depths)
        self.train_depth_mse["mean"] = update_mean(
            depth_mse_val, self.train_depth_mse["mean"], self.train_depth_mse["count"])
        self.train_depth_mse["count"] += 1
        self.train_depth_mse["max"] = max(self.train_depth_mse["max"], depth_mse_val)
        self.train_depth_mse["min"] = min(self.train_depth_mse["min"], depth_mse_val)
                    
        std_scaled_depth_mse_val = std_scaled_depth_mse(
            pred_depth=depths, target_depth=target_root_rel_depths, pred_std=uncertainties)
        self.train_std_scaled_depth_mse["mean"] = update_mean(
            std_scaled_depth_mse_val, self.train_std_scaled_depth_mse["mean"], self.train_std_scaled_depth_mse["count"])
        self.train_std_scaled_depth_mse["count"] += 1
        self.train_std_scaled_depth_mse["max"] = max(self.train_std_scaled_depth_mse["max"], std_scaled_depth_mse_val)
        self.train_std_scaled_depth_mse["min"] = min(self.train_std_scaled_depth_mse["min"], std_scaled_depth_mse_val)
        
        self.train_std["mean"] = update_mean(
            uncertainties.mean().item(), self.train_std["mean"], self.train_std["count"])
        self.train_std["count"] += 1
        self.train_std["max"] = max(self.train_std["max"], uncertainties.max().item())
        self.train_std["min"] = min(self.train_std["min"], uncertainties.min().item())
        

    def compute_iteration_losses(self, vals: Dict, batch_idx: int):
        """
        :param vals: A dictionary consisting of the various values needed to calculate the loss of a given iteration.
        :param batch_idx: The current batch index.
        :return: loss_values_per_iteration_to_print --> A dictionary containing the loss values of different loss modules
                                                        along with the overall loss for Every Iteration for Printing.
                loss_exploded ---> A flag which indicates whether the losses have exploded. If True, I will store all the checkpoints immediately.
        """        
        assert len(self.individual_losses_names) > 0
        
        # the following lambda updates the mean of a sequence of samples 
        # given the old mean, the old number of samples, and the new sample
        # it accurately computes the mean without buffering all the elements of the sequence
        update_mean = lambda new_x, old_m, old_n: (new_x + old_m * old_n) / (old_n + 1) 
        
        total_loss = 0
        losses_to_print = {}
        
        for loss_name in self.individual_losses_names:
            # get predictions, targets, and weights
            loss_keys   = self.losses_keys[loss_name]
            pred_vals   = vals[loss_keys[0]]
            target_vals = vals[loss_keys[1]]
            weight_vals = vals.get(loss_keys[2])
            
            if (
                target_vals is not None 
                and self.detach_gradient_target 
                and loss_name != 'loss_3d_pose_discriminator'):
                target_vals = target_vals.detach()
                
            # get loss module and compute loss
            vals_dict = {'pred': pred_vals, 'target': target_vals, 'weights': weight_vals}
            loss_module = self.loss_modules[loss_name]
            loss  = loss_module(**vals_dict)
            loss_val = loss.item()
            
            # store loss values
            total_loss += loss
            
            # update the loss mean for the current loss module
            old_loss_mean = self.epoch_losses[loss_name]["mean"]
            new_loss_mean = update_mean(loss_val, old_loss_mean, self.epoch_losses[loss_name]["count"])
            self.epoch_losses[loss_name]["mean"] = new_loss_mean
            self.epoch_losses[loss_name]["count"] += 1
            
            losses_to_print[loss_name] = (loss_val, new_loss_mean)
            
            loss_exploded = (
                loss_val == np.infty
                or loss_val == -np.infty
                or np.isnan(loss_val))
            if loss_exploded:
                print("[TRAINER][LOSS][WARNING]: {0} has exploded: {0}={1}".format(
                    loss_name.upper(),
                    loss_val))

        # compute the total loss
        total_loss_val = total_loss.item()
            
        # update the total loss mean
        old_total_loss_mean = self.epoch_losses[self.total_loss_key]["mean"]
        new_total_loss_mean = update_mean(total_loss_val, old_total_loss_mean, self.epoch_losses[self.total_loss_key]["count"])
        self.epoch_losses[self.total_loss_key]["mean"] = new_total_loss_mean
        self.epoch_losses[self.total_loss_key]["count"] += 1
        
        losses_to_print[self.total_loss_key] = (total_loss_val, new_total_loss_mean)

        return total_loss, losses_to_print, loss_exploded
    

    def collapse_views(self, data):
        bboxes      = data['bboxes']
        N_diff      = bboxes.size(0)
        num_cameras = bboxes.size(1)
        num_frames  = bboxes.size(2)
        R_shape     = (data['R'].size(-2), data['R'].size(-1))
        data['bboxes']                                 = data['bboxes'].reshape(-1, num_frames, 4)
        data['consider']                               = data['consider'].reshape(-1, num_frames)
        data['keypoints_det']                          = data['keypoints_det'].reshape(-1, num_frames, self.n_joints, 2)
        data['keypoints_det_norm']                     = data['keypoints_det_norm'].reshape(-1, num_frames, self.n_joints, 2)
        data['target_pose_2d']                         = data['target_pose_2d'].reshape(-1, num_frames, self.n_joints, 2)
        data['target_pose_2d_norm']                    = data['target_pose_2d_norm'].reshape(-1, num_frames, self.n_joints, 2)
        data['pelvis_cam_z']                           = data['pelvis_cam_z'].reshape(-1, num_frames)
        data['labels']                                 = data['labels'].reshape(-1, num_frames)
        data['target_pose_3d']                         = data['target_pose_3d'].reshape(-1, num_frames, self.n_joints, 3)
        data['triangulated_pose_3d']                   = data['triangulated_pose_3d'].reshape(-1, num_frames, self.n_joints, 3)
        data['R']                                      = data['R'].reshape(-1, num_frames, R_shape[0], R_shape[1])
        data['t']                                      = data['t'].reshape(-1, num_frames, 3)
        data['K']                                      = data['K'].reshape(-1, num_frames, 3, 3)
        data['dist']                                   = data['dist'].reshape(-1, num_frames, 5)
        data['target_root_rel_depth']                  = data['target_root_rel_depth'].reshape(-1, num_frames, self.n_joints)
        data['triangulated_root_rel_depth']            = data['triangulated_root_rel_depth'].reshape(-1, num_frames, self.n_joints)
        data['target_pose_3d_camera_coordinate']       = data['target_pose_3d_camera_coordinate'].reshape(-1, num_frames, self.n_joints, 3)
        data['triangulated_pose_3d_camera_coordinate'] = data['triangulated_pose_3d_camera_coordinate'].reshape(-1, num_frames, self.n_joints, 3)
        if self.inp_lifting_net_is_images:
            bs, ncs, nfs, c, h, w      = data['lifting_net_images'].size()
            data['lifting_net_images'] = data['lifting_net_images'].reshape(-1, num_frames, c, h, w)

        # TODO for Batch graphs.
        camera_indexes = torch.arange(start=0, end=num_cameras).float()
        camera_indexes = camera_indexes.reshape(1, num_cameras, 1)
        camera_indexes = camera_indexes.repeat(N_diff, 1, num_frames)
        camera_indexes = camera_indexes.reshape(-1, num_frames)
        return data, camera_indexes, R_shape
    
    
    def evaluate_models_by_plotting(self, suffix, lifting_network, return_train, phase, epoch, model_2d_pose_estimator):
        """
        Function to evaluate the learning of the 2D to 3D Lifting Network by plotting the keypoints.
        :param model_2d_pose_estimator  : The 2D Pose Estimator Model.
        :param suffix                   : The suffix to append to the saved images.
        :param lifting_network          : The 2D to 3D Pose Lifting Network.
        :param return_train             : If True, the model is train mode will be returned, i.e. model.train() or else it will be returned with model.eval()
        :param phase                    : The current phase of learning.
        :param epoch                    : The current epoch of learning.
        :return: None
        """
        print("[TRAINER][EVAL-PLOT][WARNING]: <evaluate_models_by_plotting> called.")
        
        lifting_network.eval()
        if model_2d_pose_estimator is not None:
            model_2d_pose_estimator.eval()

        if phase in ['train', 'training']:
            print("[TRAINER][EVAL-PLOT]: Will be plotting the Images for Training Phase.")
            dataloader = self.train_loader_simple
        else:
            print("[TRAINER][EVAL-PLOT]: Will be plotting the Images for Test Phase.")
            dataloader = self.test_loader

        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader, start=1):
                vals_plot = self.forward_pass(
                    lifting_network=lifting_network,
                    data=data,
                    for_mpjpe=False,
                    train_flag=False,
                    model_2d_pose_estimator=model_2d_pose_estimator)
                
                if vals_plot is None:
                    print("[TRAINER][EVAL-PLOT]: Forward pass returns None in <evaluate_models_by_plotting> function in batch ({}).".format(batch_idx))
                    continue

                print("[TRAINER][EVAL-PLOT]: Plotting the Images of Batch {} in the epoch {} for {} Phase.".format(batch_idx, epoch, phase.upper()))
                json_data_phase = self.get_json_file(vals=vals_plot, json_data_phase={})
                self.plotting_function(json_data=json_data_phase, suffix=suffix, epoch=epoch, phase=phase, save_dir=self.save_dir)
                if batch_idx == self.number_of_batches_to_plot:
                    print("[TRAINER][EVAL-PLOT]: Done Plotting Images for {} batches.".format(self.number_of_batches_to_plot))
                    break
                if self.debugging and (batch_idx == 1):
                    print("[TRAINER][EVAL-PLOT]: Done Plotting Images for 1 batch in the Debugging Mode.")
                    break

        if return_train:
            lifting_network.train()
            if model_2d_pose_estimator is not None:
                model_2d_pose_estimator.train()


    def plotting_function(self, json_data: dict, suffix: str, epoch: int, phase: str,
                          save_dir: str):
        """
        Function to Plot the Images.
        :param json_data: A dictionary consisting of the various values that needs to be plotted.
        :param suffix:    A suffix that needs to added to image filename before storing.
        :param epoch:     The current epoch of storing the images.
        :param phase:     The current phase of learning.
        :param save_dir:  The directory to store the images.
        :return: None
        """
        print("[TRAINER]: >>>>>>>>>>>>>> PLOTTING FUNCTION CALLED <<<<<<<<<<<<<<<<")
        
        key_vals = list(json_data.keys())
        save_folder = os.path.join(save_dir, 'Images')
        for key_val in key_vals:
            print("[TRAINER][PLOTTING]: Plotting in the {} phase for {} in the epoch {}.".format(phase, key_val, epoch))
            pose_information = json_data[key_val]
            plot_poses_2D_3D(save_folder=save_folder, pose_information=pose_information, key_val_plot=key_val,
                             vals_to_plot_2D=['Det_2D', 'Proj_2D-Lift', 'GT_2D'], phase=phase, suffix=suffix,
                             vals_to_plot_3D=['Anno-3D', 'Tri-3D', 'Pred-3D'], rhip_idx=self.rhip_idx,
                             lhip_idx=self.lhip_idx, neck_idx=self.neck_idx, pelvis_idx=self.pelvis_idx,
                             bones=self.bone_pairs)
        print("[TRAINER][PLOTTING]: Done")
    

    def get_inp_target_for_testing(self, poses_3d_world, poses_3d_camera, target_pose_3d, for_mpjpe, bounding_boxes=None,
                                   keypoints_det=None, triangulated_pose_3d=None, R=None, t=None, K=None, dist=None, data=None):
        # TODO: Farouk added this condition to make sure that the poses are in World Coordinates.
        assert not(self.mpjpe_poses_in_camera_coordinates), "The MPJPE poses should be in World Coordinates."
        
        poses_3d_mpjpe = poses_3d_camera if self.mpjpe_poses_in_camera_coordinates else poses_3d_world
        poses_3d_mpjpe = poses_3d_mpjpe.reshape(-1, self.n_joints, 3)
        target_pose_3d = target_pose_3d.reshape(-1, self.n_joints, 3)
        ret_vals       = {self.lifting_preds_key: poses_3d_mpjpe, self.lifting_targets_key: target_pose_3d}
        if not for_mpjpe:  # as_json_file or for_plotting:
            assert bounding_boxes       is not None
            assert keypoints_det        is not None
            assert triangulated_pose_3d is not None
            assert R    is not None
            assert t    is not None
            assert K    is not None
            assert dist is not None
            assert data is not None

            NN          = poses_3d_mpjpe.size(0)
            image_paths = data['image_paths']
            camera_ids  = data['camera_ids']
            subject_ids = data['subject_ids']
            action_ids  = data['action_ids']
            frame_ids   = list(data['frame_ids'])
            assert len(image_paths) == NN, f"The length of the image paths ({len(image_paths)}) is not equal to the number of samples ({NN}) in the batch."
            assert len(camera_ids)  == NN, f"The length of the camera ids ({len(camera_ids)}) is not equal to the number of samples ({NN}) in the batch."
            assert len(subject_ids) == NN, f"The length of the subject ids ({len(subject_ids)}) is not equal to the number of samples ({NN}) in the batch."
            assert len(action_ids)  == NN, f"The length of the action ids ({len(action_ids)}) is not equal to the number of samples ({NN}) in the batch."
            assert len(frame_ids)  == NN, f"The length of the frame ids ({len(frame_ids)}) is not equal to the number of samples ({NN}) in the batch."
            vals_extra = {'bboxes': bounding_boxes.reshape(-1, 4), 'image_paths': image_paths,
                          'camera_ids': camera_ids, 'subject_ids': subject_ids, 'frame_ids': frame_ids,
                          'keypoints_2d_det': keypoints_det.reshape(-1, self.n_joints, 2),
                          'triangulated_3d': triangulated_pose_3d.reshape(-1, self.n_joints, 3),
                          'R': R, 't': t, 'K': K, 'dist': dist}
            ret_vals = {**ret_vals, **vals_extra}
        return ret_vals

    
    @staticmethod
    def get_json_file(vals : dict, json_data_phase : dict):
        """
        Function to obtain the required data to store as a Json File.
        :param vals: A dictionary containing the necessary data to be stored in the Json File.
        :param json_data_phase: A dictionary for saving the data in the Json File.
        :return: json_data_phase.
        """
        pred_3d       = vals['pred_3d'].cpu().numpy() # The predicted 3D using the Lifting Network. # refer to self.pred_key
        tar_3d        = vals['target_3d'].cpu().numpy() # The Ground Truth 3D. # refer to self.target_key
        tri_3d        = vals['triangulated_3d'].cpu().numpy() # The Triangulated 3D.
        det_2d        = vals['keypoints_2d_det'].cpu().numpy() # The 2D keypoints detected by the 2D Pose Estimator Model.
        bboxes        = vals['bboxes'].cpu().numpy()
        image_paths   = vals['image_paths'] # It is a list of path of the images.
        camera_ids    = vals['camera_ids']  # It is a list of ids of the cameras.
        subject_ids   = vals['subject_ids'] # It is a list of subjects.
        action_ids    = vals['action_ids']  # It is a list of actions.
        frame_ids     = vals['frame_ids']   # It is a list of frames.
        R, t, K, dist = vals['R'].cpu().numpy(), vals['t'].cpu().numpy(), vals['K'].cpu().numpy(), vals['dist'].cpu().numpy()
        N             = len(image_paths)
        assert len(camera_ids)  == N
        assert len(subject_ids) == N
        assert len(action_ids)  == N
        assert len(frame_ids)   == N
        assert bboxes.shape[0]  == N
        assert R.shape[0] == N
        assert t.shape[0] == N
        assert K.shape[0] == N
        assert dist.shape[0] == N
        assert det_2d.shape[0] == N
        assert tar_3d.shape[0] == N
        assert pred_3d.shape[0] == N

        for i in range(N):
            pred_3d_i        = pred_3d[i]
            tar_3d_i         = tar_3d[i]
            tri_3d_i         = tri_3d[i]
            det_2d_i         = det_2d[i]

            R_i, t_i         = R[i], t[i]
            K_i, dist_i      = K[i], dist[i]
            rvec_i           = cv2.Rodrigues(R)[0]
            proj_2d_lift_i   = cv2.projectPoints(pred_3d_i, rvec_i, t_i, K_i, dist_i)[0].reshape(-1, 2) # Projection of the 3D Pose Predicted the 2D-3D Lifting Network.
            proj_2d_gt_i     = cv2.projectPoints(tar_3d_i, rvec_i, t_i, K_i, dist_i)[0].reshape(-1, 2)  # 2D GT pose

            bboxes_i    = bboxes[i]
            path_i      = image_paths[i]
            cam_id_i    = camera_ids[i]
            sub_id_i    = subject_ids[i]
            action_id_i = action_ids[i]
            frame_id_i  = frame_ids[i]
            info_2d_i   = {'Det_2D': det_2d_i,  'Proj_2D-Lift': proj_2d_lift_i, 'GT_2D' : proj_2d_gt_i}
            info_3d_i   = {'Anno-3D': tar_3d_i, 'Tri-3D': tri_3d_i,             'Pred-3D': pred_3d_i}
            pose_info_i = {'camera_id': cam_id_i, 'subject_id': sub_id_i, 'action_id': action_id_i,
                           'Image_Path_2D': path_i, 'bbox_2D': bboxes_i, 'image_id': frame_id_i,
                           'info_2D': info_2d_i, 'info_3D': info_3d_i}
            key_val     = ['{}'.format(sub_id_i), '{}'.format(action_id_i), '{}'.format(frame_id_i), '{}'.format(cam_id_i)]
            key_val     = '-'.join(key_val)
            json_data_phase[key_val] = pose_info_i
        return json_data_phase

