import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import parameters as par
from finetune_with_lifting_net import finetune_with_lifting_net
from utils                     import use_cuda, Logger, device


parser = argparse.ArgumentParser()
# parser = par.dataset_parameters(parser)
# parser = par.basic_parameters(parser)
# parser = par.general_optimizer_parameters(parser)
# parser = par.training_hyperparameters(parser)
# parser = par.loss_parameters(parser)
# parser = par.early_stopping_conds(parser)
# parser = par.grad_clip_norm(parser)
# parser = par.func_to_load_checkpoints(parser)
# parser = par.obtain_lifting_network(parser)
# parser = par.cpn_data_params(parser)
parser = par.get_parameters(parser=parser)
config = parser.parse_args() #args=[])

if not config.evaluate_learnt_model:
    log_dirname  = "stdout_logs"
    
    log_fname = "stdout_logs.txt"
    if config.perform_test and not(config.test_on_training_set):
        log_fname = "stdout_logs_test.txt"
    elif config.perform_test and config.test_on_training_set:
        log_fname = "stdout_logs_test_on_train.txt"
    elif config.create_stats_dataset and not(config.stats_dataset_from_test_set):
        log_fname = "stdout_logs_stats_dataset_from_train.txt"
    elif config.create_stats_dataset and config.stats_dataset_from_test_set:
        log_fname = "stdout_logs_stats_dataset_from_test.txt"
        
    if not os.path.exists(os.path.join(config.save_dir, log_dirname)):
        os.makedirs(os.path.join(config.save_dir, log_dirname))
    sys.stdout = Logger(os.path.join(config.save_dir, log_dirname, log_fname))

config.use_cuda         = use_cuda
config.device           = device
finetune_with_lifting_net(device=device, use_cuda=use_cuda, config=config)