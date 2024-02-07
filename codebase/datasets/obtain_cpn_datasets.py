import torch
import os
import dgl

from datasets.h36m_dataset       import Human36mDataset, Fusion
from datasets.utils              import define_actions
from datasets.dataset_3dhp       import Fusion as Fusion_3DHP
from torch.utils.data.dataloader import default_collate



class collate_with_graphs(object):
    def __init__(self, use_graphs): #, keys_to_consider, graph_key):
        self.use_graphs = use_graphs
        self.graph_key  = 'graph'
        # self.keys_to_consider    = keys_to_consider
        # self.graph_key           = graph_key
        # self.new_batch_dicts     = {key: [] for key in self.keys_to_consider}
        # self.key_no_modification = ['image_paths', 'camera_ids', 'frame_ids', 'action_ids', 'subject_ids', graph_key]

    def __call__(self, batch):
        samples, graphs = map(list, zip(*batch))
        final_samples   = default_collate(samples)
        if self.use_graphs:
            final_samples['batch_graphs'] = dgl.batch(graphs)
        return final_samples


def obtain_cpn_dataset(opt, dict_vals_graph):   
    if len(dict_vals_graph) == 0:
        use_graphs = False
    else:
        use_graphs = True

    collate_func = collate_with_graphs(use_graphs=use_graphs)
    root_path    = opt.root_path
    dataset_name = opt.dataset
    root_path    = os.path.join(root_path, 'cpn-data') + os.sep
    
    if dataset_name.lower() == 'h36m':
        dataset_path     =  root_path + 'data_3d_' + dataset_name + '.npz' # os.path.join(root_path , 'data_3d_' + dataset_name + '.npz')
        print("Obtaining the H36M dataset using the CPN keypoints.", dataset_path)
        dataset          = Human36mDataset(dataset_path, opt) # actions = define_actions(opt.actions)
        train_data       = Fusion(batchSize=opt.batch_size, opt=opt, train=True, dataset=dataset, root_path=root_path, MAE=opt.MAE, tds=opt.t_downsample, use_graphs=use_graphs, dict_vals_graph=dict_vals_graph)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_workers), pin_memory=True , collate_fn=collate_func)

        test_data       = Fusion(batchSize=opt.batch_size_test, opt=opt, train=False,dataset=dataset, root_path =root_path, MAE=opt.MAE, tds=opt.t_downsample, use_graphs=use_graphs, dict_vals_graph=dict_vals_graph)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size_test,  shuffle=False, num_workers=int(opt.num_workers), pin_memory=True, collate_fn=collate_func)

    elif dataset_name.lower() == 'mpi' or dataset_name.lower() == '3dhp':
        print("Obtaining the MPI_INF_3DHP dataset using the CPN keypoints.")
        print(root_path)
        train_data       = Fusion_3DHP(batchSize=opt.batch_size, opt=opt, train=True, root_path=root_path, MAE=opt.MAE, use_graphs=use_graphs, dict_vals_graph=dict_vals_graph)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=int(opt.num_workers), pin_memory=True, collate_fn=collate_func)
        
        test_data        = Fusion_3DHP(batchSize=opt.batch_size_test, opt=opt, train=False, root_path=root_path, MAE=opt.MAE, use_graphs=use_graphs, dict_vals_graph=dict_vals_graph)
        test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size_test, shuffle=False, num_workers=int(opt.num_workers), pin_memory=True, collate_fn=collate_func)

    else:
        raise NotImplementedError("We have not implemented this for another datasets.")

    return train_dataloader, test_dataloader
