import os
from pathlib import Path
import copy
from datetime import datetime
from collections import OrderedDict
import time
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm.auto import tqdm

from monai.data import partition_dataset_classes, partition_dataset

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchmetrics

import pytorch_lightning as L

import lung_repo.pytorch.lightning_GNN as gnn_modules
from lung_repo.pytorch.dataset_class import DatasetGeneratorImage
from lung_repo.pytorch.run_model_base import RunModel

class RunMalignancyModel(RunModel):
    def __init__(self, config):
        super().__init__(config=config)


    def set_data(self, resume=None):
        self.data = DatasetGeneratorImage(config=self.config)


    def set_model(self):
        """
        sets and assigns GNN model to self.model
        make sure to assign dataset before calling set_model
        """
        self.model = getattr(gnn_modules, self.config['lightning_module'])(self.config) 

    def make_folds_from_dir(self, out_file_name='preset_folds_malignancy.pkl'):
        self.fold_dict = {}
        patient_list = []
       
        for idx in range(5):
            fold_dir = self.data.data_path.joinpath(self.config['fold_dir'])
            self.fold_dict[f"fold_{idx}"] = np.unique([f.name.split('_')[0] for f in fold_dir.glob(f"F{idx+1}/label/*")])
            patient_list.extend(np.unique([f.name.split('_')[0] for f in fold_dir.glob(f"F{idx+1}/label/*")]))

        self.preset_fold_list = []
        for fold in self.fold_dict:
            self.preset_fold_list.append([patient_list.index(pat) for pat in self.fold_dict[fold]]) 
        with open(self.data.data_path.joinpath(out_file_name), 'wb') as f:
            pickle.dump(self.preset_fold_list, f)
            f.close()
       

    def set_train_test_split(self):

        self.folds = None
        if self.config['preset_folds']:
            if not os.path.exists(self.data.data_path.joinpath(self.config['preset_fold_file'])): 
                self.make_folds_from_dir(self.config['preset_fold_file'])
            self.folds = pd.read_pickle(self.data.data_path.joinpath(self.config['preset_fold_file']))
          
        else:
            #self.folds = partition_dataset_classes(range(len(self.data.patients.index.get_level_values(0).unique())), self.data.patients.groupby('patients').mean('nodes')['labels']>=0.5, num_partitions=5, shuffle=True, seed=self.config['seed'])
            self.folds = partition_dataset(range(len(self.data.patients.index.get_level_values(0).unique())), num_partitions=5, shuffle=False)
            with open(self.data.data_path.joinpath(self.config['preset_fold_file']), 'wb') as f:
                pickle.dump(self.folds, f)
                f.close()

        self.train_folds = [[2,3,4],
                            [3,4,0],
                            [4,0,1],
                            [0,1,2],
                            [1,2,3]]
        self.val_folds = [1,2,3,4,0]
        self.test_folds = [0,1,2,3,4]

        self.nested_train_folds = [
                                   [[1,2,3],[2,3,4],[3,4,1],[4,1,2]],
                                   [[2,3,4],[3,4,0],[4,0,2],[0,2,3]],
                                   [[3,4,0],[4,0,1],[0,1,3],[1,3,4]],
                                   [[4,0,1],[0,1,2],[1,2,4],[2,4,0]],
                                   [[0,1,2],[1,2,3],[2,3,0],[3,0,1]],
                                  ]
        self.nested_val_folds = [
                                 [4,1,2,3],
                                 [0,2,3,4],
                                 [1,3,4,0],
                                 [2,4,0,1],
                                 [3,0,1,2],
                                ]

        self.train_splits = [self.folds[i]+self.folds[j]+self.folds[k] for i,j,k in self.train_folds]
        self.val_splits = [self.folds[i] for i in self.val_folds]
        self.test_splits = [self.folds[i] for i in self.test_folds]

        self.aug_folds = []
        for fold_idx, fold in enumerate(self.train_splits):
            fold_pats = self.data.patients.index.get_level_values(0).unique()[fold]
            self.aug_folds.append(['_'.join(file_name.split('.')[0].split('_')[2:]) for file_name in self.data.processed_file_names if file_name.split('_')[2] in fold_pats and 'rotation' in file_name])
            aug_idx = self.data.full_patients.index.get_level_values(0).unique().get_indexer(self.aug_folds[fold_idx])
            self.train_splits[fold_idx].extend(aug_idx)

        self.nested_train_splits = [[self.folds[i]+self.folds[j]+self.folds[k] for i,j,k in nest] for nest in self.nested_train_folds]
        self.nested_val_splits = [[self.folds[i] for i in nest] for nest in self.nested_val_folds] 

        self.nested_aug_folds = []
        for nest_idx, nest in enumerate(self.nested_train_splits):
            self.nested_aug_folds.append([])
            for fold_idx, fold in enumerate(nest):
                fold_pats = self.data.patients.index.get_level_values(0).unique()[fold]
                self.nested_aug_folds[nest_idx].append(['_'.join(file_name.split('.')[0].split('_')[2:]) for file_name in self.data.processed_file_names if file_name.split('_')[2] in fold_pats and 'rotation' in file_name])
                aug_idx = self.data.full_patients.index.get_level_values(0).unique().get_indexer(self.nested_aug_folds[nest_idx][fold_idx])
                self.nested_train_splits[nest_idx][fold_idx].extend(aug_idx)
        
    
    def set_data_module(self):
        #self.data_module_cross_val = [LightningDataset(train_dataset=self.data[fold], val_dataset=self.data[self.val_splits[idx]], test_dataset=self.data[self.test_splits[idx]], batch_size=self.config['batch_size'], num_workers=16, pin_memory=True, persistent_workers=False, shuffle=True, drop_last=True) for idx, fold in enumerate(self.train_splits)] 
        self.data_module_cross_val = [LightningDataset(train_dataset=self.data[fold], val_dataset=self.data[self.val_splits[idx]], test_dataset=self.data[self.test_splits[idx]], batch_size=self.config['batch_size'], num_workers=16, pin_memory=True, persistent_workers=False, shuffle=True, drop_last=False) for idx, fold in enumerate(self.train_splits)] 

    def read_tf_weights(self):
        self.weights_df = pd.read_pickle(os.path.join(self.config['data_path'], self.config['transfer_weights_file']))

    def load_tf_weights(self, fold_idx):
        weights_df = pd.read_pickle(os.path.join(self.config['data_path'], self.config['transfer_weights_file']))

        model_weights = weights_df.loc[self.test_folds[fold_idx]]

        with torch.no_grad():
            for name, m in self.model.extractor.named_modules():
               if isinstance(m, nn.Conv3d):
                   if name == 'cn1':
                       if self.config['n_in_channels'] == 1:
                           m.weight.copy_(torch.from_numpy(model_weights[f"conv3d_weight_{name.strip('cn')}"])[:,0,:,:,:].unsqueeze(1))
                       else:
                           m.weight.copy_(torch.from_numpy(model_weights[f"conv3d_weight_{name.strip('cn')}"]))
                   else:
                       m.weight.copy_(torch.from_numpy(model_weights[f"conv3d_weight_{name.strip('cn')}"]))
               #if isinstance(m, nn.Linear):
               #    if 'linear' in name:
               #        m.weight.copy_(torch.from_numpy(model_weights[f"linear_weight_1"]).T)
               if isinstance(m, nn.BatchNorm3d):
                   m.bias.copy_(torch.from_numpy(model_weights[f"bn_beta_{name.strip('bn')}"]))
                   m.running_mean.copy_(torch.from_numpy(model_weights[f"bn_mean_{name.strip('bn')}"]))
                   m.running_var.copy_(torch.from_numpy(model_weights[f"bn_variance_{name.strip('bn')}"]))
         
            #for name, m in self.model.classify.named_modules():
            #    if isinstance(m, nn.Linear):
            #       m.weight.copy_(torch.from_numpy(model_weights[f"linear_weight_classify"]).T[1])
               m.data = m.data.float()

         
