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

### Replace with relevant directory names until I figure something else out#######
import lung_repo.pytorch.lightning_GNN as gnn_modules
from lung_repo.pytorch.dataset_class import DatasetGeneratorImage
#################################################################################

class RunModel(object):
    def __init__(self, config):
        self.config = config
        L.seed_everything(self.config['seed'])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_weights = None

        if self.config['log_dir'] is None:
            self.log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            self.log_dir = os.path.join('logs', self.config['log_dir'])

        self.metric_dir = os.path.join(self.log_dir, 'metric_dfs')
        Path(self.metric_dir).mkdir(parents=True, exist_ok=True)
        print(f"logs are located at: {self.log_dir}")
        #self.writer = SummaryWriter(self.log_dir)
        print('remember to set the data')


    def set_model(self):
        """
        sets and assigns GNN model to self.model
        make sure to assign dataset before calling set_model
        """
        self.model = getattr(gnn_modules, self.config['lightning_module'])(self.config) 


    def load_fold_weights(self, idx=None):
        if idx is None:
            raise Exception("did not specify fold to load weights from")
        self.loaded_weights = torch.load(os.path.join(self.config['weight_dir'], self.config['extractor_weight_files'][idx]))['state_dict']
        for key in self.loaded_weights.keys():
            if 'classify' in key.lower():
                self.loaded_weights.pop('classify')
        self.model.load_state_dict(self.loaded_weights, strict=False)
 

    def set_data(self, resume=None):

        self.data = DatasetGeneratorImage(config=self.config, resume=resume)


    def set_train_test_split(self):
        self.folds = None
        if self.config['preset_folds']:
            self.folds = pd.read_pickle(self.data.data_path.joinpath(self.config['preset_fold_file']))
        else:
            self.folds = partition_dataset_classes(range(len(self.data.patients)), self.data.patients['labels'], num_partitions=5, shuffle=True, seed=self.config['seed'])
            with open(self.data.data_path.joinpath(self.config['preset_fold_file']), 'wb') as f:
                pickle.dump(self.folds, f)
                f.close()

        self.train_folds = [[0,1,2],
                            [4,0,1],
                            [3,4,0],
                            [2,3,4],
                            [1,2,3]]
        self.val_folds = [3, 2, 1, 0, 4]
        self.test_folds = [4, 3, 2, 1, 0]


        self.nested_train_folds = [[[0,1,2],[1,2,3],[2,3,0],[3,0,1]],
                                   [[4,0,1],[0,1,2],[1,2,4],[2,4,0]],
                                   [[3,4,0],[4,0,1],[0,1,3],[1,3,4]],
                                   [[2,3,4],[3,4,0],[4,0,2],[0,2,3]],
                                   [[1,2,3],[2,3,4],[3,4,1],[4,1,2]]]
        self.nested_val_folds = [[3,0,1,2],
                                 [2,4,0,1],
                                 [1,3,4,0],
                                 [0,2,3,4],
                                 [4,1,2,3]]

        self.train_splits = [self.folds[i]+self.folds[j]+self.folds[k] for i,j,k in self.train_folds]
        self.val_splits = [self.folds[i] for i in self.val_folds]
        self.test_splits = [self.folds[i] for i in self.test_folds]

        ################## Add Augmentations to train splits here ###########################
        ## Below is an example from my ln malignancy project, which gets information from a MultiIndex DataFrame. Assumes that augmented patients have 'rotation' in the patient name/id 
        self.aug_folds = []
        for fold_idx, fold in enumerate(self.train_splits):
            fold_pats = self.data.patients.index.get_level_values(0).unique()[fold]
            self.aug_folds.append(['_'.join(file_name.split('.')[0].split('_')[2:]) for file_name in self.data.processed_file_names if file_name.split('_')[2] in fold_pats and 'rotation' in file_name])
            aug_idx = self.data.full_patients.index.get_level_values(0).unique().get_indexer(self.aug_folds[fold_idx])
            self.train_splits[fold_idx].extend(aug_idx)
        #####################################################################################

        self.nested_train_splits = [[self.folds[i]+self.folds[j]+self.folds[k] for i,j,k in nest] for nest in self.nested_train_folds]
        self.nested_val_splits = [[self.folds[i] for i in nest] for nest in self.nested_val_folds] 

        self.nested_aug_folds = []
        for fold_idx, fold in enumerate(self.nested_train_splits):
            self.aug_nested_folds,append([])
            for nest_idx, nest in enumerate(fold):
                fold_pats = self.data.patients.index.get_level_values(0).unique()[nest]
                self.nested_aug_folds[fold_idx].append(['_'.join(file_name.split('.')[0].split('_')[2:]) for file_name in self.data.processed_file_names if file_name.split('_')[2] in fold_pats and 'rotation' in file_name])
                aug_idx = self.data.full_patients.index.get_level_values(0).unique().get_indexer(self.aug_nested_folds[fold_idx][nest_idx])
                self.nested_train_splits[fold_idx][nest_idx].extend(aug_idx)
                 
    
    def set_data_module(self):
        #self.data_module_cross_val = [LightningDataset(train_dataset=self.data[fold], val_dataset=self.data[self.val_splits[idx]], test_dataset=self.data[self.test_splits[idx]], batch_size=self.config['batch_size'], num_workers=16, pin_memory=True, persistent_workers=False, shuffle=True, drop_last=True) for idx, fold in enumerate(self.train_splits)] 
        self.data_module_cross_val = [LightningDataset(train_dataset=self.data[fold], val_dataset=self.data[self.val_splits[idx]], test_dataset=self.data[self.test_splits[idx]], batch_size=self.config['batch_size'], num_workers=16, pin_memory=True, persistent_workers=False, shuffle=True, drop_last=False) for idx, fold in enumerate(self.train_splits)] 


    def set_nested_data_module(self):
        self.data_module_nested_cross_val = [[LightningDataset(train_dataset=self.data[nest], val_dataset=self.data[self.nested_val_splits[idx][nest_idx]], test_dataset=self.data[self.test_splits[idx]], batch_size=self.config['batch_size'], num_workers=16, pin_memory=True, persistent_workers=False, shuffle=True, drop_last=False) for nest_idx, nest in enumerate(fold)] for idx, fold in enumerate(self.nested_train_splits)]


    def set_callbacks(self, fold_idx, nest_idx=None):
        self.callbacks = []

        if nest_idx is None:
           model_name = f"top_models_fold_{fold_idx}"
        else:
           model_name = f"top_models_fold_{fold_idx}_nest_{nest_idx}"
           
        #Checkpoint options
        self.callbacks.append(L.callbacks.ModelCheckpoint(
            monitor='val_auc', 
            mode='max', 
            save_top_k=self.config['save_top_k'],
            dirpath=os.path.join(self.log_dir, model_name),
            filename='model_{epoch:02d}_{val_loss:.2f}_{val_auc:.2f}_{val_m:.2f}',
            ))
        self.callbacks.append(L.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=self.config['save_top_k'],
            dirpath=os.path.join(self.log_dir, model_name),
            filename='model_loss_{epoch:02d}_{val_loss:.2f}_{val_auc:.2f}_{val_m:.2f}',
            ))
        self.callbacks.append(L.callbacks.ModelCheckpoint(
            monitor='val_m',
            mode='max',
            save_top_k=self.config['save_top_k'],
            dirpath=os.path.join(self.log_dir, model_name),
            filename='model_m_{epoch:02d}_{val_loss:.2f}_{val_auc:.2f}_{val_m:.2f}',
            )) 
        #self.callbacks.append(L.callbacks.EarlyStopping(monitor='val_loss', patience=20, check_on_train_epoch_end=False))

        #self.callbacks.append(L.callbacks.LearningRateFinder(min_lr=1e-8, max_lr=1e-1))

    def prepare_trainers(self):
        
        self.trainers = []
        self.test_metrics = {}
        self.val_metrics = {}
        self.test_k_metrics = {}
        self.val_k_metrics = {}

        self.set_callbacks(-1)
        for callback in self.callbacks:
            if 'ModelCheckpoint' in callback.__class__.__name__:
                self.test_metrics[callback.monitor] = []
                self.val_metrics[callback.monitor] = []
                self.test_k_metrics[callback.monitor] = []
                self.val_k_metrics[callback.monitor] = []

        self.best_checkpoints = {}
        for idx in range(5):
            self.set_callbacks(idx)

            self.trainers.append(L.Trainer(
                max_epochs=self.config['n_epochs'],
                accelerator="auto",
                devices=self.config['gpu_device'] if torch.cuda.is_available() else None,
                logger=[L.loggers.CSVLogger(save_dir=os.path.join(self.log_dir, f"csvlog_fold_{idx}")), L.loggers.TensorBoardLogger(save_dir=os.path.join(self.log_dir, f"tb_fold_{idx}"))],
                callbacks=self.callbacks,
                #check_val_every_n_epoch = 1,
                #auto_lr_find=True
                ))


    def run(self, resume=False, resume_idx=None):

        if resume:
            if resume_idx is None: 
                raise Exception("need to set resume_idx")
            idx_start = resume_idx
        else:
            self.prepare_trainers()
            idx_start = 0

        for idx in range(idx_start, 5):
            L.seed_everything(self.config['seed'])
            self.set_model()

            if self.config['pretrain']:
                self.load_fold_weights(idx)
 
            self.trainers[idx].fit(self.model, datamodule=self.data_module_cross_val[idx])

            self.best_checkpoints[f"fold_{idx}"] = {callback.monitor: callback.best_model_path for callback in self.trainers[idx].callbacks if 'ModelCheckpoint' in callback.__class__.__name__}

            for monitor, best_model in self.best_checkpoints[f"fold_{idx}"].items():
                self.val_metrics[monitor].append(self.trainers[idx].validate(self.model, datamodule=self.data_module_cross_val[idx], ckpt_path=best_model)[0])
                self.test_metrics[monitor].append(self.trainers[idx].test(self.model, datamodule=self.data_module_cross_val[idx], ckpt_path=best_model)[0])


    def recover_nested_models(self):

        idx_to_recover = []
        max_idx = 0
        max_nest = 0
        for idx in range(5):
            for nest in range(4):
                if os.path.isdir(os.path.join(self.log_dir, f"top_models_fold_{idx}_nest_{nest}")):
                    idx_to_recover.append((idx, nest))
                    if idx > max_idx: 
                        max_idx = idx
                        max_nest = 0
                    if nest > max_nest: max_nest = nest

        print('restart training from last index:')
        print(idx_to_recover[-1])

        self.trainers = {}
        self.test_nested_metrics = {}
        self.val_nested_metrics = {}
        self.callbacks(-1)
        monitor_list = []

        for callback in self.callbacks:
            if 'ModelCheckpoint' in callback.__class__.__name__:
                monitor_list.append(callback.monitor)
                self.test_nested_metrics[callback.monitor] = {f"fold_{fold_idx}": {f"nest_{nest_idx}": [] for nest_idx in range(4)} for fold_idx in range(5)}
                self.val_nested_metrics[callback.monitor] = {f"fold_{fold_idx}": {f"nest_{nest_idx}": [] for nest_idx in range(4)} for fold_idx in range(5)}
    
        self.best_checkpoints = {}
        for idx in range(max_idx, 5):
            self.best_checkpoints[f"fold_{idx}"] = []
            self.trainers[f"fold_{idx}"] = []
            for nest in range(4):
                L.seed_everything(self.config['seed'])
                


    def run_nested(self, resume=False, resume_fold_idx=-1, resume_nest_idx=-1):

        if resume_fold_idx < 0:
            self.trainers = {}
            self.test_nested_metrics = {}
            self.val_nested_metrics = {}

            self.set_callbacks(-1)
            monitor_list = []
            for callback in self.callbacks:
                if 'ModelCheckpoint' in callback.__class__.__name__:
                    monitor_list.append(callback.monitor)
                    self.test_nested_metrics[callback.monitor] = {f"fold_{fold_idx}": {f"nest_{nest_idx}": [] for nest_idx in range(4)} for fold_idx in range(5)}
                    self.val_nested_metrics[callback.monitor] = {f"fold_{fold_idx}": {f"nest_{nest_idx}": [] for nest_idx in range(4)} for fold_idx in range(5)}
    
            self.best_checkpoints = {}
        for idx in range(5):
            if resume_fold_idx < idx: 
                self.best_checkpoints[f"fold_{idx}"] = []
                self.trainers[f"fold_{idx}"] = []

            for nest in range(4):
                L.seed_everything(self.config['seed'])
                if resume_fold_idx < idx and resume_nest_idx < nest:
                    self.set_model()
                    self.set_callbacks(idx, nest)

                    self.trainers[f"fold_{idx}"].append(L.Trainer(
                        max_epochs=self.config['n_epochs'],
                        accelerator="auto",
                        devices=self.config['gpu_device'] if torch.cuda.is_available() else None,
                        logger=[L.loggers.CSVLogger(save_dir=os.path.join(self.log_dir, f"csvlog_fold_{idx}_nest_{nest}")), L.loggers.TensorBoardLogger(save_dir=os.path.join(self.log_dir, f"tb_fold_{idx}_nest_{nest}"))],
                        callbacks=self.callbacks,
                        #check_val_every_n_epoch = 1,
                        #auto_lr_find=True
                        ))


                self.trainers[f"fold_{idx}"][nest].fit(self.model, datamodule=self.data_module_nested_cross_val[idx][nest])

                self.best_checkpoints[f"fold_{idx}"].append({callback.monitor: callback.best_model_path for callback in self.trainers[f"fold_{idx}"][nest].callbacks if 'ModelCheckpoint' in callback.__class__.__name__})

                for monitor, best_model in self.best_checkpoints[f"fold_{idx}"][nest].items():
                    self.val_nested_metrics[monitor][f"fold_{idx}"][f"nest_{nest}"].append(self.trainers[f"fold_{idx}"][nest].validate(self.model, datamodule=self.data_module_nested_cross_val[idx][nest], ckpt_path=best_model)[0])
                    self.test_nested_metrics[monitor][f"fold_{idx}"][f"nest_{nest}"].append(self.trainers[f"fold_{idx}"][nest].test(self.model, datamodule=self.data_module_nested_cross_val[idx][nest], ckpt_path=best_model)[0])


    def get_metrics_dataframe(self):

        self.val_metrics_df = None
        for idx, (key, values) in enumerate(self.val_metrics.items()):
            if idx == 0:
                self.val_metrics_df = pd.DataFrame(values)
                self.val_metrics_df['monitor'] = key
            else:
                tmp_df = pd.DataFrame(values)
                tmp_df['monitor'] = key
                self.val_metrics_df = pd.concat([self.val_metrics_df, tmp_df])
        self.val_metrics_df = self.val_metrics_df.set_index(['monitor', self.val_metrics_df.index])

        self.test_metrics_df = None
        for idx, (key, values) in enumerate(self.test_metrics.items()):
            if idx == 0:
                self.test_metrics_df = pd.DataFrame(values)
                self.test_metrics_df['monitor'] = key
            else:
                tmp_df = pd.DataFrame(values)
                tmp_df['monitor'] = key
                self.test_metrics_df = pd.concat([self.test_metrics_df, tmp_df])
        self.test_metrics_df = self.test_metrics_df.set_index(['monitor', self.test_metrics_df.index])

        self.val_metrics_df.to_pickle(os.path.join(self.metric_dir, 'val_best_metrics.pkl'))
        self.test_metrics_df.to_pickle(os.path.join(self.metric_dir, 'test_best_metrics.pkl'))

        self.val_metrics_df.to_csv(os.path.join(self.metric_dir, 'val_best_metrics.csv'))
        self.test_metrics_df.to_csv(os.path.join(self.metric_dir, 'test_best_metrics.csv'))



    def get_nested_metrics_dataframe(self):

        self.val_nested_metrics_df = None
        for idx, (key, values) in enumerate(self.val_nested_metrics.items()):
            for jdx, (fold_key, fold_values) in enumerate(values.items()):
               for kdx, (nest_key, nest_values) in enumerate(fold_values.items()):
                   if idx == 0 and jdx == 0 and kdx == 0:
                       self.val_nested_metrics_df = pd.DataFrame(nest_values)
                       self.val_nested_metrics_df['monitor'] = key
                       self.val_nested_metrics_df['fold'] = fold_key
                       self.val_nested_metrics_df['nest'] = nest_key
                   else:
                       tmp_df = pd.DataFrame(nest_values) 
                       tmp_df['monitor'] = key
                       tmp_df['fold'] = fold_key
                       tmp_df['nest'] = nest_key
                       self.val_nested_metrics_df = pd.concat([self.val_nested_metrics_df, tmp_df])
        self.val_nested_metrics_df = self.val_nested_metrics_df.set_index(['monitor', 'fold', 'nest'])

        self.test_nested_metrics_df = None
        for idx, (key, values) in enumerate(self.test_nested_metrics.items()):
            for jdx, (fold_key, fold_values) in enumerate(values.items()):
               for kdx, (nest_key, nest_values) in enumerate(fold_values.items()):
                   if idx == 0 and jdx == 0 and kdx == 0:
                       self.test_nested_metrics_df = pd.DataFrame(nest_values)
                       self.test_nested_metrics_df['monitor'] = key
                       self.test_nested_metrics_df['fold'] = fold_key
                       self.test_nested_metrics_df['nest'] = nest_key
                   else:
                       tmp_df = pd.DataFrame(nest_values) 
                       tmp_df['monitor'] = key
                       tmp_df['fold'] = fold_key
                       tmp_df['nest'] = nest_key
                       self.test_nested_metrics_df = pd.concat([self.test_nested_metrics_df, tmp_df])
        self.test_nested_metrics_df = self.test_nested_metrics_df.set_index(['monitor', 'fold', 'nest'])


        self.val_nested_metrics_df.to_pickle(os.path.join(self.metric_dir, 'val_best_nested_metrics.pkl'))
        self.test_nested_metrics_df.to_pickle(os.path.join(self.metric_dir, 'test_best_nested_metrics.pkl'))

        self.val_nested_metrics_df.to_csv(os.path.join(self.metric_dir, 'val_best_nested_metrics.csv'))
        self.test_nested_metrics_df.to_csv(os.path.join(self.metric_dir, 'test_best_nested_metrics.csv'))

    #def get_predictions(self):
    #    """
    #    get predictions from list of trainers stored in self.trainers
    #    requires run() to be executed as a prerequisite
    #    this will get a set of predictions for each fold
    #    """

    #    test_predictions_dict = []
    #    val_predictions_dict = []
    #    for idx, trainer in enumerate(self.trainers):
    #        test_predictions_dict.append({})
    #        val_predictions_dict.append({})
    #        tmp_test_targets = []
    #        tmp_val_targets = []
    #        best_checkpoint = trainer.checkpoint_callback.best_model_path

    #        test_predictions_dict[idx]['predictions'] = trainer.predict(trainer.model, self.data_module_cross_val[idx].test_dataloader(), ckpt_path=best_checkpoint)
    #        val_predictions_dict[idx]['predictions'] = trainer.predict(trainer.model, self.data_module_cross_val[idx].val_dataloader(), ckpt_path=best_checkpoint)
    #        for batch in self.data_module_cross_val[idx].test_dataloader():
    #            tmp_test_targets.append(batch.y)
    #        for batch in self.data_module_cross_val[idx].val_dataloader():
    #            tmp_val_targets.append(batch.y)

    #        test_predictions_dict[idx]['targets'] = torch.cat(tmp_test_targets)
    #        val_predictions_dict[idx]['targets'] = torch.cat(tmp_val_targets)

    #    self.test_predictions_df = pd.DataFrame(test_predictions_dict)
    #    self.val_predictions_df = pd.DataFrame(val_predictions_dict)

    #    self.test_predictions_df.to_pickle(os.path.join(self.metric_dir, 'test_predictions.pkl'))
    #    self.val_predictions_df.to_pickle(os.path.join(self.metric_dir, 'val_predictions.pkl'))
    def get_predictions(self):
        """
        get predictions from list of trainers stored in self.trainers
        requires run() to be executed as a prerequisite
        this will get a set of predictions for each fold
        """

        self.test_predictions_dict = {callback.monitor: [] for callback in self.trainers[0].callbacks if 'ModelCheckpoint' in callback.__class__.__name__}
        self.val_predictions_dict = {callback.monitor: [] for callback in self.trainers[0].callbacks if 'ModelCheckpoint' in callback.__class__.__name__}
        #self.test_k_predictions_dict = {callback.monitor: {f"model_{idx}": [] for idx in range(self.config['save_top_k'])} for callback in self.trainers[0].callbacks if 'ModelCheckpoint' in callback.__class__.__name__}
        #self.val_k_predictions_dict = {callback.monitor: {f"model_{idx}": [] for idx in range(self.config['save_top_k'])} for callback in self.trainers[0].callbacks if 'ModelCheckpoint' in callback.__class__.__name__}
        self.test_targets = []
        self.val_targets = []

        for idx, trainer in enumerate(self.trainers):
            tmp_test_targets = []
            tmp_val_targets = []

            for monitor, best_model in self.best_checkpoints[f"fold_{idx}"].items():
                self.test_predictions_dict[monitor].append(torch.cat(trainer.predict(trainer.model, self.data_module_cross_val[idx].test_dataloader(), ckpt_path=best_model)))
                self.val_predictions_dict[monitor].append(torch.cat(trainer.predict(trainer.model, self.data_module_cross_val[idx].val_dataloader(), ckpt_path=best_model)))

            if self.config['include_primary']:
                for batch in self.data_module_cross_val[idx].test_dataloader():
                    tmp_test_targets.append(batch.y[np.squeeze(batch.patch_type)!='primary'])
                for batch in self.data_module_cross_val[idx].val_dataloader():
                    tmp_val_targets.append(batch.y[np.squeeze(batch.patch_type)!='primary'])
            else:
                for batch in self.data_module_cross_val[idx].test_dataloader():
                    tmp_test_targets.append(batch.y)
                for batch in self.data_module_cross_val[idx].val_dataloader():
                    tmp_val_targets.append(batch.y)

            self.test_targets.append(torch.cat(tmp_test_targets))
            self.val_targets.append(torch.cat(tmp_val_targets))

        self.test_predictions_df = pd.DataFrame(self.test_predictions_dict)
        self.val_predictions_df = pd.DataFrame(self.val_predictions_dict)

        self.test_predictions_df['targets'] = self.test_targets
        self.val_predictions_df['targets'] = self.val_targets

        self.test_predictions_df.to_pickle(os.path.join(self.metric_dir, 'test_predictions.pkl'))
        self.val_predictions_df.to_pickle(os.path.join(self.metric_dir, 'val_predictions.pkl'))


    #def get_combined_metrics(self):
    #    '''
    #    run after get_predictions() to get metrics that combined all folds
    #    '''
    #    test_predictions = []
    #    test_targets = []
    #    val_predictions = []
    #    val_targets = []

    #    for idx in range(5):
    #        #test_predictions.extend(self.test_predictions_df.loc[idx, 'predictions'][0:len(self.test_predictions_df.loc[idx, 'targets'])])
    #        test_predictions.extend(self.test_predictions_df.loc[idx, 'predictions'])
    #        test_targets.extend(self.test_predictions_df.loc[idx, 'targets'])
    #        #val_predictions.extend(self.val_predictions_df.loc[idx, 'predictions'][0:len(self.val_predictions_df.loc[idx, 'targets'])])
    #        val_predictions.extend(self.val_predictions_df.loc[idx, 'predictions'])
    #        val_targets.extend(self.val_predictions_df.loc[idx, 'targets'])

    #    test_predictions = torch.tensor(test_predictions, dtype=torch.float)
    #    test_targets = torch.tensor(test_targets, dtype = torch.float)
    #    val_predictions = torch.tensor(val_predictions, dtype = torch.float)
    #    val_targets = torch.tensor(val_targets, dtype=torch.float)

    #    self.test_metrics_df.loc[len(self.test_metrics_df)] = [
    #                                                           self.model.test_auc_fn(test_predictions, test_targets).numpy(),
    #                                                           self.model.test_ap_fn(test_predictions, test_targets.to(torch.int64)).numpy(),
    #                                                           self.model.m_fn(self.model.test_sen_fn(test_predictions, test_targets), self.model.test_spe_fn(test_predictions, test_targets)).numpy(),
    #                                                           self.model.test_sen_fn(test_predictions, test_targets).numpy(),
    #                                                           self.model.test_spe_fn(test_predictions, test_targets).numpy(),
    #                                                          ]
    #    self.val_metrics_df.loc[len(self.val_metrics_df)] = [
    #                                                           self.model.loss_fn(val_predictions, val_targets).numpy(),
    #                                                           self.model.val_auc_fn(val_predictions, val_targets).numpy(),
    #                                                           self.model.val_ap_fn(val_predictions, val_targets.to(torch.int64)).numpy(),
    #                                                           self.model.m_fn(self.model.val_sen_fn(val_predictions, val_targets), self.model.val_spe_fn(val_predictions, val_targets)).numpy(),
    #                                                           self.model.val_sen_fn(val_predictions, val_targets).numpy(),
    #                                                           self.model.val_spe_fn(val_predictions, val_targets).numpy(),
    #                                                          ]
    #    self.test_metrics_df.to_csv(os.path.join(self.metric_dir, 'test_metrics.csv'))
    #    self.val_metrics_df.to_csv(os.path.join(self.metric_dir, 'val_metrics.csv'))
    def get_combined_metrics(self):
        '''
        run after get_predictions() to get metrics that combined all folds
        '''
        test_predictions = {}
        test_targets = []
        val_predictions = {}
        val_targets = []

        for monitor in self.test_predictions_df.columns:
            if 'targets' in monitor:
                continue
            test_predictions[monitor] = []
            val_predictions[monitor] = []
            for idx in range(5):
                test_predictions[monitor].extend(self.test_predictions_df.loc[idx, monitor])
                val_predictions[monitor].extend(self.val_predictions_df.loc[idx, monitor])

        for idx in range(5):
            test_targets.extend(self.test_predictions_df.loc[idx, 'targets'])
            val_targets.extend(self.val_predictions_df.loc[idx, 'targets'])

        for key in test_predictions.keys():
            test_targets = torch.stack(test_targets).to(torch.float)
            val_targets = torch.stack(val_targets).to(torch.float)
            break
   
        for key in test_predictions.keys():
            test_predictions[key] = torch.stack(test_predictions[key])
            val_predictions[key] = torch.stack(val_predictions[key])
            self.test_metrics_df.loc[(f"{key}_combined", 0), :] = [
                    self.model.test_auc_fn(test_predictions[key], test_targets).numpy(),
                    self.model.test_ap_fn(test_predictions[key], test_targets.to(torch.int64)).numpy(),
                    self.model.m_fn(self.model.test_sen_fn(test_predictions[key], test_targets), self.model.test_spe_fn(test_predictions[key], test_targets)).numpy(),
                    self.model.test_sen_fn(test_predictions[key], test_targets).numpy(),
                    self.model.test_spe_fn(test_predictions[key], test_targets).numpy(),
                    ]
            self.val_metrics_df.loc[(f"{key}_combined", 0), :] = [
                    self.model.loss_fn(val_predictions[key], val_targets).numpy(),
                    self.model.val_auc_fn(val_predictions[key], val_targets).numpy(),
                    self.model.val_ap_fn(val_predictions[key], val_targets.to(torch.int64)).numpy(),
                    self.model.m_fn(self.model.val_sen_fn(val_predictions[key], val_targets), self.model.val_spe_fn(val_predictions[key], val_targets)).numpy(),
                    self.model.val_sen_fn(val_predictions[key], val_targets).numpy(),
                    self.model.val_spe_fn(val_predictions[key], val_targets).numpy(),
                    ]
        #self.test_metrics_df.to_csv(os.path.join(self.metric_dir, 'test_metrics.csv'))
        #self.val_metrics_df.to_csv(os.path.join(self.metric_dir, 'val_metrics.csv'))

