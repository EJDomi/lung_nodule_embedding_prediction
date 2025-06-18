#!/usr/bin/env python
from pathlib import Path
import random

from math import floor
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from scipy.ndimage import center_of_mass, rotate
import scipy.io

import SimpleITK as sitk
from sklearn.preprocessing import MinMaxScaler
#from skimage.transform import rotate
from skimage.util import random_noise

import torch
from torch_geometric.data import Dataset, Data
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx


class DatasetGeneratorImage(Dataset):
    """
    generate images for pytorch dataset
    """
    def __init__(self, config, resume=None, pre_transform=None):

        self.config = config 
        self.data_path = Path(self.config['data_path'])
        self.patch_path = self.data_path.joinpath(self.config['patch_dir'])
        self.embedding_path = self.data_path.joinpath(self.config['embedding_dir'])
        self.edge_dict = pd.read_pickle(self.data_path.joinpath(f"edge_staging/{self.config['edge_file']}"))
        self.resume = resume
        self.patients = pd.read_pickle(self.data_path.joinpath(f"edge_staging/{self.config['patient_feature_file']}"))
        # change labels to one class [1,0] -> 0
        if self.config['n_classes'] == 2:
            label_mapping = {0: [1,0], 1: [0,1], -1: [0,0]}
            self.patients['y'] = self.patients['y'].map(label_mapping) 
       
        self.graph_dir = self.data_path.joinpath(f"graph_staging/{self.patch_path.name}_{self.config['edge_file'].replace('.pkl', '')}_{self.config['data_version']}")

        self.rng_rotate = np.random.default_rng(42)
        self.rng_rotate_axis = np.random.default_rng(42)
        if self.config['n_rotations'] > 0:
            self.aug_patients = self.patients.copy(deep=True)
            aug_pats = self.patients.copy(deep=True)
            for rot in range(self.config['n_rotations']):
                aug_rot_pats = aug_pats.copy(deep=True)
                aug_rot_pats.index = aug_pats.index.set_levels(aug_pats.index.levels[aug_pats.index.names.index('subject_id')] + f"_rotation_{rot+1}", level='subject_id')
                if rot == 0:
                    self.aug_patients = aug_rot_pats
                else:
                    self.aug_patients = pd.concat([self.aug_patients, aug_rot_pats])
        else:
            self.aug_patients = None

        if self.config['balance_classes']:
            if self.config['n_classes'] > 1:
                n_neg_avg = (aug_pats.groupby('subject_id')['y'].mean().str[1]<0.5).sum()
                n_pos_avg = (aug_pats.groupby('subject_id')['y'].mean().str[1]>=0.5).sum()
            elif self.config['n_classes'] == 1:
                n_neg_avg = (aug_pats.groupby('subject_id').mean('node')['y']<0.5).sum()
                n_pos_avg = (aug_pats.groupby('subject_id').mean('node')['y']>=0.5).sum()
            if self.config['true_balance_classes']:
                ratio_classes = int(floor(n_neg_avg / n_pos_avg)) - 1
            else:
                ratio_classes = int(floor(n_neg_avg / n_pos_avg))
            print(n_neg_avg, n_pos_avg)
            print(ratio_classes)
            aug_pats = self.patients.copy(deep=True)
            for rot in range(ratio_classes):
                if self.config['n_classes'] == 1:
                    aug_pos_pats = aug_pats.copy(deep=True)[aug_pats['labels']==1]
                #elif self.config['n_classes'] == 2:
                #    aug_pos_pats = aug_pats.copy(deep=True)[aug_pats['labels'].str[1]==1]
                elif self.config['n_classes'] == 2:
                    aug_pos_pats = aug_pats.copy(deep=True).loc[(aug_pats.groupby('subject_id')['y'].mean().str[1]>=1)[aug_pats.groupby('subject_id')['y'].mean().str[1]>=1].index]
                aug_pos_pats.index = aug_pos_pats.index.set_levels(aug_pos_pats.index.levels[aug_pos_pats.index.names.index('subject_id')] + f"_pos_rotation_{rot+1}", level='subject_id')
                if rot == 0:
                    self.aug_pos_patients = aug_pos_pats
                else:
                    self.aug_pos_patients = pd.concat([self.aug_pos_patients, aug_pos_pats])
                
            self.aug_patients = self.aug_patients.drop([idx for idx in self.aug_patients.index if 'rotation' not in idx[0]])
        else:
            self.aug_pos_patients = None

        self.full_patients = self.patients.copy(deep=True)
        if self.config['n_rotations'] > 0:
            self.full_patients = pd.concat([self.full_patients, self.aug_patients])
        if self.config['balance_classes']:
            self.full_patients = pd.concat([self.full_patients, self.aug_pos_patients])

        if self.config['use_radiomics']:
            self.radiomics_data = None
            print("extracting radiomics from mat files")
            for idx in tqdm(list(range(5))):
                rad_file = self.data_path.joinpath(self.config['radiomics_dir']).joinpath(f"Radio F{idx}_ct_fea.mat")
   
                rad_tmp = scipy.io.loadmat(rad_file.as_posix())['fin_fea_all']

                rad_tmp[['patient_id', 'struct']] = rad_tmp['Unnamed: 0'].str.split('__', expand=True)
                rad_tmp.drop(columns=['Unnamed: 0', 'Image', 'Mask'], inplace=True)
                rad_tmp.set_index(['patient_id', 'struct'], inplace=True)
                rad_tmp.drop(columns=[col for col in rad_tmp.columns if 'diagnostics' in col], inplace=True)

                if idx == 0:
                    self.radiomics_data = rad_tmp
                else:
                    self.radiomics_data = pd.concat([self.radiomics_data, rad_tmp])
            if self.config['scale_radiomics']:
                rad_mean = pd.read_pickle(self.config['radiomics_mean'])
                rad_std = pd.read_pickle(self.config['radiomics_std'])

                self.radiomics_data = (self.radiomics_data - rad_mean) / rad_std
            print("done extracting radiomics")
        else:
            self.radiomics_data = None
        
                            

        super(DatasetGeneratorImage, self).__init__(pre_transform=pre_transform)

    @property
    def raw_paths(self):
        return [f"{self.raw_dir}" for pat in self.patients.index.levels[0]]

    @property
    def raw_dir(self):
        return str(self.patch_path)

    @property
    def processed_dir(self):
        return str(self.graph_dir)

    @property
    def processed_file_names(self):
        file_names = [f"graph_{idx}_{pat}.pt" for idx, pat in enumerate(self.patients.index.get_level_values(0).unique())]
        if self.config['n_rotations'] > 0:
            n_files = len(file_names)
            file_names.extend([f"graph_{idx+n_files}_{pat}.pt" for idx, pat in enumerate(self.aug_patients.index.get_level_values(0).unique())])
        if self.config['balance_classes']:
            n_files = len(file_names)
            file_names.extend([f"graph_{idx+n_files}_{pat}.pt" for idx, pat in enumerate(self.aug_pos_patients.index.get_level_values(0).unique())])
        #elif self.config['balance_classes']:
        #    file_names = [f"graph_{idx}_{pat}.pt" for idx, pat in enumerate(list(self.patients.index.levels[0])+list(self.aug_pos_patients.index.levels[0]))]
        #elif self.config['n_rotations'] > 0:
        #    file_names = [f"graph_{idx}_{pat}.pt" for idx, pat in enumerate(list(self.patients.index.levels[0])+list(self.aug_patients.index.levels[0]))]

        return file_names


    def download(self):
        pass


    def process(self):
        print("processed graph files not present, starting graph production")
        self.idx = 0

        self.process_graph(self.patients, to_rotate=False)

        if self.aug_patients is not None: 
            self.process_graph(self.aug_patients, to_rotate=True) 
 
        if self.aug_pos_patients is not None:
            self.process_graph(self.aug_pos_patients, to_rotate=True) 


    def process_graph(self, patient_group, to_rotate=False):
        for full_pat, group_df in tqdm(patient_group.groupby('subject_id', sort=False)):
            if self.resume is not None:
                if self.idx < self.resume:
                    self.idx += 1
                    continue
            pat = full_pat.split('_')[0]
            graph_nx = self.edge_dict[pat]

            if to_rotate:
                self.angle = self.rng_rotate.integers(-30, high=31)
                self.rotate_axes = self.rng_rotate_axis.integers(0, high=3)

            for node in group_df.index:
                node = node[1] #index is a multiindex, with node # in second position

                node_name = f"{pat}-{node}"

                patch = scipy.io.loadmat(self.patch_path.joinpath(f"{node_name}.mat"))['tensor']

                
                node_com = group_df.loc[(full_pat, node), 'position']
                graph_nx.nodes[node]['pos'] = torch.tensor(node_com, dtype=torch.float)

                if to_rotate:
                    patch = self.apply_rotation(patch, self.angle, self.rotate_axes)

                if self.config['store_embeddings']:
                    node_embeddings = pd.read_pickle(self.embedding_path.joinpath(pat).joinpath(f"embedding_{pat}_{int(node)}.pkl"))
                graph_nx.nodes[node]['x'] = torch.tensor(np.expand_dims(patch, 0), dtype=torch.float)
                #graph_nx.nodes[node]['x'] = torch.from_numpy(np.stack((patch, seg)))

                graph_nx.nodes[node]['y'] = torch.tensor(group_df.loc[(full_pat, node), 'y'], dtype=torch.long)
                graph_nx.nodes[node]['patch_type'] = 'nodule'
                graph_nx.nodes[node]['features'] = torch.flatten(torch.from_numpy(group_df.loc[(full_pat, node), ['pos_x', 'pos_y', 'pos_z', 'spa_x', 'spa_y', 'spa_z', 'laterality']].values))
                graph_nx.nodes[node]['embeddings'] = torch.tensor(node_embeddings)
                graph_nx.graph['patient'] = full_pat
                
                del patch
              
            graph_pyg = from_networkx(graph_nx)
            norm_transform = T.Cartesian()
            graph_pyg = norm_transform(graph_pyg)
 
            torch.save(graph_pyg, f"{self.processed_dir}/graph_{self.idx}_{full_pat}.pt")
            del graph_nx
            del graph_pyg
            self.idx += 1




    def len(self):
        n_patients = len(self.patients.index.get_level_values(0).unique())
        if self.config['n_rotations'] > 0:
            n_patients += len(self.aug_patients.index.get_level_values(0).unique())
        if self.config['balance_classes']:
            n_patients += len(self.aug_pos_patients.index.get_level_values(0).unique())
        return n_patients


    def get(self, idx):
        pat = self.full_patients.index.get_level_values(0).unique()[idx]
        data = torch.load(f"{self.processed_dir}/graph_{idx}_{pat}.pt")
        return data


    def apply_rotation(self, arr, angle, rotate_axes):
        axis_tuples = [(0,1), (0,2), (1,2)]
        arr = rotate(arr, angle, axes=axis_tuples[rotate_axes], reshape=False)
        return arr



