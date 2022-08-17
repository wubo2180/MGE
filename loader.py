import networkx as nx
import numpy as np
import json
import os
import os.path as osp
import torch
import pickle
import pandas as pd
from torch_geometric.data import InMemoryDataset,Data,Dataset,download_url
from itertools import repeat, product, chain
from copy import deepcopy
import copy
from torch_geometric.utils import subgraph, dropout_adj,degree,k_hop_subgraph,to_networkx,from_networkx
import networkx as nx
import traceback
from tqdm import tqdm


class DblpDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 data_type,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.root = root
        self.data_type = data_type
        if self.data_type == 'supervised':
            self.length=299447
        elif self.data_type== 'unsupervised':
            self.length=754862
        else:
            raise NotImplementedError('Data type error!')
        super().__init__(root, transform, pre_transform, pre_filter)

        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [str(i//10000)+'/graph_'+str(i)+'.json' for i in range(self.length)]

    @property
    # def processed_file_names(self):
    #     if self.data_type == 'supervised':
    #         return ['dblpfinetune.pt']
    #     elif self.data_type== 'unsupervised':
    #         return ['dblp.pt']
    def processed_file_names(self):
        if self.data_type == 'supervised':
            return ['dblpfinetune.graph']
        elif self.data_type== 'unsupervised':
            return ['dblp.graph']
        


    def process(self):
        data_list=[]
        for i,raw_path in enumerate(self.raw_paths):
            with open(raw_path,'r') as f:
                temp = json.load(f)
            data = Data(x=torch.tensor(temp['x'],dtype=torch.float32),
                                edge_index=torch.tensor(temp['edge_index'],dtype=torch.int64),
                                edge_attr=torch.tensor(temp['edge_attr'],dtype=torch.float32),
                                center_node_idx=torch.tensor(temp['center_node_idx'],dtype=torch.int64),
                                species_id=torch.tensor(temp['species_id'],dtype=torch.int64),
                                go_target_downstream=torch.tensor(temp['go_target_downstream'],dtype=torch.int64)
                                )
                # else:
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class BioDataset(InMemoryDataset):
    def __init__(self, root,data_type, transform=None, pre_transform=None, pre_filter=None):

        self.root = root
        self.data_type = data_type
        if self.data_type == 'supervised':
            self.length=88000
        elif self.data_type== 'unsupervised':
            self.length=306925
        else:
            raise NotImplementedError('Data type error!')
        print(self.length)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        # if not empty:
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        # raise NotImplementedError('Data is assumed to be processed')
        return [str(i//10000)+'/graph_'+str(i)+'.json' for i in range(self.length)]

    @property
    def processed_file_names(self):
        if self.data_type == 'supervised':
            return ['geometric_data_processed.pt']
        elif self.data_type== 'unsupervised':
            return ['geometric_data_processed.pt']

    def process(self):
        
        keys=['x', 'edge_index', 'edge_attr', 'center_node_idx', 'species_id']
        data_list = []

        for i,raw_path in enumerate(self.raw_paths):
            # Read data from `raw_path`.
            with open(raw_path,'r') as f:
                temp = json.load(f)
            # for key in keys:
            if self.data_type== 'unsupervised':
                data = Data(x=torch.tensor(temp['x'],dtype=torch.float32),
                        edge_index=torch.tensor(temp['edge_index'],dtype=torch.int64),
                        edge_attr=torch.tensor(temp['edge_attr'],dtype=torch.float32),
                        center_node_idx=torch.tensor(temp['center_node_idx'],dtype=torch.int64),
                        species_id=torch.tensor(temp['species_id'],dtype=torch.int64)
                        )
            else:
                data = Data(x=torch.tensor(temp['x'],dtype=torch.float32),
                        edge_index=torch.tensor(temp['edge_index'],dtype=torch.int64),
                        edge_attr=torch.tensor(temp['edge_attr'],dtype=torch.float32),
                        center_node_idx=torch.tensor(temp['center_node_idx'],dtype=torch.int64),
                        species_id=torch.tensor(temp['species_id'],dtype=torch.int64),
                        go_target_downstream=torch.tensor(temp['go_target_downstream'],dtype=torch.int64),
                        go_target_pretrain=torch.tensor(temp['go_target_pretrain'],dtype=torch.int64)
                        )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
            # dir=str(i//10000)+'/'
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
class MoleculeDataset(InMemoryDataset):
    def __init__(self, root,data_type, transform=None, pre_transform=None, pre_filter=None):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: the data directory that contains a raw and processed dir
        :param data_type: either supervised or unsupervised
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        :param transform:
        :param pre_transform:
        :param pre_filter:
        """
        
        self.root = root
        self.data_type = data_type
        if self.data_type == 'chembl_filtered':
            self.length=430709
        elif self.data_type== 'bbbp':
            self.length=2039
        elif self.data_type== 'bace':
            self.length=1513
        elif self.data_type== 'toxcast':
            self.length=8576
        elif self.data_type== 'tox21':
            self.length=7831
        elif self.data_type== 'muv':
            self.length=93087
        elif self.data_type== 'hiv':
            self.length=41127
        elif self.data_type== 'sider':
            self.length=1427
        elif self.data_type== 'clintox':
            self.length=1477
        else:
            raise NotImplementedError('Data type error!')
        self.downstream_dir = [
            'dataset/bace',
            'dataset/bbbp',
            'dataset/clintox',
            'dataset/esol',
            'dataset/freesolv',
            'dataset/hiv',
            'dataset/muv',
            # 'dataset/pcba/processed/smiles.csv',
            'dataset/sider',
            'dataset/tox21',
            'dataset/toxcast'
            ]
        print(self.length)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        # if not empty:
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        # print('####')
        # raise NotImplementedError('Data is assumed to be processed')
        if self.data_type == 'chem_dataset': 
            return [str(i//10000)+'/graph_'+str(i)+'.json' for i in range(self.length)]
        else:
            return ['graph_'+str(i)+'.json' for i in range(self.length)]

    @property
    def processed_file_names(self):
        return ['geometric_data_processed.pt']
        # return ['data.pt']

    def process(self):

        keys=['x', 'edge_index', 'edge_attr', 'center_node_idx', 'species_id']
        data_list = []
        print('####')
        # print(type(self.raw_paths),len())
        for i,raw_path in enumerate(tqdm(self.raw_paths)):
            # raw_path=self.raw_paths[i]
            # Read data from `raw_path`.
            # if i>100:
            #     break
            with open(raw_path,'r') as f:
                temp = json.load(f)
            # for key in keys:
            # if self.data_type== 'unsupervised':
            if self.data_type=='chembl_filtered':
                data = Data(x=torch.tensor(temp['x'],dtype=torch.int64),
                        edge_index=torch.tensor(temp['edge_index'],dtype=torch.int64),
                        edge_attr=torch.tensor(temp['edge_attr'],dtype=torch.int64),
                        fold=torch.tensor(temp['fold'],dtype=torch.int64),
                        id=torch.tensor(temp['id'],dtype=torch.int64),
                        y=torch.tensor(temp['y'],dtype=torch.int64)
                        )
            else:
                data = Data(x=torch.tensor(temp['x'],dtype=torch.int64),
                        edge_index=torch.tensor(temp['edge_index'],dtype=torch.int64),
                        edge_attr=torch.tensor(temp['edge_attr'],dtype=torch.int64),
                        id=torch.tensor(temp['id'],dtype=torch.int64),
                        y=torch.tensor(temp['y'],dtype=torch.int64)
                        )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            # Data(edge_attr=[50, 2], edge_index=[2, 50], fold=[1], id=[1], x=[25, 2], y=[1310])
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
            # dir=str(i//10000)+'/'
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])