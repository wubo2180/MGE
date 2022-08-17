'''
Author: your name
Date: 2022-03-11 17:47:41
LastEditTime: 2022-05-29 23:43:06
LastEditors: wubo2180 15827403235@163.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /wb/MGE/code/model.py
'''

from logging import raiseExceptions
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GINEConv,GATConv,SAGEConv
# from embinitialization import EmbInitial, EmbInitial_DBLP,EmbInitial_CHEM
from info_nce import InfoNCE
# from layers import GraphESAGE,GCNEConv,GATEConv
from torch_geometric.nn import global_mean_pool,global_add_pool,global_max_pool,GlobalAttention
from torch_geometric.nn.inits import uniform, glorot, zeros
class MGE (torch.nn.Module):
    def __init__(self, args,gnn):
        super(MGE, self).__init__()
        self.args = args
        self.gnn=gnn
        self.loss=InfoNCE()
        graph_pooling=args.graph_pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(args.emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")
    
    def from_pretrained(self, model_file):
        print(f'loading pre-trained model from {model_file}')
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage)) 
    
    def forward(self, batch, optimizer):
        x = self.pool(self.gnn(batch.x, batch.edge_index, batch.edge_attr),batch.batch).squeeze()
        
        x_node=self.pool(self.gnn(batch.x, batch.edge_index_node, batch.edge_attr_node),batch.batch).squeeze()

        graph_aug=self.pool(self.gnn(batch.x, batch.edge_index_aug, batch.edge_attr_aug),batch.batch).squeeze()

        node_contrastive = self.loss(x, x_node)
        
        graph_contrastive = self.loss(x, graph_aug)
        
        optimizer.zero_grad()
        loss=node_contrastive+self.args.weight*graph_contrastive
        # loss=self.args.weight*graph_contrastive
        # loss=node_contrastive
        loss.backward()
        optimizer.step()
        return loss
        




