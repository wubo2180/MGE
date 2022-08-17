import numpy as np
import torch
# from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx,dropout_adj,subgraph,k_hop_subgraph,degree
import networkx as nx
import random

class graphTransform:
    def __init__(self,aug_ratio=0.2,aug='random3',mode='degree'):
        self.aug_ratio=aug_ratio
        self.mode=mode
        self.aug=aug

    def __call__(self, data):
        
        node_num,_=data.x.size()
        # print(max(degree(data.edge_index)))
        G=to_networkx(data)
        if self.mode=='degree':
            degree_nodes=nx.degree_centrality(G)
            degree_center_node=max(degree_nodes.keys(), key=(lambda k: degree_nodes[k]))
            k_sub_degree=k_hop_subgraph(degree_center_node,1,data.edge_index)
            data.edge_index_node,data.edge_attr_node=subgraph(k_sub_degree[0],data.edge_index,data.edge_attr,num_nodes=node_num)
        elif self.mode=='closeness_centrality':
            closeness_nodes=nx.closeness_centrality(G)
            closeness_center_node=max(closeness_nodes.keys(), key=(lambda k: closeness_nodes[k]))
            k_sub_closeness=k_hop_subgraph(closeness_center_node,1,data.edge_index)
            data.edge_index_node,data.edge_attr_node=subgraph(k_sub_closeness[0],data.edge_index,data.edge_attr,num_nodes=node_num)
        
        elif self.mode=='betweenness_centrality':
            betweenness_nodes=nx.betweenness_centrality(G)
            betweenness_center_node=max(betweenness_nodes.keys(), key=(lambda k: betweenness_nodes[k]))
            k_sub_betweenness=k_hop_subgraph(betweenness_center_node,1,data.edge_index)
            data.edge_index_node,data.edge_attr_node=subgraph(k_sub_betweenness[0],data.edge_index,data.edge_attr,num_nodes=node_num)
        else:
            raise ValueError("Invalid center.")
        # if self.aug=='random':
        #     self.aug=random.randint(0,3)
        if self.aug=='edge_perturb' :
            _, data.edge_index_aug, data.edge_attr_aug=self.perturb_edges(data)
        elif self.aug=='node_drop' :
            _, data.edge_index_aug, data.edge_attr_aug=self.node_drop(data)
        elif self.aug=='subgraph' :
            _, data.edge_index_aug, data.edge_attr_aug=self.subgraph(data)
        elif self.aug=='mask_nodes':
            _, data.edge_index_aug, data.edge_attr_aug=self.mask_nodes(data)
        else:
            raise ValueError("Invalid augmentation.")                            
        return data
    def perturb_edges(self, data):
        _, edge_num = data.edge_index.size()
        permute_num = int(edge_num * self.aug_ratio)
        idx_delete = torch.randperm(edge_num)[permute_num:]
        new_edge_index = data.edge_index[:, idx_delete]
        new_edge_attr = data.edge_attr[idx_delete]
        return data.x, new_edge_index, new_edge_attr
    def node_drop(self, data):
        node_num, _ = data.x.size()
        sub_num = int(node_num  * self.aug_ratio)
        sub_drop = np.random.choice(node_num, sub_num, replace=False)
        sub_nondrop = [n for n in range(node_num) if not n in sub_drop]
        subgraph(sub_nondrop,data.edge_index,data.edge_attr,num_nodes=node_num)
        edge_index,edge_attr=subgraph(sub_nondrop,data.edge_index,data.edge_attr,num_nodes=node_num)
        return data.x, edge_index,edge_attr
    # def node_drop(self, data):
    #     node_num, _ = data.x.size()
    #     _, edge_num = data.edge_index.size()
    #     drop_num = int(node_num  * self.aug_ratio)

    #     idx_perm = torch.randperm(node_num).numpy()

    #     idx_drop = idx_perm[:drop_num]
    #     idx_nondrop = idx_perm[drop_num:]
    #     idx_nondrop.sort()
    #     idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

    #     edge_index = data.edge_index.numpy()
    #     edge_mask = np.array([n for n in range(edge_num) if not (edge_index[0, n] in idx_drop or edge_index[1, n] in idx_drop)])

    #     edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    #     try:
    #         new_edge_index = torch.tensor(edge_index).transpose_(0, 1)
    #         new_x = data.x[idx_nondrop]
    #         new_edge_attr = data.edge_attr[edge_mask]
    #         return new_x, new_edge_index, new_edge_attr
    #     except:
    #         return data.x, data.edge_index, data.edge_attr

    def subgraph(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        sub_num = int(node_num * self.aug_ratio)

        edge_index = data.edge_index.numpy()

        idx_sub = [torch.randint(node_num, (1,)).item()]
        idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = np.random.choice(list(idx_neigh))
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

        idx_drop = [n for n in range(node_num) if not n in idx_sub]
        idx_nondrop = idx_sub
        idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
        edge_mask = np.array([n for n in range(edge_num) if (edge_index[0, n] in idx_nondrop and edge_index[1, n] in idx_nondrop)])

        edge_index = data.edge_index.numpy()
        edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
        try:
            new_edge_index = torch.tensor(edge_index).transpose_(0, 1)
            new_x = data.x[idx_nondrop]
            new_edge_attr = data.edge_attr[edge_mask]
            return new_x, new_edge_index, new_edge_attr
        except:
            return data.x, data.edge_index, data.edge_attr

    def mask_nodes(self, data):
        node_num, feat_dim = data.x.size()
        mask_num = int(node_num * self.aug_ratio)

        token = data.x.mean(dim=0)
        idx_mask = torch.randperm(node_num)[:mask_num]
        data.x[idx_mask] = token.clone().detach()

        return data.x, data.edge_index, data.edge_attr