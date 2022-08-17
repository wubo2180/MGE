import argparse
import random
from loader import BioDataset, DblpDataset,MoleculeDataset
# from loader import MoleculeDataset
from torch_geometric.data import DataLoader
# from util import TaskConstruction
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import sys
import time
from progressbar import *
from util import *
from model import MGE
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# from torch_geometric.nn import DataLoader
from torch_geometric.data import DataLoader
dir='../MGE2/'
def train(args, model, loader, optimizer):
    # train
    model.train()
    for epoch in range(1, args.epochs + 1):
        train_loss = []
        train_acc = []
        print("====epoch " + str(epoch))
        for step, batch in enumerate(tqdm(loader, desc="Iteration",ncols=80)):
            # pass
            if step>10:
                break
            batch=batch.to(args.device)
            loss= model(batch, optimizer)
            train_loss.append(loss)
        print('loss:', torch.stack(train_loss).mean().detach().cpu().item())
        # acc=0
        # print('acc:', torch.stack(train_acc).mean().detach().cpu().item())

        if epoch % 5 == 0 or epoch == 1:
            if not args.model_file == '':
                print('saving model...')
                torch.save(model.gnn.state_dict(),'./res/' + args.dataset + '/' + args.model_file + '_' + args.gnn_type + '_' + str(epoch) + "_gnn.pth")
    


def main(args):
    # set up dataset
    if args.dataset == 'bio':
        root_unsupervised = dir+'data/bio/unsupervised'
        root_unsupervised='./dataset/bio/unsupervised'
        dataset = BioDataset(root_unsupervised, data_type='unsupervised',transform=graphTransform(args.aug_ratio,args.aug,args.mode))
        from bio_model import GNN
        

    elif args.dataset == 'dblp':
        root_unsupervised = dir+'data/dblp/unsupervised'
        root_unsupervised = 'dataset/dblp/unsupervised'
        dataset = DblpDataset(root_unsupervised, data_type='unsupervised',transform=graphTransform(args.aug_ratio,args.aug,args.mode))
        from dblp_model import GNN
        
    elif args.dataset == 'chem':
        root_supervised='../GMPT-master/dataset/Chem/dataset/'
        # root_unsupervised = dir+'data/chem/chembl_filtered'
        root_unsupervised ='dataset/chem/chembl_filtered'
        dataset = MoleculeDataset(root_unsupervised,  data_type='chembl_filtered',transform=graphTransform(args.aug_ratio,args.aug,args.mode))
        from chem_model import GNN
        
    print(dataset)

    args.node_fea_dim,args.edge_fea_dim = dataset[0].x.shape[1],dataset[0].edge_attr.shape[1]

    print(args)

    loader=DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    model=MGE(args,gnn).to(args.device)
    print(model)
    # model.from_pretrained('./res/bio/bio_gin_1_gnn.pkl')
    # set up optimizer01
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)
    train(args, model, loader, optimizer)


if __name__ == "__main__":
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of meta-learning-like pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for parent tasks (default: 64)')
    # parser.add_argument('--node_batch_size', type=int, default=1,
    #                     help='input batch size for parent tasks (default: 3)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')

    # gnn setting
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--graph_pooling', type=str, default="mean")
    parser.add_argument('--model_file', type=str, default='bio', help='filename to output the pre-trained model')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    # meta-learning settings
    parser.add_argument('--weight', type=float, default=1.0,
                        help='weight (default: 1.0)')
    parser.add_argument('--aug_ratio', type=float, default=0.2)
    parser.add_argument('--mode', type=str, default="degree")
    parser.add_argument('--aug', type=str, default="node_drop")
    # dataset settings
    parser.add_argument('--dataset', type=str, default='bio',
                        help='dataset name (bio; dblp; chem)')
    parser.add_argument('--node_fea_dim', type=int, default=2,
                        help='node feature dimensions (BIO: 2; DBLP: 10; CHEM: ))')
    parser.add_argument('--edge_fea_dim', type=int, default=9,
                        help='edge feature dimensions (BIO: 9; DBLP: 1; CHEM: ))')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    # device = torch.device("cpu")
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    args.device = device
    # print(args)
    main(args)

