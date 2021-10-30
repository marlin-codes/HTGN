import math
import numpy as np
import torch
from script.config import args


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def xavier_init(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = np.random.uniform(low=-init_range, high=init_range, size=shape)
    return torch.Tensor(initial)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def prepare(data, t, detection=False):
    if detection == False:
        # obtain adj index
        edge_index = data['edge_index_list'][t].long().to(args.device)  # torch edge index
        pos_index = data['pedges'][t].long().to(args.device)  # torch edge index
        neg_index = data['nedges'][t].long().to(args.device)  # torch edge index
        new_pos_index = data['new_pedges'][t].long().to(args.device)  # torch edge index
        new_neg_index = data['new_nedges'][t].long().to(args.device)  # torch edge index
        # 2.Obtain current updated nodes
        # nodes = list(np.intersect1d(pos_index.numpy(), neg_index.numpy()))
        # 2.Obtain full related nodes
        nodes = list(np.union1d(pos_index.cpu().numpy(), neg_index.cpu().numpy()))
        weights = None
        return edge_index, pos_index, neg_index, nodes, weights, new_pos_index, new_neg_index

    if detection == True:
        train_pos_edge_index = data['gdata'][t].train_pos_edge_index.long().to(args.device)

        val_pos_edge_index = data['gdata'][t].val_pos_edge_index.long().to(args.device)
        val_neg_edge_index = data['gdata'][t].val_neg_edge_index.long().to(args.device)

        test_pos_edge_index = data['gdata'][t].test_pos_edge_index.long().to(args.device)
        test_neg_edge_index = data['gdata'][t].test_neg_edge_index.long().to(args.device)
        return train_pos_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index
