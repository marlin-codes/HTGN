import torch
import numpy as np
import scipy.sparse as sp
import pickle
import torch_geometric.utils.to_dense_adj as to_dense_adj
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils.negative_sampling import negative_sampling, structured_negative_sampling
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import softmax, k_hop_subgraph


def tuple_to_array(lot):
    out = np.array(list(lot[0]))
    for i in range(1, len(lot)):
        out = np.vstack((out, np.array(list(lot[i]))))
    return out


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def to_one_directed_edge(undirected_edge):
    return torch.from_numpy(sparse_to_tuple(sp.triu(to_scipy_sparse_matrix(undirected_edge)))[0]).transpose(1, 0)


def get_edges(edge_index_list):
    undirected_edge_list = []
    for i in range(0, len(edge_index_list)):
        edge_index, _ = remove_self_loops(
            torch.from_numpy(np.array(edge_index_list[i])).transpose(1, 0))  # remove self-loop
        undirected_edge_list.append(to_undirected(edge_index))  # convert to undirected/bi-directed edge_index
    return undirected_edge_list


def get_prediction_edges(undirected_edge_index_list):
    pos_edges_list = []
    neg_edges_list = []
    for undirected_edge in undirected_edge_index_list:
        pos_edges = to_one_directed_edge(undirected_edge)

        pos_edges_list.append(pos_edges)
        neg_edges = negative_sampling(undirected_edge, num_neg_samples=pos_edges.size(1))
        neg_edges_list.append(neg_edges)
    return pos_edges_list, neg_edges_list


def get_new_prediction_edges(undirected_edge_index_list, num_nodes):
    pos_edges_list = [torch.zeros((2, 100))]  # ignore the first pos edges
    neg_edges_list = [torch.zeros((2, 100))]  # ignore the first neg edges

    for i in range(1, len(undirected_edge_index_list)):
        current_edges = to_one_directed_edge(undirected_edge_index_list[i])
        last_edges = to_one_directed_edge(undirected_edge_index_list[i - 1])

        edges_perm = current_edges[0] * num_nodes + current_edges[1]  # hash current edges
        last_edges_perm = last_edges[0] * num_nodes + last_edges[1]  # hash last edges

        perm = np.setdiff1d(edges_perm, np.intersect1d(edges_perm, last_edges_perm))  # new edges: edge-edge^last_edge
        edges_pos = np.vstack(np.divmod(perm, num_nodes)).transpose().astype(np.long)  # convert perm to indices
        edges_pos = torch.from_numpy(edges_pos).transpose(1, 0)

        pos_edges_list.append(edges_pos)
        neg_edges_list.append(negative_sampling(to_undirected(edges_pos), num_neg_samples=edges_pos.size(1)))

    return pos_edges_list, neg_edges_list
