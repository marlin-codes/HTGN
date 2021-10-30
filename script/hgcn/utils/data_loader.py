from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import numpy as np
import torch
import torch_geometric.transforms as T
import os


def read_label(dir, task='node_classification'):
    if task == 'node_classification':
        f_path = dir + os.sep + 'labels.txt'
        fin_labels = open(f_path)
        labels = []
        node_id_mapping = dict()
        for new_id, line in enumerate(fin_labels.readlines()):
            old_id, label = line.strip().split()
            labels.append(int(label))
            node_id_mapping[old_id] = new_id
        fin_labels.close()
    else:
        labels = None
        nodes = []
        with open(dir + os.sep + 'edges.txt') as ef:
            for line in ef.readlines():
                nodes.extend(line.strip().split()[:2])
        nodes = sorted(list(set(nodes)))
        node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(nodes)}
    return labels, node_id_mapping


def read_edges(dir, node_id_mapping):
    edges = []
    fin_edges = open(dir + os.sep + 'edges.txt')
    for line in fin_edges.readlines():
        node1, node2 = line.strip().split()[:2]
        edges.append([node_id_mapping[node1], node_id_mapping[node2]])
    fin_edges.close()
    return edges


def split_dataset(n_samples, val_ratio, test_ratio, stratify=None):
    train_indices, test_indices = train_test_split(list(range(n_samples)), test_size=test_ratio, stratify=stratify,
                                                   random_state=1234)
    train_indices, val_indices = train_test_split(list(np.unique(train_indices)), test_size=val_ratio,
                                                  stratify=stratify, random_state=1234)

    train_mask = get_mask(train_indices, n_samples)
    val_mask = get_mask(val_indices, n_samples)
    test_mask = get_mask(test_indices, n_samples)
    return train_mask, val_mask, test_mask


def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return torch.tensor(mask, dtype=torch.long)


def loader(dataset, dir='../sdata'):
    dir = dir + os.sep + dataset
    labels, node_id_mapping = read_label(dir)
    edges = read_edges(dir, node_id_mapping)
    edge_index = torch.tensor(edges, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    data = Data(x=None, y=y, edge_index=edge_index.transpose(1, 0), num_nodes=y.size(0), train_mask=None, val_mask=None,
                test_mask=None, num_classes=None)
    T.OneHotDegree(y.size(0))(data)
    normalized_x = T.NormalizeFeatures()
    data = normalized_x(data)
    data.train_mask, data.val_mask, data.test_mask = split_dataset(data.num_nodes, val_ratio=0.1, test_ratio=0.1)
    data.num_classes = torch.unique(y).size(0)
    print('==' * 20)
    print('dataset:\n\t num nodes:{}\n\t num edges:{}\n\t feature dim:{}\n\t'.format(data.num_nodes, data.num_edges,
                                                                                     data.num_features))
    return data


if __name__ == '__main__':
    loader('usa')
