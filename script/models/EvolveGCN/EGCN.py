import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from script.models.EvolveGCN.GCNCONV import GCNConv

from script.models.BaseModel import BaseModel


# EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs
# For more information, please refer to https://arxiv.org/abs/1902.10191

class EvolveGCN(BaseModel):
    def __init__(self, args):
        super(EvolveGCN, self).__init__(args)
        self.input_dim = args.nfeat
        self.hidden_dim = 2 * args.nhid
        self.output_dim = args.nhid
        self.egcn_type = args.egcn_type

        self.GRCU_layers1 = GRCU(self.input_dim, self.hidden_dim, egcn_type=self.egcn_type)
        self.GRCU_layers2 = GRCU(self.hidden_dim, self.output_dim, egcn_type=self.egcn_type)
        assert self.egcn_type in ['EGCNO', 'EGCNH']

    def forward(self, A_list, Nodes_list=None):
        if Nodes_list is None:
            Nodes_list = [self.feat] * len(A_list)
        Nodes_list = self.GRCU_layers1(A_list, Nodes_list, activation=True)
        Nodes_list = self.GRCU_layers2(A_list, Nodes_list, activation=False)
        return Nodes_list


class GRCU(torch.nn.Module):
    def __init__(self, input_dim, output_dim, egcn_type='EGCNH'):
        super(GRCU, self).__init__()
        self.evolve_weights = mat_GRU_cell(input_dim, output_dim, egcn_type)
        self.egcn_type = egcn_type
        self.GCN_init_weights = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_param(self.GCN_init_weights)
        self.gcn = GCNConv(input_dim, output_dim, cached=False)

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, A_list, node_embs_list, activation=True, mask_list=None):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, edge_index in enumerate(A_list):
            node_embs = node_embs_list[t]
            # first evolve the weights from the initial and use the new weights with the node_embs
            if self.egcn_type == 'EGCNO':
                GCN_weights = self.evolve_weights(GCN_weights)
            elif self.egcn_type == 'EGCNH':
                if mask_list is not None:
                    GCN_weights = self.evolve_weights(GCN_weights, node_embs, mask_list[t])
                else:
                    GCN_weights = self.evolve_weights(GCN_weights, node_embs)
            else:
                raise Exception('Unsupported EvolveGCN type!')

            node_embs = self.gcn(node_embs, edge_index, GCN_weights)
            if activation:
                node_embs = nn.functional.rrelu(node_embs)
            out_seq.append(node_embs)
        return out_seq


class mat_GRU_cell(torch.nn.Module):
    def __init__(self, input_dim, output_dim, egcn_type='EGCNH'):
        super(mat_GRU_cell, self).__init__()
        self.egcn_type = egcn_type
        self.update = mat_GRU_gate(input_dim, output_dim, torch.nn.Sigmoid())
        self.reset = mat_GRU_gate(input_dim, output_dim, torch.nn.Sigmoid())
        self.htilda = mat_GRU_gate(input_dim, output_dim, torch.nn.Tanh())
        self.choose_topk = TopK(feats=input_dim, k=output_dim)
        assert self.egcn_type in ['EGCNO', 'EGCNH']

    def forward(self, prev_Q, prev_Z=None, mask=None):
        if self.egcn_type == 'EGCNO':
            z_topk = prev_Q
        elif self.egcn_type == 'EGCNH':
            z_topk = self.choose_topk(prev_Z, mask)
        else:
            raise Exception('Unsupported EvolveGCN type!')
        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)
        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)
        new_Q = (1 - update) * prev_Q + update * h_cap
        return new_Q


def pad_with_last_val(vect, k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    if vect.size(0) == 0:
        return torch.arange(k)
    pad = torch.ones(k - vect.size(0),
                     dtype=torch.long,
                     device=device) * vect[-1]
    vect = torch.cat([vect, pad])
    return vect


class mat_GRU_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation):
        super(mat_GRU_gate, self).__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.FloatTensor(rows, rows))
        self.reset_param(self.W)
        self.U = Parameter(torch.FloatTensor(rows, rows))
        self.reset_param(self.U)
        self.bias = Parameter(torch.zeros(rows, cols))
        self.reset_param(self.bias)

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        return self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)


class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super(TopK, self).__init__()
        self.scorer = Parameter(torch.FloatTensor(feats, 1))
        self.reset_param(self.scorer)
        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask=None):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        if mask is None:
            mask = torch.zeros_like(scores) if torch.cuda.is_available() else torch.zeros_like(scores)
        scores = scores + mask
        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]
        if topk_indices.size(0) < self.k:
            topk_indices = pad_with_last_val(topk_indices, self.k)
        tanh = torch.nn.Tanh()
        if isinstance(node_embs, torch.sparse.FloatTensor) or \
                isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))
        # we need to transpose the output
        return out.t()
