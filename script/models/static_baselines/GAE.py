import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from script.config import args
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import negative_sampling, add_self_loops, remove_self_loops

EPS = 1e-15


class GCNNet(torch.nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(2 * args.nhid, 2 * args.nhid).to(args.device)
        self.conv2 = GCNConv(2 * args.nhid, args.nhid).to(args.device)
        self.x = nn.Parameter(torch.ones(args.num_nodes, args.nfeat), requires_grad=True).to(args.device)
        self.lin = nn.Linear(args.nfeat, 2 * args.nhid)

    def reset_parameters(self):
        glorot(self.x)
        glorot(self.lin.weight)

    def decoder(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def compute_loss(self, z, pos_edges, neg_edges=None):
        pos_loss = -torch.log(self.decoder(z, pos_edges) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edges, _ = remove_self_loops(pos_edges)
        pos_edges, _ = add_self_loops(pos_edges)

        if neg_edges is None:
            neg_edges = negative_sampling(pos_edges, z.size(0), num_neg_samples=pos_edges.size(1) * args.sampling_times)
        neg_loss = -torch.log(1 - self.decoder(z, neg_edges) + EPS).mean()
        return pos_loss + neg_loss

    def predict(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1)).to(args.device)
        neg_y = z.new_zeros(neg_edge_index.size(1)).to(args.device)
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = self.decoder(z, pos_edge_index)
        neg_pred = self.decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return [roc_auc_score(y, pred), average_precision_score(y, pred)]

    def forward(self, edge_index, x):
        if x is None:
            x = self.x
        x = self.lin(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
