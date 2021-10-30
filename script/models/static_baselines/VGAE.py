import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from script.config import args
import torch.nn as nn
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import negative_sampling

# refer to : https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py

EPS = 1e-15
MAX_LOGSTD = 10


class VGAENet(torch.nn.Module):
    def __init__(self):
        super(VGAENet, self).__init__()
        self.conv1 = GCNConv(2 * args.nhid, 2 * args.nhid).to(args.device)

        self.conv_mu = GCNConv(2 * args.nhid, args.nhid)
        self.conv_logstd = GCNConv(2 * args.nhid, args.nhid)
        self.x = nn.Parameter(torch.ones(args.num_nodes, args.nfeat), requires_grad=True).to(args.device)
        self.lin = nn.Linear(args.nfeat, 2 * args.nhid)

    def reset_parameters(self):
        glorot(self.x)
        glorot(self.lin.weight)

    def forward(self, edge_index, x):
        if x is None:
            x = self.x
        x = self.lin(x)
        x = F.relu(self.conv1(x, edge_index))

        self.__mu__, self.__logstd__ = self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def decoder(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.
        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def recon_loss(self, z, pos_edges, neg_edges=None):
        pos_loss = -torch.log(self.decoder(z, pos_edges) + EPS).mean()
        if neg_edges is None:
            neg_edges = negative_sampling(pos_edges, num_neg_samples=pos_edges.size(1) * args.sampling_times)
        neg_loss = -torch.log(1 - self.decoder(z, neg_edges) + EPS).mean()
        return pos_loss + neg_loss

    def compute_loss(self, z, pos_edges, neg_edges=None):
        return self.recon_loss(z, pos_edges, neg_edges) + (1. / args.num_nodes) * self.kl_loss()

    def predict(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return [roc_auc_score(y, pred), average_precision_score(y, pred)]
