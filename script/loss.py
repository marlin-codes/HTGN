import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from script.config import args
# from script.utils.util import negative_sampling
from script.hgcn.manifolds import PoincareBall
from torch_geometric.utils import negative_sampling
from script.utils.util import logger

device = args.device

EPS = 1e-15
MAX_LOGVAR = 10


class ReconLoss(nn.Module):
    def __init__(self, args):
        super(ReconLoss, self).__init__()
        self.negative_sampling = negative_sampling
        self.sampling_times = args.sampling_times
        self.r = 2.0
        self.t = 1.0
        self.sigmoid = True
        self.manifold = PoincareBall()
        self.use_hyperdecoder = args.use_hyperdecoder and args.model == 'HTGN'
        logger.info('using hyper decoder' if self.use_hyperdecoder else "not using hyper decoder")

    @staticmethod
    def maybe_num_nodes(index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

    def decoder(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def hyperdeoder(self, z, edge_index):
        def FermiDirac(dist):
            probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
            return probs

        edge_i = edge_index[0]
        edge_j = edge_index[1]
        z_i = torch.nn.functional.embedding(edge_i, z)
        z_j = torch.nn.functional.embedding(edge_j, z)
        dist = self.manifold.sqdist(z_i, z_j, c=1.0)
        return FermiDirac(dist)

    def forward(self, z, pos_edge_index, neg_edge_index=None):
        decoder = self.hyperdeoder if self.use_hyperdecoder else self.decoder
        pos_loss = -torch.log(
            decoder(z, pos_edge_index) + EPS).mean()
        if neg_edge_index == None:
            neg_edge_index = negative_sampling(pos_edge_index,
                                               num_neg_samples=pos_edge_index.size(1) * self.sampling_times)
        neg_loss = -torch.log(1 - decoder(z, neg_edge_index) + EPS).mean()

        return pos_loss + neg_loss

    def predict(self, z, pos_edge_index, neg_edge_index):
        decoder = self.hyperdeoder if self.use_hyperdecoder else self.decoder

        pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
        neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = decoder(z, pos_edge_index)
        neg_pred = decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAEloss(ReconLoss):
    def __init__(self, args):
        super(VGAEloss, self).__init__(args)

    def kl_loss(self, mu=None, logvar=None):
        mu = self.__mu__ if mu is None else mu
        logvar = self.__logvar__ if logvar is None else logvar.clamp(
            max=MAX_LOGVAR)
        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))

    def forward(self, x, pos_edge_index, neg_edge_index):
        z, mu, logvar = x
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        reconloss = pos_loss + neg_loss
        klloss = (1 / z.size(0)) * self.kl_loss(mu=mu, logvar=logvar)

        return reconloss + klloss
