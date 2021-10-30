import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from script.models.BaseModel import BaseModel

MAX_LOGSTD = 10


# refer to: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gat.py
class DGAT(BaseModel):
    def __init__(self, args):
        super(DGAT, self).__init__(args)
        self.layer1 = GATConv(2 * args.nhid, args.nhid // 2, args.heads, dropout=args.dropout)
        self.layer2 = GATConv(args.nhid // 2 * args.heads, args.nhid, heads=1, dropout=args.dropout, concat=False)
        self.dropout1 = args.dropout
        self.dropout2 = args.dropout
        self.act = F.elu


# refer to https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py
class DGCN(BaseModel):
    def __init__(self, args):
        super(DGCN, self).__init__(args)
        self.layer1 = GCNConv(2 * args.nhid, 2 * args.nhid)
        self.layer2 = GCNConv(2 * args.nhid, args.nhid)
        self.dropout1 = 0
        self.dropout2 = args.dropout
        self.act = F.relu
