"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax, add_self_loops
from torch_scatter import scatter, scatter_add
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot, zeros
from script.hgcn.manifolds import PoincareBall
import itertools


class HGATConv(nn.Module):
    """
    Hyperbolic graph convolution layer.。
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, act=F.leaky_relu,
                 dropout=0.6, att_dropout=0.6, use_bias=True, heads=2, concat=False):
        super(HGATConv, self).__init__()
        out_features = out_features * heads
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout=dropout, use_bias=use_bias)
        self.agg = HypAttAgg(manifold, c_in, out_features, att_dropout, heads=heads, concat=concat)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold
        self.c_in = c_in

    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HGCNConv(nn.Module):
    """
    Hyperbolic graph convolution layer, from hgcn。
    """

    def __init__(self, manifold, in_features, out_features, c_in=1.0, c_out=1.0, dropout=0.6, act=F.leaky_relu,
                 use_bias=True):
        super(HGCNConv, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout=dropout)
        self.agg = HypAgg(manifold, c_in, out_features, bias=use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold
        self.c_in = c_in

    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout=0.6, use_bias=True):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HypAggAtt(MessagePassing):
    """
    Hyperbolic aggregation layer using degree.
    """

    def __init__(self, manifold, c, out_features, bias=True):
        super(HypAggAtt, self).__init__()
        self.manifold = manifold
        self.c = c
        self.use_bias = bias
        self.mlp = nn.Sequential(nn.Linear(out_features * 2, 1))

    def forward(self, x, edge_index=None):
        x_tangent = self.manifold.logmap0(x, c=self.c)

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        edge_i = edge_index[0]
        edge_j = edge_index[1]
        x_j = torch.nn.functional.embedding(edge_j, x_tangent)
        x_i = torch.nn.functional.embedding(edge_i, x_tangent)

        norm = self.mlp(torch.cat([x_i, x_j], dim=1))
        norm = softmax(norm, edge_i, x_i.size(0)).view(-1, 1)
        support = norm.view(-1, 1) * x_j
        support_t_curv = scatter(support, edge_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
        output = self.manifold.proj(self.manifold.expmap0(support_t_curv, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAgg(MessagePassing):
    """
    Hyperbolic aggregation layer using degree.
    """

    def __init__(self, manifold, c, out_features, bias=True):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.manifold = PoincareBall()
        self.c = c
        self.use_bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        zeros(self.bias)
        self.mlp = nn.Sequential(nn.Linear(out_features * 2, 1))

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index=None):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        edge_index, norm = self.norm(edge_index, x.size(0), dtype=x.dtype)
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_j = torch.nn.functional.embedding(node_j, x_tangent)
        support = norm.view(-1, 1) * x_j
        support_t = scatter(support, node_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAttAgg(MessagePassing):
    def __init__(self, manifold, c, out_features, att_dropout=0.6, heads=1, concat=False):
        super(HypAttAgg, self).__init__()
        self.manifold = manifold
        self.dropout = att_dropout
        self.out_channels = out_features // heads
        self.negative_slope = 0.2
        self.heads = heads
        self.c = c
        self.concat = concat
        self.att_i = Parameter(torch.Tensor(1, heads, self.out_channels), requires_grad=True)
        self.att_j = Parameter(torch.Tensor(1, heads, self.out_channels), requires_grad=True)
        glorot(self.att_i)
        glorot(self.att_j)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x.size(self.node_dim))

        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]

        x_tangent0 = self.manifold.logmap0(x, c=self.c)  # project to origin
        x_i = torch.nn.functional.embedding(edge_index_i, x_tangent0)
        x_j = torch.nn.functional.embedding(edge_index_j, x_tangent0)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=x_i.size(0))
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        support_t = scatter(x_j * alpha.view(-1, self.heads, 1), edge_index_i, dim=0)

        if self.concat:
            support_t = support_t.view(-1, self.heads * self.out_channels)
        else:
            support_t = support_t.mean(dim=1)
        support_t = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)

        return support_t

    '''
    def forward(self, x, edge_index):
        x_tangent0 = self.manifold.logmap0(x, c=self.c)  # project to origin
        out = self.propagate(edge_index, x=x_tangent0, num_nodes=x.size(0),original_x=x)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        out = self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)
        return out
    def message(self, edge_index_i, x_i, x_j, num_nodes,original_x_i, original_x_j):
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)
        if False:  # Compute distance
            dist = self.manifold.dist(original_x_i, original_x_j, self.c)
            dist = softmax(dist, edge_index_i, num_nodes).reshape(-1, 1)
            alpha = alpha * dist

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)
    '''


# refer to: https://github.com/ferrine/hyrnn/blob/master/hyrnn/nets.py
class HypGRU(nn.Module):
    def __init__(self, args):
        super(HypGRU, self).__init__()
        self.manifold = PoincareBall()
        self.nhid = args.nhid
        self.weight_ih = Parameter(torch.Tensor(3 * args.nhid, args.nhid), requires_grad=True).to(args.device)
        self.weight_hh = Parameter(torch.Tensor(3 * args.nhid, args.nhid), requires_grad=True).to(args.device)
        if args.bias:
            bias = nn.Parameter(torch.zeros(3, args.nhid) * 1e-5, requires_grad=False)
            self.bias = self.manifold.expmap0(bias).to(args.device)
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.nhid)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, hyperx, hyperh):
        out = self.mobius_gru_cell(hyperx, hyperh, self.weight_ih, self.weight_hh, self.bias)
        return out

    def mobius_gru_cell(self, input, hx, weight_ih, weight_hh, bias, nonlin=None, ):
        W_ir, W_ih, W_iz = weight_ih.chunk(3)
        b_r, b_h, b_z = bias
        W_hr, W_hh, W_hz = weight_hh.chunk(3)

        z_t = self.manifold.logmap0(self.one_rnn_transform(W_hz, hx, W_iz, input, b_z)).sigmoid()
        r_t = self.manifold.logmap0(self.one_rnn_transform(W_hr, hx, W_ir, input, b_r)).sigmoid()

        rh_t = self.manifold.mobius_pointwise_mul(r_t, hx)
        h_tilde = self.one_rnn_transform(W_hh, rh_t, W_ih, input, b_h)

        if nonlin is not None:
            h_tilde = self.manifold.mobius_fn_apply(nonlin, h_tilde)
        delta_h = self.manifold.mobius_add(-hx, h_tilde)
        h_out = self.manifold.mobius_add(hx, self.manifold.mobius_pointwise_mul(z_t, delta_h))
        return h_out

    def one_rnn_transform(self, W, h, U, x, b):
        W_otimes_h = self.manifold.mobius_matvec(W, h)
        U_otimes_x = self.manifold.mobius_matvec(U, x)
        Wh_plus_Ux = self.manifold.mobius_add(W_otimes_h, U_otimes_x)
        return self.manifold.mobius_add(Wh_plus_Ux, b)

    def mobius_linear(self, input, weight, bias=None, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None):
        if hyperbolic_input:
            output = self.manifold.mobius_matvec(weight, input)
        else:
            output = torch.nn.functional.linear(input, weight)
            output = self.manifold.expmap0(output)
        if bias is not None:
            if not hyperbolic_bias:
                bias = self.manifold.expmap0(bias)
            output = self.manifold.mobius_add(output, bias)
        if nonlin is not None:
            output = self.manifold.mobius_fn_apply(nonlin, output)
        output = self.manifold.project(output)
        return output
