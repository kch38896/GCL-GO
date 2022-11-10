import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Seq_encode(nn.Module):
    def __init__(self, nhid, dropout):
        super(Seq_encode, self).__init__()
        self.dropout = dropout

        self.FC = nn.Sequential(nn.Linear(1280, nhid[0]),
                                nn.ReLU(), nn.Dropout(self.dropout),
                                nn.Linear(nhid[0], nhid[1]*2))


    def forward(self, seq_embed):
        seq_out = self.FC(seq_embed) # seq_out size : [batch_size, 256]

        return seq_out


class Net(nn.Module):
    def __init__(self, seq_feature, go_feature, nhid, kernel_size, dropout):
        super(Net, self).__init__()
        self.dropout = dropout
        self.mlp = nn.Sequential(nn.Linear(go_feature, nhid[0]),
                                 nn.ReLU(), nn.Dropout(self.dropout),
                                 nn.Linear(nhid[0], nhid[1]))

        self.gc1 = GraphConvolution(go_feature, nhid[0])
        self.gc2 = GraphConvolution(nhid[0], nhid[1])

        self.seq_encode = Seq_encode(nhid, dropout)

    def forward(self, seq_embed, go_embed, adj):
        h_semantic = self.mlp(go_embed)

        x = F.relu(self.gc1(go_embed, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        h_structure = F.relu(self.gc2(x, adj))

        seq_out = self.seq_encode(seq_embed) # input seq size : [batch_size, 26, maxlen]
        go_out = torch.cat([h_semantic, h_structure], dim=1)
        go_out = go_out.transpose(0, 1)  # size : [256, num_go_term]

        pred = torch.mm(seq_out, go_out)
        pred = torch.sigmoid(pred)

        return h_semantic, h_structure, pred

