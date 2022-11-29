from abc import ABC

import torch
import torch.nn as nn
import torch_cluster
import torch_geometric
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
import torch
from torch.nn import Linear as Lin, BatchNorm1d as BN


class RNNAgg(MessagePassing, ABC):
    def __init__(self, in_dim, out_dim):
        super(RNNAgg, self).__init__('add')
        self.weight = torch.nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.batch = None
        self.lstm = nn.LSTM(in_dim, out_dim // 2, bidirectional=True, num_layers=2, batch_first=True)

    def forward(self, x, assign_index, batch):
        self.batch = batch
        m_size = x.size(0)
        global_x = self.propagate(assign_index, size=(m_size, m_size), x=x)
        return global_x

    def message(self, x_i, edge_index):
        return x_i

    def update(self, inputs):
        x, mask = torch_geometric.utils.to_dense_batch(x=inputs, batch=self.batch)
        x, _ = self.lstm(x)
        x = x[mask]
        return x
