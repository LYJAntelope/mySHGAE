"""
GAT的模型
"""
import torch
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv

# GAT构造
class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(in_channels, 64, heads=5)
        self.conv2 = GATConv(64 * 5, out_channels)

        # self.conv1 = GATConv(in_channels, 128)
        # self.conv2 = GATConv(128, out_channels)
        # self.conv1 = GCNConv(in_channels, 128)
        # self.conv2 = GCNConv(128, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    # def decode_val(self, z, pos_edge_index, neg_edge_index):
    #     edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    #     a, b = z[edge_index[0]], z[edge_index[1]]
    #     fenzi = (a * b).sum(dim=-1)
    #     fenmu = torch.sqrt((a * a).sum(dim=-1)) * torch.sqrt((b * b).sum(dim=-1))
    #     return fenzi/fenmu

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()