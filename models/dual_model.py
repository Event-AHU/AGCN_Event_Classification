# -- coding: utf-8 --**
# GCN model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import voxel_grid, max_pool, max_pool_x, GMMConv
from timm.models.layers import Mlp, DropPath
import os
import sys
import pdb
import torch.nn.functional as F

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left_conv1 = GMMConv(in_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn1 = torch.nn.BatchNorm1d(out_channel)
        self.left_conv2 = GMMConv(out_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn2 = torch.nn.BatchNorm1d(out_channel)

        self.shortcut_conv = GMMConv(in_channel, out_channel, dim=3, kernel_size=1)
        self.shortcut_bn = torch.nn.BatchNorm1d(out_channel)

    def forward(self, data):
        data.x = F.elu(self.left_bn2(
            self.left_conv2(F.elu(self.left_bn1(self.left_conv1(data.x, data.edge_index, data.edge_attr))),
                            data.edge_index, data.edge_attr)) + self.shortcut_bn(
            self.shortcut_conv(data.x, data.edge_index, data.edge_attr)))

        return data

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_v = GMMConv(16, 64, dim=3, kernel_size=5)
        self.bn_v = torch.nn.BatchNorm1d(64)

        self.conv_p = GMMConv(1, 64, dim=3, kernel_size=5)
        self.bn_p = torch.nn.BatchNorm1d(64)

        self.block1 = ResidualBlock(64,128)
        self.block2 = ResidualBlock(128,256)
        self.block3 = ResidualBlock(256,512)

        self.block4 = ResidualBlock(64,128)
        self.block5 = ResidualBlock(128,256)
        self.block6 = ResidualBlock(256,512)

        self.fc1 = torch.nn.Linear(1024,512)
        self.bn = torch.nn.BatchNorm1d(512)
        self.drop_out = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(512,300)

    def forward(self, data_v,data_p):

        virtual_node_index_v=data_v.ptr[1:]-1
        virtual_node_index_p=data_p.ptr[1:]-1
        data_v.x = F.elu(self.bn_v(self.conv_v(data_v.x, data_v.edge_index, data_v.edge_attr)))
        data_p.x = F.elu(self.bn_p(self.conv_p(data_p.x, data_p.edge_index, data_p.edge_attr)))

        data_v = self.block1(data_v) 
        data_v = self.block2(data_v)
        data_v = self.block3(data_v)

        data_p = self.block4(data_p) 
        data_p = self.block5(data_p) 
        data_p = self.block6(data_p)

        x_v = data_v.x[virtual_node_index_v,:]
        x_p = data_p.x[virtual_node_index_p,:]
        x=torch.cat((x_v,x_p),dim=1)

        x = self.fc1(x)
        x = F.elu(x)
        x = self.bn(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)