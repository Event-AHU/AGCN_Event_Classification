import os
import pdb
import csv
import numpy as np
import cv2
import torch
import random
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from spconv.pytorch.utils import PointToVoxel

def calculate_edges(data, r=5):
    # threshold of radius
    d = 32
    # scaling factor to tune the difference between temporal and spatial resolution
    alpha = 1
    beta = 1
    data_size = data.shape[0]
    # max number of edges is 1000000,
    edges = np.zeros([100000, 2])
    # get t, x,
    points = data[:, 0:3]

    row_num = 0
    edge_sum = 0
    #300 302 245
    for i in range(data_size - 1):
        count = 0
        distance_matrix = points[i + 1 : data_size + 1, 0:3]
        distance_matrix[:, 1:3] = distance_matrix[:, 1:3] - points[i, 1:3]
        distance_matrix[:, 0] = distance_matrix[:, 0] - points[i, 0]
        distance_matrix = np.square(distance_matrix)
        distance_matrix[:, 0] *= alpha
        distance_matrix[:, 1:3] *= beta
        # calculate the distance of each pair of events
        distance = np.sqrt(np.sum(distance_matrix, axis=1))
        index = np.where(distance <= r)

        # save the edges
        if index:
            index = index[0].tolist()
            for id in index:
                edges[row_num, 0] = i
                edges[row_num + 1, 1] = i
                edges[row_num, 1] = int(id) + i + 1
                edges[row_num + 1, 0] = int(id) + i + 1
                row_num = row_num + 2
                count = count + 1
                edge_sum = edge_sum +2
                if count > d:
                    break
        if edge_sum > 40000:
            break
    edges = edges[~np.all(edges == 0, axis=1)]
    edges = np.transpose(edges)

    # add virtual edges:
    virtual_index=data_size
    virtual_edges=np.zeros([2*data_size, 2])
    row=0
    for i in range(data_size):
        virtual_edges[row,0]=virtual_index
        virtual_edges[row,1]=i
        virtual_edges[row+1,0]=i
        virtual_edges[row+1,1]=virtual_index
        row = row+2
    virtual_edges = np.transpose(virtual_edges)
    new_edges = np.hstack((edges,virtual_edges))
    return new_edges

# get polarity as the feature of the node
def extract_feature(data):
    data_size = data.shape[0]
    feature = np.zeros([data_size, 1])
    for i in range(data_size):
        if data[i, 3] == 1:
            feature[i, 0] = +1
        else:
            feature[i, 0] = -1
    virtual_feature=np.ones([1, 1])
    new_feature=np.vstack((feature,virtual_feature))
    return new_feature


def extract_position(data):
    data_size = data.shape[0]
    position = np.zeros([data_size, 3])
    for i in range(data_size):
        position[i, :] = data[i, 0:3]
    p1 = int(position[:,0].sum()/data_size)
    p2 = int(position[:,1].sum()/data_size)
    p3 = int(position[:,2].sum()/data_size)
    virtual_position = np.array([p1,p2,p3])
    new_position=np.vstack((position,virtual_position))
    return new_position

def generate_graph(data):
    feature = extract_feature(data)
    position = extract_position(data)
    edges = calculate_edges(data,5)
 
    return feature,position,edges

if __name__ == '__main__':
    data_path = '/amax/DATA/yuanchengguo/celex_action_recognition/data/ASL/ASL_Nonuniform_downsample'
    save_path = '/amax/DATA/yuanchengguo/celex_action_recognition/data/ASL'
    device = torch.device("cuda:0")

    class_dir = os.listdir(data_path)
    # class_dir.sort(key=lambda x:x.split('.')[0])
    # label_dict = {lbl: i for i, lbl in enumerate(class_dir)}
    for classID in range(len(class_dir)):
        cls = class_dir[classID]
        label = classID
        cls_name = cls
        fileLIST = os.listdir(os.path.join(data_path, cls))
        if not os.path.exists(os.path.join(save_path,'p2g_r5', cls_name)):
            os.makedirs(os.path.join(save_path,'p2g_r5',  cls_name))
        for FileID in tqdm(range(len(fileLIST))):
            file_Name = fileLIST[FileID]
            save_name = file_Name.split('.')[0]
            if os.path.exists(os.path.join(save_path,'p2g_r5', cls_name, '{}.pt'.format(save_name))):
                continue
            read_path = os.path.join(data_path, cls, file_Name)
            file = sio.loadmat(read_path)

            data = file["points"]
            feature,position,edges = generate_graph(data)

            feature = torch.tensor(feature)[:, :].float()
            edge_index = torch.tensor(
                np.array(edges).astype(np.int32), dtype=torch.long
            )
            pos = torch.tensor(np.array(position), dtype=torch.float32)
            label_idx = torch.tensor(int(label), dtype=torch.long)
            data = Data(
                x=feature, edge_index=edge_index, pos=pos, y=label_idx.unsqueeze(0)
            )

            G_save_path =os.path.join(save_path,'p2g_r5', cls_name, '{}.pt'.format(save_name))

            torch.save(data, G_save_path)