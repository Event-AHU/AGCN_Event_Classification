# -- coding: utf-8 --**
# train the Run EV-Gait-3DGraph model

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import argparse
import pdb
from tqdm import tqdm
import os
import torchvision
import torch.distributed as dist
from torch import autograd
import logging
import sys
sys.path.append("..")
from config import Config
from models.dual_model import Net as dual_model
from datasets.dual_graph_dataset import EV_Gait_3DGraph_Dataset as dual_dataset


if __name__ == '__main__':
    # pdb.set_trace()
    if not os.path.exists(Config.log_dir):
        os.makedirs(Config.log_dir)

    if not os.path.exists(Config.model_dir):
        os.makedirs(Config.model_dir)

    parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda", default="0", help="The GPU ID")
    parser.add_argument("--epoch", default=150, type=int, help="The GPU ID")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--num_works", default=16, type=int)
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(filename=Config.graph_train_log_path, level=logging.DEBUG)

    dist.init_process_group(backend='nccl')
    args.nprocs = torch.cuda.device_count()
    model = dual_model()

    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomScale([0.96, 1]), T.RandomTranslate(0.001)])

    train_dataset = dual_dataset(
        Config.DVS128_root_dir, mode='train',split='dir',transform=train_data_aug
    )
    num_samples = len(train_dataset)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_works, pin_memory=True, drop_last=True)
    
    test_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomScale([0.999, 1])])
    test_dataset = dataset_factory(
        Config.DVS128_root_dir, mode='test',split='dir',transform=test_data_aug
    )
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,num_workers=args.num_works, pin_memory=True)
  
    # train
    
    for epoch in range(1, args.epoch+1):
        model.train()

        if epoch == 60:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.0001

        if epoch == 110:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.00001

        correct = 0
        total = 0
        total_loss = 0
        train_sampler.set_epoch(epoch)
        with tqdm(total=num_samples, desc=f'Epoch {epoch}/{args.epoch}', unit='sample') as pbar:
            for batch in train_loader:

                optimizer.zero_grad()
                label = batch[0].cuda(args.local_rank,non_blocking=True)
                data_voxel = batch[1].cuda(args.local_rank,non_blocking=True)
                data_point = batch[2].cuda(args.local_rank,non_blocking=True)
                end_point = model(data_voxel,data_point)
                loss = F.nll_loss(end_point, label)

                pred = end_point.max(1)[1]
                total += len(label)
                correct += pred.eq(label).sum()
                total_loss += float(loss)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(len(label))
        if dist.get_rank() == 0:
            logging.info("epoch: {}, train acc is {}, loss is {}".format(epoch, float(correct) / total,total_loss/len(train_loader)))
            print("epoch: {}, train acc is {},loss is {}".format(epoch, float(correct) / total,total_loss/len(train_loader)))
        
        if dist.get_rank() == 0 and epoch >=90 and epoch %10==0:    
            torch.save(model.module.state_dict(), Config.gcn_model_name.format(epoch))

    # test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for index,batch in enumerate(tqdm(test_loader)):

            label = batch[0].cuda(args.local_rank,non_blocking=True)
            data_voxel = batch[1].cuda(args.local_rank,non_blocking=True)
            end_point,d1,d2,d3= batch[2].cuda(args.local_rank,non_blocking=True)
            end_point = model(data_voxel,data_point)
            
            pred = end_point.max(1)[1]
            total += len(label)
            correct += pred.eq(label).sum().item()

        logging.info("test acc is {}".format(float(correct) / total))
        if dist.get_rank() == 0:
            print("test acc is {}".format(float(correct) / total))
            torch.save(model.module.state_dict(), Config.gcn_model_name.format(epoch))

