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

import logging
import sys
sys.path.append("..")
from config import Config
from models.dual_model import Net as dual_model
from datasets.dual_graph_dataset import EV_Gait_3DGraph_Dataset as dual_dataset



if __name__ == '__main__':
    if not os.path.exists(Config.log_dir):
        os.makedirs(Config.log_dir)

    if not os.path.exists(Config.model_dir):
        os.makedirs(Config.model_dir)

    parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda", default="0", help="The GPU ID")
    parser.add_argument("--model_name", default="Test_EV_Gait_3DGraph.pkl")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--num_works", default=8, type=int)
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
    args = parser.parse_args()

    # seed_everything(1234)
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(filename=Config.graph_train_log_path, level=logging.DEBUG)

    dist.init_process_group(backend='nccl')
    args.nprocs = torch.cuda.device_count()

    model = dual_model()
    # pdb.set_trace()
    model.load_state_dict(torch.load(os.path.join(Config.model_dir, args.model_name),map_location='cpu'),True)
    torch.cuda.set_device(args.local_rank)
    
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
    test_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomScale([0.99999, 1])])
    test_dataset = dual_dataset(
        Config.Har_root_dir, mode='test',split='txt',transform=test_data_aug
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,num_workers=args.num_works, pin_memory=True)

    # test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for index,batch in enumerate(tqdm(test_loader)):

            label = batch[0].cuda(args.local_rank,non_blocking=True)
            data_voxel = batch[1].cuda(args.local_rank,non_blocking=True)
            data_point = batch[2].cuda(args.local_rank,non_blocking=True)
            end_point = model(data_voxel,data_point)

            pred = end_point.max(1)[1]
            total += len(label)
            correct += pred.eq(label).sum().item()
        logging.info("test acc is {}".format(float(correct) / total))
        print("test acc is {}".format(float(correct) / total))
