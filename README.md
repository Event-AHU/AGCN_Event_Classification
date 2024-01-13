# AGCN_Event_Classification 
Official PyTorch implementation of "Jiang, Bo and Yuan, Chengguo and Wang, Xiao and Bao, Zhimin and Zhu, Lin and Tian, Yonghong and Tang, Jin (2023). **Point-Voxel Absorbing Graph Representation Learning for Event Stream based Recognition**. arXiv preprint arXiv:2306.05239." 
[[Paper](https://arxiv.org/abs/2306.05239)] 



## Abstract 
Sampled point and voxel methods are usually employed to downsample the dense events into sparse ones. After that, one popular way is to leverage a graph model which treats the sparse points/voxels as nodes and adopts graph neural networks (GNNs) to learn the representation of event data. Although good performance can be obtained, however, their results are still limited mainly due to two issues. (1) Existing event GNNs generally adopt the additional max (or mean) pooling layer to summarize all node embeddings into a single graph-level representation for the whole event data representation. However, this approach fails to capture the importance of graph nodes and also fails to be fully aware of the node representations. (2) Existing methods generally employ either a sparse point or voxel graph representation model which thus lacks consideration of the complementary between these two types of representation models. To address these issues, in this paper, we propose a novel dual point-voxel absorbing graph representation learning for event stream data representation. To be specific, given the input event stream, we first transform it into the sparse event cloud and voxel grids and build dual absorbing graph models for them respectively. Then, we design a novel absorbing graph convolutional network (AGCN) for our dual absorbing graph representation and learning. The key aspect of the proposed AGCN is its ability to effectively capture the importance of nodes and thus be fully aware of node representations in summarizing all node representations through the introduced absorbing nodes. Finally, the event representations of dual learning branches are concatenated together to extract the complementary information of two cues. The output is then fed into a linear layer for event data classification. Extensive experiments on multiple event-based classification benchmark datasets fully validated the effectiveness of our framework.


## Our Proposed Framework 

<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/AGCN_Event_Classification/blob/main/figure/framework.jpg" alt="Framework"/>
</p> 

## Environment Setting 
    Python 3.8
    Pytorch 
    numpy
    scipy
    Pytorch Geometric
    torch-cluster 1.5.9
    torch-geometric 1.7.0
    torch-scatter 2.0.6
    torch-sparse 0.6.9
    torch-spline-conv 1.2.1
    spconv 2.3.3
    Matlab
## Dataset Download and Pre-processing 
    DVS128-Gait-Day:https://pan.baidu.com/s/1F3Uo-fVy1S7n5plmEXFl4w,extraction code: mk55
    ASL-DVS:https: //www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=0
    N-MNIST: https://www.garrickorchard.com/datasets/n-mnist
### Non_UPS_Downsample and Voxelization
    The ASL data set is used as an example to downsample the source file.
    cd downsample/ASL_Non-UPS_downsample
    matlab -nodesktop -nosplash -r "main, exit()"
    cd voxelization
    python raw2voxel.py
### generate absorbing graph
    cd generate_graph
    center point to absorbing graph
    python p2g.py
    voxel to absorbing graph
    python v2g.py
## Training and Testing 
    train
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 4789 --nproc_per_node=1 train_p_v.py --epoch 150 --batch_size 16 --num_works 8
    test
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 test_p_v_graph.py --model_name xxx.pkl --batch_size 8
## Experimental Results 

<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/AGCN_Event_Classification/blob/main/figure/feature_vis.jpg" alt="feature_vis"/>
</p> 


<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/AGCN_Event_Classification/blob/main/figure/top5_hardvs.jpg" alt="top5_hardvs"/>
</p> 



## Citation 
If you find this work helps your research, please cite the following paper and give us a **star**. 
```bibtex
@article{jiang2023eventAGCN,
  title={Point-Voxel Absorbing Graph Representation Learning for Event Stream based Recognition},
  author={Jiang, Bo and Yuan, Chengguo and Wang, Xiao and Bao, Zhimin and Zhu, Lin and Tian, Yonghong and Tang, Jin},
  journal={arXiv preprint arXiv:2306.05239},
  year={2023}
}
```
if you have any problems with this work, please leave an issue. 
